//! Standalone stackful initializer: [`init`]/[`StackfulInit`]/
//! [`StackfulBuilderImpl`] — the "return from setup already running as a
//! ULT" entry point mirroring ComposableThreads'
//! `basic_scheduler<P>::initializer`
//! (`include/cmpth/wss/basic_scheduler.hpp`'s
//! `class basic_scheduler<P>::initializer`).
//!
//! # Mechanism
//!
//! The C++ original's constructor does a *child-first* fork
//! (`thread_funcs_type::fork_child_first`): the freshly forked child
//! immediately runs the scheduler loop, and the *parent* — the
//! constructor's own caller — is what gets pushed to the local deque as an
//! ordinary, stealable continuation. Control "returns" from the
//! constructor only once something (usually the very scheduler loop that
//! was just forked) pops that continuation back off and switches into it —
//! so by the time the caller's own next line runs, it is already running as
//! a schedulable ULT, migratable like any other.
//!
//! [`init`] mirrors this with this crate's own primitives:
//! [`ContextSwitcher::suspend_to_new`] is the same child-first-fork
//! primitive [`spawn`](crate::resumable::stackful::thread::spawn) uses; its
//! "child" here is the scheduler loop.
//!
//! One cmpth-rs-specific wrinkle the C++ side doesn't have to think about:
//! the stackful-only [`RunnableItem`](crate::resumable::common::system::RunnableItem)
//! impl (`resumable::stackful::worker`) unconditionally
//! records *whatever was running right before a switch* as the worker's
//! `root_cont` (the fallback `pop_or_root`
//! resumes when the local deque is empty) — C++'s equivalent
//! (`basic_worker::execute`) instead *dynamically* marks/unmarks the
//! current task as root around each switch
//! (`mark_cur_task_as_root`/`unmark_cur_task_as_root`), so it doesn't care
//! which physical stack is driving the loop. cmpth-rs has no such
//! mark/unmark hook — `is_root()` is a fixed, permanent property baked into
//! a descriptor at construction. So whichever descriptor calls `execute()`
//! in a loop (i.e. whichever stack drives `worker_idle_loop`) *must* be
//! the one flagged `is_root()`, same as every ordinary worker's own native
//! OS-thread stack in `worker_loop`.
//!
//! That fixes which side of the fork gets which role: **the scheduler
//! loop** keeps the worker's `root_desc()` identity (exactly like
//! `worker_loop`'s bootstrap — just running on a freshly, separately
//! allocated stack here instead of the OS thread's native one), and **the
//! calling OS thread's own native stack** — which must *not* be
//! `is_root()` (an `is_root()` continuation sitting in the ordinary deque
//! would get misrouted straight into the root-continuation slot instead of
//! back onto the deque, the moment anything cancels a conditional suspend
//! against it — see [`crate::resumable::stackful::worker::StackfulWorker::cond_suspend_to_sched`]) —
//! gets a purpose-built non-root, no-real-stack descriptor
//! (`InitState::caller_desc`) instead of the worker's `root_desc()`.
//! `S::Desc::alloc_with(StackMem::None, false)` already produces exactly
//! that combination (`is_root: false`, no stack storage) with no new
//! constructor needed — the same call [`new_root`](crate::resumable::common::desc::TaskDescAlloc::new_root)
//! makes internally, just with the root flag left off.
//!
//! [`StackfulInit`]'s `Drop` mirrors the C++ destructor: `suspend_to_sched`
//! parks the calling continuation (wherever it has migrated to by then —
//! "This worker may not be the root worker" applies here just as it does in
//! the C++ comment) and hands control to whichever worker's scheduler loop
//! picks it up next. Once every worker observes `finished`, worker 0's own
//! forked loop joins the other workers' OS threads and `exit_to_cont`s back
//! into the parked continuation, resuming `Drop`'s own remaining code (TLS
//! teardown) — always on the *original* calling OS thread, since a context
//! switch never changes which OS thread is executing, only which stack it
//! runs on, and the scheduler loop forked here never runs anywhere else.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::traits::common::TlsSlot;
use crate::traits::stackful::{JoinHandleLike, ThreadSystem};
use crate::traits::system::stackful::{StackfulBuilder, StackfulInitSystem};
use crate::resumable::common::deque::WorkerRunQueue;
use crate::resumable::common::desc::{HasExternalQueue, RunningTaskToken, SuspendedTaskToken, TaskDescAlloc};
use crate::resumable::common::external_queue::ExternalQueue;
use crate::resumable::common::pool::{DescPool, DynamicPool};
use crate::resumable::common::scheduler::{recursion_pool_threshold, worker_idle_loop, worker_loop, Scheduler};
use crate::resumable::common::system::WorkerSystem;
use crate::resumable::common::stack::{StackAlloc as _, StackMem, UltStackMemory as _};
use crate::resumable::common::thread::{align_down, JoinHandle};
use crate::resumable::common::worker::{LocalQueue, UltWorker, WorkerOps};
use crate::resumable::stackful::desc::{HasCtx, StackfulTaskDesc};
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::stackful::worker::{ContextSwitcher, StackfulWorker};

// ---------------------------------------------------------------------------
// InitState
// ---------------------------------------------------------------------------

/// State shared between `StackfulInit::drop` and the scheduler-loop closure
/// built in [`init`]: heap-allocated (`Arc`) rather than reached through a
/// raw `*const StackfulInit<S>`, specifically so neither ever needs
/// `StackfulInit`'s own address to be stable — `init` returns `StackfulInit`
/// by value, which its caller is free to move (into a local variable, a
/// struct field, ...) after construction.
struct InitState<S: StackfulSchedulerSystem>
where
    S::Desc: StackfulTaskDesc,
{
    /// Pseudo-descriptor for the *calling OS thread's own native stack* —
    /// the "parent" side of `init`'s child-first fork. `is_root() ==
    /// false` (unlike a worker's own `root_desc()`) and has no real stack
    /// storage of its own (`StackMem::None`, same as a real `root_desc()`)
    /// — see the module doc comment for why it can't just reuse
    /// `root_desc()` itself. Embedded here (not in `UltWorker`, which
    /// already has its own permanent `root_desc()` playing the *scheduler
    /// loop's* identity) purely so it gets a stable heap address before
    /// anything ever takes a pointer to it.
    caller_desc: S::Desc,
    /// Populated by `StackfulInit::drop`'s `suspend_to_sched` callback (the
    /// continuation to resume once the scheduler loop tears down);
    /// consumed by the scheduler-loop closure's final `exit_to_cont`.
    /// `Cell`, not an atomic slot: both sides run on the same OS thread
    /// this ever touches (the original calling one — see this module's
    /// doc comment), just at different times, sequenced by `finished`'s own
    /// Release (in `drop`, *after* this is populated) / Acquire (in
    /// `worker_idle_loop`) pair — no separate synchronization is needed
    /// here.
    dest_cont: Cell<Option<SuspendedTaskToken<S::Desc>>>,
}

// ---------------------------------------------------------------------------
// StackfulInit — the RAII guard
// ---------------------------------------------------------------------------

/// RAII guard returned by [`StackfulBuilderImpl::init`] /
/// [`StackfulBuilder::init`]. See the module doc comment for the full
/// fork/suspend/exit mechanism, and [`StackfulBuilder::init`]'s own doc
/// comment for the panic-across-`Drop` and `!Send`-across-suspension
/// hazards.
pub struct StackfulInit<S: StackfulSchedulerSystem>
where
    S::Desc: StackfulTaskDesc,
{
    shared: Arc<Scheduler<S>>,
    state: Arc<InitState<S>>,
    /// `S::ExternalQueue::run_service`, spawned as an ordinary task once
    /// `NEEDS_SERVICE` says it's needed (e.g. `PollerUltQueue`'s poller
    /// ULT) — `None` when the external queue has no service to run (e.g.
    /// the default `StealPathQueue`). `Drop` stops and joins it *before*
    /// the rest of teardown, so the closure below can safely hold a strong
    /// `Arc<Scheduler<S>>` with no cycle: by the time the scheduler itself
    /// could be dropped, this handle is already gone.
    service_handle: Option<JoinHandle<S, ()>>,
}

/// Standalone init: bring up `num_workers` workers, then return with the
/// calling OS thread's own continuation already running as an ordinary,
/// stealable task on worker 0's deque. See the module doc comment.
pub fn init<S>(num_workers: usize, stack_size: usize) -> StackfulInit<S>
where
    S: StackfulSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>,
    S::Desc: StackfulTaskDesc,
{
    assert!(num_workers >= 1, "need at least one worker");
    assert!(
        S::Worker::current().is_none(),
        "cmpth: nested init() of the same system on one thread"
    );

    // Resolve this system's TLS slot index now, single-threaded, before any
    // worker OS thread starts — see `TlsSlot::warm_up`.
    S::worker_tls().warm_up();

    let workers: Box<[UltWorker<S>]> = (0..num_workers).map(UltWorker::new).collect();
    let stealers = workers.iter().map(|w| w.deque.stealer()).collect();
    let shared = Arc::new(Scheduler {
        workers,
        stealers,
        finished: std::sync::atomic::AtomicBool::new(false),
        external_queue: S::ExternalQueue::default(),
        stack_size,
        task_pool: S::Pool::new_pool(num_workers, stack_size),
        async_task_pool: S::AsyncPool::new_pool(num_workers, S::ASYNC_POOL_SIZE),
        recursion_pool: S::RecursionPool::new(num_workers, recursion_pool_threshold::<S>()),
    });
    for w in shared.workers.iter() {
        w.bind_scheduler(Arc::as_ptr(&shared));
    }

    // Build the calling native stack's own pseudo-descriptor — see
    // `InitState::caller_desc`'s doc comment for why this can't just be
    // `wk0.root_desc()`. `commit_as_ctx`/`set_external_queue` mirror
    // `spawn`'s/`fork_parent_first`'s identical setup for any freshly built
    // ctx-bearing descriptor (a no-op for a plain stackful-only `Owned`,
    // load-bearing for a dual system's ctx/poll_fn union).
    let mut caller_desc = S::Desc::alloc_with(StackMem::None, false);
    {
        // SAFETY: `caller_desc` is a fresh local value nobody else has a
        // pointer to yet — trivially exclusive.
        let mut token = unsafe { SuspendedTaskToken::from_raw(&mut caller_desc as *mut S::Desc) };
        token.commit_as_ctx();
        token.set_external_queue(&shared.external_queue as *const _);
        let _ = token.into_raw();
    }
    let state = Arc::new(InitState { caller_desc, dest_cont: Cell::new(None) });

    // Bootstrap worker 0 on the calling OS thread: TLS + `cur_task` = the
    // native stack's own pseudo-descriptor built above (now living at a
    // stable address inside `state`). Done *before* the fork below — its
    // shim captures whatever is `cur_task` right now as the continuation to
    // publish (`prev`), so it has to already be set.
    let wk0 = &shared.workers[0];
    S::worker_tls().set(wk0 as *const UltWorker<S> as *mut UltWorker<S>);
    let caller_task = unsafe {
        // SAFETY: `state.caller_desc` was just moved into `state` above and
        // has never been wrapped in a token since — trivially exclusive.
        RunningTaskToken::from_raw(&state.caller_desc as *const S::Desc as *mut S::Desc)
    };
    wk0.set_cur_task(caller_task);

    // Start worker OS threads 1..num_workers — identical to `run`'s own
    // worker startup.
    let handles: Vec<_> = (1..num_workers)
        .map(|i| {
            let shared = Arc::clone(&shared);
            S::Base::spawn(move || worker_loop(&shared.workers[i], &shared))
        })
        .collect();

    // A real stack for the scheduler loop to run on — separate from any
    // descriptor's own stack storage, since the loop's *identity*
    // descriptor is `wk0.root_desc()`, which (like every worker's
    // `root_desc()`) never owns real stack storage of its own. Freed by
    // `Drop`ping it from the `exit_to_cont` callback below, once the loop
    // has switched off of it for good.
    let sched_stack = S::StackAlloc::alloc_stack(stack_size);
    let exec_top = align_down(sched_stack.stack_top() as usize, 16) as *mut u8;
    let root_desc_ptr = wk0.root_desc() as *const S::Desc as *mut S::Desc;

    let state2 = Arc::clone(&state);
    let shared3 = Arc::clone(&shared);
    let scheduler_loop = move |wk: &UltWorker<S>, prev: SuspendedTaskToken<S::Desc>| {
        // Running on the scheduler loop's own fresh stack now, as
        // `wk0.root_desc()` (see this module's doc comment for why).
        // `prev` is the *calling* OS thread's own continuation — publish it
        // as an ordinary, stealable task, same as C++
        // `on_fork_child_first`'s `wk.local_push_top(parent_cont)` — this
        // is the moment `init`'s caller becomes a schedulable ULT.
        wk.push(prev.into());

        // Ordinary dispatch loop, same one every worker OS thread runs,
        // until `StackfulInit::drop` sets `finished`.
        worker_idle_loop(wk, &shared3);

        // Join the other workers' OS threads — mirrors C++
        // `on_fork_root`'s `finish_workers()`.
        //
        // `dest_cont` is guaranteed populated by the time `take()` runs
        // below, but *not* because `StackfulInit::drop` writes it before
        // setting `finished` — it writes it after (the continuation only
        // exists once `suspend_to_sched` has saved it). The real guarantee
        // is the join above, and it holds either way `drop` can be reached:
        //
        // - `drop` ran on worker 0: it shares this OS thread, so its
        //   `suspend_to_sched` callback is what switched control back into
        //   this loop in the first place — the write already happened.
        // - `drop` ran on some worker k != 0: k's OS thread is one of
        //   `handles`, and it cannot return from `worker_loop` (and so
        //   cannot be joined) until the ULT running `drop` suspends. That
        //   suspension *is* the write. So joining k happens-after it.
        for h in handles {
            h.join();
        }
        let dest = state2.dest_cont.take()
            .expect("cmpth: scheduler loop finished before the initializer's Drop populated dest_cont");

        // Abandon the loop's own context and switch into the parked
        // teardown continuation — mirrors C++ `on_fork_root`'s
        // `exit_to_cont`/`on_root_exit`: free the loop's own (now
        // permanently abandoned) stack on the *destination* stack, since
        // nothing needs the old one anymore.
        wk.exit_to_cont(dest, move |_wk| {
            drop(sched_stack);
        });
    };
    // SAFETY: `root_desc_ptr` points at `wk0.root_desc()`, embedded by
    // value in `UltWorker` and, same as `worker_loop`'s identical
    // reasoning, never wrapped in a token before this — the first token
    // ever constructed for it, so exclusivity is trivial. `wk0`'s
    // `cur_task` was set to `state.caller_desc`'s token immediately above,
    // satisfying `suspend_to_new`'s implicit contract that a task is
    // currently running to be captured as `prev`.
    unsafe { wk0.suspend_to_new(exec_top, root_desc_ptr, scheduler_loop) };

    // Resumed here once something — typically the very scheduler loop just
    // forked above, but any worker's `pop_or_root` may get there first —
    // switches back into the continuation pushed above: exactly the
    // "control returns from the constructor already running as a ULT"
    // moment the C++ initializer's own constructor comment describes. May
    // be running on a different worker (migration) than the one `init` was
    // called on.
    //
    // *Now* — and only now — is `spawn` available: it needs `UltWorker::
    // current()`, which is exactly what the switch above just established.
    // If the external queue needs a service task (`PollerUltQueue`'s
    // poller ULT), spawn it as an ordinary, stealable task like any other.
    // The closure captures a clone of `shared` rather than reaching it
    // through a `Weak` — safe because `Drop` below joins this handle before
    // the scheduler can ever be torn down, so the strong `Arc` never
    // outlives it.
    let service_handle = if <S::ExternalQueue as ExternalQueue<S>>::NEEDS_SERVICE {
        let shared_for_service = Arc::clone(&shared);
        Some(crate::resumable::stackful::thread::spawn::<S, (), _>(move || {
            shared_for_service.external_queue.run_service();
        }))
    } else {
        None
    };

    StackfulInit { shared, state, service_handle }
}

impl<S> Drop for StackfulInit<S>
where
    S: StackfulSchedulerSystem,
    S::Desc: StackfulTaskDesc,
    S::Worker: StackfulWorker<S>,
{
    fn drop(&mut self) {
        // A context switch (below) while an unwind is in flight is unsound:
        // the unwinder's in-flight state is OS-thread-local, but the ULT
        // resuming next may run on a different OS thread, or on the same
        // one only after other tasks — which may themselves unwind — have
        // run in between. `StackfulBuilder::run` never lets an unwind reach
        // here (it catches around `init`/`drop` and re-raises afterward);
        // standalone `init` callers are responsible for the same
        // discipline, and this is the last point that can still refuse
        // rather than corrupt scheduler state.
        if std::thread::panicking() {
            eprintln!(
                "cmpth: panic unwinding across a stackful initializer's Drop \
                 is unsupported (it would context-switch while unwinding); \
                 aborting. Use `Builder::run` instead of standalone `init()` \
                 if the panic needs to propagate to the caller."
            );
            std::process::abort();
        }

        // Stop and join the external queue's service task (if any) before
        // anything else: this is what makes the strong `Arc<Scheduler<S>>`
        // the spawn closure in `init` captured safe to hold — by the time
        // `finished` is set and the rest of teardown runs below, that
        // closure (and its `Arc`) is already gone, so there is no cycle to
        // worry about. A panic inside `run_service` propagates from here
        // the same way any other spawned task's panic would.
        self.shared.external_queue.stop_service();
        if let Some(h) = self.service_handle.take() {
            JoinHandleLike::join(h);
        }

        self.shared.finished.store(true, Ordering::Release);

        // This worker may not be worker 0 — the ULT `init` returned as may
        // have migrated any number of times since.
        let wk = S::Worker::current()
            .expect("cmpth: StackfulInit dropped outside any worker of its own system");

        let state = &self.state;
        let root_wk = wk.suspend_to_sched(|_wk, cont| {
            let old = state.dest_cont.replace(Some(cont));
            debug_assert!(old.is_none(), "cmpth: overwriting a live dest_cont");
        });

        // Resumed here once the scheduler loop's own `exit_to_cont` switches
        // back — always worker 0, on the *original* calling OS thread: a
        // context switch never changes which OS thread is executing, only
        // which stack it runs on, and the loop forked in `init` never runs
        // anywhere else.
        debug_assert_eq!(root_wk.num(), 0, "cmpth: initializer teardown resumed on the wrong worker");
        S::worker_tls().set(std::ptr::null_mut());
    }
}

// ---------------------------------------------------------------------------
// StackfulBuilderImpl
// ---------------------------------------------------------------------------

/// [`StackfulBuilder`] implementation shared by every `resumable`-backed
/// stackful system (blanket-derived in
/// `resumable::stackful::system` for any `S: ThreadSystem +
/// StackfulSchedulerSystem`).
pub struct StackfulBuilderImpl<S> {
    num_workers: Option<usize>,
    stack_size: Option<usize>,
    _marker: PhantomData<fn() -> S>,
}

impl<S> StackfulBuilderImpl<S> {
    pub(crate) fn new() -> Self {
        StackfulBuilderImpl { num_workers: None, stack_size: None, _marker: PhantomData }
    }
}

impl<S> StackfulBuilder<S> for StackfulBuilderImpl<S>
where
    S: StackfulSchedulerSystem + StackfulInitSystem<Init = StackfulInit<S>> + WorkerSystem<Worker = UltWorker<S>>,
    S::Desc: StackfulTaskDesc,
{
    fn workers(mut self, n: usize) -> Self {
        self.num_workers = Some(n);
        self
    }

    fn stack_size(mut self, bytes: usize) -> Self {
        self.stack_size = Some(bytes);
        self
    }

    fn init(self) -> S::Init {
        init::<S>(
            self.num_workers.unwrap_or_else(crate::os::available_parallelism),
            self.stack_size.unwrap_or(S::STACK_SIZE),
        )
    }
}
