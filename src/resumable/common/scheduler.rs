//! [`Scheduler`]: worker set shared by every flavor, plus the worker idle
//! loop. Flavor-specific entry points live in
//! [`stackful::init::init`](crate::resumable::stackful::init::init) and
//! [`stackless::scheduler::run_async`](crate::resumable::stackless::scheduler::run_async).

use std::alloc::Layout;
use std::sync::atomic::Ordering;

use crate::traits::common::TlsSlot;
use crate::traits::stackful::ThreadSystem;
use crate::resumable::common::deque::{Steal, WorkerRunQueue};
use crate::resumable::common::external_queue::ExternalQueue;
use crate::resumable::common::system::{RunnableItem, SchedulerSystem, WorkerSystem};
use crate::resumable::common::worker::{LocalQueue, UltWorker};

/// State shared by all workers of one scheduler instance. Worker-layer
/// (`S: WorkerSystem`): only the pools/run-queue/external-queue axis, no
/// dispatch (`execute`/`free_finished_desc`) — shared by stackful-only,
/// dual, and (eventually) stackless-only systems alike — only
/// [`init`](crate::resumable::stackful::init::init) (the stackful entry
/// point) needs the stackful extension.
pub struct Scheduler<S: WorkerSystem> {
    pub(crate) workers: Box<[UltWorker<S>]>,
    /// Cloneable stealer handles, one per worker, indexed the same as
    /// `workers`. A thief reaches a victim's run queue exclusively through
    /// this table now, never through `workers[victim]` itself — `UltWorker`
    /// is full of `Cell` fields whose `unsafe impl Sync` justification is
    /// "only the owning base thread touches these", so keeping thieves off
    /// it entirely (rather than merely disciplined about which field they
    /// touch) turns that justification from a convention into something
    /// structural. Populated once, at construction, by mapping over
    /// `workers` (see the two `init`/`run_async` call sites) — never
    /// mutated afterward.
    pub(crate) stealers: Box<[<S::RunQueue as WorkerRunQueue<S::SuspendedToken>>::Stealer]>,
    pub(crate) finished: std::sync::atomic::AtomicBool,
    pub(crate) external_queue: S::ExternalQueue,
    /// Per-task ULT stack size for the stackful `spawn`/`fork_parent_first`
    /// paths and the scheduler loop's own stack, in bytes — set from
    /// [`StackfulBuilder::stack_size`](crate::traits::system::stackful::StackfulBuilder::stack_size)
    /// (default: `S::STACK_SIZE`) by [`init`](crate::resumable::stackful::init::init).
    /// `0` on a stackless-only system (`run_async`'s construction site),
    /// which has no real stacks to size — mirrors the existing precedent on
    /// [`PoolSystem::AsyncPool`](crate::resumable::common::system::PoolSystem::AsyncPool), where a stackful-only system must
    /// name an async pool it never allocates from.
    pub(crate) stack_size: usize,
    pub(crate) task_pool: S::Pool,
    /// Separate from `task_pool`: `spawn_async` needs a much smaller fixed
    /// slot size than a ULT stack, and a dual system needs both live at once
    /// (see [`PoolSystem::AsyncPool`](crate::resumable::common::system::PoolSystem::AsyncPool)).
    pub(crate) async_task_pool: S::AsyncPool,
    /// Pool backing [`crate::resumable::stackless::thread::recurse`] — see
    /// [`PoolSystem::RecursionPool`](crate::resumable::common::system::PoolSystem::RecursionPool) for why this needs none of
    /// [`task_pool`](Self::task_pool)'s `TaskDesc`/stealing-specific
    /// construction, just the same fixed-slot free-list mechanism.
    pub(crate) recursion_pool: S::RecursionPool,
}

/// Threshold `S::RecursionPool` is configured with — reuses
/// [`PoolSystem::ASYNC_POOL_SIZE`](crate::resumable::common::system::PoolSystem::ASYNC_POOL_SIZE) (both are "small `Future` storage"
/// budgets) rather than adding a second, near-duplicate per-system
/// constant; align 16 covers realistic recursive-`async fn` frames without
/// needing its own knob either.
pub(crate) fn recursion_pool_threshold<S: SchedulerSystem>() -> Layout {
    Layout::from_size_align(S::ASYNC_POOL_SIZE, 16)
        .expect("cmpth: ASYNC_POOL_SIZE not a valid Layout size for align 16")
}

unsafe impl<S: SchedulerSystem> Send for Scheduler<S> {}
unsafe impl<S: SchedulerSystem> Sync for Scheduler<S> {}

/// `shared` is an explicit parameter, not read back out of `wk` — every
/// caller already holds the `Scheduler<S>` it just built or was handed.
pub(crate) fn worker_loop<S>(wk: &UltWorker<S>, shared: &Scheduler<S>)
where
    S: SchedulerSystem + WorkerSystem<Worker = UltWorker<S>>,
{
    S::worker_tls().set(wk as *const UltWorker<S> as *mut UltWorker<S>);
    // SAFETY: `root_desc` is embedded by value in `UltWorker` and only ever
    // reachable through `wk.root_desc() -> &S::Desc` (never a token) until
    // this very call, which runs once per worker at startup — this is the
    // first token ever constructed for it, so exclusivity is trivial.
    let root_task = unsafe {
        crate::resumable::common::desc::RunningTaskToken::from_raw(
            wk.root_desc() as *const _ as *mut _,
        )
    };
    wk.set_cur_task(root_task);

    worker_idle_loop(wk, shared);

    S::worker_tls().set(std::ptr::null_mut());
}

/// The idle/dispatch loop itself, factored out of [`worker_loop`] so the
/// stackful initializer (`resumable::stackful::init`) can drive it on a
/// bootstrap it sets up differently: TLS is already pointed at `wk` and
/// `cur_task` is already populated (by the child-first fork's switch shim,
/// not by the two lines above) before that caller ever reaches this
/// function, and TLS teardown happens later too (on whichever OS thread
/// ends up running the initializer's `Drop`, not necessarily this one) — so
/// neither belongs inside this shared core.
///
/// `shared` is an explicit parameter, not read back out of `wk` — every
/// caller already holds the `Scheduler<S>` it just built or was handed.
pub(crate) fn worker_idle_loop<S>(wk: &UltWorker<S>, shared: &Scheduler<S>)
where
    S: SchedulerSystem + WorkerSystem<Worker = UltWorker<S>>,
{
    let mut idle_rounds = 0u32;
    while !shared.finished.load(Ordering::Acquire) {
        if let Some(c) = wk.try_pop() {
            c.run_on(wk);
            idle_rounds = 0;
            continue;
        }
        // `try_steal` distinguishes "every victim was genuinely empty"
        // (`Steal::Empty`) from "some victim had work but it couldn't be
        // taken right now" (`Steal::Retry`, e.g. lost a CAS race) — see
        // `Steal`'s own doc comment. A `Retry` round must not count toward
        // `idle_rounds` below: there is known work nearby, so backing off
        // to `S::Base::yield_now()` on its account would be a real
        // regression, not just noise.
        let steal = wk.try_steal();
        if let Steal::Success(c) = steal {
            c.run_on(wk);
            idle_rounds = 0;
            continue;
        }
        if let Some(c) = shared.external_queue.try_pop() {
            S::SuspendedToken::from(c).run_on(wk);
            idle_rounds = 0;
            continue;
        }
        std::hint::spin_loop();
        if matches!(steal, Steal::Empty) {
            idle_rounds += 1;
            if idle_rounds & 0x3F == 0 {
                S::Base::yield_now();
            }
        }
    }
}
