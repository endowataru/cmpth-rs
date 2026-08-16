//! Stackful thread functions: child-first fork, exit, blocking `.join()`.
//! See [`common::thread`](crate::resumable::common::thread) for the shared
//! [`JoinHandle`] type both
//! this and [`stackless::thread`](crate::resumable::stackless::thread)
//! produce.

use std::any::Any;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::traits::stackful::{ContextPolicy, HandoffTaskDesc, JoinHandleLike, Transfer};
use crate::resumable::common::system::PoolSystem;
use crate::resumable::common::thread::{align_down, drop_stack_result, JoinHandle, StackResult};
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::common::desc::{HasExternalQueue, SuspendedTaskToken, TaskDesc, TaskDescCore, TaskExitSink};
use crate::resumable::stackful::desc::{HasCtx, StackfulTaskDesc};
use crate::resumable::common::worker::{DescWorkerOps, LocalQueue, TaskPool, WorkerOps};
use crate::resumable::stackful::worker::{BranchWarmPool, ContextSwitcher, StackfulWorker};

// ---------------------------------------------------------------------------
// spawn (child-first fork)
// ---------------------------------------------------------------------------

// Reserve space at the top of a task's stack (high addresses) for its
// closure and result slot.  The execution stack gets the rest below.
//
//   stack_top (high)
//   ┌─────────────────┐
//   │ StackResult<T>  │  ← result_addr
//   ├─────────────────┤
//   │ F               │  ← f_addr
//   ├─────────────────┤
//   │ (exec stack)    │  ← exec_top and below
//   └─────────────────┘ ← stack base
//
// A pure function of `stack_top` and the two types' layouts, so both the
// side that writes the closure in (before any switch/push) and the side
// that reads it back out (`spawn`'s child, or `branch_entry` below, which
// has no closure capture to carry it) compute the exact same addresses
// without needing an extra header or side channel.
fn branch_layout<F, T>(stack_top: usize) -> (*mut F, *mut StackResult<T>) {
    let result_layout = std::alloc::Layout::new::<StackResult<T>>();
    let f_layout = std::alloc::Layout::new::<F>();

    let result_addr = align_down(stack_top - result_layout.size(), result_layout.align());
    let f_addr = align_down(result_addr.wrapping_sub(f_layout.size()), f_layout.align().max(1));

    (f_addr as *mut F, result_addr as *mut StackResult<T>)
}

/// Spawn a ULT.  Child-first: the child starts immediately on this worker and
/// the parent's continuation is pushed to the deque for stealing.
///
/// The closure `F` and the result slot `StackResult<T>` are placed directly on
/// the child's stack, avoiding two heap allocations that the old Box-erasure
/// approach required.
pub fn spawn<S, T, F>(f: F) -> JoinHandle<S, T>
where
    S: StackfulSchedulerSystem,
    S::Worker: ContextSwitcher<S> + DescWorkerOps<S>,
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
    <S as PoolSystem>::Desc: StackfulTaskDesc,
{
    let wk = <S::Worker as WorkerOps<S>>::current().expect("cmpth: spawn called outside a worker");
    let mut token = wk.alloc_task(true);
    token.commit_as_ctx();
    token.set_external_queue(wk.external_queue() as *const _);
    let stack_top = token.as_desc().stack_top() as usize;
    let desc = token.into_raw();

    let (f_ptr, result_ptr) = branch_layout::<F, T>(stack_top);
    let exec_top = align_down(f_ptr as usize, 16) as *mut u8;

    // Write the closure onto the child's stack before switching.
    unsafe { f_ptr.write(f) };

    let child = move |wk: &S::Worker, prev: SuspendedTaskToken<S::Desc>| {
        // Running on the child's stack.  Publish the parent for stealing, run
        // the closure, then exit via exit_with_result.
        wk.push(prev.into());
        let val = catch_unwind(AssertUnwindSafe(|| unsafe { f_ptr.read() }()));
        // The closure may have suspended and resumed on a different worker,
        // so re-derive which one we're on now.
        let wk = <S::Worker as WorkerOps<S>>::current().expect("cmpth: worker vanished");
        debug_assert!(std::ptr::eq(wk.cur_task(), desc));
        exit_with_result::<S, T>(wk, wk.cur_task_ref(), result_ptr, val)
    };
    // SAFETY: `desc` was freshly allocated above and, after the token was
    // released via `into_raw`, has never been re-tokenized — still
    // exclusively owned here.
    unsafe { wk.suspend_to_new(exec_top, desc, child) };

    JoinHandle { desc, result_ptr, result_drop: drop_stack_result::<T>, _marker: PhantomData }
}

// ---------------------------------------------------------------------------
// parallel_call (fork-parent-first, pool-backed via make_context, with a
// warm-reuse cache that skips make_context on repeat calls)
// ---------------------------------------------------------------------------

/// Type-erased dispatch function for a `parallel_call` branch: the
/// type-specific half of [`branch_entry`], stored as *data* (a function
/// pointer written into a fixed stack slot) rather than baked into the
/// `make_context`-built ctx frame as machine code. See [`branch_entry`]'s
/// doc comment for why this indirection exists.
type DispatchFn = unsafe extern "C" fn(usize) -> !;

/// Size of the reserved slot (at `stack_top - DISPATCH_SLOT_SIZE`) holding
/// the branch's [`DispatchFn`].
const DISPATCH_SLOT_SIZE: usize = std::mem::size_of::<DispatchFn>();

/// Fixed budget (below the dispatch slot) reserved for `Fb` +
/// `StackResult<Rb>` when a branch is eligible for
/// [`BranchWarmPool`]'s reuse cache (see [`header_fits`]). Generous enough
/// for realistic closures (a handful of captured `Vec`s/primitives —
/// `nqueens_parallel_invoke`'s branch closures, the widest in this crate's
/// own benches, run ~65 bytes); oversized ones fall back to the fully
/// dynamic path below, exactly like every `parallel_call` did before this
/// cache existed — just never warm-cached.
const BRANCH_HEADER_BUDGET: usize = 256;

/// Whether `Fb`/`StackResult<Rb>` fit [`BRANCH_HEADER_BUDGET`]. `size`/
/// `align` are used as a conservative combined padding estimate per slot —
/// this only needs to guarantee no overlap with `exec_top`, not be tight.
/// `Layout::new` is `const`, so for any concrete `Fb`/`Rb` this reduces to
/// a compile-time-constant `bool`; the `if` reading it in [`parallel_call`]
/// is expected to fold to a single branch at codegen, though nothing here
/// depends on that for correctness — a stray runtime check would just be
/// one perfectly-predicted branch (same outcome every call at a given call
/// site).
fn header_fits<Fb, Rb>() -> bool {
    let f = std::alloc::Layout::new::<Fb>();
    let r = std::alloc::Layout::new::<StackResult<Rb>>();
    f.size() + f.align() + r.size() + r.align() + DISPATCH_SLOT_SIZE <= BRANCH_HEADER_BUDGET
}

/// Run `a` and `b`, potentially in parallel: `b` is built as a real,
/// switchable task and pushed to the ordinary run queue *without* a context
/// switch (`S::Ctx::make_context` prepares the context; nothing ever
/// switches into it here), then `a` runs inline on the caller's own,
/// unmodified stack. If `b` was never stolen, popping it back by pointer
/// identity affords running it as a plain function call — no context is
/// ever touched. If it *was* stolen, the ordinary dispatch path
/// (`RunnableItem::run_on`) already knows how to run a
/// `SuspendedTaskToken`; a `make_context`-built one is indistinguishable
/// from an ordinarily-suspended one to that machinery, so the only new code
/// needed is `branch_entry` itself, and finishing joins the existing
/// `JoinHandle` protocol verbatim.
///
/// When `Fb`/`StackResult<Rb>` fit the fixed header budget, an un-stolen
/// branch's descriptor goes to [`BranchWarmPool`] instead of back to the
/// general pool: its `ctx` was never touched (never switched into), so a
/// later call on this same worker can reuse it — same stack, same
/// `exec_top`, same cached ctx frame — skipping `alloc_task`,
/// `commit_as_ctx`, `set_external_queue`, and `make_context` entirely, down
/// to just writing the new closure and dispatch function.
///
/// `Fa`/`Ra` don't need `Send + 'static` here — `a` never leaves the
/// caller's stack — but callers (the `ScopedStackfulTaskSystem` blanket
/// impl) may still pass values satisfying stricter bounds; this function
/// simply doesn't require them.
pub fn parallel_call<S, Ra, Fa, Rb, Fb>(a: Fa, b: Fb) -> (Ra, Rb)
where
    S: StackfulSchedulerSystem,
    S::Worker: ContextSwitcher<S> + DescWorkerOps<S> + BranchWarmPool<S>,
    Fa: FnOnce() -> Ra,
    Fb: FnOnce() -> Rb + Send + 'static,
    Rb: Send + 'static,
    <S as PoolSystem>::Desc: StackfulTaskDesc,
{
    let wk = <S::Worker as WorkerOps<S>>::current().expect("cmpth: parallel_call called outside a worker");

    let fits = header_fits::<Fb, Rb>();
    let warm = if fits { wk.branch_warm_pop() } else { None };

    let (desc, f_ptr, result_ptr, dispatch_slot) = match warm {
        Some(desc) => {
            // Warm: `ctx` and the fixed-offset header are exactly as the
            // previous un-stolen round trip left them — no alloc_task, no
            // commit_as_ctx/set_external_queue (worker-stable, set once
            // when this descriptor was first built), no make_context.
            let stack_top = unsafe { (*desc).stack_top() } as usize;
            let dispatch_slot = (stack_top - DISPATCH_SLOT_SIZE) as *mut DispatchFn;
            let (f_ptr, result_ptr) = branch_layout::<Fb, Rb>(stack_top - DISPATCH_SLOT_SIZE);
            (desc, f_ptr, result_ptr, dispatch_slot)
        }
        None => {
            // Cold: either the cache was empty, or `Fb`/`Rb` don't fit the
            // warm budget. Build a context exactly as before — the only
            // difference from the pre-warm-cache version is `entry` is now
            // the type-erased `branch_entry::<S>` (see its doc comment),
            // and `exec_top` is the fixed warm-cacheable offset whenever
            // `fits` allows a later reuse, or the dynamic one otherwise.
            let mut token = wk.alloc_task(true);
            token.commit_as_ctx();
            token.set_external_queue(wk.external_queue() as *const _);
            let stack_top = token.as_desc().stack_top() as usize;
            let dispatch_slot = (stack_top - DISPATCH_SLOT_SIZE) as *mut DispatchFn;
            let (f_ptr, result_ptr) = branch_layout::<Fb, Rb>(stack_top - DISPATCH_SLOT_SIZE);
            let exec_top = if fits {
                align_down(stack_top - BRANCH_HEADER_BUDGET, 16)
            } else {
                align_down(f_ptr as usize, 16)
            } as *mut u8;
            let ctx = unsafe { S::Ctx::make_context(exec_top, branch_entry::<S>, std::ptr::null_mut()) };
            token.set_ctx(ctx.0);
            (token.desc(), f_ptr, result_ptr, dispatch_slot)
        }
    };

    // Write `b` and the type-specific dispatch shim onto the not-yet-(and
    // maybe never-)switched-to stack.
    unsafe { f_ptr.write(b) };
    unsafe { dispatch_slot.write(dispatch_shim::<S, Fb, Rb>) };

    // SAFETY: `desc` is either freshly allocated above (never wrapped in a
    // token before) or popped from `BranchWarmPool`, which only ever holds
    // descriptors this same function pushed after popping them back
    // un-stolen (never shared, never handed to a thief) — exclusively ours
    // either way.
    wk.push(unsafe { SuspendedTaskToken::from_raw(desc) }.into());

    let ra = a();

    // `a` may have suspended and resumed on a different worker (any nested
    // `parallel_call`/`.join()` inside it can migrate the calling ULT), so
    // re-derive which one we're actually on now before touching a queue or
    // pool — same reason `spawn`'s child re-derives `wk` after running its
    // closure. Using the stale `wk` here would silently operate on a
    // *different* worker's deque/pool from a different OS thread.
    let wk = <S::Worker as WorkerOps<S>>::current().expect("cmpth: worker vanished");

    match wk.try_pop() {
        Some(raw) => {
            let popped: SuspendedTaskToken<S::Desc> = raw.into();
            if std::ptr::eq(popped.desc(), desc) {
                // Not stolen: `ctx` was never touched. If it fits the warm
                // budget, cache it for a later call on this worker instead
                // of returning it to the general pool; either way, run `b`
                // as a plain function call — no context ever touched.
                let desc = popped.into_raw();
                if fits {
                    wk.branch_warm_push(desc);
                } else {
                    unsafe { wk.free_task(desc) };
                }
                let rb = unsafe { f_ptr.read() }();
                (ra, rb)
            } else {
                // Something else was on top; put it back and fall through
                // to the stolen path below.
                wk.push(popped.into());
                let handle = JoinHandle::<S, Rb> { desc, result_ptr, result_drop: drop_stack_result::<Rb>, _marker: PhantomData };
                (ra, JoinHandleLike::join(handle))
            }
        }
        None => {
            let handle = JoinHandle::<S, Rb> { desc, result_ptr, result_drop: drop_stack_result::<Rb>, _marker: PhantomData };
            (ra, JoinHandleLike::join(handle))
        }
    }
}

/// [`EntryFn`](crate::traits::stackful::EntryFn) for a `parallel_call`
/// branch that got stolen: the first (and only) time this context is ever
/// switched into. Generic only in `S` — never in the branch's `Fb`/`Rb` —
/// so the same 56-byte `make_context`-built ctx frame can be reused
/// ([`BranchWarmPool`]) across calls of *different* closure types on the
/// same worker: the type-specific behavior lives in the [`DispatchFn`]
/// pointer stored as data in the fixed slot at
/// `stack_top - DISPATCH_SLOT_SIZE` (written by [`parallel_call`] on every
/// push, warm or cold alike), not baked into this function's own machine
/// code the way it would be if this were still generic over `Fb`/`Rb`.
///
/// Runs on the destination stack *after* the generic dispatch bookkeeping
/// (`RunnableItem::run_on` → `suspend_shim`) already parked the thief's own
/// previous continuation as its `root_cont` — this function never needs to
/// touch `prev` itself, unlike `spawn`'s child closure (which switches
/// directly via `save_context`, bypassing that generic wrapper).
unsafe extern "C" fn branch_entry<S>(_transfer: Transfer, _arg: *mut ()) -> !
where
    S: StackfulSchedulerSystem,
    S::Worker: ContextSwitcher<S> + DescWorkerOps<S>,
    <S as PoolSystem>::Desc: StackfulTaskDesc,
{
    let wk = <S::Worker as WorkerOps<S>>::current().expect("cmpth: worker vanished");
    let desc = wk.cur_task();
    let stack_top = unsafe { (*desc).stack_top() } as usize;
    let dispatch_fn = unsafe { ((stack_top - DISPATCH_SLOT_SIZE) as *const DispatchFn).read() };
    unsafe { dispatch_fn(stack_top) }
}

/// Type-specific half of a stolen `parallel_call` branch's execution —
/// the [`DispatchFn`] [`branch_entry`] reads from the fixed slot and
/// tail-calls, monomorphized once per `(Fb, Rb)`. Recomputes
/// `f_ptr`/`result_ptr` from `stack_top` via [`branch_layout`] (anchored
/// below the dispatch slot — the same formula regardless of whether this
/// branch used the fixed warm-cache `exec_top` or the dynamic
/// oversized-fallback one, since that choice only affects where
/// `exec_top`/the ctx frame sit, never this computation), runs the
/// closure, and finishes via [`exit_with_result`] exactly as `spawn`'s
/// child does.
unsafe extern "C" fn dispatch_shim<S, Fb, Rb>(stack_top: usize) -> !
where
    S: StackfulSchedulerSystem,
    S::Worker: ContextSwitcher<S> + DescWorkerOps<S>,
    Fb: FnOnce() -> Rb + Send + 'static,
    Rb: Send + 'static,
    <S as PoolSystem>::Desc: StackfulTaskDesc,
{
    let wk = <S::Worker as WorkerOps<S>>::current().expect("cmpth: worker vanished");
    let desc = wk.cur_task();
    let (f_ptr, result_ptr) = branch_layout::<Fb, Rb>(stack_top - DISPATCH_SLOT_SIZE);

    let val = catch_unwind(AssertUnwindSafe(|| unsafe { f_ptr.read() }()));
    // The closure may have suspended and resumed on a different worker, so
    // re-derive which one we're on now (same as `spawn`'s child).
    let wk = <S::Worker as WorkerOps<S>>::current().expect("cmpth: worker vanished");
    debug_assert!(std::ptr::eq(wk.cur_task(), desc));
    exit_with_result::<S, Rb>(wk, wk.cur_task_ref(), result_ptr, val)
}

// ---------------------------------------------------------------------------
// exit helpers
// ---------------------------------------------------------------------------

/// [`TaskExitSink`] for [`exit_with_result`]: a dropped `JoinHandle`'s
/// result lives on the exiting task's own (still-allocated) stack, so
/// `reclaim` must drop it in place before freeing the descriptor.
struct ExitWithResultSink<'a, S: StackfulSchedulerSystem, T> {
    wk: &'a S::Worker,
    desc_ptr: *mut S::Desc,
    result_ptr: *mut StackResult<T>,
}

impl<'a, S: StackfulSchedulerSystem, T: Send + 'static> TaskExitSink<S::Desc>
    for ExitWithResultSink<'a, S, T>
where
    <S as PoolSystem>::Desc: StackfulTaskDesc,
    S::Worker: TaskPool<S>,
{
    fn resume(&self, cont: <S::Desc as TaskDesc>::Suspended) {
        self.wk.push(cont.into());
    }

    fn reclaim(&self) {
        // The handle was dropped while we were exiting: the result already
        // sits on our (still-allocated) stack.
        unsafe {
            self.result_ptr.drop_in_place();
            self.wk.free_task(self.desc_ptr);
        }
    }
}

/// Exit a spawned task.
///
/// A parked sync joiner and the abandoned (detached) state are both
/// *stable* (the joiner cannot act until resumed; a dropped handle never
/// comes back), so a plain Acquire read selects those paths —
/// `try_take_handoff_target`/`is_abandoned`. Anything else (`RUNNING` or a
/// registered async waker) can still change concurrently — late joiner
/// registration, waker replacement, detach — so the exit callback publishes
/// `FINISHED` with a `swap` *after* the context switch
/// (`finish_and_settle`) and settles whichever party it finds in the old
/// value.
fn exit_with_result<S: StackfulSchedulerSystem, T: Send + 'static>(
    wk: &S::Worker,
    desc: &S::Desc,
    result_ptr: *mut StackResult<T>,
    val: Result<T, Box<dyn Any + Send>>,
) -> !
where
    <S as PoolSystem>::Desc: StackfulTaskDesc,
    S::Worker: StackfulWorker<S> + TaskPool<S>,
{
    let desc_ptr = desc as *const S::Desc as *mut S::Desc;
    if let Some(j_token) = desc.try_take_handoff_target() {
        // Direct handoff: switch straight to the parked joiner.
        let sr = match val { Ok(v) => StackResult::Ok(v), Err(e) => StackResult::Err(e) };
        unsafe { result_ptr.write(sr) };
        wk.exit_to_cont(j_token, move |_wk| {
            desc.commit_finished();
        })
    } else if desc.is_abandoned() {
        // No handle: drop val on the task's own stack before the context
        // switch so destructors run correctly.
        drop(val);
        wk.exit_to_sched(move |wk| unsafe { wk.free_task(desc_ptr) })
    } else {
        let sr = match val { Ok(v) => StackResult::Ok(v), Err(e) => StackResult::Err(e) };
        unsafe { result_ptr.write(sr) };
        wk.exit_to_sched(move |wk| {
            let sink = ExitWithResultSink::<S, T> { wk, desc_ptr, result_ptr };
            desc.finish_and_settle(&sink);
        })
    }
}

// ---------------------------------------------------------------------------
// blocking JoinHandle::join
// ---------------------------------------------------------------------------

// Blocking `.join()`: inherently stackful (parks the calling ULT via
// `cond_suspend_to_sched`), so this is a separate impl block bounded on
// `StackfulSchedulerSystem` rather than widening the base block in
// `common::thread` — a stackless-only `JoinHandle` (from `spawn_async`) only
// ever gets `.await`ed (see `stackless::thread`'s `Future for JoinHandle`),
// never `.join()`ed.
impl<S: StackfulSchedulerSystem, T: Send + 'static> JoinHandle<S, T>
where
    S::Desc: StackfulTaskDesc,
    S::Worker: StackfulWorker<S>,
{
    pub fn join(self) -> Result<T, Box<dyn Any + Send>> {
        let wk = <S::Worker as WorkerOps<S>>::current().expect("cmpth: join called outside a worker");

        // Fast path: the child already exited.  Child-first spawn guarantees
        // this whenever the parent continuation was not stolen, so the whole
        // fork-join hot path lands here.  FINISHED is published with Release
        // after the result write; the Acquire read makes the result visible.
        if self.desc_ref().is_finished() {
            return self.take_result(wk);
        }

        // Slow path: register this task as the sync joiner with one CAS.
        // cond_suspend cancels the suspension when the child finished in the
        // meantime (the CAS loses to the exit path's swap).
        // The returned worker is the one we resumed on — no TLS re-read.
        let desc = self.desc_ref();
        let wk = wk.cond_suspend_to_sched(move |_wk, prev| {
            let joiner = prev.take().expect("cond_suspend contract");
            match desc.try_register_joiner(joiner) {
                Ok(()) => {}
                // Cancel: hand the token straight back so `cond_suspend_to_sched`
                // sees `prev` still holding it and resumes at once.
                Err(joiner) => *prev = Some(joiner),
            }
        });

        debug_assert!(desc.is_finished());
        self.take_result(wk)
    }
}

impl<S: StackfulSchedulerSystem, T: Send + 'static> JoinHandleLike<T> for JoinHandle<S, T>
where
    S::Desc: StackfulTaskDesc,
    S::Worker: StackfulWorker<S>,
{
    fn join(self) -> T {
        match JoinHandle::join(self) {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        }
    }
}
