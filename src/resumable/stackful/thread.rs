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
use crate::resumable::stackful::worker::{ContextSwitcher, StackfulWorker};

// ---------------------------------------------------------------------------
// spawn (child-first fork)
// ---------------------------------------------------------------------------

/// Spawn a ULT.  Child-first: the child starts immediately on this worker and
/// the parent's continuation is pushed to the deque for stealing.
///
/// The closure `F` and the result slot `StackResult<T>` are placed directly on
/// the child's stack, avoiding two heap allocations that the old Box-erasure
/// approach required.
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
// parallel_call (fork-parent-first, pool-backed via make_context)
// ---------------------------------------------------------------------------

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
/// needed is [`branch_entry`] itself, and finishing joins the existing
/// `JoinHandle` protocol verbatim.
///
/// `Fa`/`Ra` don't need `Send + 'static` here — `a` never leaves the
/// caller's stack — but callers (the `ScopedStackfulTaskSystem` blanket
/// impl) may still pass values satisfying stricter bounds; this function
/// simply doesn't require them.
pub fn parallel_call<S, Ra, Fa, Rb, Fb>(a: Fa, b: Fb) -> (Ra, Rb)
where
    S: StackfulSchedulerSystem,
    S::Worker: ContextSwitcher<S> + DescWorkerOps<S>,
    Fa: FnOnce() -> Ra,
    Fb: FnOnce() -> Rb + Send + 'static,
    Rb: Send + 'static,
    <S as PoolSystem>::Desc: StackfulTaskDesc,
{
    let wk = <S::Worker as WorkerOps<S>>::current().expect("cmpth: parallel_call called outside a worker");

    let mut token = wk.alloc_task(true);
    token.commit_as_ctx();
    token.set_external_queue(wk.external_queue() as *const _);
    let stack_top = token.as_desc().stack_top() as usize;

    let (f_ptr, result_ptr) = branch_layout::<Fb, Rb>(stack_top);
    let exec_top = align_down(f_ptr as usize, 16) as *mut u8;

    // Write `b` onto the not-yet-(and maybe never-)switched-to stack.
    unsafe { f_ptr.write(b) };

    // Prepare a context that, if ever switched into, enters `branch_entry`
    // — without switching now. `arg` is unused: `branch_entry` recomputes
    // `f_ptr`/`result_ptr` from `stack_top` via `branch_layout`, same as
    // this function just did, rather than threading them through as a
    // separate payload.
    let ctx = unsafe { S::Ctx::make_context(exec_top, branch_entry::<S, Fb, Rb>, std::ptr::null_mut()) };
    token.set_ctx(ctx.0);

    let desc = token.desc();
    wk.push(token.into());

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
                // Not stolen: discard the never-switched-into context and
                // stack reservation, and run `b` as a plain function call.
                unsafe { wk.free_task(popped.into_raw()) };
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
/// switched into. Runs on the destination stack *after* the generic
/// dispatch bookkeeping (`RunnableItem::run_on` → `suspend_shim`) already
/// parked the thief's own previous continuation as its `root_cont` — this
/// function never needs to touch `prev` itself, unlike `spawn`'s child
/// closure (which switches directly via `save_context`, bypassing that
/// generic wrapper).
unsafe extern "C" fn branch_entry<S, Fb, Rb>(_transfer: Transfer, _arg: *mut ()) -> !
where
    S: StackfulSchedulerSystem,
    S::Worker: ContextSwitcher<S> + DescWorkerOps<S>,
    Fb: FnOnce() -> Rb + Send + 'static,
    Rb: Send + 'static,
    <S as PoolSystem>::Desc: StackfulTaskDesc,
{
    // Raw fn pointer, no closure capture available — re-derive everything
    // via TLS, same as `spawn`'s child re-derives `wk` after running its
    // closure (in case of migration); here we need it up front too, to
    // locate `f_ptr`/`result_ptr` on our own stack.
    let wk = <S::Worker as WorkerOps<S>>::current().expect("cmpth: worker vanished");
    let desc = wk.cur_task();
    let stack_top = unsafe { (*desc).stack_top() } as usize;
    let (f_ptr, result_ptr) = branch_layout::<Fb, Rb>(stack_top);

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
