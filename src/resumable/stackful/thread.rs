//! Stackful thread functions: fork (child-first and parent-first), exit,
//! blocking `.join()`. See
//! [`common::thread`](crate::resumable::common::thread) for the shared
//! [`JoinHandle`] type both
//! this and [`stackless::thread`](crate::resumable::stackless::thread)
//! produce.

use std::any::Any;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::traits::stackful::{ContextPolicy, HandoffTaskDesc, JoinHandleLike, Transfer};
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::thread::{align_down, drop_stack_result, JoinHandle, StackResult};
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::common::desc::{HasScheduler, SuspendedTaskToken, TaskDesc, TaskDescAlloc, TaskDescCore, TaskExitSink};
use crate::resumable::stackful::desc::{HasCtx, StackfulTaskDesc};
use crate::resumable::common::worker::{LocalQueue, TaskPool, UltWorker, WorkerOps};
use crate::resumable::stackful::worker::{ContextSwitcher, StackfulWorker};

// Still needed for fork_parent_first (root task entry).
pub(crate) type ErasedBody = Box<dyn FnOnce() -> Box<dyn Any + Send> + Send>;

// ---------------------------------------------------------------------------
// spawn (child-first fork)
// ---------------------------------------------------------------------------

/// Spawn a ULT.  Child-first: the child starts immediately on this worker and
/// the parent's continuation is pushed to the deque for stealing.
///
/// The closure `F` and the result slot `StackResult<T>` are placed directly on
/// the child's stack, avoiding two heap allocations that the old Box-erasure
/// approach required.
pub fn spawn<S, T, F>(f: F) -> JoinHandle<S, T>
where
    S: StackfulSchedulerSystem,
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
    <S as SchedulerSystem>::Desc: StackfulTaskDesc,
{
    let wk = UltWorker::<S>::current().expect("cmpth: spawn called outside a worker");
    let desc = wk.alloc_task(true, S::STACK_SIZE);
    let stack_top = {
        // SAFETY: `desc` was just freshly allocated by `alloc_task` and has
        // never been wrapped in a token before — trivially exclusive.
        let mut token = unsafe { SuspendedTaskToken::from_raw(desc) };
        token.commit_as_ctx();
        token.set_scheduler(wk.shared.get());
        let stack_top = token.as_desc().stack_top() as usize;
        let _ = token.into_raw();
        stack_top
    };

    // Reserve space at the top of the child's stack (high addresses) for the
    // closure and the result slot.  The execution stack gets the rest below.
    //
    //   stack_top (high)
    //   ┌─────────────────┐
    //   │ StackResult<T>  │  ← result_addr
    //   ├─────────────────┤
    //   │ F               │  ← f_addr
    //   ├─────────────────┤
    //   │ (exec stack)    │  ← exec_top and below
    //   └─────────────────┘ ← stack base

    let result_layout = std::alloc::Layout::new::<StackResult<T>>();
    let f_layout = std::alloc::Layout::new::<F>();

    let result_addr = align_down(stack_top - result_layout.size(), result_layout.align());
    let f_addr = align_down(result_addr.wrapping_sub(f_layout.size()), f_layout.align().max(1));
    let exec_top = align_down(f_addr, 16) as *mut u8;

    let result_ptr = result_addr as *mut StackResult<T>;
    let f_ptr = f_addr as *mut F;

    // Write the closure onto the child's stack before switching.
    unsafe { f_ptr.write(f) };

    let child = move |wk: &UltWorker<S>, prev| {
        // Running on the child's stack.  Publish the parent for stealing, run
        // the closure, then exit via exit_with_result.
        wk.push_local_top(prev);
        let val = catch_unwind(AssertUnwindSafe(|| unsafe { f_ptr.read() }()));
        // The closure may have suspended and resumed on a different worker,
        // so re-derive which one we're on now.
        let wk = UltWorker::<S>::current().expect("cmpth: worker vanished");
        debug_assert!(std::ptr::eq(wk.cur_task(), desc));
        exit_with_result(wk, wk.cur_task_ref(), result_ptr, val)
    };
    // SAFETY: `desc` was freshly allocated above and, after the token built
    // at line 46 was released via `into_raw`, has never been re-tokenized —
    // still exclusively owned here.
    unsafe { wk.suspend_to_new(exec_top, desc, child) };

    JoinHandle { desc, result_ptr, result_drop: drop_stack_result::<T>, _marker: PhantomData }
}

/// Parent-first fork: package `body` as a ready continuation without running
/// it.  Used for the root task of `run` and by [`PollerUltQueue::on_start`].
///
/// `scheduler` is stored on the descriptor for external-thread wake support.
pub(crate) fn fork_parent_first<S: StackfulSchedulerSystem>(body: ErasedBody, scheduler: *const crate::resumable::common::scheduler::Scheduler<S>) -> SuspendedTaskToken<S::Desc>
where
    <S as SchedulerSystem>::Desc: StackfulTaskDesc,
{
    use crate::resumable::common::stack::StackAlloc as _;
    // Allocated directly (not through S::Pool), like `fork_async_parent_first`'s
    // one-off root async descriptor: this runs once per `run`/`PollerUltQueue::on_start`
    // call, so pooling it has nothing to gain. Still wrapped via `Node::wrap_fresh`
    // (not a bare `Box::new`) and marked `oversized` unconditionally, so its
    // eventual dealloc (through the pool, like any other finished task) can
    // recover the node via `Node::node_of` and always raw-frees it.
    let payload = S::Desc::alloc_with(S::StackAlloc::alloc_stack(S::STACK_SIZE).into(), false);
    let desc = crate::resumable::common::pool::Node::wrap_fresh(0, true, payload);
    // SAFETY: `desc` was just freshly allocated above and has never been
    // wrapped in a token before — trivially exclusive.
    let mut token = unsafe { SuspendedTaskToken::from_raw(desc) };
    token.commit_as_ctx();
    token.set_scheduler(scheduler);
    let arg = Box::into_raw(Box::new(body));
    let ctx = unsafe {
        S::Ctx::make_context(token.as_desc().stack_top(), task_entry::<S>, arg as *mut ())
    };
    token.init_saved_context(ctx.0);
    token
}

unsafe extern "C" fn task_entry<S: StackfulSchedulerSystem>(transfer: Transfer, arg: *mut ()) -> ! where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    let wk = unsafe { &*(transfer.0 as *const UltWorker<S>) };
    let desc = wk.cur_task();
    let body = *unsafe { Box::from_raw(arg as *mut ErasedBody) };
    let result = catch_unwind(AssertUnwindSafe(body));
    // See spawn: the body may have suspended and resumed on a different
    // worker, so re-derive which one we're on now.
    let wk = UltWorker::<S>::current().expect("cmpth: worker vanished");
    debug_assert!(std::ptr::eq(wk.cur_task(), desc));
    // `task_entry` only ever runs a `fork_parent_first` body (`run`'s root
    // task, `PollerUltQueue`'s poller ULT) — both always detached (no
    // `JoinHandle`, see `fork_parent_first`'s `has_handle: false`), so
    // nobody is ever positioned to collect this result. Drop it here rather
    // than storing it on the descriptor only to have `reinit`/`free_task`
    // drop it later unread.
    drop(result);
    exit(wk, wk.cur_task_ref())
}

// ---------------------------------------------------------------------------
// exit helpers
// ---------------------------------------------------------------------------

/// [`TaskExitSink`] for [`exit_with_result`]: a dropped `JoinHandle`'s
/// result lives on the exiting task's own (still-allocated) stack, so
/// `reclaim` must drop it in place before freeing the descriptor — unlike
/// [`ExitSink`] (used by [`exit`], whose task never had a result anyone
/// could observe).
struct ExitWithResultSink<'a, S: StackfulSchedulerSystem, T> {
    wk: &'a UltWorker<S>,
    desc_ptr: *mut S::Desc,
    result_ptr: *mut StackResult<T>,
}

impl<'a, S: StackfulSchedulerSystem, T: Send + 'static> TaskExitSink<S::Desc>
    for ExitWithResultSink<'a, S, T>
where
    <S as SchedulerSystem>::Desc: StackfulTaskDesc,
{
    fn resume(&self, cont: <S::Desc as TaskDesc>::Suspended) {
        self.wk.push_local_top(cont);
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

/// [`TaskExitSink`] for [`exit`] (parent-first/detached-only tasks): no
/// result was ever written for anyone to observe (`task_entry` already
/// dropped it before calling this), so `reclaim` only needs to free the
/// descriptor.
struct ExitSink<'a, S: StackfulSchedulerSystem> {
    wk: &'a UltWorker<S>,
    desc_ptr: *mut S::Desc,
}

impl<'a, S: StackfulSchedulerSystem> TaskExitSink<S::Desc> for ExitSink<'a, S>
where
    <S as SchedulerSystem>::Desc: StackfulTaskDesc,
{
    fn resume(&self, cont: <S::Desc as TaskDesc>::Suspended) {
        self.wk.push_local_top(cont);
    }

    fn reclaim(&self) {
        unsafe { self.wk.free_task(self.desc_ptr) };
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
    wk: &UltWorker<S>,
    desc: &S::Desc,
    result_ptr: *mut StackResult<T>,
    val: Result<T, Box<dyn Any + Send>>,
) -> ! where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
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
            let sink = ExitWithResultSink { wk, desc_ptr, result_ptr };
            desc.finish_and_settle(&sink);
        })
    }
}

/// Exit for parent-first tasks (`fork_parent_first`): `task_entry` already
/// dropped the result before calling this (see its own comment) — every
/// `fork_parent_first` task starts, and stays, abandoned, so the
/// handoff-target/running-with-a-handle paths below are unreachable in
/// practice for this caller, kept only because this shares the same state
/// machine as `exit_with_result`.
fn exit<S: StackfulSchedulerSystem>(wk: &UltWorker<S>, desc: &S::Desc) -> ! where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    let desc_ptr = desc as *const S::Desc as *mut S::Desc;
    if let Some(j_token) = desc.try_take_handoff_target() {
        wk.exit_to_cont(j_token, move |_wk| {
            desc.commit_finished();
        })
    } else if desc.is_abandoned() {
        wk.exit_to_sched(move |wk| unsafe { wk.free_task(desc_ptr) })
    } else {
        wk.exit_to_sched(move |wk| {
            let sink = ExitSink { wk, desc_ptr };
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
{
    pub fn join(self) -> Result<T, Box<dyn Any + Send>> {
        let wk = UltWorker::<S>::current().expect("cmpth: join called outside a worker");

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
{
    fn join(self) -> T {
        match JoinHandle::join(self) {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        }
    }
}
