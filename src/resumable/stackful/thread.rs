//! Stackful thread functions: fork (child-first and parent-first), exit,
//! blocking `.join()`. See
//! [`common::thread`](crate::resumable::common::thread) for the shared
//! [`JoinHandle`] type both
//! this and [`stackless::thread`](crate::resumable::stackless::thread)
//! produce.

use std::any::Any;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::context::{ContextPolicy, Transfer};
use crate::traits::stackful::JoinHandleLike;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::thread::{align_down, drop_stack_result, JoinHandle, StackResult};
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::common::desc::{HasBaseOwned, JoinState, SuspendedTaskToken, TaskDesc, TaskDescCore, TaskDescAlloc};
use crate::resumable::stackful::desc::{HasCtx, StackfulTaskDesc};
use crate::resumable::common::worker::{LocalQueue, TaskPool, UltWorker, Worker};
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
        let mut token = SuspendedTaskToken(desc);
        token.commit_as_ctx();
        token.base_mut().scheduler = wk.shared.get() as *const ();
        if let Some(slot) = token.base().slot {
            unsafe { (*slot).system_id.set(crate::resumable::common::lookup::system_id::<S>()) };
        }
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

    wk.suspend_to_new(exec_top, desc, move |wk, prev| {
        // Running on the child's stack.  Publish the parent for stealing, run
        // the closure, then exit via exit_with_result.
        wk.push_local_top(prev);
        let val = catch_unwind(AssertUnwindSafe(|| unsafe { f_ptr.read() }()));
        // The closure may have suspended and resumed on a different worker,
        // but every resume records the worker in the descriptor — cheaper
        // than a TLS lookup.
        let wk = unsafe { &*(crate::resumable::common::desc::peek_worker(desc) as *const UltWorker<S>) };
        debug_assert!(std::ptr::eq(wk, UltWorker::<S>::current().expect("cmpth: worker vanished")));
        debug_assert!(std::ptr::eq(wk.cur_task(), desc));
        exit_with_result(wk, wk.cur_task_ref(), result_ptr, val)
    });

    JoinHandle { desc, result_ptr, result_drop: drop_stack_result::<T>, _marker: PhantomData }
}

/// Parent-first fork: package `body` as a ready continuation without running
/// it.  Used for the root task of `run` and by [`PollerUltQueue::on_start`].
///
/// `scheduler` is a type-erased `*const Scheduler<S>` stored on the
/// descriptor for external-thread wake support.
pub(crate) fn fork_parent_first<S: StackfulSchedulerSystem>(body: ErasedBody, scheduler: *const ()) -> SuspendedTaskToken<S::Desc> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    use crate::resumable::common::stack::StackAlloc as _;
    // Allocated directly (not through S::Pool), like `fork_async_parent_first`'s
    // one-off root async descriptor: this runs once per `run`/`PollerUltQueue::on_start`
    // call, so pooling it has nothing to gain. Still wrapped via `Node::wrap_fresh`
    // (not a bare `Box::new`) and marked `oversized` unconditionally, so its
    // eventual dealloc (through the pool, like any other finished task) can
    // recover the node via `Node::node_of` and always raw-frees it.
    let payload = S::Desc::alloc_with(S::StackAlloc::alloc_stack(S::STACK_SIZE).into(), false);
    let desc = crate::resumable::common::pool::Node::wrap_fresh(0, true, payload);
    let mut token = SuspendedTaskToken(desc);
    token.commit_as_ctx();
    token.base_mut().scheduler = scheduler;
    if let Some(slot) = token.base().slot {
        unsafe { (*slot).system_id.set(crate::resumable::common::lookup::system_id::<S>()) };
    }
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
    // See spawn: the descriptor tracks the current worker across migrations.
    let wk = unsafe { &*(crate::resumable::common::desc::peek_worker(desc) as *const UltWorker<S>) };
    debug_assert!(std::ptr::eq(wk, UltWorker::<S>::current().expect("cmpth: worker vanished")));
    debug_assert!(std::ptr::eq(wk.cur_task(), desc));
    wk.cur_task_token_mut().base_mut().result = Some(result);
    exit(wk, wk.cur_task_ref())
}

// ---------------------------------------------------------------------------
// exit helpers
// ---------------------------------------------------------------------------

/// Exit a spawned task.
///
/// One atomic decides everything.  A parked sync joiner and the detached
/// state are both *stable* (the joiner cannot act until resumed; a dropped
/// handle never comes back), so a plain Acquire read selects those paths.
/// Anything else (`RUNNING` or a registered async waker) can still change
/// concurrently — late joiner registration, waker replacement, detach — so
/// the exit callback publishes `FINISHED` with a `swap` *after* the context
/// switch and settles whichever party it finds in the old value.
fn exit_with_result<S: StackfulSchedulerSystem, T: Send + 'static>(
    wk: &UltWorker<S>,
    desc: &S::Desc,
    result_ptr: *mut StackResult<T>,
    val: Result<T, Box<dyn Any + Send>>,
) -> ! where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    let desc_ptr = desc as *const S::Desc as *mut S::Desc;
    match desc.read_join_state() {
        JoinState::SyncJoiner(j_desc) => {
            // Direct handoff: switch straight to the parked joiner.
            let sr = match val { Ok(v) => StackResult::Ok(v), Err(e) => StackResult::Err(e) };
            unsafe { result_ptr.write(sr) };
            wk.exit_to_cont(SuspendedTaskToken(j_desc), move |_wk| {
                desc.commit_finished();
            })
        }
        JoinState::Detached => {
            // No handle: drop val on the task's own stack before the context
            // switch so destructors run correctly.
            drop(val);
            wk.exit_to_sched(move |wk| unsafe { wk.free_task(desc_ptr) })
        }
        _ => {
            let sr = match val { Ok(v) => StackResult::Ok(v), Err(e) => StackResult::Err(e) };
            unsafe { result_ptr.write(sr) };
            wk.exit_to_sched(move |wk| {
                match desc.publish_finished() {
                    // No joiner appeared: the JoinHandle collects the result.
                    JoinState::Running => {}
                    // A joiner registered while we were exiting.
                    JoinState::SyncJoiner(j) => wk.push_local_top(SuspendedTaskToken(j)),
                    JoinState::AsyncWaker(w) => unsafe { Box::from_raw(w) }.wake(),
                    JoinState::AsyncJoiner(j) => S::wake_async_joiner(j),
                    // The handle was dropped while we were exiting: the
                    // result already sits on our (still-allocated) stack.
                    JoinState::Detached => unsafe {
                        result_ptr.drop_in_place();
                        wk.free_task(desc_ptr);
                    },
                    JoinState::Finished => unreachable!("cmpth: double task exit"),
                }
            })
        }
    }
}

/// Exit for parent-first tasks (`fork_parent_first`): the result, if kept,
/// is already in `desc.result`.  Same state machine as `exit_with_result`.
fn exit<S: StackfulSchedulerSystem>(wk: &UltWorker<S>, desc: &S::Desc) -> ! where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    let desc_ptr = desc as *const S::Desc as *mut S::Desc;
    match desc.read_join_state() {
        JoinState::SyncJoiner(j_desc) => {
            wk.exit_to_cont(SuspendedTaskToken(j_desc), move |_wk| {
                desc.commit_finished();
            })
        }
        // Root tasks start in this state; desc.result drops with the desc.
        JoinState::Detached => wk.exit_to_sched(move |wk| unsafe { wk.free_task(desc_ptr) }),
        _ => wk.exit_to_sched(move |wk| {
            match desc.publish_finished() {
                JoinState::Running => {}
                JoinState::SyncJoiner(j) => wk.push_local_top(SuspendedTaskToken(j)),
                JoinState::AsyncWaker(w) => unsafe { Box::from_raw(w) }.wake(),
                JoinState::AsyncJoiner(j) => S::wake_async_joiner(j),
                JoinState::Detached => unsafe { wk.free_task(desc_ptr) },
                JoinState::Finished => unreachable!("cmpth: double task exit"),
            }
        }),
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
            let joiner = prev.as_ref().expect("cond_suspend contract").desc();
            if unsafe { desc.try_register_sync_joiner(joiner) } {
                let _ = prev.take().expect("cond_suspend contract").into_raw();
            }
            // else: leave `prev` in place -> cancel, resume at once
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
