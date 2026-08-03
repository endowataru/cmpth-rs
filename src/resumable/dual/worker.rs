//! Dual-only `SchedulerSystem::execute`/`StackfulSchedulerSystem::pop_or_root`/
//! `SchedulerSystem::free_finished_desc` bodies: a popped continuation may
//! be either a real ULT or a `spawn_async` task, so dispatch needs the
//! `poll_fn` tag check the stackful-only/stackless-only bodies don't pay
//! for. See [`common::worker`](crate::resumable::common::worker) and
//! [`stackful::worker`](crate::resumable::stackful::worker) for the shared
//! machinery this builds on.

use crate::resumable::common::deque::WorkerDeque;
use crate::resumable::common::worker::{LocalQueue, TaskPool, UltWorker};
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::stackful::worker::{ContextSwitcher, StackfulLocalQueue};
use crate::resumable::common::desc::SuspendedTaskToken;
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::stackless::desc::AsyncTaskDesc;
use crate::resumable::common::pool::DescPool;

/// `execute` body for dual systems: today's original logic — check
/// `poll_fn` first, and either poll inline or perform a real context switch.
pub fn execute_dual<S>(wk: &UltWorker<S>, cont: SuspendedTaskToken<S::Desc>)
where
    S: StackfulSchedulerSystem,
    S::Desc: StackfulTaskDesc + AsyncTaskDesc,
{
    let desc = cont.desc();
    if cont.is_poll_fn_dispatch() {
        let poll_fn = cont.poll_fn()
            .expect("cmpth: descriptor committed to poll_fn dispatch but poll_fn unset");
        let _ = cont.into_raw(); // consumed; no context switch
        crate::resumable::stackless::worker::run_async_poll(wk, desc, poll_fn);
    } else {
        // Sync ULT: context switch as usual.
        let wk2 = wk.suspend_to_cont(cont, |wk, prev| wk.set_root_cont(prev));
        debug_assert!(std::ptr::eq(wk2 as *const UltWorker<S>, wk as *const UltWorker<S>));
    }
}

/// `pop_or_root` body for dual systems: today's original logic — an async
/// task popped off the top has no saved context to switch into, so requeue
/// it and fall back to the root (scheduler-loop) continuation instead.
pub fn pop_or_root_dual<S>(wk: &UltWorker<S>) -> SuspendedTaskToken<S::Desc>
where
    S: StackfulSchedulerSystem,
    S::Desc: StackfulTaskDesc + AsyncTaskDesc,
{
    if let Some(c) = wk.deque.try_pop_top() {
        if c.is_poll_fn_dispatch() {
            // Async tasks have no saved context; they can only be executed
            // by the scheduler loop via execute().  Push the async task back
            // to the LIFO end and return root so the scheduler loop handles it.
            wk.deque.push_top(c);
        } else {
            return c;
        }
    }
    wk.take_root_cont()
}

/// `free_finished_desc` body for dual systems: async tasks go through
/// `S::AsyncPool` (a separate pool from the ULT-stack `S::Pool`, see
/// [`SchedulerSystem::AsyncPool`](crate::resumable::common::system::SchedulerSystem::AsyncPool));
/// everything else goes through the ULT-stack pool as usual.
///
/// # Safety
/// No other references to `desc` may exist after this call (same contract
/// as [`TaskPool::free_task`]/[`DescPool::dealloc`]).
pub unsafe fn free_finished_desc_dual<S>(wk: &UltWorker<S>, desc: *mut S::Desc)
where
    S: StackfulSchedulerSystem,
    S::Desc: StackfulTaskDesc + AsyncTaskDesc,
{
    // SAFETY: `desc` is finished (about to be freed) — `TaskDesc::join_state`'s
    // own contract guarantees the exit path never touches the descriptor
    // again after publishing `FINISHED`, so no other token can exist for
    // it; safe to construct one transiently just to read the dispatch tag.
    if unsafe { crate::resumable::common::desc::SuspendedTaskToken::from_raw(desc) }.is_poll_fn_dispatch() {
        unsafe { wk.shared().async_task_pool.dealloc(wk.num(), desc) };
    } else {
        unsafe { wk.free_task(desc) };
    }
}
