//! Dual-only `SchedulerSystem::execute`/`StackfulSchedulerSystem::pop_or_root`/
//! `SchedulerSystem::free_finished_desc` bodies: a popped continuation may
//! be either a real ULT or a `spawn_async` task, so dispatch needs the
//! `poll_fn` tag check the stackful-only/stackless-only bodies don't pay
//! for. See [`common::worker`](crate::resumable::common::worker) and
//! [`stackful::worker`](crate::resumable::stackful::worker) for the shared
//! machinery this builds on.

use crate::resumable::common::deque::WorkerRunQueue;
use crate::resumable::common::worker::{AsyncTaskPool, TaskPool, UltWorker};
use crate::resumable::common::system::{DescScheduler, SchedulerSystem};
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::stackful::worker::{ContextSwitcher, StackfulLocalQueue};
use crate::resumable::common::desc::SuspendedTaskToken;
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::stackless::desc::AsyncTaskDesc;

/// `execute` body for dual systems: today's original logic — check
/// `poll_fn` first, and either poll inline or perform a real context switch.
///
/// Lowest rung reachable: the fold trait [`StackfulSchedulerSystem`], not
/// lower — same as [`crate::resumable::stackful::worker::execute_stackful`]
/// for the sync-ULT branch (needs `ContextSwitcher`/`StackfulLocalQueue`,
/// which need `DescScheduler + StackfulWorkerSystem`), and the poll_fn
/// branch's `crate::resumable::stackless::worker::run_async_poll` needs
/// `StacklessSchedulerSystem` (itself `DescScheduler`-gated) regardless.
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
///
/// Lowest rung: [`DescScheduler`] + `S::Desc: AsyncTaskDesc` — same `Item`
/// pinning need as
/// [`crate::resumable::stackful::worker::pop_or_root_stackful`] (`wk.deque`
/// round-trips `S::Item`), plus `AsyncTaskDesc` for `is_poll_fn_dispatch`.
/// No context switch happens here, so — unlike `execute_dual` — this needs
/// neither `StackfulWorkerSystem` nor `StackfulTaskDesc`.
pub fn pop_or_root_dual<S>(wk: &UltWorker<S>) -> SuspendedTaskToken<S::Desc>
where
    S: DescScheduler,
    S::Desc: AsyncTaskDesc,
{
    if let Some(c) = wk.deque.try_pop() {
        if c.is_poll_fn_dispatch() {
            // Async tasks have no saved context; they can only be executed
            // by the scheduler loop via execute().  Push the async task back
            // so the scheduler loop handles it.
            wk.deque.push(c);
        } else {
            return c;
        }
    }
    wk.take_root_cont()
}

/// `free_finished_desc` body for dual systems: async tasks go through
/// `S::AsyncPool` (a separate pool from the ULT-stack `S::Pool`, see
/// [`PoolSystem::AsyncPool`](crate::resumable::common::system::PoolSystem::AsyncPool));
/// everything else goes through the ULT-stack pool as usual.
///
/// # Safety
/// No other references to `desc` may exist after this call (same contract
/// as [`TaskPool::free_task`]/[`AsyncTaskPool::free_async_task`]).
///
/// Lowest rung: plain [`SchedulerSystem`] + `S::Desc: AsyncTaskDesc` — the
/// biggest drop of the three dual functions: `desc` is a raw `*mut S::Desc`
/// throughout (never `S::Item`), so this needs neither `DescScheduler` (no
/// `Item`/`Worker` pinning) nor `StackfulWorkerSystem`/`StackfulTaskDesc` (no
/// context switch).
pub unsafe fn free_finished_desc_dual<S>(wk: &UltWorker<S>, desc: *mut S::Desc)
where
    S: SchedulerSystem,
    S::Desc: AsyncTaskDesc,
{
    // SAFETY: `desc` is finished (about to be freed) — `TaskDesc::join_state`'s
    // own contract guarantees the exit path never touches the descriptor
    // again after publishing `FINISHED`, so no other token can exist for
    // it; safe to construct one transiently just to read the dispatch tag.
    if unsafe { crate::resumable::common::desc::SuspendedTaskToken::from_raw(desc) }.is_poll_fn_dispatch() {
        unsafe { wk.free_async_task(desc) };
    } else {
        unsafe { wk.free_task(desc) };
    }
}
