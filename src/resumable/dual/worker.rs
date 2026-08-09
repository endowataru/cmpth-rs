//! Dual-only `StackfulSchedulerSystem::pop_or_root` body and the
//! [`RunnableItem`]/[`ReclaimableDesc`] impls for [`DualTaskDesc`]: a popped
//! continuation may be either a real ULT or a `spawn_async` task, so
//! dispatch needs the `poll_fn` tag check the stackful-only/stackless-only
//! bodies don't pay for. See
//! [`common::worker`](crate::resumable::common::worker) and
//! [`stackful::worker`](crate::resumable::stackful::worker) for the shared
//! machinery this builds on.

use crate::resumable::common::deque::WorkerRunQueue;
use crate::resumable::common::worker::{AsyncTaskPool, TaskPool, UltWorker};
use crate::resumable::common::system::{DescScheduler, ReclaimableDesc, RunnableItem, WorkerSystem};
use crate::resumable::stackful::system::StackfulWorkerSystem;
use crate::resumable::stackful::worker::{ContextSwitcher, StackfulLocalQueue};
use crate::resumable::common::desc::SuspendedTaskToken;
use crate::resumable::dual::desc::DualTaskDesc;
use crate::resumable::stackless::desc::AsyncTaskDesc;

/// `cont` may be either a real ULT or a `spawn_async` task — check
/// `poll_fn` first, and either poll inline or perform a real context switch.
///
/// Bound: `StackfulWorkerSystem + DescScheduler<Desc = DualTaskDesc<S>>` —
/// strictly below `SchedulerSystem`, same reasoning as
/// [`stackful::worker`](crate::resumable::stackful::worker)'s
/// `RunnableItem` impl for the sync-ULT branch (`ContextSwitcher`/
/// `StackfulLocalQueue` need exactly this). The poll_fn branch's
/// `crate::resumable::stackless::worker::run_async_poll` needs
/// `StacklessSchedulerSystem`, which becomes derivable for `S` from this
/// same bound once this impl (plus the matching `ReclaimableDesc` impl
/// below) exist — `StacklessSchedulerSystem` doesn't need to be named here.
///
/// `Desc` is pinned only via `DescScheduler<Desc = ...>`, not restated on
/// `StackfulWorkerSystem`, for the reason spelled out on `stackful::worker`'s
/// `RunnableItem` impl (and on `StackfulWorkerSystem`'s own doc comment —
/// this dual impl is in fact the concrete case that first exposed the
/// problem: `HasPollFn`, needed here for `is_poll_fn_dispatch`/`poll_fn`
/// and transitively by `AsyncTaskDesc`/`StacklessSchedulerSystem` for
/// `run_async_poll`, stopped normalizing entirely as long as
/// `StackfulWorkerSystem` carried *any* nested bound on `Desc`, even one
/// with nothing to do with `HasPollFn`).
impl<S: StackfulWorkerSystem + DescScheduler<Desc = DualTaskDesc<S>>>
    RunnableItem<S> for SuspendedTaskToken<DualTaskDesc<S>>
{
    fn run_on(self, wk: &UltWorker<S>) {
        let desc = self.desc();
        if self.is_poll_fn_dispatch() {
            let poll_fn = self.poll_fn()
                .expect("cmpth: descriptor committed to poll_fn dispatch but poll_fn unset");
            let _ = self.into_raw(); // consumed; no context switch
            crate::resumable::stackless::worker::run_async_poll(wk, desc, poll_fn);
        } else {
            // Sync ULT: context switch as usual.
            let wk2 = wk.suspend_to_cont(self, |wk, prev| wk.set_root_cont(prev));
            debug_assert!(std::ptr::eq(wk2 as *const UltWorker<S>, wk as *const UltWorker<S>));
        }
    }
}

/// `pop_or_root` body for dual systems: today's original logic — an async
/// task popped off the top has no saved context to switch into, so requeue
/// it and fall back to the root (scheduler-loop) continuation instead.
///
/// Lowest rung: plain `WorkerSystem` + `S::Desc: AsyncTaskDesc` — same
/// conversion need as
/// [`crate::resumable::stackful::worker::pop_or_root_stackful`] (`wk.deque`
/// round-trips `S::SuspendedToken`, converted via the `Into`/`From` bounds on
/// [`crate::resumable::common::system::WorkerSystem::SuspendedToken`] rather
/// than an equality pin), plus `AsyncTaskDesc` for `is_poll_fn_dispatch`. No
/// context switch happens here, so — unlike `execute_dual` — this needs
/// neither `StackfulWorkerSystem` nor `StackfulTaskDesc`.
pub fn pop_or_root_dual<S>(wk: &UltWorker<S>) -> SuspendedTaskToken<S::Desc>
where
    S: WorkerSystem,
    S::Desc: AsyncTaskDesc,
{
    if let Some(c) = wk.deque.try_pop() {
        let c: SuspendedTaskToken<S::Desc> = c.into();
        if c.is_poll_fn_dispatch() {
            // Async tasks have no saved context; they can only be executed
            // by the scheduler loop via execute().  Push the async task back
            // so the scheduler loop handles it.
            wk.deque.push(c.into());
        } else {
            return c;
        }
    }
    wk.take_root_cont()
}

/// Async tasks go through `S::AsyncPool` (a separate pool from the
/// ULT-stack `S::Pool`, see
/// [`PoolSystem::AsyncPool`](crate::resumable::common::system::PoolSystem::AsyncPool));
/// everything else goes through the ULT-stack pool as usual.
///
/// Bound: plain [`DescScheduler`] — `desc` is a raw `*mut Self` throughout
/// (never `S::Item`), and `wk.free_task`/`wk.free_async_task` are
/// `WorkerSystem`-gated, so this needs neither `StackfulWorkerSystem` nor
/// any fold of `SchedulerSystem`. `DescScheduler` itself is only needed to
/// pin `Desc = DualTaskDesc<S>` so `Self` resolves.
impl<S: DescScheduler<Desc = DualTaskDesc<S>>> ReclaimableDesc<S> for DualTaskDesc<S> {
    unsafe fn reclaim(wk: &UltWorker<S>, desc: *mut Self) {
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
}
