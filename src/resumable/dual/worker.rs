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
use crate::resumable::common::system::{ReclaimableDesc, RunnableItem, WorkerSystem};
use crate::resumable::stackful::system::StackfulWorkerSystem;
use crate::resumable::stackful::worker::{ContextSwitcher, StackfulLocalQueue};
use crate::resumable::common::desc::SuspendedTaskToken;
use crate::resumable::dual::desc::DualTaskDesc;
use crate::resumable::stackless::desc::AsyncTaskDesc;

/// `cont` may be either a real ULT or a `spawn_async` task — check
/// `poll_fn` first, and either poll inline or perform a real context switch.
///
/// Bound: `StackfulWorkerSystem + WorkerSystem<SuspendedToken =
/// SuspendedTaskToken<Self::Desc>, Desc = DualTaskDesc<S>>`, plus
/// `S::Worker: ContextSwitcher<S> + StackfulLocalQueue<S> + AsyncTaskPool<S>`
/// — strictly below `SchedulerSystem`, and no `DescScheduler` (no `Worker`
/// identity pin at all): the sync-ULT branch only needs `ContextSwitcher`/
/// `StackfulLocalQueue` as *capabilities* on `S::Worker` (same reasoning as
/// the stackful-only `RunnableItem` impl), and the poll_fn branch's
/// `crate::resumable::stackless::worker::run_async_poll` now takes `&S::Worker`
/// too (its old `polling_async`/`yield_requested` field reads were promoted
/// to `WorkerOps` methods). `AsyncTaskPool` is needed transitively: `run_async_poll`
/// requires `S: StacklessSchedulerSystem` -> `SchedulerSystem` ->
/// `Desc: ReclaimableDesc<Self>`, and the matching `ReclaimableDesc` impl
/// below needs exactly this.
///
/// `SuspendedToken` still has to be pinned, though (same reasoning as the
/// stackless-only `RunnableItem` impl's doc comment): `SchedulerSystem`'s
/// own blanket derive rule is stated as `Self::SuspendedToken:
/// RunnableItem<Self>` (the *opaque* associated type), only provable if
/// `S::SuspendedToken` is known equal to the concrete
/// `SuspendedTaskToken<DualTaskDesc<S>>` this impl is written for.
///
/// `Desc` is pinned directly on `WorkerSystem`, not restated on
/// `StackfulWorkerSystem`, for the reason spelled out on `stackful::worker`'s
/// `RunnableItem` impl (and on `StackfulWorkerSystem`'s own doc comment —
/// this dual impl is in fact the concrete case that first exposed the
/// problem: `HasPollFn`, needed here for `is_poll_fn_dispatch`/`poll_fn`
/// and transitively by `AsyncTaskDesc`/`StacklessSchedulerSystem` for
/// `run_async_poll`, stopped normalizing entirely as long as
/// `StackfulWorkerSystem` carried *any* nested bound on `Desc`, even one
/// with nothing to do with `HasPollFn`).
impl<S> RunnableItem<S> for SuspendedTaskToken<DualTaskDesc<S>>
where
    S: StackfulWorkerSystem
        + WorkerSystem<SuspendedToken = SuspendedTaskToken<<S as crate::resumable::common::system::PoolSystem>::Desc>, Desc = DualTaskDesc<S>>,
    S::Worker: ContextSwitcher<S> + StackfulLocalQueue<S> + AsyncTaskPool<S>,
{
    fn run_on(self, wk: &S::Worker) {
        let desc = self.desc();
        if self.is_poll_fn_dispatch() {
            let poll_fn = self.poll_fn()
                .expect("cmpth: descriptor committed to poll_fn dispatch but poll_fn unset");
            let _ = self.into_raw(); // consumed; no context switch
            crate::resumable::stackless::worker::run_async_poll::<S>(wk, desc, poll_fn);
        } else {
            // Sync ULT: context switch as usual.
            let wk2 = wk.suspend_to_cont(self, |wk, prev| wk.set_root_cont(prev));
            debug_assert!(std::ptr::eq(wk2 as *const S::Worker, wk as *const S::Worker));
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
/// Bound: plain [`WorkerSystem`], `Desc` pinned directly (not via
/// `DescScheduler`) — `desc` is a raw
/// `*mut Self` throughout (never `S::SuspendedToken`), `wk.free_task` is
/// reachable through `S::Worker: WorkerOps<S>` alone, and `wk.free_async_task`
/// needs the separate `S::Worker: AsyncTaskPool<S>` bound stated below (not a
/// `WorkerOps` supertrait) — so this needs neither `Worker = UltWorker<S>`
/// nor any fold of `SchedulerSystem`.
impl<S: WorkerSystem<Desc = DualTaskDesc<S>>> ReclaimableDesc<S> for DualTaskDesc<S>
where
    S::Worker: AsyncTaskPool<S>,
{
    unsafe fn reclaim(wk: &S::Worker, desc: *mut Self) {
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
