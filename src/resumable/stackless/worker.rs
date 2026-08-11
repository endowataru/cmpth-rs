//! Stackless-only/dual dispatch: driving a `spawn_async` task's poll loop,
//! and the [`RunnableItem`]/[`ReclaimableDesc`] impls for
//! [`StacklessOnlyTaskDesc`]. See
//! [`common::worker`](crate::resumable::common::worker) for the base traits
//! and [`UltWorker<S>`](crate::resumable::common::worker::UltWorker) itself.

use std::task::{RawWaker, Waker};

use crate::resumable::common::worker::{AsyncTaskPool, DescWorkerOps, LocalQueue};
use crate::resumable::common::system::{PoolSystem, ReclaimableDesc, RunnableItem, WorkerSystem};
use crate::resumable::stackless::system::StacklessSchedulerSystem;
use crate::resumable::common::desc::{RunningTaskToken, SuspendedTaskToken};
use crate::resumable::stackless::desc::{StacklessOnlyTaskDesc, WakerTaskDesc};
use crate::resumable::stackless::desc::{TaskPollFn, TaskPollResult};

/// Drive one async task's poll to completion or a suspend point. Called
/// from the dual `RunnableItem` impl (`resumable::dual::worker`, when
/// `desc.poll_fn` is `Some`) and from the stackless-only one below
/// (always). Needs `S: StacklessSchedulerSystem`: polling a `spawn_async`
/// task never touches context-switch machinery, so a stackless-only system
/// needs this exactly as much as a dual one does.
///
/// A `loop`, not a single poll: when a completion reports
/// [`TaskPollResult::ReadyAndContinue`] (its completion directly claimed a
/// waiting `AsyncJoiner`), this continues straight into that descriptor's
/// own poll on the next iteration instead of returning control to the
/// outer dispatch loop — symmetric transfer, skipping a deque push/pop
/// round trip for the common case where a parent was waiting on exactly
/// the task that just finished. A `loop` rather than a recursive call, so
/// an arbitrarily long completion chain (however deep the fork-join
/// recursion) costs no native call-stack depth.
///
/// # Safety (invariant, not `unsafe fn`)
/// `desc` must already be exclusively owned by the caller for the duration
/// of this call — every caller establishes this via a real
/// `SuspendedTaskToken` (consumed with `into_raw()`, or implicitly via
/// `try_pop()`'s own single-consumer guarantee plus the token's `Drop`
/// being a no-op) before passing the raw pointer in.
pub(crate) fn run_async_poll<S>(
    wk: &S::Worker,
    mut desc: *mut S::Desc,
    mut poll_fn: TaskPollFn<S::Desc>,
) where
    S: StacklessSchedulerSystem,
    S::Worker: DescWorkerOps<S>,
{
    // Whatever this worker was polling (if anything) before this call —
    // restored once the chain below is done. Unlike the pre-2026-07-30
    // version, the restore happens *before* a task we're done with
    // becomes reachable by another thread (deque push), not after —
    // otherwise `polling_async` briefly claims we're still driving a
    // descriptor that has already left synchronous driving, which
    // `JoinHandle::poll`'s fast path could observe as a false positive.
    let prev_polling = wk.polling_async();

    loop {
        // One relay point per iteration (`desc` is reassigned on symmetric
        // transfer below, so this can't be hoisted above the loop).
        let desc_ref: &S::Desc = unsafe { &*desc };

        // Mark as POLLING so the waker's state machine works correctly.
        desc_ref.mark_polling();

        let raw = RawWaker::new(desc as *const (), crate::resumable::stackless::waker::async_task_private_vtable::<S>());
        let waker = unsafe { Waker::from_raw(raw) };

        // Record that `desc` is the task this worker is polling right now,
        // so `JoinHandle::poll` (reachable synchronously from `poll_fn`
        // below via any `.await` on a child) can recognize its ambient
        // waker as this task's own instead of boxing a fresh one.
        wk.set_polling_async(desc);

        let mut cx = std::task::Context::from_waker(&waker);
        let result = unsafe { poll_fn(desc, &mut cx) };

        // waker is dropped here; drop_async_private is a no-op for PRIVATE mode.
        drop(waker);

        // Consume any `yield_now()` request made during *this* poll --
        // unconditionally, before branching on `result`, so a `true` left
        // by a task that turns out to complete (`Ready`/`ReadyAndContinue`)
        // rather than actually self-wake-and-park never leaks into
        // whatever this worker polls next. Only the `Pending` arm below
        // ever acts on it.
        let yield_requested = wk.take_yield_requested();

        match result {
            TaskPollResult::Ready => {
                wk.set_polling_async(prev_polling);
                return;
            }
            TaskPollResult::Pending => {
                // Park, unless a wake raced in during poll() -- then
                // re-queue immediately instead. `polling_async` is
                // restored *before* the deque push, not after: once
                // pushed, `desc` is immediately stealable by another
                // worker, so the marker must stop claiming we're driving
                // it before that happens, not a couple of statements
                // later.
                let parked = desc_ref.park_after_poll();
                wk.set_polling_async(prev_polling);
                if !parked {
                    // SAFETY: `park_after_poll` returning `false` means no
                    // wake raced in — this call is still the sole owner of
                    // `desc` (same contract as this function's own entry
                    // invariant, still holding since poll_fn returned
                    // control back to us without handing `desc` to anyone
                    // else).
                    let token = unsafe { SuspendedTaskToken::from_raw(desc) };
                    if yield_requested {
                        // Fair yield: behind whatever this worker already
                        // had queued, not ahead of it -- see
                        // `StacklessTaskSystem::yield_now`.
                        wk.defer(token.into());
                    } else {
                        wk.push(token.into());
                    }
                }
                return;
            }
            TaskPollResult::ReadyAndContinue(next) => {
                // SAFETY: per `TaskPollResult::ReadyAndContinue`'s own
                // contract, `next` is only ever constructed from a
                // `try_wake_state` `ClaimedParked` outcome, which proves
                // nobody else can be concurrently polling that descriptor.
                poll_fn = unsafe { RunningTaskToken::from_raw(next) }.poll_fn().expect(
                    "cmpth: symmetric-transfer target has no poll_fn (not a spawn_async task)",
                );
                desc = next;
                // loop: poll `next` directly, no deque round trip.
            }
        }
    }
}

/// Every popped continuation is a `spawn_async` task, so always poll — no
/// `poll_fn` tag check, because there is nothing else it could be.
///
/// Bound: `WorkerSystem<SuspendedToken = SuspendedTaskToken<Self::Desc>,
/// Desc = StacklessOnlyTaskDesc<S>>` — strictly below `SchedulerSystem`, and
/// no `Worker` pin at all: `run_async_poll` now takes `wk: &S::Worker`
/// (its old `polling_async`/`yield_requested` field reads were promoted to
/// `WorkerOps` methods, so it no longer needs the concrete `UltWorker<S>`
/// type). `SuspendedToken` still has to be pinned, though: `run_async_poll`
/// needs `S: StacklessSchedulerSystem`, which extends `SchedulerSystem`,
/// whose own blanket derive rule is stated as `Self::SuspendedToken:
/// RunnableItem<Self>` (the *opaque* associated type) — only provable if
/// `S::SuspendedToken` is known equal to the concrete
/// `SuspendedTaskToken<StacklessOnlyTaskDesc<S>>` this impl is written for
/// (an impl for a concrete type doesn't automatically count as one for an
/// unrelated opaque type). It becomes derivable for `S` from this same
/// bound once this impl (plus the matching `ReclaimableDesc` impl below)
/// exist, so it doesn't need to be named here.
impl<S: WorkerSystem<SuspendedToken = SuspendedTaskToken<<S as PoolSystem>::Desc>> + PoolSystem<Desc = StacklessOnlyTaskDesc<S>>>
    RunnableItem<S> for SuspendedTaskToken<StacklessOnlyTaskDesc<S>>
where
    // Needed transitively: `run_async_poll` requires `S: StacklessSchedulerSystem`,
    // which requires `S::Desc: ReclaimableDesc<S>` (via `SchedulerSystem`),
    // and the matching `ReclaimableDesc` impl below needs exactly this.
    S::Worker: AsyncTaskPool<S> + DescWorkerOps<S>,
{
    fn run_on(self, wk: &S::Worker) {
        let desc = self.desc();
        let poll_fn = self.poll_fn()
            .expect("cmpth: execute_async called on a continuation with no poll_fn (not a spawn_async task)");
        let _ = self.into_raw(); // consumed; no context switch
        run_async_poll::<S>(wk, desc, poll_fn);
    }
}

/// Every descriptor is a `spawn_async` allocation, so always route it
/// through `S::AsyncPool` (which itself decides pool-return vs. raw-free
/// based on whether the descriptor's `Node` wrapper was marked oversized at
/// allocation time).
///
/// Bound: plain [`WorkerSystem`], `Desc` pinned directly (not via
/// `DescScheduler`) — `wk.free_async_task`
/// needs `S::Worker: AsyncTaskPool<S>` (stated below; not a `WorkerOps`
/// supertrait); `desc` is a raw `*mut Self`, never `S::SuspendedToken`, so
/// this needs neither `Worker = UltWorker<S>` nor any fold of
/// `SchedulerSystem`.
impl<S: WorkerSystem + PoolSystem<Desc = StacklessOnlyTaskDesc<S>>> ReclaimableDesc<S> for StacklessOnlyTaskDesc<S>
where
    S::Worker: AsyncTaskPool<S>,
{
    unsafe fn reclaim(wk: &S::Worker, desc: *mut Self) {
        unsafe { wk.free_async_task(desc) };
    }
}
