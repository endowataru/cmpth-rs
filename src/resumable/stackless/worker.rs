//! Stackless-only/dual dispatch: driving a `spawn_async` task's poll loop,
//! and the stackless-only `SchedulerSystem::execute`/`free_finished_desc`
//! bodies. See [`common::worker`](crate::resumable::common::worker) for the
//! base traits and [`UltWorker<S>`](crate::resumable::common::worker::UltWorker)
//! itself.

use std::task::{RawWaker, Waker};

use crate::resumable::common::worker::{LocalQueue, UltWorker};
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::desc::{RunningTaskToken, SuspendedTaskToken};
use crate::resumable::stackless::desc::WakerTaskDesc;
use crate::resumable::stackless::desc::{AsyncTaskDesc, TaskPollFn, TaskPollResult};
use crate::resumable::common::pool::DescPool;

/// Drive one async task's poll to completion or a suspend point. Called
/// from `execute_dual` (when `desc.poll_fn` is `Some`) and from
/// [`execute_async`] (always). Base-level (`S: SchedulerSystem`): polling a
/// `spawn_async` task never touches context-switch machinery, so a
/// stackless-only system needs this exactly as much as a dual one does.
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
/// `pop_local()`'s own single-consumer guarantee plus the token's `Drop`
/// being a no-op) before passing the raw pointer in.
pub(crate) fn run_async_poll<S>(
    wk: &UltWorker<S>,
    mut desc: *mut S::Desc,
    mut poll_fn: TaskPollFn<S::Desc>,
) where
    S: SchedulerSystem,
    S::Desc: AsyncTaskDesc,
{
    // Whatever this worker was polling (if anything) before this call —
    // restored once the chain below is done. Unlike the pre-2026-07-30
    // version, the restore happens *before* a task we're done with
    // becomes reachable by another thread (deque push), not after —
    // otherwise `polling_async` briefly claims we're still driving a
    // descriptor that has already left synchronous driving, which
    // `JoinHandle::poll`'s fast path could observe as a false positive.
    let prev_polling = wk.polling_async.get();

    loop {
        // One relay point per iteration (`desc` is reassigned on symmetric
        // transfer below, so this can't be hoisted above the loop).
        let desc_ref: &S::Desc = unsafe { &*desc };

        // Mark as POLLING so the waker's state machine works correctly.
        desc_ref.mark_polling();

        // Same bookkeeping the stackful switch shims do on every real
        // context switch: publish `wk` on the descriptor itself (and, for
        // an arena-backed AsyncPool, on the cell slot too). Lets anything
        // holding a pointer into this task's own arena cell — e.g.
        // `JoinHandle::poll`'s `self` address, see
        // `worker_from_async_arena_addr` — find `wk` via address masking
        // instead of a TLS lookup.
        // SAFETY: covered by this function's own contract above — `desc`
        // is already exclusively owned by the caller. The temporary token
        // is just let go afterward (no `Drop` glue, so nothing is
        // invalidated) since `desc` itself stays the loop's working
        // pointer — `poll_fn` below needs it raw, not as a token.
        unsafe { RunningTaskToken::from_raw(desc) }.mark_resumed_on(wk as *const UltWorker<S> as *const ());

        let raw = RawWaker::new(desc as *const (), crate::resumable::stackless::waker::async_task_private_vtable::<S>());
        let waker = unsafe { Waker::from_raw(raw) };

        // Record that `desc` is the task this worker is polling right now,
        // so `JoinHandle::poll` (reachable synchronously from `poll_fn`
        // below via any `.await` on a child) can recognize its ambient
        // waker as this task's own instead of boxing a fresh one.
        wk.polling_async.set(desc);

        let mut cx = std::task::Context::from_waker(&waker);
        let result = unsafe { poll_fn(desc, &mut cx) };

        // waker is dropped here; drop_async_private is a no-op for PRIVATE mode.
        drop(waker);

        match result {
            TaskPollResult::Ready => {
                wk.polling_async.set(prev_polling);
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
                wk.polling_async.set(prev_polling);
                if !parked {
                    // SAFETY: `park_after_poll` returning `false` means no
                    // wake raced in — this call is still the sole owner of
                    // `desc` (same contract as this function's own entry
                    // invariant, still holding since poll_fn returned
                    // control back to us without handing `desc` to anyone
                    // else).
                    wk.push_local_top(unsafe { SuspendedTaskToken::from_raw(desc) });
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

/// `execute` body for stackless-only systems: every popped continuation is
/// a `spawn_async` task, so always poll — no `poll_fn` tag check, because
/// there is nothing else it could be.
pub fn execute_async<S>(wk: &UltWorker<S>, cont: SuspendedTaskToken<S::Desc>)
where
    S: SchedulerSystem,
    S::Desc: AsyncTaskDesc,
{
    let desc = cont.desc();
    let poll_fn = cont.poll_fn()
        .expect("cmpth: execute_async called on a continuation with no poll_fn (not a spawn_async task)");
    let _ = cont.into_raw(); // consumed; no context switch
    run_async_poll(wk, desc, poll_fn);
}

/// `free_finished_desc` body for stackless-only systems: every descriptor
/// is a `spawn_async` allocation, so always route it through `S::AsyncPool`
/// (which itself decides pool-return vs. raw-free based on whether the
/// descriptor's `Node` wrapper was marked oversized at allocation time).
///
/// # Safety
/// No other references to `desc` may exist after this call (same contract
/// as [`DescPool::dealloc`]).
pub unsafe fn free_finished_desc_async<S>(wk: &UltWorker<S>, desc: *mut S::Desc)
where
    S: SchedulerSystem,
{
    unsafe { wk.shared().async_task_pool.dealloc(wk.num(), desc) };
}
