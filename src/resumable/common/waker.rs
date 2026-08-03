//! Waker-adjacent machinery shared by the stackful ([`stackful::waker`](crate::resumable::stackful::waker)),
//! stackless ([`stackless::waker`](crate::resumable::stackless::waker)) `RawWaker`
//! vtable families, and stackful `block_on`'s [`ResumablePoller`](crate::resumable::stackful::waker::ResumablePoller):
//! pushing a woken continuation to a deque, dropping the last SHARED
//! reference, and the core POLLING/PARKED/NOTIFIED park/wake state machine.
//!
//! The state machine lives here (not in `WakerTaskDesc`) because it's
//! needed by two things with genuinely different backing storage:
//! `WakerTaskDesc` (`stackless/desc.rs`) stores it on the task descriptor
//! itself (`spawn_async` has no stack, so the descriptor is the only stable
//! anchor to hang wake state on); `ResumablePoller` stores it in a
//! block_on-call-scoped box (a real ULT already has a stable "where to
//! resume" via its saved context, so nothing needs to live on the
//! descriptor). Both drive the exact same proven CAS logic against
//! whatever `&AtomicUsize` they own.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::resumable::common::scheduler::Scheduler;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::worker::{LocalQueue, UltWorker, Worker};
use crate::resumable::common::desc::{HasBaseOwned, SuspendedTaskToken};
use crate::resumable::common::external_queue::ExternalQueue;

pub use crate::traits::common::WakeOutcome;

// ---------------------------------------------------------------------------
// waker state encoding
//
// bit 63:    EVER_SHARED — set on first clone of a waker; sticky forever.
//            Purely a PRIVATE/SHARED vtable-dispatch hint (see stackful/
//            stackless waker.rs) — nothing here counts clones, since
//            nothing ever frees based on a count reaching zero (descriptor/
//            slot lifetime is governed by the join protocol or the
//            enclosing block_on call, never by waker ref count).
// bits 0-1:  IDLE/POLLING/PARKED/NOTIFIED state.
// ---------------------------------------------------------------------------
pub(crate) const IDLE:        usize = 0;
pub(crate) const POLLING:     usize = 1;
pub(crate) const PARKED:      usize = 2;
pub(crate) const NOTIFIED:    usize = 3;
pub(crate) const EVER_SHARED: usize = 1 << 63;
pub(crate) const STATE_MASK:  usize = 3;

/// Reset to POLLING unconditionally. Called whenever a poll is about to
/// begin (`UltPoller::new`/`ResumablePoller::new`, `run_async_poll`'s
/// pre-poll mark).
#[inline]
pub(crate) fn mark_polling(state: &AtomicUsize) {
    state.store(POLLING, Ordering::Release);
}

/// Reset to IDLE unconditionally. Called when a poll session ends
/// (`UltPoller`/`ResumablePoller`'s `Drop`, `poll_spawned_task` invalidating
/// the waker before publishing the result so a racing `wake()` becomes a
/// no-op).
#[inline]
pub(crate) fn mark_idle(state: &AtomicUsize) {
    state.store(IDLE, Ordering::Release);
}

/// `wait`'s cond_suspend commit decision, run from inside the suspend shim
/// after the context is already saved: read the current state and either
/// commit to PARKED (`true`) or, if a wake already raced in during `poll()`
/// (state == NOTIFIED), cancel by resetting to POLLING (`false`) so the
/// caller re-polls immediately instead of parking.
///
/// CAS loop, matching [`park_after_poll`]. A plain load-then-store here
/// raced with [`try_wake_state`]'s CAS: if a concurrent `wake()` claimed
/// POLLING -> NOTIFIED between this method's load and store, the store
/// (computed from the stale POLLING read) clobbered NOTIFIED back to
/// PARKED. `try_wake_state` had already returned `SetNotified` (task
/// "still running, will notice on its own") and so never pushed a
/// continuation -- the task committed to PARKED with nobody left to wake
/// it, a permanent lost-wakeup livelock.
pub(crate) fn decide_park(state: &AtomicUsize) -> bool {
    loop {
        let refs = state.load(Ordering::Acquire);
        let s = refs & STATE_MASK;
        if s == NOTIFIED {
            let new = (refs & !STATE_MASK) | POLLING;
            if state.compare_exchange(refs, new, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                return false;
            }
        } else {
            let new = (refs & !STATE_MASK) | PARKED;
            if state.compare_exchange(refs, new, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                return true;
            }
        }
    }
}

/// `run_async_poll`'s post-poll transition after a `Pending` result: try to
/// park (POLLING -> PARKED, returns `true`). If a wake raced in during the
/// poll (state == NOTIFIED), claims a re-poll instead (NOTIFIED -> POLLING,
/// returns `false` so the caller re-queues the task immediately rather than
/// parking it).
///
/// CAS loop — unlike [`decide_park`], concurrent wakers can race this
/// transition (matches the original call site exactly).
#[inline]
pub(crate) fn park_after_poll(state: &AtomicUsize) -> bool {
    loop {
        let refs = state.load(Ordering::Acquire);
        let s = refs & STATE_MASK;
        if s == NOTIFIED {
            let new = (refs & !STATE_MASK) | POLLING;
            if state.compare_exchange(refs, new, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                return false;
            }
        } else {
            let new = (refs & !STATE_MASK) | PARKED;
            if state.compare_exchange(refs, new, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                return true;
            }
        }
    }
}

/// Core wake CAS loop, shared by every wake path (`stackful::waker`'s
/// `try_wake`, `stackless::waker`'s `try_wake_async`, and
/// `ResumablePoller`'s vtable).
pub(crate) fn try_wake_state(state: &AtomicUsize) -> WakeOutcome {
    loop {
        let refs = state.load(Ordering::Acquire);
        let s = refs & STATE_MASK;
        match s {
            s if s == POLLING => {
                let new = (refs & !STATE_MASK) | NOTIFIED;
                match state.compare_exchange(refs, new, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => return WakeOutcome::SetNotified,
                    Err(_) => continue,
                }
            }
            s if s == PARKED => {
                let new = (refs & !STATE_MASK) | POLLING;
                match state.compare_exchange(refs, new, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => return WakeOutcome::ClaimedParked,
                    Err(_) => continue,
                }
            }
            s if s == NOTIFIED => return WakeOutcome::NoOp,
            s if s == IDLE => return WakeOutcome::NoOp,
            _ => unreachable!("unexpected waker state: {:#x}", refs),
        }
    }
}

/// Takes an already-constructed token (rather than a raw pointer it would
/// have to re-validate) so the `ClaimedParked` callers that already hold
/// exclusive ownership at their call site don't have to hand off a raw
/// pointer just to have this function immediately reconstruct a token from
/// it — one `from_raw` per genuine ownership transfer, not two.
pub(crate) fn push_continuation<S: SchedulerSystem>(token: SuspendedTaskToken<S::Desc>) {
    match UltWorker::<S>::current() {
        Some(wk) => wk.push_local_top(token),
        None => {
            let scheduler = token.base().scheduler;
            assert!(
                !scheduler.is_null(),
                "cmpth: wake() called from outside ULT scheduler \
                 and task has no scheduler reference"
            );
            let scheduler = unsafe { &*(scheduler as *const Scheduler<S>) };
            scheduler.external_queue.push(token);
        }
    }
}

/// Bridge from `RawWaker`'s type-erased `*const ()` data pointer to a safe
/// `&D`. This is the one relay point the waker vtable boundary genuinely
/// needs to stay `unsafe` at — `RawWaker`'s contract, not anything about
/// `D`, is what makes `ptr`'s validity unprovable to the type system.
/// Every vtable function should call this exactly once, at the top, and use
/// only safe `&self` methods on the result afterward.
///
/// # Safety
/// `ptr` must be a live `*const D` disguised as `*const ()` (i.e. exactly
/// the data pointer a `RawWaker` for this `D` was constructed with).
pub(crate) unsafe fn desc_from_erased<D>(ptr: *const ()) -> &'static D {
    unsafe { &*(ptr as *const D) }
}

/// SHARED-vtable drop: a no-op. Kept as a named function (rather than
/// inlined away) so every SHARED vtable can point at the same symbol.
/// There is no ref count to decrement: nothing here ever frees based on a
/// waker clone count reaching zero (descriptor/slot lifetime is governed
/// by the join protocol or the enclosing block_on call), so tracking one
/// would be pure overhead — confirmed dead code the first time this was
/// written (the count was incremented/decremented but its value was never
/// read anywhere, back to the crate's initial commit).
pub(crate) fn drop_shared(_ptr: *const ()) {}
