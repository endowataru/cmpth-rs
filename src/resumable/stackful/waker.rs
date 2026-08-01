//! Async waker integration for stackful/dual `block_on`. See
//! [`common::waker`](crate::resumable::common::waker) for the pieces shared
//! with [`stackless::waker`](crate::resumable::stackless::waker), and this
//! module's parent for the full PRIVATE/SHARED state-machine design.
//!
//! The waker data pointer is `*mut S::Desc`.  Two vtables are used:
//!
//! * **PRIVATE** (`waker_refs & EVER_SHARED == 0`): the waker has never been
//!   cloned.  State transitions are driven by `waker_refs` bits 0-1
//!   (POLLING/PARKED/NOTIFIED).
//!
//! * **SHARED** (`waker_refs & EVER_SHARED != 0`): at least one clone exists.
//!   Bits 2-62 hold the ref count; state bits 0-1 are still used for
//!   POLLING/PARKED/NOTIFIED.  The EVER_SHARED flag is sticky.
//!
//! Both modes use `ctx` for the suspend/resume handshake:
//!   null     = task is running (POLLING or NOTIFIED)
//!   non-null = task is parked, value is the saved context pointer
//!
//! # Waker lifetime
//!
//! Wakers created by `block_on_ult` are valid for exactly the duration of
//! the `block_on` call.  SHARED wakers given to external subsystems must not
//! outlive the enclosing `block_on` invocation.  Wake calls after `block_on`
//! returns are silently ignored (the ctx CAS fails because the task is
//! running again).
//!
//! # Wake from outside the scheduler
//!
//! `wake()` in PARKED state requires finding a worker deque to push the
//! continuation.  Currently, this uses `UltWorker::<S>::current()` and
//! therefore requires that `wake()` is called from a thread running this
//! scheduler.  Calling `wake()` from an OS thread that is not a ULT worker
//! will panic.

use std::marker::PhantomData;
use std::ptr::NonNull;
use std::task::{Context, RawWaker, RawWakerVTable, Waker};

use crate::traits::Poller;
use crate::traits::stackful::{noop_waker, ThreadSystem};
use crate::resumable::common::desc::{WakeOutcome, WakerTaskDesc};
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::waker::{drop_shared, push_continuation};
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::common::worker::{UltWorker, Worker};
use crate::resumable::stackful::worker::StackfulWorker;

// ---------------------------------------------------------------------------
// Vtable singletons (one per concrete ThreadSystem type S)
// ---------------------------------------------------------------------------

struct PrivateVtable<S>(PhantomData<S>);
impl<S: StackfulSchedulerSystem> PrivateVtable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        clone_private::<S>,
        wake_private::<S>,
        wake_by_ref_private::<S>,
        drop_private::<S>,
    );
}

struct SharedVtable<S>(PhantomData<S>);
impl<S: StackfulSchedulerSystem> SharedVtable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        clone_shared::<S>,
        wake_shared::<S>,
        wake_by_ref_shared::<S>,
        drop_shared::<S>,
    );
}

fn private_vtable<S: StackfulSchedulerSystem>() -> &'static RawWakerVTable where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    &PrivateVtable::<S>::VTABLE
}

fn shared_vtable<S: StackfulSchedulerSystem>() -> &'static RawWakerVTable where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    &SharedVtable::<S>::VTABLE
}

// ---------------------------------------------------------------------------
// UltPoller — Poller implementation for ULT systems
// ---------------------------------------------------------------------------

/// [`Poller`] implementation for ULT systems.
///
/// In ULT mode (`desc` is `Some`): stores a real [`Waker`] backed by
/// `waker_refs` in the current descriptor.  [`wait`](Poller::wait) uses
/// `cond_suspend_to_sched` with NOTIFIED-cancel logic.
///
/// In fallback mode (`desc` is `None`, called from outside the scheduler):
/// stores a no-op waker and [`wait`](Poller::wait) busy-polls via
/// `S::Base::yield_now`.
///
/// This type is `!Send`: it is bound to the same ULT.  In cmpth, `!Send`
/// means "bound to the same ULT", not "bound to the same OS thread" — the
/// scheduler moves the entire ULT stack atomically on steal.
pub struct UltPoller<S: StackfulSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    desc: Option<NonNull<S::Desc>>,
    waker: Waker,
    _marker: PhantomData<S>,
}

impl<S: StackfulSchedulerSystem> Poller for UltPoller<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    fn new() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        match UltWorker::<S>::current() {
            Some(wk) => {
                let desc = wk.cur_task();
                unsafe { (*desc).mark_polling() };
                let raw = RawWaker::new(desc as *const (), private_vtable::<S>());
                let waker = unsafe { Waker::from_raw(raw) };
                UltPoller {
                    desc: NonNull::new(desc),
                    waker,
                    _marker: PhantomData,
                }
            }
            None => UltPoller {
                desc: None,
                waker: noop_waker(),
                _marker: PhantomData,
            },
        }
    }

    fn context<'a>(&'a self) -> Context<'a> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        Context::from_waker(&self.waker)
    }

    fn wait(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        match self.desc {
            Some(desc) => {
                let desc = desc.as_ptr();
                UltWorker::<S>::current()
                    .expect("UltPoller::wait called from outside scheduler")
                    .cond_suspend_to_sched(|_wk, prev_opt| {
                        // wake() fired during poll(): decide_park cancels
                        // (resets to POLLING) and returns false; otherwise
                        // it commits to PARKED and we consume prev_opt.
                        if unsafe { (*desc).decide_park() } {
                            let _ = prev_opt.take().unwrap().into_raw();
                        }
                    });
            }
            None => S::Base::yield_now(),
        }
    }
}

impl<S: StackfulSchedulerSystem> Drop for UltPoller<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    fn drop(&mut self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        if let Some(desc) = self.desc {
            // Reset before self.waker drops, so racing wakers see IDLE.
            unsafe { desc.as_ref().mark_idle() };
        }
        // self.waker drops here (calls drop_private / drop_shared as needed).
    }
}

// ---------------------------------------------------------------------------
// Core wake logic (shared between PRIVATE and SHARED paths)
// ---------------------------------------------------------------------------

/// Attempt to wake the task.  State machine:
///   POLLING  → NOTIFIED  (task is running; it will re-poll automatically)
///   PARKED   → POLLING   (task is suspended; push the continuation)
///   NOTIFIED → (no-op)
///   IDLE     → (no-op, stale wake after block_on returned)
///
/// Returns `true` if a continuation was pushed (caller must not decrement refs
/// before this; the push races with the task freeing itself).
///
/// # Safety
/// `desc` must point to a live `DualTaskDesc`.
unsafe fn try_wake<S: StackfulSchedulerSystem>(desc: *const S::Desc) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    let desc = desc as *mut S::Desc;
    if let WakeOutcome::ClaimedParked = unsafe { (*desc).try_wake_state() } {
        // ctx is a plain field now (see HasCtx's doc comment) — this
        // debug_assert-only read was already relying on exactly that
        // invariant even when ctx was atomic: the Acquire CAS inside
        // try_wake_state already happened-after the ctx store (Release) in
        // cond_shim, so no ordering of ctx's own is needed here either way.
        // `desc` is genuinely parked here (ClaimedParked), so constructing
        // a transient token just to peek is sound.
        let _ctx = crate::resumable::common::desc::SuspendedTaskToken(desc).peek_saved_context();
        debug_assert!(!_ctx.is_null());
        unsafe { push_continuation::<S>(desc) };
    }
}

// ---------------------------------------------------------------------------
// PRIVATE vtable functions
// ---------------------------------------------------------------------------

unsafe fn clone_private<S: StackfulSchedulerSystem>(ptr: *const ()) -> RawWaker where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    let desc = ptr as *const S::Desc;
    unsafe { (*desc).transition_to_shared() };
    // The clone uses the SHARED vtable; the original retains PRIVATE vtable
    // but its wake_private/drop_private check EVER_SHARED and dispatch correctly.
    RawWaker::new(ptr, shared_vtable::<S>())
}

unsafe fn wake_private<S: StackfulSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    // wake() consumes the waker.  For PRIVATE, there is no ref count to
    // decrement (the waker is part of the block_on frame).  If EVER_SHARED
    // was set after construction, delegate to the SHARED path for the drop.
    unsafe { wake_by_ref_private::<S>(ptr) };
    unsafe { drop_private::<S>(ptr) };
}

unsafe fn wake_by_ref_private<S: StackfulSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    let desc = ptr as *const S::Desc;
    if unsafe { (*desc).is_ever_shared() } {
        // Transitioned to SHARED after construction; use SHARED wake logic.
        unsafe { wake_by_ref_shared::<S>(ptr) };
    } else {
        unsafe { try_wake::<S>(desc) };
    }
}

unsafe fn drop_private<S: StackfulSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    let desc = ptr as *const S::Desc;
    if unsafe { (*desc).is_ever_shared() } {
        // The original waker is being dropped; treat like a SHARED drop.
        unsafe { drop_shared::<S>(ptr) };
    }
    // Pure PRIVATE: the waker is owned by block_on's stack frame; no action.
}

// ---------------------------------------------------------------------------
// SHARED vtable functions
// ---------------------------------------------------------------------------

unsafe fn clone_shared<S: StackfulSchedulerSystem>(ptr: *const ()) -> RawWaker where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    let desc = ptr as *const S::Desc;
    unsafe { (*desc).incr_shared_ref() };
    RawWaker::new(ptr, shared_vtable::<S>())
}

unsafe fn wake_shared<S: StackfulSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    unsafe { wake_by_ref_shared::<S>(ptr) };
    unsafe { drop_shared::<S>(ptr) };
}

unsafe fn wake_by_ref_shared<S: StackfulSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    unsafe { try_wake::<S>(ptr as *const S::Desc) };
}
