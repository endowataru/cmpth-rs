//! Async waker integration for ULT-based systems.
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
//! Both modes use `ctx` (AtomicPtr) for the suspend/resume handshake:
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
use std::sync::atomic::Ordering;
use std::task::{Context, RawWaker, RawWakerVTable, Waker};

use crate::traits::Poller;
use crate::traits::thread_system::{noop_waker, ThreadSystem};
use crate::ult::desc::{StackfulTaskDesc, SuspendedUlt, WakeOutcome, WakerTaskDesc};
use crate::ult::external_queue::ExternalQueue;
use crate::ult::scheduler::Scheduler;
use crate::ult::system::{SchedulerSystem, UltSchedulerSystem};
use crate::ult::desc::TaskDesc;
use crate::ult::worker::{LocalQueue, StackfulWorker, UltWorker, Worker};

// ---------------------------------------------------------------------------
// Vtable singletons (one per concrete UltSystem type S)
// ---------------------------------------------------------------------------

struct PrivateVtable<S>(PhantomData<S>);
impl<S: UltSchedulerSystem> PrivateVtable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        clone_private::<S>,
        wake_private::<S>,
        wake_by_ref_private::<S>,
        drop_private::<S>,
    );
}

struct SharedVtable<S>(PhantomData<S>);
impl<S: UltSchedulerSystem> SharedVtable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        clone_shared::<S>,
        wake_shared::<S>,
        wake_by_ref_shared::<S>,
        drop_shared::<S>,
    );
}

fn private_vtable<S: UltSchedulerSystem>() -> &'static RawWakerVTable where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    &PrivateVtable::<S>::VTABLE
}

fn shared_vtable<S: UltSchedulerSystem>() -> &'static RawWakerVTable where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
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
pub struct UltPoller<S: UltSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    desc: Option<NonNull<S::Desc>>,
    waker: Waker,
    _marker: PhantomData<S>,
}

impl<S: UltSchedulerSystem> Poller for UltPoller<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    fn new() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        match UltWorker::<S>::current() {
            Some(wk) => {
                let desc = wk.cur_task.get();
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

impl<S: UltSchedulerSystem> Drop for UltPoller<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    fn drop(&mut self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        if let Some(desc) = self.desc {
            // Reset before self.waker drops, so racing wakers see IDLE.
            unsafe { desc.as_ref().mark_idle() };
        }
        // self.waker drops here (calls drop_private / drop_shared as needed).
    }
}

// ---------------------------------------------------------------------------
// Push continuation to a worker deque (used by wake)
// ---------------------------------------------------------------------------

/// # Safety
/// `desc` must be a currently-suspended task whose ctx has just been cleared.
unsafe fn push_continuation<S: SchedulerSystem>(desc: *mut S::Desc) where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    match UltWorker::<S>::current() {
        Some(wk) => wk.push_local_top(SuspendedUlt(desc)),
        None => {
            let scheduler = unsafe { (*desc).scheduler().get() };
            assert!(
                !scheduler.is_null(),
                "cmpth: wake() called from outside ULT scheduler \
                 and task has no scheduler reference"
            );
            let scheduler = unsafe { &*(scheduler as *const Scheduler<S>) };
            scheduler.external_queue.push(SuspendedUlt(desc));
        }
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
/// `desc` must point to a live `BasicTaskDesc`.
unsafe fn try_wake<S: UltSchedulerSystem>(desc: *const S::Desc) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    let desc = desc as *mut S::Desc;
    if let WakeOutcome::ClaimedParked = unsafe { (*desc).try_wake_state() } {
        // The ctx store (Release) in cond_shim happened-before the Acquire
        // CAS inside try_wake_state; we can now load ctx safely.
        let _ctx = unsafe { (*desc).ctx().load(Ordering::Relaxed) };
        debug_assert!(!_ctx.is_null());
        unsafe { push_continuation::<S>(desc) };
    }
}

// ---------------------------------------------------------------------------
// PRIVATE vtable functions
// ---------------------------------------------------------------------------

unsafe fn clone_private<S: UltSchedulerSystem>(ptr: *const ()) -> RawWaker where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    let desc = ptr as *const S::Desc;
    unsafe { (*desc).transition_to_shared() };
    // The clone uses the SHARED vtable; the original retains PRIVATE vtable
    // but its wake_private/drop_private check EVER_SHARED and dispatch correctly.
    RawWaker::new(ptr, shared_vtable::<S>())
}

unsafe fn wake_private<S: UltSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    // wake() consumes the waker.  For PRIVATE, there is no ref count to
    // decrement (the waker is part of the block_on frame).  If EVER_SHARED
    // was set after construction, delegate to the SHARED path for the drop.
    unsafe { wake_by_ref_private::<S>(ptr) };
    unsafe { drop_private::<S>(ptr) };
}

unsafe fn wake_by_ref_private<S: UltSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    let desc = ptr as *const S::Desc;
    if unsafe { (*desc).is_ever_shared() } {
        // Transitioned to SHARED after construction; use SHARED wake logic.
        unsafe { wake_by_ref_shared::<S>(ptr) };
    } else {
        unsafe { try_wake::<S>(desc) };
    }
}

unsafe fn drop_private<S: UltSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    let desc = ptr as *const S::Desc;
    if unsafe { (*desc).is_ever_shared() } {
        // The original waker is being dropped; treat like a SHARED drop.
        unsafe { drop_shared::<S>(ptr) };
    }
    // Pure PRIVATE: the waker is owned by block_on's stack frame; no action.
}

// ---------------------------------------------------------------------------
// Async task vtable
//
// Same as the block_on waker vtable but `try_wake_async` is used instead of
// `try_wake`.  The only difference: async tasks never store a saved `ctx`
// pointer (no context switch happens), so the `debug_assert!(!ctx.is_null())`
// in `try_wake` must not fire.
// ---------------------------------------------------------------------------

/// Like `try_wake` but skips the ctx non-null assertion.  Used for async
/// tasks where PARKED simply means "not in the deque", not "context saved".
///
/// Also the wake-side counterpart of `JoinState::AsyncJoiner` (see
/// `TaskDesc::try_register_async_joiner`): called directly, bypassing the
/// `Waker`/`RawWakerVTable` indirection entirely, since the registering side
/// (`JoinHandle::poll`) only takes that path when it already knows — from
/// `UltWorker::polling_async` — that going through a real `Waker` would have
/// dispatched here anyway.
pub(crate) unsafe fn try_wake_async<S: SchedulerSystem>(desc: *const S::Desc) where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    let desc = desc as *mut S::Desc;
    if let WakeOutcome::ClaimedParked = unsafe { (*desc).try_wake_state() } {
        // No ctx to load for async tasks; just push to deque.
        unsafe { push_continuation::<S>(desc) };
    }
}

struct AsyncPrivateVtable<S>(PhantomData<S>);
impl<S: SchedulerSystem> AsyncPrivateVtable<S> where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        clone_async_private::<S>,
        wake_async_private::<S>,
        wake_by_ref_async_private::<S>,
        drop_async_private::<S>,
    );
}

struct AsyncSharedVtable<S>(PhantomData<S>);
impl<S: SchedulerSystem> AsyncSharedVtable<S> where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        clone_async_shared::<S>,
        wake_async_shared::<S>,
        wake_by_ref_async_shared::<S>,
        drop_shared::<S>,
    );
}

pub(crate) fn async_task_private_vtable<S: SchedulerSystem>() -> &'static RawWakerVTable where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    &AsyncPrivateVtable::<S>::VTABLE
}

unsafe fn clone_async_private<S: SchedulerSystem>(ptr: *const ()) -> RawWaker where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    let desc = ptr as *const S::Desc;
    unsafe { (*desc).transition_to_shared() };
    RawWaker::new(ptr, &AsyncSharedVtable::<S>::VTABLE)
}

unsafe fn clone_async_shared<S: SchedulerSystem>(ptr: *const ()) -> RawWaker where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    let desc = ptr as *const S::Desc;
    unsafe { (*desc).incr_shared_ref() };
    RawWaker::new(ptr, &AsyncSharedVtable::<S>::VTABLE)
}

unsafe fn wake_async_private<S: SchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    unsafe { wake_by_ref_async_private::<S>(ptr) };
    unsafe { drop_async_private::<S>(ptr) };
}

unsafe fn wake_by_ref_async_private<S: SchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    let desc = ptr as *const S::Desc;
    if unsafe { (*desc).is_ever_shared() } {
        unsafe { wake_by_ref_async_shared::<S>(ptr) };
    } else {
        unsafe { try_wake_async::<S>(desc) };
    }
}

unsafe fn drop_async_private<S: SchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    let desc = ptr as *const S::Desc;
    if unsafe { (*desc).is_ever_shared() } {
        unsafe { drop_shared::<S>(ptr) };
    }
    // Pure PRIVATE: waker is owned by run_async_poll's stack frame; no action.
}

unsafe fn wake_async_shared<S: SchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    unsafe { wake_by_ref_async_shared::<S>(ptr) };
    unsafe { drop_shared::<S>(ptr) };
}

unsafe fn wake_by_ref_async_shared<S: SchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    unsafe { try_wake_async::<S>(ptr as *const S::Desc) };
}

// ---------------------------------------------------------------------------
// SHARED vtable functions
// ---------------------------------------------------------------------------

unsafe fn clone_shared<S: UltSchedulerSystem>(ptr: *const ()) -> RawWaker where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    let desc = ptr as *const S::Desc;
    unsafe { (*desc).incr_shared_ref() };
    RawWaker::new(ptr, shared_vtable::<S>())
}

unsafe fn wake_shared<S: UltSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    unsafe { wake_by_ref_shared::<S>(ptr) };
    unsafe { drop_shared::<S>(ptr) };
}

unsafe fn wake_by_ref_shared<S: UltSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    unsafe { try_wake::<S>(ptr as *const S::Desc) };
}

unsafe fn drop_shared<S: SchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    let desc = ptr as *const S::Desc;
    // If this was the last SHARED reference, the task is either still
    // running (block_on not done) or has already finished (block_on
    // returned with IDLE state).  Either way, no cleanup is needed:
    // BasicTaskDesc lifetime is managed by the scheduler, not by waker refs.
    unsafe { (*desc).decr_shared_ref() };
}
