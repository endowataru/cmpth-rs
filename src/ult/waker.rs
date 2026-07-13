//! Async waker integration for ULT-based systems.
//!
//! The waker data pointer is `*mut UltDesc`.  Two vtables are used:
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
use crate::ult::desc::{
    SuspendedUlt, UltDesc, EVER_SHARED, IDLE, NOTIFIED, PARKED, POLLING, REF_ONE, STATE_MASK,
};
use crate::ult::external_queue::ExternalQueue;
use crate::ult::scheduler::Scheduler;
use crate::ult::system::UltSchedulerSystem;
use crate::ult::worker::{LocalQueue, UltWorker, Worker};

// ---------------------------------------------------------------------------
// Vtable singletons (one per concrete UltSystem type S)
// ---------------------------------------------------------------------------

struct PrivateVtable<S>(PhantomData<S>);
impl<S: UltSchedulerSystem> PrivateVtable<S> {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        clone_private::<S>,
        wake_private::<S>,
        wake_by_ref_private::<S>,
        drop_private::<S>,
    );
}

struct SharedVtable<S>(PhantomData<S>);
impl<S: UltSchedulerSystem> SharedVtable<S> {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        clone_shared::<S>,
        wake_shared::<S>,
        wake_by_ref_shared::<S>,
        drop_shared::<S>,
    );
}

fn private_vtable<S: UltSchedulerSystem>() -> &'static RawWakerVTable {
    &PrivateVtable::<S>::VTABLE
}

fn shared_vtable<S: UltSchedulerSystem>() -> &'static RawWakerVTable {
    &SharedVtable::<S>::VTABLE
}

// ---------------------------------------------------------------------------
// UltPoller — Poller implementation for ULT systems
// ---------------------------------------------------------------------------

/// [`Poller`] implementation for ULT systems.
///
/// In ULT mode (`desc` is `Some`): stores a real [`Waker`] backed by
/// `waker_refs` in the current [`UltDesc`].  [`wait`](Poller::wait) uses
/// `cond_suspend_to_sched` with NOTIFIED-cancel logic.
///
/// In fallback mode (`desc` is `None`, called from outside the scheduler):
/// stores a no-op waker and [`wait`](Poller::wait) busy-polls via
/// `S::Base::yield_now`.
///
/// This type is `!Send`: it is bound to the same ULT.  In cmpth, `!Send`
/// means "bound to the same ULT", not "bound to the same OS thread" — the
/// scheduler moves the entire ULT stack atomically on steal.
pub struct UltPoller<S: UltSchedulerSystem> {
    desc: Option<NonNull<UltDesc>>,
    waker: Waker,
    _marker: PhantomData<S>,
}

impl<S: UltSchedulerSystem> Poller for UltPoller<S> {
    fn new() -> Self {
        match UltWorker::<S>::current() {
            Some(wk) => {
                let desc = wk.cur_task.get();
                unsafe { (*desc).waker_refs.store(POLLING, Ordering::Release) };
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

    fn context<'a>(&'a self) -> Context<'a> {
        Context::from_waker(&self.waker)
    }

    fn wait(&self) {
        match self.desc {
            Some(desc) => {
                let desc = desc.as_ptr();
                UltWorker::<S>::current()
                    .expect("UltPoller::wait called from outside scheduler")
                    .cond_suspend_to_sched(|_wk, prev_opt| {
                        let refs = unsafe { (*desc).waker_refs.load(Ordering::Relaxed) };
                        let state = refs & STATE_MASK;
                        if state == NOTIFIED {
                            // wake() fired during poll(); cancel park and re-poll.
                            let new = (refs & !STATE_MASK) | POLLING;
                            unsafe { (*desc).waker_refs.store(new, Ordering::Release) };
                        } else {
                            // Commit: POLLING → PARKED.
                            let new = (refs & !STATE_MASK) | PARKED;
                            unsafe { (*desc).waker_refs.store(new, Ordering::Release) };
                            let _ = prev_opt.take().unwrap().into_raw();
                        }
                    });
            }
            None => S::Base::yield_now(),
        }
    }
}

impl<S: UltSchedulerSystem> Drop for UltPoller<S> {
    fn drop(&mut self) {
        if let Some(desc) = self.desc {
            // Reset before self.waker drops, so racing wakers see IDLE.
            unsafe { desc.as_ref().waker_refs.store(IDLE, Ordering::Release) };
        }
        // self.waker drops here (calls drop_private / drop_shared as needed).
    }
}

// ---------------------------------------------------------------------------
// Push continuation to a worker deque (used by wake)
// ---------------------------------------------------------------------------

/// # Safety
/// `desc` must be a currently-suspended task whose ctx has just been cleared.
unsafe fn push_continuation<S: UltSchedulerSystem>(desc: *mut UltDesc) {
    match UltWorker::<S>::current() {
        Some(wk) => wk.push_local_top(SuspendedUlt(desc)),
        None => {
            let scheduler = unsafe { (*desc).scheduler };
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
/// `desc` must point to a live `UltDesc`.
unsafe fn try_wake<S: UltSchedulerSystem>(desc: *const UltDesc) {
    let desc = desc as *mut UltDesc;
    loop {
        let refs = unsafe { (*desc).waker_refs.load(Ordering::Acquire) };
        let state = refs & STATE_MASK;

        match state {
            s if s == POLLING => {
                // Task is running; request a re-poll by setting NOTIFIED.
                let new = (refs & !STATE_MASK) | NOTIFIED;
                match unsafe {
                    (*desc).waker_refs.compare_exchange(refs, new, Ordering::AcqRel, Ordering::Acquire)
                } {
                    Ok(_) => return,
                    Err(_) => continue,
                }
            }
            s if s == PARKED => {
                // Task is suspended; claim it and push continuation.
                let new = (refs & !STATE_MASK) | POLLING;
                match unsafe {
                    (*desc).waker_refs.compare_exchange(refs, new, Ordering::AcqRel, Ordering::Acquire)
                } {
                    Ok(_) => {
                        // The ctx store (Release) in cond_shim happened-before
                        // our Acquire CAS; we can now load ctx safely.
                        let _ctx = unsafe { (*desc).ctx.load(Ordering::Relaxed) };
                        debug_assert!(!_ctx.is_null());
                        unsafe { push_continuation::<S>(desc) };
                        return;
                    }
                    Err(_) => continue,
                }
            }
            s if s == NOTIFIED => return, // already notified
            s if s == IDLE => return,     // stale wake
            _ => unreachable!("unexpected waker_refs: {:#x}", refs),
        }
    }
}

// ---------------------------------------------------------------------------
// PRIVATE vtable functions
// ---------------------------------------------------------------------------

unsafe fn clone_private<S: UltSchedulerSystem>(ptr: *const ()) -> RawWaker {
    let desc = ptr as *const UltDesc;
    // Transition to SHARED: set EVER_SHARED, init ref count to 2
    // (original waker + this new clone).
    //
    // Use a CAS loop to preserve the current state bits while setting
    // EVER_SHARED.  Concurrent wake() may change the state bits.
    loop {
        let old = unsafe { (*desc).waker_refs.load(Ordering::Relaxed) };
        let new = EVER_SHARED | (2 * REF_ONE) | (old & STATE_MASK);
        match unsafe {
            (*desc).waker_refs.compare_exchange(old, new, Ordering::Release, Ordering::Relaxed)
        } {
            Ok(_) => break,
            Err(_) => continue,
        }
    }
    // The clone uses the SHARED vtable; the original retains PRIVATE vtable
    // but its wake_private/drop_private check EVER_SHARED and dispatch correctly.
    RawWaker::new(ptr, shared_vtable::<S>())
}

unsafe fn wake_private<S: UltSchedulerSystem>(ptr: *const ()) {
    // wake() consumes the waker.  For PRIVATE, there is no ref count to
    // decrement (the waker is part of the block_on frame).  If EVER_SHARED
    // was set after construction, delegate to the SHARED path for the drop.
    unsafe { wake_by_ref_private::<S>(ptr) };
    unsafe { drop_private::<S>(ptr) };
}

unsafe fn wake_by_ref_private<S: UltSchedulerSystem>(ptr: *const ()) {
    let desc = ptr as *const UltDesc;
    let refs = unsafe { (*desc).waker_refs.load(Ordering::Relaxed) };
    if refs & EVER_SHARED != 0 {
        // Transitioned to SHARED after construction; use SHARED wake logic.
        unsafe { wake_by_ref_shared::<S>(ptr) };
    } else {
        unsafe { try_wake::<S>(desc) };
    }
}

unsafe fn drop_private<S: UltSchedulerSystem>(ptr: *const ()) {
    let desc = ptr as *const UltDesc;
    let refs = unsafe { (*desc).waker_refs.load(Ordering::Relaxed) };
    if refs & EVER_SHARED != 0 {
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
unsafe fn try_wake_async<S: UltSchedulerSystem>(desc: *const UltDesc) {
    let desc = desc as *mut UltDesc;
    loop {
        let refs = unsafe { (*desc).waker_refs.load(Ordering::Acquire) };
        let state = refs & STATE_MASK;
        match state {
            s if s == POLLING => {
                let new = (refs & !STATE_MASK) | NOTIFIED;
                match unsafe {
                    (*desc).waker_refs.compare_exchange(refs, new, Ordering::AcqRel, Ordering::Acquire)
                } {
                    Ok(_) => return,
                    Err(_) => continue,
                }
            }
            s if s == PARKED => {
                let new = (refs & !STATE_MASK) | POLLING;
                match unsafe {
                    (*desc).waker_refs.compare_exchange(refs, new, Ordering::AcqRel, Ordering::Acquire)
                } {
                    Ok(_) => {
                        // No ctx to load for async tasks; just push to deque.
                        unsafe { push_continuation::<S>(desc) };
                        return;
                    }
                    Err(_) => continue,
                }
            }
            s if s == NOTIFIED => return,
            s if s == IDLE => return,
            _ => unreachable!("unexpected waker_refs: {:#x}", refs),
        }
    }
}

struct AsyncPrivateVtable<S>(PhantomData<S>);
impl<S: UltSchedulerSystem> AsyncPrivateVtable<S> {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        clone_async_private::<S>,
        wake_async_private::<S>,
        wake_by_ref_async_private::<S>,
        drop_async_private::<S>,
    );
}

struct AsyncSharedVtable<S>(PhantomData<S>);
impl<S: UltSchedulerSystem> AsyncSharedVtable<S> {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        clone_async_shared::<S>,
        wake_async_shared::<S>,
        wake_by_ref_async_shared::<S>,
        drop_shared::<S>,
    );
}

pub(crate) fn async_task_private_vtable<S: UltSchedulerSystem>() -> &'static RawWakerVTable {
    &AsyncPrivateVtable::<S>::VTABLE
}

unsafe fn clone_async_private<S: UltSchedulerSystem>(ptr: *const ()) -> RawWaker {
    let desc = ptr as *const UltDesc;
    loop {
        let old = unsafe { (*desc).waker_refs.load(Ordering::Relaxed) };
        let new = EVER_SHARED | (2 * REF_ONE) | (old & STATE_MASK);
        match unsafe {
            (*desc).waker_refs.compare_exchange(old, new, Ordering::Release, Ordering::Relaxed)
        } {
            Ok(_) => break,
            Err(_) => continue,
        }
    }
    RawWaker::new(ptr, &AsyncSharedVtable::<S>::VTABLE)
}

unsafe fn clone_async_shared<S: UltSchedulerSystem>(ptr: *const ()) -> RawWaker {
    let desc = ptr as *const UltDesc;
    unsafe { (*desc).waker_refs.fetch_add(REF_ONE, Ordering::Relaxed) };
    RawWaker::new(ptr, &AsyncSharedVtable::<S>::VTABLE)
}

unsafe fn wake_async_private<S: UltSchedulerSystem>(ptr: *const ()) {
    unsafe { wake_by_ref_async_private::<S>(ptr) };
    unsafe { drop_async_private::<S>(ptr) };
}

unsafe fn wake_by_ref_async_private<S: UltSchedulerSystem>(ptr: *const ()) {
    let desc = ptr as *const UltDesc;
    let refs = unsafe { (*desc).waker_refs.load(Ordering::Relaxed) };
    if refs & EVER_SHARED != 0 {
        unsafe { wake_by_ref_async_shared::<S>(ptr) };
    } else {
        unsafe { try_wake_async::<S>(desc) };
    }
}

unsafe fn drop_async_private<S: UltSchedulerSystem>(ptr: *const ()) {
    let desc = ptr as *const UltDesc;
    let refs = unsafe { (*desc).waker_refs.load(Ordering::Relaxed) };
    if refs & EVER_SHARED != 0 {
        unsafe { drop_shared::<S>(ptr) };
    }
    // Pure PRIVATE: waker is owned by run_async_poll's stack frame; no action.
}

unsafe fn wake_async_shared<S: UltSchedulerSystem>(ptr: *const ()) {
    unsafe { wake_by_ref_async_shared::<S>(ptr) };
    unsafe { drop_shared::<S>(ptr) };
}

unsafe fn wake_by_ref_async_shared<S: UltSchedulerSystem>(ptr: *const ()) {
    unsafe { try_wake_async::<S>(ptr as *const UltDesc) };
}

// ---------------------------------------------------------------------------
// SHARED vtable functions
// ---------------------------------------------------------------------------

unsafe fn clone_shared<S: UltSchedulerSystem>(ptr: *const ()) -> RawWaker {
    let desc = ptr as *const UltDesc;
    // Increment ref count (bits 2+) without touching state bits.
    unsafe { (*desc).waker_refs.fetch_add(REF_ONE, Ordering::Relaxed) };
    RawWaker::new(ptr, shared_vtable::<S>())
}

unsafe fn wake_shared<S: UltSchedulerSystem>(ptr: *const ()) {
    unsafe { wake_by_ref_shared::<S>(ptr) };
    unsafe { drop_shared::<S>(ptr) };
}

unsafe fn wake_by_ref_shared<S: UltSchedulerSystem>(ptr: *const ()) {
    unsafe { try_wake::<S>(ptr as *const UltDesc) };
}

unsafe fn drop_shared<S: UltSchedulerSystem>(ptr: *const ()) {
    let desc = ptr as *const UltDesc;
    // Decrement ref count.  If this was the last SHARED reference, the task
    // is either still running (block_on not done) or has already finished
    // (block_on returned with IDLE state).  Either way, no cleanup is needed:
    // UltDesc lifetime is managed by the scheduler, not by waker refs.
    unsafe { (*desc).waker_refs.fetch_sub(REF_ONE, Ordering::Release) };
}
