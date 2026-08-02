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

use std::cell::Cell;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::task::{Context, RawWaker, RawWakerVTable, Waker};

use crate::traits::Poller;
use crate::traits::stackful::{noop_waker, ThreadSystem};
use crate::resumable::stackless::desc::WakerTaskDesc;
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::waker::{self, WakeOutcome, desc_from_erased, drop_shared, push_continuation};
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
        drop_shared,
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
                wk.cur_task_ref().mark_polling();
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
                let desc: &S::Desc = unsafe { desc.as_ref() };
                UltWorker::<S>::current()
                    .expect("UltPoller::wait called from outside scheduler")
                    .cond_suspend_to_sched(|_wk, prev_opt| {
                        // wake() fired during poll(): decide_park cancels
                        // (resets to POLLING) and returns false; otherwise
                        // it commits to PARKED and we consume prev_opt.
                        if desc.decide_park() {
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
// ResumablePoller — Poller implementation for stackful-only systems
// ---------------------------------------------------------------------------

/// Owner-exclusive-once-PARKED state for [`ResumablePoller`]: the same
/// POLLING/PARKED/NOTIFIED core [`WakerTaskDesc`] uses, driven against a
/// block_on-call-scoped allocation instead of a task descriptor field —
/// see `common/desc.rs`'s module doc comment for why `block_on` doesn't
/// need the descriptor-embedded version. `cont` rides on `state`'s CAS the
/// same way `ctx` rides on the switch shims' context-slot CAS (see
/// [`HasCtx`](crate::resumable::stackful::desc::HasCtx)'s doc comment):
/// written before the PARKED-committing CAS, read only by whichever
/// `wake()` call wins the matching PARKED->POLLING CAS, so a plain `Cell`
/// needs no atomicity of its own.
struct ResumablePollerSlot<S: StackfulSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    state: AtomicUsize,
    cont: Cell<*mut S::Desc>,
    _marker: PhantomData<S>,
}

// Sound for the same reason task descriptors themselves are `Send + Sync`
// despite raw-pointer/Cell fields: `cont`'s exclusive access is proven by
// `state`'s own CAS, not by the type system, so `Cell` here is safe to
// share across the threads that concurrently call `wake()`/`wake_by_ref()`.
unsafe impl<S: StackfulSchedulerSystem> Send for ResumablePollerSlot<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}
unsafe impl<S: StackfulSchedulerSystem> Sync for ResumablePollerSlot<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}

struct ResumableVtable<S>(PhantomData<S>);
impl<S: StackfulSchedulerSystem> ResumableVtable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        resumable_clone::<S>,
        resumable_wake::<S>,
        resumable_wake_by_ref::<S>,
        resumable_drop::<S>,
    );
}

/// `Arc`-backed: unlike a task descriptor (whose lifetime is governed by
/// the join protocol, independent of how many `Waker` clones exist), this
/// slot's *only* owner is whichever `Waker`/`ResumablePoller` values are
/// currently alive — so, unlike `WakerTaskDesc`'s dead ref count (nothing
/// there was ever freed based on it), this ref count is load-bearing: it's
/// what lets a `Waker` clone that outlives the `block_on` call (a
/// documented misuse, but one that must stay memory-safe, not become UB)
/// still point at valid memory. `Arc::into_raw`/`from_raw` do the counting;
/// no hand-rolled atomics needed here.
unsafe fn resumable_clone<S: StackfulSchedulerSystem>(ptr: *const ()) -> RawWaker where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    unsafe { Arc::increment_strong_count(ptr as *const ResumablePollerSlot<S>) };
    RawWaker::new(ptr, &ResumableVtable::<S>::VTABLE)
}

unsafe fn resumable_wake<S: StackfulSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    unsafe { resumable_wake_by_ref::<S>(ptr) };
    unsafe { resumable_drop::<S>(ptr) };
}

unsafe fn resumable_wake_by_ref<S: StackfulSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    let slot: &ResumablePollerSlot<S> = unsafe { desc_from_erased(ptr) };
    if let WakeOutcome::ClaimedParked = waker::try_wake_state(&slot.state) {
        let desc = slot.cont.get();
        unsafe { push_continuation::<S>(desc) };
    }
}

unsafe fn resumable_drop<S: StackfulSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    drop(unsafe { Arc::from_raw(ptr as *const ResumablePollerSlot<S>) });
}

/// [`Poller`] implementation for stackful-only (`S::Desc: StackfulTaskDesc`
/// only — no [`WakerTaskDesc`]) systems' `block_on`. Structurally identical
/// in spirit to [`UltPoller`], differing only in where the park/wake state
/// lives: a boxed (here, `Arc`ed) slot local to this `block_on` call
/// instead of a field on every task descriptor, regardless of whether that
/// descriptor's task ever calls `block_on`.
///
/// `wait`'s commit/cancel shape mirrors
/// [`StackfulResumable::wait_with_cond`](crate::traits::stackful::StackfulResumable::wait_with_cond)
/// (as implemented by [`StackfulOnlyResumableCore`](crate::resumable::stackful::suspended::StackfulOnlyResumableCore))
/// exactly — the same "wake raced in during poll" race that mutex/condvar
/// already resolve via `cond_suspend_to_sched`'s commit/cancel closure, no
/// separate PRIVATE/SHARED distinction needed since there's no ref count
/// to gate a fast path on (cloning is always just `Arc::increment_strong_count`).
///
/// `!Send`: bound to the same ULT, matching `UltPoller`.
pub struct ResumablePoller<S: StackfulSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    slot: Option<Arc<ResumablePollerSlot<S>>>,
    waker: Waker,
    _marker: PhantomData<S>,
}

impl<S: StackfulSchedulerSystem> Poller for ResumablePoller<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    fn new() -> Self {
        match UltWorker::<S>::current() {
            Some(_wk) => {
                let slot = Arc::new(ResumablePollerSlot::<S> {
                    state: AtomicUsize::new(waker::POLLING),
                    cont: Cell::new(std::ptr::null_mut()),
                    _marker: PhantomData,
                });
                let ptr = Arc::into_raw(Arc::clone(&slot)) as *const ();
                let raw = RawWaker::new(ptr, &ResumableVtable::<S>::VTABLE);
                let waker = unsafe { Waker::from_raw(raw) };
                ResumablePoller { slot: Some(slot), waker, _marker: PhantomData }
            }
            None => ResumablePoller { slot: None, waker: noop_waker(), _marker: PhantomData },
        }
    }

    fn context<'a>(&'a self) -> Context<'a> {
        Context::from_waker(&self.waker)
    }

    fn wait(&self) {
        match &self.slot {
            Some(slot) => {
                UltWorker::<S>::current()
                    .expect("ResumablePoller::wait called from outside scheduler")
                    .cond_suspend_to_sched(|_wk, prev| {
                        // Release: publishes both the context saved just
                        // before this callback (via `prev`'s own switch-shim
                        // ordering) and `cont` itself — see
                        // `ResumablePollerSlot`'s doc comment.
                        slot.cont.set(prev.as_ref().expect("cond_suspend contract").desc());
                        // wake() fired during poll(): decide_park cancels
                        // (resets to POLLING) and returns false; otherwise
                        // it commits to PARKED and we consume prev.
                        if waker::decide_park(&slot.state) {
                            let _ = prev.take().expect("cond_suspend contract").into_raw();
                        }
                        // else: leave `prev` in place -> cancel, resume at once
                    });
            }
            None => S::Base::yield_now(),
        }
    }
}

impl<S: StackfulSchedulerSystem> Drop for ResumablePoller<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    fn drop(&mut self) {
        if let Some(slot) = &self.slot {
            // Reset before self.waker drops, so racing wakers see IDLE
            // instead of reading a stale `cont`.
            waker::mark_idle(&slot.state);
        }
        // self.waker drops here (calls resumable_drop, releasing its own
        // Arc strong ref — the slot itself only actually frees once every
        // outstanding clone has done the same).
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
    let desc_ptr = desc as *mut S::Desc;
    let desc: &S::Desc = unsafe { &*desc };
    if let WakeOutcome::ClaimedParked = desc.try_wake_state() {
        // ctx is a plain field now (see HasCtx's doc comment) — this
        // debug_assert-only read was already relying on exactly that
        // invariant even when ctx was atomic: the Acquire CAS inside
        // try_wake_state already happened-after the ctx store (Release) in
        // cond_shim, so no ordering of ctx's own is needed here either way.
        // `desc` is genuinely parked here (ClaimedParked), so constructing
        // a transient token just to peek is sound.
        let _ctx = crate::resumable::common::desc::SuspendedTaskToken(desc_ptr).peek_saved_context();
        debug_assert!(!_ctx.is_null());
        unsafe { push_continuation::<S>(desc_ptr) };
    }
}

// ---------------------------------------------------------------------------
// PRIVATE vtable functions
// ---------------------------------------------------------------------------

unsafe fn clone_private<S: StackfulSchedulerSystem>(ptr: *const ()) -> RawWaker where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    let desc: &S::Desc = unsafe { desc_from_erased(ptr) };
    desc.transition_to_shared();
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
    let desc: &S::Desc = unsafe { desc_from_erased(ptr) };
    if desc.is_ever_shared() {
        // Transitioned to SHARED after construction; use SHARED wake logic.
        unsafe { wake_by_ref_shared::<S>(ptr) };
    } else {
        unsafe { try_wake::<S>(desc as *const S::Desc) };
    }
}

unsafe fn drop_private<S: StackfulSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    let desc: &S::Desc = unsafe { desc_from_erased(ptr) };
    if desc.is_ever_shared() {
        // The original waker is being dropped; treat like a SHARED drop.
        drop_shared(ptr);
    }
    // Pure PRIVATE: the waker is owned by block_on's stack frame; no action.
}

// ---------------------------------------------------------------------------
// SHARED vtable functions
// ---------------------------------------------------------------------------

unsafe fn clone_shared<S: StackfulSchedulerSystem>(ptr: *const ()) -> RawWaker where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    // No ref count to bump — see `common::waker::drop_shared`'s doc comment.
    RawWaker::new(ptr, shared_vtable::<S>())
}

unsafe fn wake_shared<S: StackfulSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    unsafe { wake_by_ref_shared::<S>(ptr) };
    drop_shared(ptr);
}

unsafe fn wake_by_ref_shared<S: StackfulSchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    unsafe { try_wake::<S>(ptr as *const S::Desc) };
}
