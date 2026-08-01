//! Async waker integration for `spawn_async` tasks.
//!
//! Same PRIVATE/SHARED state machine as
//! [`stackful::waker`](crate::resumable::stackful::waker)'s `block_on`
//! waker, but `try_wake_async` is used instead of `try_wake`.  The only
//! difference: async tasks never store a saved `ctx` pointer (no context
//! switch happens), so `try_wake`'s `debug_assert!(!ctx.is_null())` doesn't
//! apply here.

use std::marker::PhantomData;
use std::task::{RawWaker, RawWakerVTable};

use crate::resumable::common::desc::{WakeOutcome, WakerTaskDesc};
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::waker::{desc_from_erased, drop_shared, push_continuation};

/// Like `try_wake` (in `stackful::waker`) but skips the ctx non-null
/// assertion.  Used for async tasks where PARKED simply means "not in the
/// deque", not "context saved".
///
/// Also the wake-side counterpart of `JoinState::AsyncJoiner` (see
/// `TaskDesc::try_register_async_joiner`): called directly, bypassing the
/// `Waker`/`RawWakerVTable` indirection entirely, since the registering side
/// (`JoinHandle::poll`) only takes that path when it already knows — from
/// `UltWorker::polling_async` — that going through a real `Waker` would have
/// dispatched here anyway.
pub(crate) unsafe fn try_wake_async<S: SchedulerSystem>(desc: *const S::Desc) where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    let desc_ptr = desc as *mut S::Desc;
    let desc: &S::Desc = unsafe { &*desc };
    if let WakeOutcome::ClaimedParked = desc.try_wake_state() {
        // No ctx to load for async tasks; just push to deque.
        unsafe { push_continuation::<S>(desc_ptr) };
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
    let desc: &S::Desc = unsafe { desc_from_erased(ptr) };
    desc.transition_to_shared();
    RawWaker::new(ptr, &AsyncSharedVtable::<S>::VTABLE)
}

unsafe fn clone_async_shared<S: SchedulerSystem>(ptr: *const ()) -> RawWaker where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    let desc: &S::Desc = unsafe { desc_from_erased(ptr) };
    desc.incr_shared_ref();
    RawWaker::new(ptr, &AsyncSharedVtable::<S>::VTABLE)
}

unsafe fn wake_async_private<S: SchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    unsafe { wake_by_ref_async_private::<S>(ptr) };
    unsafe { drop_async_private::<S>(ptr) };
}

unsafe fn wake_by_ref_async_private<S: SchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    let desc: &S::Desc = unsafe { desc_from_erased(ptr) };
    if desc.is_ever_shared() {
        unsafe { wake_by_ref_async_shared::<S>(ptr) };
    } else {
        unsafe { try_wake_async::<S>(desc as *const S::Desc) };
    }
}

unsafe fn drop_async_private<S: SchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    let desc: &S::Desc = unsafe { desc_from_erased(ptr) };
    if desc.is_ever_shared() {
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
