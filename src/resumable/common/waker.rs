//! Waker-adjacent machinery shared by both the stackful/dual
//! ([`stackful::waker`](crate::resumable::stackful::waker)) and stackless
//! ([`stackless::waker`](crate::resumable::stackless::waker)) `RawWaker`
//! vtable families: pushing a woken continuation to a deque, and dropping
//! the last SHARED reference. Both families' `..._shared`/
//! `..._private`-delegation functions call directly into these.

use crate::resumable::common::scheduler::Scheduler;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::worker::{LocalQueue, UltWorker, Worker};
use crate::resumable::common::desc::{HasBaseOwned, SuspendedTaskToken, WakerTaskDesc};
use crate::resumable::common::external_queue::ExternalQueue;

/// # Safety
/// `desc` must be a currently-suspended task whose ctx has just been cleared.
pub(crate) unsafe fn push_continuation<S: SchedulerSystem>(desc: *mut S::Desc) where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    let token = SuspendedTaskToken(desc);
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

pub(crate) unsafe fn drop_shared<S: SchedulerSystem>(ptr: *const ()) where <S as SchedulerSystem>::Desc: WakerTaskDesc {
    let desc: &S::Desc = unsafe { desc_from_erased(ptr) };
    // If this was the last SHARED reference, the task is either still
    // running (block_on not done) or has already finished (block_on
    // returned with IDLE state).  Either way, no cleanup is needed:
    // DualTaskDesc lifetime is managed by the scheduler, not by waker refs.
    desc.decr_shared_ref();
}
