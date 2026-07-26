//! [`SuspendedFuture`] — the pure-async wait-slot from
//! `docs/sync-async-unification.md`.

use std::marker::PhantomData;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::task::{Context, Waker};

use crate::traits::{Resumable, StacklessResumable};
use crate::resumable::stackless::system::StacklessSystem;

/// Pure-async wait slot: holds zero or one registered [`Waker`].
///
/// Deliberately holds a *standard* `Waker`, not cmpth's internal `UltDesc`
/// pointer — this is what lets a primitive built on this type (e.g. an
/// async-only mutex) compose with any executor, not just cmpth's own,
/// matching [`crate::future::yield_now`]'s executor-agnostic philosophy.
pub struct SuspendedFuture<S: StacklessSystem> {
    waker: AtomicPtr<Waker>,
    _marker: PhantomData<S>,
}

unsafe impl<S: StacklessSystem> Send for SuspendedFuture<S> {}
unsafe impl<S: StacklessSystem> Sync for SuspendedFuture<S> {}

impl<S: StacklessSystem> Default for SuspendedFuture<S> {
    fn default() -> Self {
        SuspendedFuture { waker: AtomicPtr::new(ptr::null_mut()), _marker: PhantomData }
    }
}

impl<S: StacklessSystem> Resumable<S> for SuspendedFuture<S> {
    fn is_set(&self) -> bool {
        !self.waker.load(Ordering::Acquire).is_null()
    }

    fn notify(&self) {
        let ptr = self.waker.swap(ptr::null_mut(), Ordering::AcqRel);
        if !ptr.is_null() {
            let w = unsafe { Box::from_raw(ptr) };
            w.wake();
        }
    }
}

impl<S: StacklessSystem> StacklessResumable<S> for SuspendedFuture<S> {
    fn register(&self, cx: &mut Context<'_>) {
        let boxed = Box::new(cx.waker().clone());
        let ptr = Box::into_raw(boxed);
        // Release: publishes the waker before a concurrent notify() can
        // observe it via the Acquire swap above.
        let old = self.waker.swap(ptr, Ordering::AcqRel);
        debug_assert!(old.is_null(), "SuspendedFuture::register called on an already-set slot");
    }
}
