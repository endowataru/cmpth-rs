//! ULT-layer parked-continuation interface and default implementation.

use std::marker::PhantomData;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::traits::{Resumable, StackfulResumable};
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::common::desc::SuspendedTaskToken;
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::common::worker::{LocalQueue, UltWorker, Worker};
use crate::resumable::stackful::worker::{ContextSwitcher, StackfulWorker};

/// Raw parked-continuation storage: a single atomic slot, the publication
/// point between the parking worker and a concurrent notifier. Implementing
/// this opts a type into the [`Resumable`]/[`StackfulResumable`] operations
/// below for free via the blanket impls — the same two-tier relationship as
/// [`TaskDescCore`](crate::resumable::common::desc::TaskDescCore)/[`TaskDesc`](crate::resumable::common::desc::TaskDesc).
/// Swap in a different struct (e.g. one with profiling counters) by
/// implementing `cont()`.
pub trait StackfulOnlyResumableCore: Send + Default
where
    <Self::StackfulSchedulerSystem as SchedulerSystem>::Desc: crate::resumable::stackful::desc::StackfulTaskDesc,
{
    type StackfulSchedulerSystem: StackfulSchedulerSystem;

    /// Access the raw continuation slot.
    ///
    /// The slot is atomic because it is the publication point between the
    /// parking worker and a concurrent notifier: the parker stores the slot
    /// with `Release` *after* the context save, and consumers take it with an
    /// `Acquire` swap.  On weakly-ordered machines (AArch64) a plain store
    /// here can become visible before the saved context does, letting the
    /// notifier resume a continuation whose frame is not yet written.
    fn cont(&self) -> &AtomicPtr<<Self::StackfulSchedulerSystem as SchedulerSystem>::Desc>;

    // --- helpers shared by the blanket impls below ---------------------------

    fn take_cont(&self) -> SuspendedTaskToken<<Self::StackfulSchedulerSystem as SchedulerSystem>::Desc> {
        // `swap` (not load+store) so that concurrent take/cancel pairs can
        // never both obtain the continuation.
        let c = self.cont().swap(ptr::null_mut(), Ordering::Acquire);
        assert!(!c.is_null(), "StackfulOnlyResumableCore: no parked continuation");
        // SAFETY: `c` is a pointer published via `into_raw()` by
        // `wait_with`/`wait_with_cond` (`Release` store into this same
        // `cont()` slot); the `Acquire` swap above pairs with that store
        // and the slot only ever holds one such pointer at a time, so this
        // swap is the sole consumer.
        unsafe { SuspendedTaskToken::from_raw(c) }
    }

    fn wk() -> &'static UltWorker<Self::StackfulSchedulerSystem> {
        UltWorker::<Self::StackfulSchedulerSystem>::current()
            .expect("cmpth: StackfulOnlyResumableCore operation called outside a worker")
    }
}

/// Blanket: any [`StackfulOnlyResumableCore`] automatically implements the
/// top-level wait-slot traits. `enter`/`swap` always perform a real context
/// switch here — the slot can only ever hold a real continuation, unlike
/// `DualResumable`, which may hold an async waiter and has to fall back to
/// a plain wake internally.
impl<T: StackfulOnlyResumableCore> Resumable<T::StackfulSchedulerSystem> for T {
    fn is_set(&self) -> bool {
        !self.cont().load(Ordering::Acquire).is_null()
    }

    fn notify(&self) {
        let c = self.take_cont();
        Self::wk().push_local_top(c);
    }
}

impl<T: StackfulOnlyResumableCore> StackfulResumable<T::StackfulSchedulerSystem> for T {
    fn wait_with<F: FnOnce()>(&self, f: F) {
        type D<T> = <<T as StackfulOnlyResumableCore>::StackfulSchedulerSystem as SchedulerSystem>::Desc;
        let slot = self.cont() as *const AtomicPtr<D<Self>>;
        Self::wk().suspend_to_sched(move |_wk, prev| {
            // Release: publishes the context saved just before this callback.
            unsafe { (*slot).store(prev.into_raw(), Ordering::Release) };
            f();
        });
    }

    fn wait_with_cond<F: FnOnce() -> bool>(&self, f: F) {
        type D<T> = <<T as StackfulOnlyResumableCore>::StackfulSchedulerSystem as SchedulerSystem>::Desc;
        let slot = self.cont() as *const AtomicPtr<D<Self>>;
        Self::wk().cond_suspend_to_sched(move |_wk, prev| {
            unsafe {
                (*slot).store(prev.take().unwrap().into_raw(), Ordering::Release)
            };
            if !f() {
                let c = unsafe { (*slot).swap(ptr::null_mut(), Ordering::Acquire) };
                debug_assert!(!c.is_null());
                // SAFETY: same reasoning as `take_cont` — `c` was published
                // into this same slot a few lines up by this same closure
                // (`Release` store), and this `Acquire` swap is the sole
                // consumer of that publish.
                *prev = Some(unsafe { SuspendedTaskToken::from_raw(c) });
            }
        });
    }

    fn enter(&self) {
        let wk = Self::wk();
        let c = self.take_cont();
        wk.suspend_to_cont(c, |wk, prev| wk.push_local_top(prev));
    }

    fn swap(&self, next: &Self) {
        type D<T> = <<T as StackfulOnlyResumableCore>::StackfulSchedulerSystem as SchedulerSystem>::Desc;
        debug_assert!(!self.is_set());
        let wk = Self::wk();
        let c = next.take_cont();
        let slot = self.cont() as *const AtomicPtr<D<Self>>;
        wk.suspend_to_cont(c, move |_wk, prev| {
            unsafe { (*slot).store(prev.into_raw(), Ordering::Release) };
        });
    }
}

/// Single-slot parked-continuation implementation.  Implements
/// [`StackfulOnlyResumableCore`] by providing just the `cont()` accessor;
/// all behaviour comes from the blanket [`Resumable`]/[`StackfulResumable`]
/// impls above.
pub struct BasicStackfulOnlyResumable<S: StackfulSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    cont: AtomicPtr<S::Desc>,
    _marker: PhantomData<S>,
}

unsafe impl<S: StackfulSchedulerSystem> Send for BasicStackfulOnlyResumable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}

impl<S: StackfulSchedulerSystem> Default for BasicStackfulOnlyResumable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    fn default() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc { Self::new() }
}

impl<S: StackfulSchedulerSystem> BasicStackfulOnlyResumable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    pub const fn new() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        BasicStackfulOnlyResumable { cont: AtomicPtr::new(ptr::null_mut()), _marker: PhantomData }
    }
}

impl<S: StackfulSchedulerSystem> StackfulOnlyResumableCore for BasicStackfulOnlyResumable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    type StackfulSchedulerSystem = S;
    fn cont(&self) -> &AtomicPtr<S::Desc> where <S as SchedulerSystem>::Desc: StackfulTaskDesc { &self.cont }
}
