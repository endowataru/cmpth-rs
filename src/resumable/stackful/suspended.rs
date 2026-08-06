//! ULT-layer parked-continuation interface and default implementation.

use std::marker::PhantomData;
use std::sync::atomic::Ordering;

use crate::traits::{Resumable, StackfulResumable};
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::common::desc::SuspendedTaskToken;
use crate::interchange::AtomicSlot;
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::common::worker::{LocalQueue, UltWorker, WorkerOps};
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
    /// parking worker and a concurrent notifier: the parker publishes the
    /// slot with `Release` *after* the context save, and consumers take it
    /// with an `Acquire` swap.  On weakly-ordered machines (AArch64) a plain
    /// store here can become visible before the saved context does, letting
    /// the notifier resume a continuation whose frame is not yet written.
    fn cont(&self) -> &AtomicSlot<SuspendedTaskToken<<Self::StackfulSchedulerSystem as SchedulerSystem>::Desc>>;

    // --- helpers shared by the blanket impls below ---------------------------

    fn take_cont(&self) -> SuspendedTaskToken<<Self::StackfulSchedulerSystem as SchedulerSystem>::Desc> {
        // `take` (not load+store) so that concurrent take/cancel pairs can
        // never both obtain the continuation.
        self.cont().take(Ordering::Acquire)
            .expect("StackfulOnlyResumableCore: no parked continuation")
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
        self.cont().is_set(Ordering::Acquire)
    }

    fn notify(&self) {
        let c = self.take_cont();
        Self::wk().push(c);
    }
}

impl<T: StackfulOnlyResumableCore> StackfulResumable<T::StackfulSchedulerSystem> for T {
    fn wait_with<F: FnOnce()>(&self, f: F) {
        type D<T> = <<T as StackfulOnlyResumableCore>::StackfulSchedulerSystem as SchedulerSystem>::Desc;
        let slot = self.cont() as *const AtomicSlot<SuspendedTaskToken<D<Self>>>;
        Self::wk().suspend_to_sched(move |_wk, prev| {
            // Release: publishes the context saved just before this callback.
            // SAFETY: `slot` outlives this callback (it's `&self`'s own
            // field, borrowed for as long as `wait_with` is suspended).
            unsafe { (*slot).publish(prev, Ordering::Release) };
            f();
        });
    }

    fn wait_with_cond<F: FnOnce() -> bool>(&self, f: F) {
        type D<T> = <<T as StackfulOnlyResumableCore>::StackfulSchedulerSystem as SchedulerSystem>::Desc;
        let slot = self.cont() as *const AtomicSlot<SuspendedTaskToken<D<Self>>>;
        Self::wk().cond_suspend_to_sched(move |_wk, prev| {
            // SAFETY: same as `wait_with` — `slot` outlives this callback.
            unsafe { (*slot).publish(prev.take().unwrap(), Ordering::Release) };
            if !f() {
                // SAFETY: `slot` outlives this callback; `take` pairs with
                // this same closure's `publish` a few lines up.
                let c = unsafe { (*slot).take(Ordering::Acquire) };
                *prev = Some(c.expect("StackfulOnlyResumableCore: wait_with_cond cancel raced"));
            }
        });
    }

    fn enter(&self) {
        let wk = Self::wk();
        let c = self.take_cont();
        wk.suspend_to_cont(c, |wk, prev| wk.push(prev));
    }

    fn swap(&self, next: &Self) {
        type D<T> = <<T as StackfulOnlyResumableCore>::StackfulSchedulerSystem as SchedulerSystem>::Desc;
        debug_assert!(!self.is_set());
        let wk = Self::wk();
        let c = next.take_cont();
        let slot = self.cont() as *const AtomicSlot<SuspendedTaskToken<D<Self>>>;
        wk.suspend_to_cont(c, move |_wk, prev| {
            // SAFETY: `slot` outlives this callback (it's `self`'s own
            // field, and `self` outlives the suspend/resume it spans).
            unsafe { (*slot).publish(prev, Ordering::Release) };
        });
    }
}

/// Single-slot parked-continuation implementation.  Implements
/// [`StackfulOnlyResumableCore`] by providing just the `cont()` accessor;
/// all behaviour comes from the blanket [`Resumable`]/[`StackfulResumable`]
/// impls above.
pub struct BasicStackfulOnlyResumable<S: StackfulSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    cont: AtomicSlot<SuspendedTaskToken<S::Desc>>,
    _marker: PhantomData<S>,
}

unsafe impl<S: StackfulSchedulerSystem> Send for BasicStackfulOnlyResumable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}

impl<S: StackfulSchedulerSystem> Default for BasicStackfulOnlyResumable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    fn default() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc { Self::new() }
}

impl<S: StackfulSchedulerSystem> BasicStackfulOnlyResumable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    pub const fn new() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        BasicStackfulOnlyResumable { cont: AtomicSlot::empty(), _marker: PhantomData }
    }
}

impl<S: StackfulSchedulerSystem> StackfulOnlyResumableCore for BasicStackfulOnlyResumable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    type StackfulSchedulerSystem = S;
    fn cont(&self) -> &AtomicSlot<SuspendedTaskToken<S::Desc>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc { &self.cont }
}
