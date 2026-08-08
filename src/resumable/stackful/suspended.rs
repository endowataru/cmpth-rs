//! ULT-layer parked-continuation interface and default implementation.

use std::marker::PhantomData;
use std::sync::atomic::Ordering;

use crate::traits::{Resumable, StackfulResumable};
use crate::resumable::common::system::{DescScheduler, PoolSystem};
use crate::resumable::stackful::system::StackfulWorkerSystem;
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
    <Self::StackfulWorkerSystem as PoolSystem>::Desc: crate::resumable::stackful::desc::StackfulTaskDesc,
{
    type StackfulWorkerSystem: StackfulWorkerSystem;

    /// Access the raw continuation slot.
    ///
    /// The slot is atomic because it is the publication point between the
    /// parking worker and a concurrent notifier: the parker publishes the
    /// slot with `Release` *after* the context save, and consumers take it
    /// with an `Acquire` swap.  On weakly-ordered machines (AArch64) a plain
    /// store here can become visible before the saved context does, letting
    /// the notifier resume a continuation whose frame is not yet written.
    fn cont(&self) -> &AtomicSlot<SuspendedTaskToken<<Self::StackfulWorkerSystem as PoolSystem>::Desc>>;

    // --- helpers shared by the blanket impls below ---------------------------

    fn take_cont(&self) -> SuspendedTaskToken<<Self::StackfulWorkerSystem as PoolSystem>::Desc> {
        // `take` (not load+store) so that concurrent take/cancel pairs can
        // never both obtain the continuation.
        self.cont().take(Ordering::Acquire)
            .expect("StackfulOnlyResumableCore: no parked continuation")
    }

    /// `Self::StackfulWorkerSystem: DescScheduler` is not part of this
    /// trait's own associated-type bound (only plain `StackfulWorkerSystem`
    /// — see that bound's declaration above): the equality constraint on
    /// [`StackfulWorkerSystem::SuspendedThread`]
    /// (`StackfulOnlyResumableCore<StackfulWorkerSystem = Self>`) is written
    /// from *inside* the `StackfulWorkerSystem` trait, where `Self` is only
    /// known to be `StackfulWorkerSystem`, not `DescScheduler` — so the
    /// bound has to live here instead, on the one method that actually needs
    /// a live worker. Combined with the trait-level `StackfulWorkerSystem`
    /// bound, this is enough to derive the fold trait
    /// [`StackfulSchedulerSystem`](crate::resumable::stackful::system::StackfulSchedulerSystem)
    /// (its blanket impl needs exactly `SchedulerSystem + StackfulWorkerSystem
    /// + DescScheduler<Desc: ...>`, and `DescScheduler` already implies
    /// `SchedulerSystem`), which is what makes `UltWorker::current()` below
    /// resolve.
    fn wk() -> &'static UltWorker<Self::StackfulWorkerSystem>
    where
        Self::StackfulWorkerSystem: DescScheduler,
    {
        UltWorker::<Self::StackfulWorkerSystem>::current()
            .expect("cmpth: StackfulOnlyResumableCore operation called outside a worker")
    }
}

/// Blanket: any [`StackfulOnlyResumableCore`] automatically implements the
/// top-level wait-slot traits. `enter`/`swap` always perform a real context
/// switch here — the slot can only ever hold a real continuation, unlike
/// `DualResumable`, which may hold an async waiter and has to fall back to
/// a plain wake internally.
impl<T: StackfulOnlyResumableCore> Resumable<T::StackfulWorkerSystem> for T
where
    T::StackfulWorkerSystem: DescScheduler,
{
    fn is_set(&self) -> bool {
        self.cont().is_set(Ordering::Acquire)
    }

    fn notify(&self) {
        let c = self.take_cont();
        Self::wk().push(c);
    }
}

impl<T: StackfulOnlyResumableCore> StackfulResumable<T::StackfulWorkerSystem> for T
where
    T::StackfulWorkerSystem: DescScheduler,
{
    fn wait_with<F: FnOnce()>(&self, f: F) {
        type D<T> = <<T as StackfulOnlyResumableCore>::StackfulWorkerSystem as PoolSystem>::Desc;
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
        type D<T> = <<T as StackfulOnlyResumableCore>::StackfulWorkerSystem as PoolSystem>::Desc;
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
        type D<T> = <<T as StackfulOnlyResumableCore>::StackfulWorkerSystem as PoolSystem>::Desc;
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
pub struct BasicStackfulOnlyResumable<S: StackfulWorkerSystem> where <S as PoolSystem>::Desc: StackfulTaskDesc {
    cont: AtomicSlot<SuspendedTaskToken<S::Desc>>,
    _marker: PhantomData<S>,
}

unsafe impl<S: StackfulWorkerSystem> Send for BasicStackfulOnlyResumable<S> where <S as PoolSystem>::Desc: StackfulTaskDesc {}

impl<S: StackfulWorkerSystem> Default for BasicStackfulOnlyResumable<S> where <S as PoolSystem>::Desc: StackfulTaskDesc {
    fn default() -> Self where <S as PoolSystem>::Desc: StackfulTaskDesc { Self::new() }
}

impl<S: StackfulWorkerSystem> BasicStackfulOnlyResumable<S> where <S as PoolSystem>::Desc: StackfulTaskDesc {
    pub const fn new() -> Self where <S as PoolSystem>::Desc: StackfulTaskDesc {
        BasicStackfulOnlyResumable { cont: AtomicSlot::empty(), _marker: PhantomData }
    }
}

impl<S: StackfulWorkerSystem> StackfulOnlyResumableCore for BasicStackfulOnlyResumable<S> where <S as PoolSystem>::Desc: StackfulTaskDesc {
    type StackfulWorkerSystem = S;
    fn cont(&self) -> &AtomicSlot<SuspendedTaskToken<S::Desc>> where <S as PoolSystem>::Desc: StackfulTaskDesc { &self.cont }
}
