//! ULT-layer parked-continuation interface and default implementation.

use std::marker::PhantomData;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::traits::{Resumable, StackfulResumable};
use crate::ult::system::{SchedulerSystem, UltSchedulerSystem};
use crate::ult::desc::{StackfulTaskDesc, SuspendedUlt};
use crate::ult::worker::{ContextSwitcher, LocalQueue, StackfulWorker, UltWorker, Worker};

/// ULT-layer interface for parked-continuation slots.
///
/// Implementors supply only the [`cont`](Self::cont) accessor; all scheduling
/// operations are provided as default methods built on [`UltWorker`].  Swap
/// in a different struct (e.g. one with profiling counters) by implementing
/// `cont()` and overriding whichever methods need customisation.
///
/// The blanket `impl<T: UltSuspendedThread> Resumable/StackfulResumable for T`
/// then automatically satisfies the top-level wait-slot interface
/// (`docs/sync-async-unification.md`).
pub trait UltSuspendedThread: Send + Default
where
    <Self::UltSchedulerSystem as SchedulerSystem>::Desc: crate::ult::desc::StackfulTaskDesc,
{
    type UltSchedulerSystem: UltSchedulerSystem;

    /// Access the raw continuation slot.
    ///
    /// The slot is atomic because it is the publication point between the
    /// parking worker and a concurrent notifier: the parker stores the slot
    /// with `Release` *after* the context save, and consumers take it with an
    /// `Acquire` swap.  On weakly-ordered machines (AArch64) a plain store
    /// here can become visible before the saved context does, letting the
    /// notifier resume a continuation whose frame is not yet written.
    fn cont(&self) -> &AtomicPtr<<Self::UltSchedulerSystem as SchedulerSystem>::Desc>;

    // --- helpers ------------------------------------------------------------

    fn take_cont(&self) -> SuspendedUlt<<Self::UltSchedulerSystem as SchedulerSystem>::Desc> {
        // `swap` (not load+store) so that concurrent take/cancel pairs can
        // never both obtain the continuation.
        let c = self.cont().swap(ptr::null_mut(), Ordering::Acquire);
        assert!(!c.is_null(), "UltSuspendedThread: no parked continuation");
        SuspendedUlt(c)
    }

    fn wk() -> &'static UltWorker<Self::UltSchedulerSystem> {
        UltWorker::<Self::UltSchedulerSystem>::current()
            .expect("cmpth: UltSuspendedThread operation called outside a worker")
    }

    // --- default implementations --------------------------------------------

    fn is_set_impl(&self) -> bool {
        !self.cont().load(Ordering::Acquire).is_null()
    }

    fn wait_with_impl<F: FnOnce()>(&self, f: F) {
        type D<T> = <<T as UltSuspendedThread>::UltSchedulerSystem as SchedulerSystem>::Desc;
        let slot = self.cont() as *const AtomicPtr<D<Self>>;
        Self::wk().suspend_to_sched(move |_wk, prev| {
            // Release: publishes the context saved just before this callback.
            unsafe { (*slot).store(prev.into_raw(), Ordering::Release) };
            f();
        });
    }

    fn wait_with_cond_impl<F: FnOnce() -> bool>(&self, f: F) {
        type D<T> = <<T as UltSuspendedThread>::UltSchedulerSystem as SchedulerSystem>::Desc;
        let slot = self.cont() as *const AtomicPtr<D<Self>>;
        Self::wk().cond_suspend_to_sched(move |_wk, prev| {
            unsafe {
                (*slot).store(prev.take().unwrap().into_raw(), Ordering::Release)
            };
            if !f() {
                let c = unsafe { (*slot).swap(ptr::null_mut(), Ordering::Acquire) };
                debug_assert!(!c.is_null());
                *prev = Some(SuspendedUlt(c));
            }
        });
    }

    fn notify_impl(&self) {
        let c = self.take_cont();
        Self::wk().push_local_top(c);
    }

    fn enter_impl(&self) {
        let wk = Self::wk();
        let c = self.take_cont();
        wk.suspend_to_cont(c, |wk, prev| wk.push_local_top(prev));
    }

    fn swap_impl(&self, next: &Self) {
        type D<T> = <<T as UltSuspendedThread>::UltSchedulerSystem as SchedulerSystem>::Desc;
        debug_assert!(!self.is_set_impl());
        let wk = Self::wk();
        let c = next.take_cont();
        let slot = self.cont() as *const AtomicPtr<D<Self>>;
        wk.suspend_to_cont(c, move |_wk, prev| {
            unsafe { (*slot).store(prev.into_raw(), Ordering::Release) };
        });
    }
}

/// Blanket: any `UltSuspendedThread` automatically implements the top-level
/// wait-slot traits. `enter`/`swap` always perform a real context switch
/// here — the slot can only ever hold a real continuation, unlike
/// `SuspendedTask`, which may hold an async waiter and has to fall back to
/// a plain wake internally.
impl<T: UltSuspendedThread> Resumable<T::UltSchedulerSystem> for T {
    fn is_set(&self) -> bool { self.is_set_impl() }
    fn notify(&self) { self.notify_impl() }
}

impl<T: UltSuspendedThread> StackfulResumable<T::UltSchedulerSystem> for T {
    fn wait_with<F: FnOnce()>(&self, f: F) { self.wait_with_impl(f) }
    fn wait_with_cond<F: FnOnce() -> bool>(&self, f: F) { self.wait_with_cond_impl(f) }
    fn enter(&self) { self.enter_impl() }
    fn swap(&self, next: &Self) { self.swap_impl(next) }
}

/// Single-slot parked-continuation implementation.  Implements
/// [`UltSuspendedThread`] by providing just the `cont()` accessor; all
/// behaviour comes from the default methods.
pub struct BasicSuspendedThread<S: UltSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    cont: AtomicPtr<S::Desc>,
    _marker: PhantomData<S>,
}

unsafe impl<S: UltSchedulerSystem> Send for BasicSuspendedThread<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}

impl<S: UltSchedulerSystem> Default for BasicSuspendedThread<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    fn default() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc { Self::new() }
}

impl<S: UltSchedulerSystem> BasicSuspendedThread<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    pub const fn new() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        BasicSuspendedThread { cont: AtomicPtr::new(ptr::null_mut()), _marker: PhantomData }
    }
}

impl<S: UltSchedulerSystem> UltSuspendedThread for BasicSuspendedThread<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    type UltSchedulerSystem = S;
    fn cont(&self) -> &AtomicPtr<S::Desc> where <S as SchedulerSystem>::Desc: StackfulTaskDesc { &self.cont }
}
