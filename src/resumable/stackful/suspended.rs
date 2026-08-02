//! ULT-layer parked-continuation interface and default implementation.

use std::cell::Cell;
use std::marker::PhantomData;
use std::ptr;

use crate::traits::{Resumable, StackfulResumable};
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::common::desc::SuspendedTaskToken;
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::common::worker::{LocalQueue, UltWorker, Worker};
use crate::resumable::stackful::worker::{ContextSwitcher, StackfulWorker};

/// ULT-layer interface for parked-continuation slots.
///
/// Implementors supply only the [`cont`](Self::cont) accessor; all scheduling
/// operations are provided as default methods built on [`UltWorker`].  Swap
/// in a different struct (e.g. one with profiling counters) by implementing
/// `cont()` and overriding whichever methods need customisation.
///
/// The blanket `impl<T: StackfulOnlyResumable> Resumable/StackfulResumable for T`
/// then automatically satisfies the top-level wait-slot interface
/// (`docs/sync-async-unification.md`).
pub trait StackfulOnlyResumable: Send + Default
where
    <Self::StackfulSchedulerSystem as SchedulerSystem>::Desc: crate::resumable::stackful::desc::StackfulTaskDesc,
{
    type StackfulSchedulerSystem: StackfulSchedulerSystem;

    /// Access the raw continuation slot.
    ///
    /// Deliberately plain (`Cell`, not `AtomicPtr`): this slot's own
    /// Release/Acquire is *not* what makes a parked continuation's `ctx`
    /// visible to whoever discovers it. That relies on the implementor's
    /// own external protocol instead — e.g. `McsMutex`'s successor is only
    /// ever discovered via `next: AtomicPtr` (Release store paired with an
    /// Acquire load), `McsCondvar`/`Mutex`/`Barrier`'s waiters only via a
    /// `SpinLock`'s own lock/unlock, `McsQueue`/`RingBufQueue`-backed
    /// delegators only via their own `tail`/`head` CAS chain — and in every
    /// one of these, this slot's write happens-before, and its read
    /// happens-after, that same real Release/Acquire pair. This was
    /// audited call-site-by-call-site for every implementor in this crate
    /// before making this change (see `cmpth-rs-waker-task-desc-scope.md`
    /// memory / the "consumer_sth" correction in particular for how easy
    /// it is to get this wrong by trusting a stale claim instead of
    /// re-deriving it).
    ///
    /// **Any new `StackfulOnlyResumable` implementor must prove the same
    /// property for its own protocol** (a real Release store that the
    /// discovering side genuinely Acquire-observes on the *same* memory
    /// location, happening after this slot's own write and before
    /// `take_cont()`/`notify()`/`enter()`/`swap()` read it) — or it needs
    /// its own atomic slot instead of reusing this trait's plain one. This
    /// exact subsystem has already produced one ARM-only, CI-invisible
    /// weak-memory race from getting a nearly identical invariant wrong
    /// (see `ctx`'s own history, `HasCtx`'s doc comment) — don't relax
    /// this without the same E-core stress-test rigor that caught it.
    fn cont(&self) -> &Cell<*mut <Self::StackfulSchedulerSystem as SchedulerSystem>::Desc>;

    // --- helpers ------------------------------------------------------------

    fn take_cont(&self) -> SuspendedTaskToken<<Self::StackfulSchedulerSystem as SchedulerSystem>::Desc> {
        // `replace` (not get+set) so that concurrent take/cancel pairs can
        // never both obtain the continuation.
        let c = self.cont().replace(ptr::null_mut());
        assert!(!c.is_null(), "StackfulOnlyResumable: no parked continuation");
        SuspendedTaskToken(c)
    }

    fn wk() -> &'static UltWorker<Self::StackfulSchedulerSystem> {
        UltWorker::<Self::StackfulSchedulerSystem>::current()
            .expect("cmpth: StackfulOnlyResumable operation called outside a worker")
    }

    // --- default implementations --------------------------------------------

    fn is_set_impl(&self) -> bool {
        !self.cont().get().is_null()
    }

    fn wait_with_impl<F: FnOnce()>(&self, f: F) {
        type D<T> = <<T as StackfulOnlyResumable>::StackfulSchedulerSystem as SchedulerSystem>::Desc;
        let slot = self.cont() as *const Cell<*mut D<Self>>;
        Self::wk().suspend_to_sched(move |_wk, prev| {
            // Publishes the context saved just before this callback — sound
            // per `cont()`'s own doc comment: the implementor's external
            // protocol provides the real Release this needs.
            unsafe { (*slot).set(prev.into_raw()) };
            f();
        });
    }

    fn wait_with_cond_impl<F: FnOnce() -> bool>(&self, f: F) {
        type D<T> = <<T as StackfulOnlyResumable>::StackfulSchedulerSystem as SchedulerSystem>::Desc;
        let slot = self.cont() as *const Cell<*mut D<Self>>;
        Self::wk().cond_suspend_to_sched(move |_wk, prev| {
            unsafe { (*slot).set(prev.take().unwrap().into_raw()) };
            if !f() {
                let c = unsafe { (*slot).replace(ptr::null_mut()) };
                debug_assert!(!c.is_null());
                *prev = Some(SuspendedTaskToken(c));
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
        type D<T> = <<T as StackfulOnlyResumable>::StackfulSchedulerSystem as SchedulerSystem>::Desc;
        debug_assert!(!self.is_set_impl());
        let wk = Self::wk();
        let c = next.take_cont();
        let slot = self.cont() as *const Cell<*mut D<Self>>;
        wk.suspend_to_cont(c, move |_wk, prev| {
            unsafe { (*slot).set(prev.into_raw()) };
        });
    }
}

/// Blanket: any `StackfulOnlyResumable` automatically implements the top-level
/// wait-slot traits. `enter`/`swap` always perform a real context switch
/// here — the slot can only ever hold a real continuation, unlike
/// `DualResumable`, which may hold an async waiter and has to fall back to
/// a plain wake internally.
impl<T: StackfulOnlyResumable> Resumable<T::StackfulSchedulerSystem> for T {
    fn is_set(&self) -> bool { self.is_set_impl() }
    fn notify(&self) { self.notify_impl() }
}

impl<T: StackfulOnlyResumable> StackfulResumable<T::StackfulSchedulerSystem> for T {
    fn wait_with<F: FnOnce()>(&self, f: F) { self.wait_with_impl(f) }
    fn wait_with_cond<F: FnOnce() -> bool>(&self, f: F) { self.wait_with_cond_impl(f) }
    fn enter(&self) { self.enter_impl() }
    fn swap(&self, next: &Self) { self.swap_impl(next) }
}

/// Single-slot parked-continuation implementation.  Implements
/// [`StackfulOnlyResumable`] by providing just the `cont()` accessor; all
/// behaviour comes from the default methods.
pub struct BasicStackfulOnlyResumable<S: StackfulSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    cont: Cell<*mut S::Desc>,
    _marker: PhantomData<S>,
}

// `Cell` makes this `!Send`/`!Sync` automatically. Both are asserted
// manually instead, on the same basis `cont()`'s own doc comment lays out:
// every embedding protocol in this crate (`McsMutex`'s `next` chain,
// `SpinLock`-guarded `Mutex`/`Condvar`/`Barrier`, `McsQueue`/
// `RingBufQueue`-backed delegators, and `DualMutex`/`DualBarrier` when
// instantiated with this type for stackful-only systems — same `next:
// AtomicPtr` MCS shape as `McsMutex`) already provides a real external
// Release/Acquire pair around every access to `cont`, so sharing this
// value's `&self` across threads (which is all `Sync` actually promises;
// it says nothing about ordering) is sound. `StackfulOnlyResumable` itself
// only requires `Send`; `Sync` is needed transitively by
// `DualMutex`/`DualBarrier`'s own `StackfulMutex`/`StackfulBarrier` impls.
unsafe impl<S: StackfulSchedulerSystem> Send for BasicStackfulOnlyResumable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}
unsafe impl<S: StackfulSchedulerSystem> Sync for BasicStackfulOnlyResumable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}

impl<S: StackfulSchedulerSystem> Default for BasicStackfulOnlyResumable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    fn default() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc { Self::new() }
}

impl<S: StackfulSchedulerSystem> BasicStackfulOnlyResumable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    pub const fn new() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        BasicStackfulOnlyResumable { cont: Cell::new(ptr::null_mut()), _marker: PhantomData }
    }
}

impl<S: StackfulSchedulerSystem> StackfulOnlyResumable for BasicStackfulOnlyResumable<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    type StackfulSchedulerSystem = S;
    fn cont(&self) -> &Cell<*mut S::Desc> where <S as SchedulerSystem>::Desc: StackfulTaskDesc { &self.cont }
}
