//! Shared root types used by both the stackful and stackless flavors:
//! [`TaskSystem`], [`Resumable`], [`DualMutex`]/[`DualBarrier`],
//! [`TlsAnchor`]/[`TlsSlot`].

use std::sync::atomic::AtomicUsize;

use crate::traits::stackful::{StackfulBarrier, StackfulMutex};
use crate::traits::stackless::{StacklessBarrier, StacklessMutex};

/// Declares that a system provides an efficient (work-stealing) scheduler
/// as its execution model — the shared foundation both
/// [`ThreadSystem`](crate::traits::stackful::ThreadSystem) (spawn/join) and
/// the `scoped` family (`ScopedStackfulTaskSystem`/`ScopedStacklessTaskSystem`,
/// in [`crate::traits::scoped`]) build on: both assume the same efficient
/// scheduling underneath, just expose different capabilities on top of it.
pub trait TaskSystem: Sized + Send + Sync + 'static {
    /// This worker's own index among its `num_workers()` peers (stable for
    /// the lifetime of the calling task/thread). Not meaningful outside a
    /// managed worker pool — a system with no such pool (e.g. `OsSystem`,
    /// whose "workers" are just whatever OS threads happen to be running)
    /// always reports `0`.
    fn worker_num() -> usize;

    /// Number of parallel workers (OS threads or ULT worker threads).
    fn num_workers() -> usize;
}

/// The durable capability every wait-slot has, regardless of what kind of
/// waiter (if any) is currently parked: a real ULT continuation, a
/// registered async [`Waker`](std::task::Waker), or nothing. Unlike
/// `is_set`'s answer, which changes per instance over time, this trait
/// itself is a fixed property of the type — same spirit as `Send`/`Sync`.
pub trait Resumable<S>: Default {
    /// True if a waiter is currently parked here.
    fn is_set(&self) -> bool;

    /// Wake whatever is parked here, if anything. Cheap and direct for a
    /// real ULT continuation; goes through `Waker::wake` only when the slot
    /// actually holds a registered async waiter.
    fn notify(&self);
}

/// Return value of [`StackfulBarrier::wait`], mirroring
/// `std::sync::BarrierWaitResult`.
pub struct BarrierWaitResult {
    pub is_leader: bool,
}

impl BarrierWaitResult {
    pub fn is_leader(&self) -> bool { self.is_leader }
}

/// A barrier usable from either calling convention — see [`DualMutex`] for
/// the same pattern applied to mutexes. The interface owns the name here
/// too: the concrete generic-over-N type
/// (`resumable::common::sync::DualBarrier`) is re-exported under an alias
/// (`UltDualBarrier`) at the crate root to make room.
pub trait DualBarrier: Sized + Send + Sync + StackfulBarrier + StacklessBarrier {}

impl<M: StackfulBarrier + StacklessBarrier> DualBarrier for M {}

/// A mutex usable from either calling convention. Blanket-derived: any type
/// implementing both flavors gets this for free, so it exists purely as a
/// convenience bound for generic code that wants "works either way" as one
/// name (`S::Mutex: DualMutex<T>`) instead of spelling out both traits.
pub trait DualMutex<T: Send>: StackfulMutex<T> + StacklessMutex<T> {}

impl<T: Send, M: StackfulMutex<T> + StacklessMutex<T>> DualMutex<T> for M {}

/// The untyped storage behind a [`TlsSlot`]: a lazily assigned slot index.
///
/// Statics inside associated functions cannot mention `Self` or generic
/// parameters, which is what forces per-system TLS slots to be spelled out
/// concretely (by hand or by the `ult_system!` macro).  `TlsAnchor` breaks
/// that constraint: the static is untyped, and [`TlsSlot::from_anchor`]
/// views it as the typed slot.  `worker_tls` then becomes the same two
/// lines for every system:
///
/// ```ignore
/// fn worker_tls() -> &'static <Self::Base as ThreadSystem>::ThreadSpecific<UltWorker<Self>> {
///     static A: TlsAnchor = TlsAnchor::new();
///     TlsSlot::from_anchor(&A)
/// }
/// ```
pub struct TlsAnchor {
    pub(crate) index: AtomicUsize,
}

/// Sentinel for [`TlsAnchor::index`]: no slot assigned yet.
pub(crate) const TLS_ANCHOR_UNASSIGNED: usize = usize::MAX;

impl TlsAnchor {
    pub const fn new() -> Self {
        TlsAnchor { index: AtomicUsize::new(TLS_ANCHOR_UNASSIGNED) }
    }
}

impl Default for TlsAnchor {
    fn default() -> Self {
        Self::new()
    }
}

/// A thread-specific `*mut T` slot.
///
/// `INIT` allows a slot to be placed in a `static`, which is how a system
/// anchors the per-worker pointer of a nested scheduler.
///
/// # Implementation contract
///
/// Implementors must be `#[repr(transparent)]` wrappers around
/// [`TlsAnchor`] so that [`from_anchor`](Self::from_anchor) can reinterpret
/// a shared untyped static as the typed slot.
pub trait TlsSlot<T: 'static>: Sync + 'static {
    const INIT: Self;

    /// View an untyped anchor as this slot type (see [`TlsAnchor`]).
    fn from_anchor(anchor: &'static TlsAnchor) -> &'static Self;

    /// Get the value for the current thread (null if never set).
    fn get(&self) -> *mut T;

    /// Set the value for the current thread.
    fn set(&self, p: *mut T);

    /// Like [`get`](Self::get), but callable from code the caller may
    /// inline: only sound when the OS thread is guaranteed not to change
    /// between the read and its use. `OsTls::get` must forbid inlining
    /// because a suspended ULT can resume on a *different* OS thread,
    /// and an inlined read could get CSE'd across that opaque
    /// context-switch call, reading the wrong thread's slot.
    /// Stackless-only code (no context switches: a `Future::poll` call
    /// never migrates OS threads mid-call) has no such hazard, so it can
    /// use this instead of paying for a call it doesn't need protection
    /// from. Defaults to the safe [`get`](Self::get); implementors that
    /// can offer a genuinely inlinable path override it.
    #[inline]
    fn get_inline(&self) -> *mut T {
        self.get()
    }

    /// Eagerly resolve whatever one-time internal state this slot needs
    /// (e.g. `OsTls`'s array index) before the hot path ever calls
    /// `get`/`set`/`get_inline`. Called once, single-threaded, at scheduler
    /// construction (see `Scheduler::new`'s callers in `resumable::scheduler`) —
    /// well before any worker OS thread starts, so the real first-use
    /// assignment race this guards against in [`get`](Self::get)'s slow
    /// path never actually happens in practice.
    ///
    /// Default: no-op. Implementations whose "assign once" step isn't on a
    /// hot path (e.g. `UltTls`, used far less often than a `spawn`/`join`
    /// hot loop) don't need to override this.
    fn warm_up(&self) {}
}
