//! The threading-system interface (ComposableThreads' `ult_itf` equivalent).
//!
//! [`ThreadSystem`] is the single interface bundle implemented both by the
//! bottom layer (`OsSystem`) and by every ULT scheduler layered on top of it.
//! Because the interface is identical at every level, a ULT scheduler can be
//! nested on top of another ULT scheduler for verification
//! (`OsSystem` -> `UltSystem` -> `UltUltSystem`).

use std::future::Future;
use std::pin::pin;
use std::task::{Poll, RawWaker, RawWakerVTable, Waker};

use crate::traits::{Delegator, DelegatorConsumer, Poller, StackfulBarrier, StackfulMutex};

/// Threading system interface bundle — swap the entire backend by changing
/// one type parameter.
pub trait ThreadSystem: Sized + Send + Sync + 'static {
    /// Drives a single `block_on` call; the customisation point for async
    /// integration.  See [`Poller`].
    type Poller: Poller;

    /// Block the current thread/ULT until `future` completes.
    ///
    /// On a ULT system this suspends only the calling ULT; the OS thread
    /// underneath keeps running other tasks.
    ///
    /// ```
    /// use cmpth::ThreadSystem;
    ///
    /// cmpth::default::run(2, || {
    ///     let x = cmpth::DefaultUltSystem::block_on(async { 6 * 7 });
    ///     assert_eq!(x, 42);
    /// });
    /// ```
    ///
    /// The default implementation drives the future through [`Self::Poller`].
    fn block_on<F, T>(f: F) -> T
    where
        F: Future<Output = T> + Send,
        T: Send,
    {
        let pol = Self::Poller::new();
        let mut f = pin!(f);
        loop {
            match f.as_mut().poll(&mut pol.context()) {
                Poll::Ready(v) => return v,
                Poll::Pending => pol.wait(),
            }
        }
    }

    /// Yield the current thread/ULT so other tasks can run.
    fn yield_now();

    /// Spawn a new thread or ULT; returns a handle that can be joined.
    type JoinHandle<T: Send + 'static>: JoinHandleLike<T>;
    fn spawn<T, F>(f: F) -> Self::JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static;

    /// Mutex type for this system.
    type Mutex<T: Send>: StackfulMutex<T> + Send + Sync;

    /// Barrier type for this system.
    type Barrier: StackfulBarrier + Send + Sync;

    /// Parked-continuation handle for this system.
    type SuspendedThread: Send + Default;

    /// Delegator type for this system.
    type Delegator<C: DelegatorConsumer<Self>>: Delegator<Self, C>;

    /// Thread-specific storage slot: one `*mut T` per thread (or per ULT) of
    /// this system.  A nested scheduler stores its per-worker pointer here,
    /// which is why a single slot per level is enough — everything else is
    /// reached through the worker pointer.
    type ThreadSpecific<T: 'static>: TlsSlot<T>;

    /// Number of parallel workers (OS threads or ULT worker threads).
    fn num_workers() -> usize;
}

/// Common interface for join handles returned by [`ThreadSystem::spawn`].
pub trait JoinHandleLike<T: Send + 'static>: Send {
    fn join(self) -> T;
}

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
    pub(crate) index: std::sync::atomic::AtomicUsize,
}

/// Sentinel for [`TlsAnchor::index`]: no slot assigned yet.
pub(crate) const TLS_ANCHOR_UNASSIGNED: usize = usize::MAX;

impl TlsAnchor {
    pub const fn new() -> Self {
        TlsAnchor { index: std::sync::atomic::AtomicUsize::new(TLS_ANCHOR_UNASSIGNED) }
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
    /// construction (see `Scheduler::new`'s callers in `ult::scheduler`) —
    /// well before any worker OS thread starts, so the real first-use
    /// assignment race this guards against in [`get`](Self::get)'s slow
    /// path never actually happens in practice.
    ///
    /// Default: no-op. Implementations whose "assign once" step isn't on a
    /// hot path (e.g. `UltTls`, used far less often than a `spawn`/`join`
    /// hot loop) don't need to override this.
    fn warm_up(&self) {}
}

// ---------------------------------------------------------------------------
// Shared no-op waker (used by OsPoller and UltPoller fallback)
// ---------------------------------------------------------------------------

/// A waker whose `wake()` is a no-op.  Used for busy-polling fallbacks where
/// the poll loop drives re-polling itself (via `yield_now`).
pub(crate) fn noop_waker() -> Waker {
    static VTABLE: RawWakerVTable =
        RawWakerVTable::new(|p| RawWaker::new(p, &VTABLE), |_| {}, |_| {}, |_| {});
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}
