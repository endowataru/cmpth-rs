//! Shared root types used by both the stackful and stackless flavors:
//! [`TaskSystem`], [`TaskDesc`], [`Resumable`], [`DualMutex`]/[`DualBarrier`],
//! [`TlsAnchor`]/[`TlsSlot`].

use std::ops::DerefMut;
use std::sync::atomic::AtomicUsize;
use std::task::Waker;

use crate::traits::stackful::{StackfulBarrier, StackfulMutex};
use crate::traits::stackless::{StacklessBarrier, StacklessMutex};

// ---------------------------------------------------------------------------
// TaskDesc
// ---------------------------------------------------------------------------

/// Decoded view of a task descriptor's join-protocol state — who (if
/// anyone) is waiting on this task, or whether it has already finished.
pub enum JoinState<D> {
    /// Task alive, nobody waiting.
    Running,
    /// Result written (or the task was detached-and-cleaned).
    Finished,
    /// The `JoinHandle` was dropped early; the exit path cleans up.
    Detached,
    /// A parked sync joiner, registered via
    /// [`TaskDesc::try_register_sync_joiner`].
    SyncJoiner(*mut D),
    /// A registered async waker — used when the polling task's waker isn't
    /// verifiably one of this system's own (foreign executor, or no worker
    /// at all).
    AsyncWaker(*mut Waker),
    /// Same role as `AsyncWaker`, but unboxed: the polling task's own
    /// descriptor, reachable directly because its waker is known (by
    /// construction) to be this system's own poll-loop waker.
    AsyncJoiner(*mut D),
}

/// A task descriptor: the per-task join-protocol state (has it finished?
/// who's waiting?) that every scheduling flavor needs, regardless of how it
/// represents a task's stack, poll function, or anything else about *how*
/// the task actually runs.
///
/// Bodyless — pure behavior, no storage representation implied. Implement
/// this directly for a completely custom descriptor, or implement
/// [`TaskDescCore`](crate::resumable::common::desc::TaskDescCore) instead
/// to get this crate's own word-based join-protocol algorithm for free via
/// a blanket impl.
///
/// `Owned` is this descriptor's owner-exclusive data (whatever a concrete
/// implementation wants to store there — e.g. this crate's own worker/
/// slot/result/tls bookkeeping) reachable *only* through a live
/// [`Suspended`](Self::Suspended)/[`Running`](Self::Running) token: holding
/// one of these tokens is itself the proof of exclusive access, the same
/// "the token proves the precondition" pattern `MutexGuard`/`RefMut` use.
/// `TaskDesc` says nothing about *how* that exclusivity is implemented
/// (`UnsafeCell`, a lock, whatever) — only that a token gives `&mut Owned`
/// via `DerefMut`.
pub trait TaskDesc: Send + Sync + Sized + 'static {
    /// Owner-exclusive data, reached only through a live `Suspended`/
    /// `Running` token.
    type Owned;

    /// Owning handle to a suspended (parked) task — proof that nothing
    /// else can be concurrently accessing its `Owned` data.
    type Suspended: DerefMut<Target = Self::Owned> + Send;

    /// Owning handle to the task currently running — same exclusivity
    /// proof as `Suspended`, for the task actively executing rather than
    /// parked.
    type Running: DerefMut<Target = Self::Owned> + Send;

    /// Read and decode the current join state.
    fn read_join_state(&self) -> JoinState<Self>;

    /// Fast check for the hot join path: is the task already finished?
    fn is_finished(&self) -> bool;

    /// Direct-handoff exit: the exiting task already switched straight
    /// into the parked sync joiner's continuation — just publish
    /// `Finished` so the joiner (now running) observes its result is
    /// ready.
    fn commit_finished(&self);

    /// General-case exit: publish `Finished` and return whichever party
    /// the old state names, so the caller can settle it (wake a
    /// late-registered joiner/waker, or notice the handle was dropped).
    fn publish_finished(&self) -> JoinState<Self>;

    /// Try to register `joiner` (a parked sync joiner's descriptor) as
    /// this task's waiter. Returns `false` if the task turned out to
    /// already be finished (caller should cancel its own suspension and
    /// proceed immediately) — otherwise commits `joiner`.
    ///
    /// # Safety
    /// `joiner` must be a stable pointer to a currently-parked task
    /// descriptor for as long as it might be woken through this slot.
    unsafe fn try_register_sync_joiner(&self, joiner: *mut Self) -> bool;

    /// Try to mark this task detached (no handle left to collect the
    /// result). Returns `true` if the task was already finished (caller
    /// now owns the result and the descriptor) — otherwise commits
    /// detached.
    fn try_mark_detached(&self) -> bool;
}

/// Outcome of a wake attempt against a POLLING/PARKED/NOTIFIED state
/// machine — shared by `WakerTaskDesc`'s
/// `try_wake_state` (stackless: `spawn_async`) and stackful `block_on`'s
/// `Poller` (a real ULT's wait state uses the same three-state shape, just
/// not anchored on a descriptor).
pub enum WakeOutcome {
    /// Was POLLING; now NOTIFIED. The task will notice on its next state
    /// check and re-poll; there is no continuation to push.
    SetNotified,
    /// Was PARKED; now POLLING. The caller owns delivering the
    /// continuation (push to a worker deque or the external queue).
    ClaimedParked,
    /// Was already NOTIFIED, or IDLE (a stale wake after the poll session
    /// ended). Nothing to do.
    NoOp,
}

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
/// registered async [`Waker`], or nothing. Unlike
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
/// concretely (by hand, or in each concrete `UltIdentity`/`UltAsyncIdentity`
/// implementor's own `worker_tls_anchor`).  `TlsAnchor` breaks
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
