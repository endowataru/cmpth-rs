//! [`WorkerSystem`]/[`SchedulerSystem`] — the base worker-lookup and
//! dispatch traits shared by every flavor (stackful, stackless, dual).
//! Extended by
//! [`StackfulSchedulerSystem`](crate::resumable::stackful::system::StackfulSchedulerSystem)
//! (real-stack capability) and
//! [`StacklessTaskSystem`](crate::resumable::stackless::system::StacklessTaskSystem)
//! (async-task capability).

use crate::traits::common::TaskSystem;
use crate::traits::stackful::{NestableSystem, SpawnableStackfulTaskSystem};
use crate::resumable::common::deque::WorkerRunQueue;
use crate::resumable::common::external_queue::ExternalQueue;
use crate::resumable::common::desc::{SuspendedTaskToken, TaskDescAlloc};
use crate::resumable::common::lookup::CurrentLookup;
use crate::resumable::common::pool::{DescPool, DynamicPool};
use crate::resumable::common::worker::{LocalQueue, WorkerOps};

/// Storage-related subset of [`SchedulerSystem`]: the descriptor type and
/// the pools/external-queue that store and move it around. Split out as its
/// own supertrait so this axis can be named independently of the rest of
/// [`SchedulerSystem`] (worker/run-queue/dispatch).
///
/// `Desc` lives here (not on [`SchedulerSystem`] itself) because every other
/// member of this trait is typed by it (`Pool: DescPool<Self::Desc>`, etc.);
/// [`SchedulerSystem`] inherits it via the supertrait bound, so `Self::Desc`
/// keeps resolving unchanged for any `S: SchedulerSystem`.
pub trait PoolSystem: Sized + Send + Sync + 'static {
    /// Task descriptor type for this system.
    type Desc: TaskDescAlloc;

    /// Descriptor pool implementation for this system, used by the stackful
    /// `spawn` path (fixed-size ULT stacks, `STACK_SIZE` on
    /// [`StackfulSchedulerSystem`](crate::resumable::stackful::system::StackfulSchedulerSystem)).
    type Pool: DescPool<Self::Desc>;

    /// Descriptor pool used by `spawn_async` (variable-size Future storage,
    /// capped at `ASYNC_POOL_SIZE`) — deliberately a *separate* pool/type
    /// from [`Pool`](Self::Pool) rather than sharing it: a dual system needs
    /// both a large fixed-size ULT-stack pool and a small fixed-size
    /// async-task pool live at once, and the two have nothing in common
    /// beyond both implementing [`DescPool`]. Required on every
    /// `SchedulerSystem` (even stackful-only ones that never call
    /// `spawn_async`) purely so the type is nameable uniformly; an unused
    /// `AsyncPool` costs nothing beyond declaring it, matching the existing
    /// precedent of stackless-only systems declaring an unused
    /// [`Pool`](Self::Pool).
    type AsyncPool: DescPool<Self::Desc>;

    /// Fixed slot size for [`AsyncPool`](Self::AsyncPool). Futures that fit
    /// are served from its free list; larger ones fall back to a one-off
    /// allocation (see [`DescPool::alloc`]).
    const ASYNC_POOL_SIZE: usize;

    /// Frame-only pool backing [`crate::resumable::stackless::thread::recurse`] — the same
    /// fixed-slot free-list mechanism [`Pool`](Self::Pool)/[`AsyncPool`](Self::AsyncPool)
    /// use ([`crate::resumable::common::pool::DynamicPool`]'s doc comment has the full
    /// layering), just without any `TaskDescAlloc`-specific construction:
    /// no descriptor, no join-protocol state, nothing schedulable — a
    /// recursion frame is never pushed to a deque, stolen, or joined by
    /// anyone but its immediate caller.
    type RecursionPool: DynamicPool;

    /// Queue for continuations pushed by external (non-worker) OS threads.
    type ExternalQueue: ExternalQueue<Self>;
}

/// Worker-layer subset of what used to be [`SchedulerSystem`]: the threading
/// base, the run-queue/item types, current-worker lookup, and the TLS slot
/// that anchors it. Split out as its own supertrait so this axis — "how do I
/// find/drive a worker" — can be named independently of dispatch
/// ([`RunnableItem::run_on`]/[`ReclaimableDesc::reclaim`]), which now lives
/// off `SchedulerSystem` entirely, on capability traits that must not
/// require it.
pub trait WorkerSystem: PoolSystem {
    /// The threading system this scheduler runs on.
    type Base: SpawnableStackfulTaskSystem + NestableSystem;

    /// The unit that goes on a worker run queue / the external queue. Every
    /// concrete system sets this to `SuspendedTaskToken<Self::Desc>` — kept
    /// as its own associated type (rather than folding it into `RunQueue`'s
    /// bound directly) so [`WorkerRunQueue`] and [`ExternalQueue`] stay
    /// generic over "whatever this system moves through them," with no need
    /// to name `SuspendedTaskToken`/`Self::Desc` themselves.
    ///
    /// The `Into`/`From` bounds let call sites cross between this opaque
    /// type and the crate's own concrete `SuspendedTaskToken<Self::Desc>`
    /// (e.g. a token freshly built from a raw descriptor pointer, or one
    /// pulled out of `ExternalQueue::try_pop`, which is declared at the
    /// `PoolSystem` level and so can only speak the concrete type) without
    /// pinning the two equal — every concrete system today sets
    /// `SuspendedToken = SuspendedTaskToken<Self::Desc>` as a literal type
    /// alias, so both bounds are satisfied for free by `std`'s blanket
    /// `impl<T> From<T> for T`.
    type SuspendedToken: Send
        + Into<SuspendedTaskToken<Self::Desc>>
        + From<SuspendedTaskToken<Self::Desc>>;

    /// Work-stealing run queue implementation. `+ Default` here (rather than
    /// as a [`WorkerRunQueue`] supertrait) so that trait's contract stays
    /// scoped to the queue behavior itself.
    type RunQueue: WorkerRunQueue<Self::SuspendedToken> + Default;

    /// Current-worker lookup policy.
    type Lookup: CurrentLookup<Self>;

    /// The concrete worker type driving this scheduler. Every system today
    /// sets this to [`UltWorker<Self>`](crate::resumable::common::worker::UltWorker) — kept as its own associated type
    /// (rather than [`worker_tls`](Self::worker_tls)/[`CurrentLookup`]
    /// naming `UltWorker<Self>` directly) so those interfaces stay generic
    /// over "whatever struct implements [`WorkerOps`] for this system,"
    /// with no need to name `UltWorker` itself.
    type Worker: WorkerOps<Self>;

    /// The one TLS slot that stores the worker pointer for this scheduler
    /// level.  Each concrete system gets its own `static`, anchored by the
    /// function body of this implementation.
    fn worker_tls() -> &'static <Self::Base as NestableSystem>::ThreadSpecific<Self::Worker>;
}

/// Execution capability of a scheduling item: what it means to run one.
/// Replaces the former `SchedulerSystem::execute` — the body was never a
/// choice the system author made, it was fully determined by the descriptor
/// flavor, so it belongs on the item type. Implemented on
/// `SuspendedTaskToken<Desc>` once per descriptor flavor (stackful-only,
/// dual, stackless-only) — see `resumable::stackful::worker`,
/// `resumable::dual::worker`, `resumable::stackless::worker` for the bodies.
pub trait RunnableItem<S: WorkerSystem> {
    fn run_on(self, wk: &S::Worker);
}

/// Reclamation capability of a finished descriptor. Replaces the former
/// `SchedulerSystem::free_finished_desc`, for the same reason: which pool
/// (if any) a finished descriptor returns to is fully determined by the
/// descriptor flavor, not a per-system choice.
pub trait ReclaimableDesc<S: WorkerSystem> {
    /// # Safety
    /// No other references to `desc` may exist after this call.
    unsafe fn reclaim(wk: &S::Worker, desc: *mut Self);
}

/// Dispatch subset of the base scheduler-system trait shared by every
/// flavor (stackful, stackless, dual): running a popped continuation and
/// freeing a finished descriptor.  Independent of whether tasks are stackful
/// ULTs, stackless `spawn_async` futures, or both.
///
/// A pure fold point, not implemented directly by concrete systems: the
/// dispatch bodies formerly required here as `execute`/`free_finished_desc`
/// methods now live on [`RunnableItem`]/[`ReclaimableDesc`], implemented per
/// descriptor flavor rather than per system — see those traits. Blanket-derived
/// for any `WorkerSystem` whose `Item`/`Desc` satisfy those capabilities, with
/// both bounds nested directly in this trait's own supertrait bound list (not
/// a separate `where`-clause): that is what lets every function merely
/// bounded `S: SchedulerSystem` get `Self::Item: RunnableItem<Self>` and
/// `Self::Desc: ReclaimableDesc<Self>` for free, with no need to restate
/// either (verified empirically — a `where`-clause form does not propagate
/// this way; see [`StackfulSchedulerSystem`](crate::resumable::stackful::system::StackfulSchedulerSystem)'s
/// doc comment for the same rule applied to its own nested bounds).
///
/// Deliberately does **not** name a context-switch policy or stack
/// allocator: a stackless-only system has no real stack to switch into, so
/// requiring one here would force it to name machinery it never uses. See
/// [`StackfulSchedulerSystem`](crate::resumable::stackful::system::StackfulSchedulerSystem) for the stackful extension.
pub trait SchedulerSystem: WorkerSystem<SuspendedToken: RunnableItem<Self>, Desc: ReclaimableDesc<Self>> {}

impl<S: WorkerSystem<SuspendedToken: RunnableItem<S>, Desc: ReclaimableDesc<S>>> SchedulerSystem for S {}

// ---------------------------------------------------------------------------
// Blanket TaskSystem for every WorkerSystem
// ---------------------------------------------------------------------------

/// Every `resumable`-backed system (stackful, stackless, or dual alike)
/// assumes the same work-stealing scheduler underneath, so `TaskSystem` is
/// blanket-derived here rather than implemented per flavor — one impl
/// covers `SpawnableStackfulTaskSystem`'s (stackful) and `StacklessTaskSystem`'s
/// (stackless) supertrait requirement alike.
///
/// Bounded on plain `WorkerSystem`, not `DescScheduler`:
/// [`WorkerOps::current`]/
/// `num`/`num_workers` are `LocalQueue`/`WorkerOps` trait methods reachable
/// through `S::Worker` alone — this needs no dispatch capability, and
/// neither method ever names `S::Item`/`S::Desc`, so there is nothing here
/// for `DescScheduler`'s `Item`/`Worker` pinning to actually buy. Only the
/// true base case (`OsSystem`, which isn't even a `WorkerSystem` — no
/// managed worker pool) needs its own hand-written impl, in `os.rs`.
impl<S: WorkerSystem> TaskSystem for S {
    fn worker_num() -> usize {
        match S::Worker::current() {
            Some(wk) => wk.num(),
            None => 0,
        }
    }

    fn num_workers() -> usize {
        match S::Worker::current() {
            Some(wk) => wk.num_workers(),
            None => 1,
        }
    }
}
