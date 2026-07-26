//! [`SchedulerSystem`] — the base scheduler-system trait shared by every
//! flavor (stackful, stackless, dual). Extended by
//! [`UltSchedulerSystem`](crate::resumable::stackful::system::UltSchedulerSystem)
//! (real-stack capability) and
//! [`AsyncTaskSystem`](crate::resumable::stackless::system::AsyncTaskSystem)
//! (async-task capability).

use crate::traits::thread_system::ThreadSystem;
use crate::resumable::common::deque::WorkerDeque;
use crate::resumable::common::external_queue::ExternalQueue;
use crate::resumable::common::desc::{SuspendedUlt, TaskDescAlloc};
use crate::resumable::common::pool::{DescPool, DynamicPool};
use crate::resumable::common::worker::UltWorker;

/// Base system interface required by [`UltWorker`] and
/// [`Scheduler`](crate::resumable::common::scheduler::Scheduler), independent of whether
/// tasks are stackful ULTs, stackless `spawn_async` futures, or both.
///
/// Deliberately does **not** name a context-switch policy or stack
/// allocator: a stackless-only system has no real stack to switch into, so
/// requiring one here would force it to name machinery it never uses. See
/// [`UltSchedulerSystem`](crate::resumable::stackful::system::UltSchedulerSystem) for the stackful extension.
pub trait SchedulerSystem: Sized + Send + Sync + 'static {
    /// The threading system this scheduler runs on.
    type Base: ThreadSystem;

    /// Task descriptor type for this system. Every concrete system today
    /// sets this to `BasicTaskDesc`; the associated type exists so
    /// `SuspendedUlt`/`WorkerDeque`/`DescPool`/the worker traits never
    /// hardcode a concrete descriptor, in preparation for narrower
    /// stackful-only/stackless-only descriptor types later.
    type Desc: TaskDescAlloc;

    /// Work-stealing deque implementation.
    type Deque: WorkerDeque<Self::Desc>;

    /// Descriptor pool implementation for this system, used by the stackful
    /// `spawn` path (fixed-size ULT stacks, `STACK_SIZE` on
    /// [`UltSchedulerSystem`](crate::resumable::stackful::system::UltSchedulerSystem)).
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
    /// precedent of stackless-only systems declaring an unused [`Pool`].
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

    /// Current-worker lookup policy.
    type Lookup: crate::resumable::common::lookup::CurrentLookup<Self>;

    /// Queue for continuations pushed by external (non-worker) OS threads.
    type ExternalQueue: ExternalQueue<Self>;

    /// The one TLS slot that stores the worker pointer for this scheduler
    /// level.  Each concrete system gets its own `static`, anchored by the
    /// function body of this implementation.
    fn worker_tls() -> &'static <Self::Base as ThreadSystem>::ThreadSpecific<UltWorker<Self>>;

    /// Run one continuation popped off a deque/root/external-queue.
    ///
    /// Required, with **no default**: the correct body depends entirely on
    /// which task flavors this system supports, and a base `SchedulerSystem`
    /// can't know that. Every concrete system supplies this directly by
    /// calling exactly one of the free functions in `worker.rs`:
    ///
    /// - stackful-only: [`crate::resumable::stackful::worker::execute_stackful`] (always a
    ///   real context switch — `Self::Desc` need not even implement
    ///   `AsyncTaskDesc`, so there is no tag to check).
    /// - dual: [`crate::resumable::dual::worker::execute_dual`] (today's poll_fn check).
    /// - stackless-only (added when that flavor lands): always polls.
    ///
    /// This is ordinary trait-method overriding, not specialization: each
    /// concrete marker struct gets exactly one `impl SchedulerSystem for
    /// Self` block, so the compiler picks the right body statically.
    fn execute(wk: &UltWorker<Self>, cont: SuspendedUlt<Self::Desc>);

    /// Free a finished task's descriptor once its `JoinHandle` is done with
    /// it (`take_result`/`Drop`, both in `thread.rs`).
    ///
    /// Required, with **no default**, for the same reason as [`execute`]:
    /// stackful-only frees always go through the pool
    /// ([`crate::resumable::stackful::worker::free_finished_desc_stackful`]); stackless-only
    /// descriptors always bypass the pool (variable-size `spawn_async`
    /// allocations — [`crate::resumable::stackless::worker::free_finished_desc_async`]); dual
    /// systems check `poll_fn` first
    /// ([`crate::resumable::dual::worker::free_finished_desc_dual`]).
    ///
    /// [`execute`]: Self::execute
    fn free_finished_desc(wk: &UltWorker<Self>, desc: *mut Self::Desc);
}
