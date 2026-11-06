//! Placement policy: where new and woken tasks go on the worker that
//! creates or wakes them.
//!
//! * [`SpawnOrder`]: a stackful `spawn` runs the child at once and leaves
//!   the parent for thieves (child-first, the default: work-first
//!   scheduling, a fork-join tree stays depth-first on each worker), or
//!   queues the child and lets the parent go on (parent-first: what
//!   `spawn_async` and `parallel_call`'s second branch always do).
//! * [`WakeTo`]: a task woken on a worker goes to the front of its queue
//!   (runs next: the waker's cache is still warm with what it touched, the
//!   default) or to the back (behind what was queued: fairer to tasks
//!   already waiting).  A task woken from outside the pool always goes
//!   through the external queue.
//!
//! A *yielding* task always goes to the back of its worker's queue
//! ([`WorkerRunQueue::defer`](super::run_queue::WorkerRunQueue::defer)).
//!
//! A system picks one via `WorkerSystem::Placement`; implementations:
//! `resumable::common::placement`.

/// Order of a stackful `spawn`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpawnOrder {
    ChildFirst,
    ParentFirst,
}

/// Where a task woken on a worker goes in that worker's queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WakeTo {
    /// The worker's `push`: runs next.
    Front,
    /// The worker's `defer`: after the tasks already queued.
    Back,
}

/// Placement of new and woken tasks.  Constants only: each choice compiles
/// to one path.
pub trait PlacementPolicy: Send + Sync + 'static {
    const SPAWN: SpawnOrder = SpawnOrder::ChildFirst;
    const WAKE: WakeTo = WakeTo::Front;
}
