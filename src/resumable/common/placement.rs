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
//! Where a *yielding* task goes is the [`YieldPolicy`](crate::YieldPolicy)'s
//! choice.  Swap the implementation via [`crate::WorkerSystem::Placement`].

/// Order of a stackful `spawn`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpawnOrder {
    ChildFirst,
    ParentFirst,
}

/// Where a task woken on a worker goes in that worker's queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WakeTo {
    /// [`push`](crate::resumable::common::worker::LocalQueue::push): runs next.
    Front,
    /// [`defer`](crate::resumable::common::worker::LocalQueue::defer): after
    /// the tasks already queued.
    Back,
}

/// Placement of new and woken tasks.  Constants only: each choice compiles
/// to one path.
pub trait PlacementPolicy: Send + Sync + 'static {
    const SPAWN: SpawnOrder = SpawnOrder::ChildFirst;
    const WAKE: WakeTo = WakeTo::Front;
}

/// Child-first spawn, woken tasks to the front.  The default.
pub struct StandardPlacement;
impl PlacementPolicy for StandardPlacement {}

/// Parent-first spawn.  (A policy generic over the two enums would need
/// `adt_const_params`, so the combinations are spelled out; write your own
/// impl for anything else.)
pub struct ParentFirst;
impl PlacementPolicy for ParentFirst {
    const SPAWN: SpawnOrder = SpawnOrder::ParentFirst;
}

/// Woken tasks go behind the tasks already queued.
pub struct WakeBack;
impl PlacementPolicy for WakeBack {
    const WAKE: WakeTo = WakeTo::Back;
}

/// [`ParentFirst`] and [`WakeBack`].
pub struct ParentFirstWakeBack;
impl PlacementPolicy for ParentFirstWakeBack {
    const SPAWN: SpawnOrder = SpawnOrder::ParentFirst;
    const WAKE: WakeTo = WakeTo::Back;
}
