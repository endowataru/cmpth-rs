//! Placement policies ([`PlacementPolicy`]).

pub use crate::traits::component::placement::{PlacementPolicy, SpawnOrder, WakeTo};

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
