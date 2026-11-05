//! Scheduler observers ([`SchedObserver`]).

pub use crate::traits::component::dispatch::Site;
pub use crate::traits::component::observe::{SchedObserver, StealOutcome};

/// Observes nothing.  The default.
pub struct NoObserver;

impl SchedObserver for NoObserver {}
