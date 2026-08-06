//! [`DualBarrier`]/[`DualMutex`] — synchronization primitives usable from
//! either calling convention, spanning both flavors in one file the same
//! way [`crate::traits::scoped`] does (see that module's own doc comment).

pub use crate::traits::system::sync::{DualBarrier, DualMutex};
