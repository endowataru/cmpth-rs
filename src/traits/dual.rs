//! [`DualBarrier`]/[`DualMutex`] — synchronization primitives usable from
//! either calling convention, spanning both flavors in one file the same
//! way [`crate::traits::scoped`] does (see that module's own doc comment).

use crate::traits::stackful::{StackfulBarrier, StackfulMutex};
use crate::traits::stackless::{StacklessBarrier, StacklessMutex};

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
