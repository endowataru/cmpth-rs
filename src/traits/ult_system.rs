//! [`UltSystem`] and [`AsyncWorkerSystem`] — the two independent
//! capability traits a concrete system composes (`docs/sync-async-unification.md`).
//!
//! Kept separate from [`UltContextSystem`](crate::ult::system::UltContextSystem)/
//! [`UltSchedulerSystem`], which stay in
//! `ult::system`: those reference implementation-layer types
//! (`UltWorker`) directly in their signatures, so they belong with the
//! scheduler implementation, not the interface layer.

use crate::traits::{Delegator, DelegatorConsumer, StackfulBarrier, StackfulMutex, StacklessBarrier, StacklessMutex};
use crate::ult::system::UltSchedulerSystem;

/// Full ULT system configuration trait.
///
/// Implement this on a concrete marker struct to define a complete ULT system.
/// The blanket `impl<S: UltSystem> ThreadSystem for S` automatically provides
/// all threading-system methods; there is no need to write
/// `impl ThreadSystem for …` separately.
///
/// Use [`ult_system!`](crate::ult_system) to implement this trait.
pub trait UltSystem: UltSchedulerSystem {
    /// Mutex type for this system.
    type Mutex<T: Send>: StackfulMutex<T> + Send + Sync;

    /// Barrier type for this system.
    type Barrier: StackfulBarrier + Send + Sync;

    /// Delegator type for this system.
    type Delegator<C: DelegatorConsumer<Self>>: Delegator<Self, C>;

    /// Start `num_workers` workers on the base system and run `root` as the
    /// first task.  Returns when `root` and all spawned tasks complete.
    fn run<F>(num_workers: usize, root: F)
    where
        F: FnOnce() + Send + 'static,
    {
        crate::ult::scheduler::run::<Self, F>(num_workers, root)
    }
}

/// Marker for systems that can run stackless `spawn_async`-style tasks.
///
/// Independent of [`UltSystem`] (see `docs/sync-async-unification.md`): a
/// system can implement `UltSystem` only (real ULTs, no async),
/// `AsyncWorkerSystem` only (no stack machinery at all — not yet expressible
/// here since `spawn_async` is still defined in terms of `UltSystem`, see
/// the doc's "open / not yet decided"), or both (today's `DefaultUltSystem`).
///
/// Deliberately not blanket-implemented for every `UltSystem`: that would
/// make a genuinely async-free configuration inexpressible.
pub trait AsyncWorkerSystem: Sized + Send + Sync + 'static {
    /// Mutex type for this system's stackless (`.await`-based) callers.
    type Mutex<T: Send>: StacklessMutex<T> + Send + Sync;

    /// Barrier type for this system's stackless callers.
    type Barrier: StacklessBarrier + Send + Sync;
}
