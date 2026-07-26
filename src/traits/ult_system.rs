//! [`UltSystem`] and [`AsyncWorkerSystem`] — the two independent
//! capability traits a concrete system composes (`docs/sync-async-unification.md`).
//!
//! Kept separate from [`SchedulerSystem`](crate::resumable::common::system::SchedulerSystem)/
//! [`UltSchedulerSystem`](crate::resumable::stackful::system::UltSchedulerSystem), which stay in
//! `resumable::system`: those reference implementation-layer types
//! (`UltWorker`) directly in their signatures, so they belong with the
//! scheduler implementation, not the interface layer. `UltSystem` is a
//! sibling of `UltSchedulerSystem`, not built on top of it (no supertrait
//! relationship) — a concrete marker struct implements both independently
//! (via [`ult_system!`](crate::ult_system)), but generic code that only
//! needs scheduler internals (almost all of `resumable/`) should bound on
//! `UltSchedulerSystem` alone rather than pulling in `UltSystem`'s
//! user-facing `Mutex`/`Barrier`/`Delegator` capabilities it never uses.
//! `run` *is* a trait method here, despite needing `Self: UltSchedulerSystem`
//! to implement (every implementor is a concrete marker struct that also
//! implements `UltSchedulerSystem` directly, so the bound is satisfied at
//! each `impl` site without this trait's own signature needing to name it) —
//! `bench/examples/quick.rs` calls it generically over an abstract
//! `S: UltSystem` to run the same benchmark body against several system
//! configurations, so it can't be inherent-only.

use crate::traits::{Delegator, DelegatorConsumer, StackfulBarrier, StackfulMutex, StacklessBarrier, StacklessMutex};
use crate::traits::thread_system::ThreadSystem;

/// Full ULT system configuration trait.
///
/// Implement this on a concrete marker struct to define a complete ULT system.
/// The blanket `impl<S: UltSystem + UltSchedulerSystem> ThreadSystem for S`
/// automatically provides all threading-system methods; there is no need to
/// write `impl ThreadSystem for …` separately.
///
/// Use [`ult_system!`](crate::ult_system) to implement this trait.
pub trait UltSystem: Sized + Send + Sync + 'static {
    /// Mutex type for this system.
    type Mutex<T: Send>: StackfulMutex<T> + Send + Sync;

    /// Barrier type for this system.
    type Barrier: StackfulBarrier + Send + Sync;

    /// Delegator type for this system.
    type Delegator<C: DelegatorConsumer<Self>>: Delegator<Self, C>
    where
        Self: ThreadSystem;

    /// Start `num_workers` workers and run `root` as the first task.
    /// Returns when `root` and all spawned tasks complete.
    fn run<F>(num_workers: usize, root: F)
    where
        F: FnOnce() + Send + 'static;
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
