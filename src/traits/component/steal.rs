//! Steal policy: which victims an idle worker tries, and in what order.
//!
//! A policy sees only the pool's stealer table and the thief's own index,
//! and keeps whatever it needs between rounds (a cursor, a random state) in
//! a per-worker [`StealPolicy::State`].  It never touches the run queues
//! beyond [`RunQueueStealer::try_steal`], so it is independent of the queue
//! implementation and of the idle policy.
//!
//! A system picks one via `WorkerSystem::Steal`; implementations:
//! `resumable::common::steal`.

use super::run_queue::{RunQueueStealer, Steal};

/// Victim selection for one worker pool.
///
/// # Contract
///
/// * `try_steal` is called by the worker `me` only, once per search round
///   that found nothing in its own queue; `victims[me]` is its own queue
///   and must not be robbed.
/// * It returns [`Steal::Retry`] if a victim it tried had work that could
///   not be taken right now, so that the idle policy does not count the
///   round toward backing off.
/// * A policy need not try every victim in one round: the caller searches
///   again after an empty round.
pub trait StealPolicy: Send + Sync + 'static {
    /// Per-worker state, touched only by its owning worker.
    type State;

    /// The state of worker `me`, built once when its pool is built.
    fn new_state(me: usize) -> Self::State;

    /// One steal round.  Only called when there are at least two workers.
    fn try_steal<T: Send, St: RunQueueStealer<T>>(state: &Self::State, me: usize, victims: &[St]) -> Steal<T>;
}
