//! Scheduler observer: the scheduler reports its events to
//! `WorkerSystem::Observe`, so a counter or a trace can be attached to a
//! system without editing the scheduler, and every measurement counts the
//! same events at the same places.
//!
//! Every hook is a static function with an empty default body, called with
//! the worker's index in its pool; `NoObserver` (the default) compiles to
//! nothing.  An observer keeps its own storage (per-worker counters in a
//! `static`, say) and must be cheap: the hooks run on the scheduler's
//! paths.

use super::dispatch::Site;
use super::run_queue::Steal;

/// The outcome of one steal round, without the stolen item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StealOutcome {
    Success,
    Empty,
    Retry,
}

impl StealOutcome {
    #[inline(always)]
    pub fn of<T>(s: &Steal<T>) -> StealOutcome {
        match s {
            Steal::Success(_) => StealOutcome::Success,
            Steal::Empty => StealOutcome::Empty,
            Steal::Retry => StealOutcome::Retry,
        }
    }
}

/// Scheduler events, reported by worker index.
pub trait SchedObserver: Send + Sync + 'static {
    /// Worker `me` ran a steal round.
    #[inline(always)]
    fn steal(_me: usize, _outcome: StealOutcome) {}

    /// Worker `me` picked a task at `site`.
    #[inline(always)]
    fn dispatch(_me: usize, _site: Site) {}

    /// Worker `me` searched everywhere and found nothing (one round).
    #[inline(always)]
    fn idle(_me: usize) {}

    /// Worker `me` is about to block its thread (reported by the idle
    /// policy; see [`IdleContext::parking`](super::idle::IdleContext::parking)).
    #[inline(always)]
    fn park(_me: usize) {}

    /// Worker `me` woke from [`park`](Self::park).
    #[inline(always)]
    fn unpark(_me: usize) {}
}
