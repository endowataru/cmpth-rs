//! Idle policy: what a worker does when it has looked everywhere and found
//! nothing, and how a producer tells it that has changed.
//!
//! The two halves are one protocol, which is why they are one trait:
//! [`no_work_found`](IdlePolicy::no_work_found) (consumer side, called from
//! the shared idle loop) and [`new_work`](IdlePolicy::new_work) (producer
//! side, called after every publication of stealable work) must agree on how
//! a sleeping worker is woken without a lost wakeup. A policy that never
//! sleeps (`SpinIdle`) has nothing to agree on, and `new_work` compiles
//! away entirely.
//!
//! A system picks one via `WorkerSystem::Idle`; implementations:
//! `resumable::common::idle`.

use std::sync::atomic::AtomicBool;
use std::time::Duration;

/// The duties an idle worker can serve: the system's duty registry, or
/// nothing.  Every call is cheap when there are no duties.
pub trait IdleDuties {
    /// Run the duties that are half due.  Idle policies call it where they
    /// already do something slow (yield the thread, or before sleeping).
    fn run_idle();

    /// How long a worker should spin before it parks, given the duties'
    /// wait periods; 0 without duties.
    fn wait_spin_ns() -> u64;

    /// Before parking: if the worker should keep the duties' due times, set
    /// `timed` and return how long it may sleep.
    fn park_timeout(timed: &mut bool) -> Option<Duration>;

    /// A worker that kept the due times (`timed`) found work; clears
    /// `timed`.
    fn idle_done(timed: &mut bool);

    /// The pool shuts down: give back the due times if one of its workers
    /// keeps them.
    fn release_timed();
}

/// What an idle policy may use from the worker that calls it: whether the
/// last round saw work it could not take, the pool's `finished` flag, a
/// yield of the base thread, the system's duties and its observer.
pub trait IdleContext {
    type Duties: IdleDuties;

    /// The calling worker's index in its pool.
    fn worker(&self) -> usize;

    /// The round saw a victim that had work but could not be robbed right
    /// now ([`Steal::Retry`](super::run_queue::Steal::Retry)): not evidence
    /// of idleness.
    fn retry(&self) -> bool;

    /// Set once the pool stops; a policy that blocks must return promptly
    /// once it is (and [`IdlePolicy::shutdown`] wakes it).
    fn finished(&self) -> &AtomicBool;

    /// Yield the *base* thread: the base need not be an OS thread, so never
    /// `std::thread::yield_now`.
    fn yield_now(&self);

    /// Report that this worker is about to block its thread...
    fn parking(&self);

    /// ... and that it woke up.
    fn woke(&self);

    /// [`IdleDuties::run_idle`] of the system's duties.
    #[inline(always)]
    fn run_duties(&self) {
        Self::Duties::run_idle();
    }

    /// [`IdleDuties::wait_spin_ns`] of the system's duties.
    #[inline(always)]
    fn duty_wait_spin_ns(&self) -> u64 {
        Self::Duties::wait_spin_ns()
    }

    /// [`IdleDuties::park_timeout`] of the system's duties.
    #[inline(always)]
    fn duty_park_timeout(&self, timed: &mut bool) -> Option<Duration> {
        Self::Duties::park_timeout(timed)
    }

    /// [`IdleDuties::idle_done`] of the system's duties.
    #[inline(always)]
    fn duty_idle_done(&self, timed: &mut bool) {
        Self::Duties::idle_done(timed)
    }
}

/// Idle/wakeup protocol of one worker pool. One value is shared by all
/// workers of a pool (it lives in `PoolCore`).
///
/// # Contract
///
/// * `new_work` is called by *any* thread after it made an item visible to
///   thieves (local `push`/`defer`, external-queue push). It must be cheap
///   when nobody is idle.
/// * `no_work_found` is called by an idle worker after each search round
///   that found nothing. The caller re-searches after it returns, so a policy
///   may use consecutive calls as "announce intent to sleep", then "re-check",
///   then "sleep" (see `ParkIdle`).  The re-check is a full search only if
///   the system's [`StealPolicy`](super::steal::StealPolicy) tries every
///   victim in a round (the standard ones but `RandomSteal` do); otherwise
///   a sleeper may miss work that another worker will run itself.
/// * `work_found` is called when a search succeeds, ending the idle episode.
/// * `shutdown` is called once after `finished` has been set; every blocked
///   worker must return from `no_work_found` promptly.
///
/// The calls get an [`IdleContext`]: the system's duties, its observer and
/// the base thread's yield.  A policy should run the duties
/// ([`IdleContext::run_duties`]) where it already does something slow, as
/// the standard ones do; a policy that never does leaves them to the
/// keeper.  `shutdown` gets the duties alone.
pub trait IdlePolicy: Default + Send + Sync + 'static {
    /// Per-worker state for one idle episode (round counters, a sleep ticket).
    type Episode: Default;

    fn new_work(&self);

    fn work_found<C: IdleContext>(&self, ep: &mut Self::Episode, cx: &C);

    fn no_work_found<C: IdleContext>(&self, ep: &mut Self::Episode, cx: &C);

    fn shutdown<D: IdleDuties>(&self);

    /// Told the pool's worker count once, when the pool is built.
    fn set_workers(&self, _n: usize) {}
}
