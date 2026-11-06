//! Duties: work that someone must check every so often because nothing
//! will announce its result (a transport's progress engine that serves
//! other processes, requests whose completion only a poll detects, timers).
//!
//! A [`DutyService`] registers duties and decides who runs them; a system
//! names the one its idle workers serve
//! ([`DutySystem::Duties`](crate::traits::DutySystem::Duties)).

use std::future::Future;
use std::time::{Duration, Instant};

/// Where a duty's `poll` may run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DutyPlace {
    /// Any thread, including a worker's idle loop: `poll` must not block
    /// or suspend.
    Anywhere,
    /// The service's own thread only: `poll` may block it (briefly).
    Keeper,
}

/// A registered duty; dropping it unregisters it, and its `poll` never runs
/// again after the drop returns.
pub trait DutyLike: Send + Sync + 'static {
    type Toucher: DutyToucherLike<Waiter = Self::Waiter>;
    /// A registered waiter; dropping it unregisters it.
    type Waiter: Send + 'static;

    /// Change the obligation: at least every `period`, backing off up to
    /// `max_period` while `poll` finds nothing to do; `None`: only while
    /// there are waiters.
    fn set_period(&self, period: Option<(Duration, Duration)>);

    /// Change the period while there are waiters.
    fn set_wait_period(&self, period: Duration);

    /// Ask for a poll at `at` (or none); the poll clears it, so a duty that
    /// wants the next one sets it again from its `poll`.
    fn set_deadline(&self, at: Option<Instant>);

    /// Report that the progress this duty stands for just happened anyway.
    fn touch(&self);

    /// Register a waiter: until it is dropped, the duty runs every wait
    /// period.  Whoever detects the completion wakes the waiting task; the
    /// duty only makes sure someone looks.
    fn waiter(&self) -> Self::Waiter;

    /// A handle that only touches and registers waiters, for code that
    /// makes the progress but does not own the duty.  Touching an
    /// unregistered duty does nothing.
    fn toucher(&self) -> Self::Toucher;
}

/// See [`DutyLike::toucher`].
pub trait DutyToucherLike: Clone + Send + Sync + 'static {
    type Waiter: Send + 'static;

    /// See [`DutyLike::touch`].
    fn touch(&self);

    /// See [`DutyLike::waiter`].
    fn waiter(&self) -> Self::Waiter;
}

/// Registers duties and runs them.
pub trait DutyService: Send + Sync + 'static {
    type Duty: DutyLike;
    type Sleep: Future<Output = ()> + Send + 'static;

    /// Register `poll` to run at least every `period` (backing off up to
    /// `max_period` while it returns false, i.e. did nothing), and every
    /// `period` while it has waiters.
    fn duty(period: Duration, max_period: Duration, place: DutyPlace, poll: impl Fn() -> bool + Send + Sync + 'static) -> Self::Duty;

    /// Register `poll` to run only while it has waiters, every
    /// `wait_period`, or at its deadline.
    fn on_demand(wait_period: Duration, place: DutyPlace, poll: impl Fn() -> bool + Send + Sync + 'static) -> Self::Duty;

    /// A future that resolves after `d`: the task suspends meanwhile, and
    /// whoever runs the duties wakes it.
    fn sleep(d: Duration) -> Self::Sleep;
}
