//! [`Poller`] — customisation point for `block_on`.

use std::task::Context;

/// Drives a single `block_on` invocation.
///
/// `Poller` is to `block_on` what `SuspendedThread` is to `wait_with`: a thin
/// type that encapsulates the system-specific park/wake mechanism, leaving the
/// poll loop itself as a generic default on [`ThreadSystem`].
///
/// Implementations are always stack-local inside `block_on`.  They are `!Send`
/// by convention — bound to the same ULT, not to a specific OS thread.  In
/// cmpth, `!Send` means "bound to the same ULT", not "bound to the same OS
/// thread": work-stealing moves the entire ULT stack atomically, so a `!Send`
/// value is safe across `yield_now` even when the ULT migrates.
///
/// [`Drop`] performs cleanup (e.g. resetting `waker_refs` to `IDLE`).
///
/// [`ThreadSystem`]: crate::traits::thread_system::ThreadSystem
pub trait Poller {
    /// Initialise for the current thread/ULT.
    fn new() -> Self;

    /// Return a [`Context`] whose [`Waker`] resumes the current thread/ULT.
    ///
    /// [`Waker`]: std::task::Waker
    fn context<'a>(&'a self) -> Context<'a>;

    /// Suspend until the waker fires.
    ///
    /// For ULT systems this uses `cond_suspend_to_sched` and handles the race
    /// where `wake()` fires between `poll()` returning `Pending` and the actual
    /// suspension (NOTIFIED → re-poll without parking).
    fn wait(&self);
}
