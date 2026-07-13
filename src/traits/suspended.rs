//! [`SuspendedThread`] — the parked-continuation interface.

use crate::traits::thread_system::ThreadSystem;

/// Interface for a parked-continuation slot.
///
/// A `SuspendedThread` holds zero or one suspended continuation.  The slot is
/// externally synchronized by the owning primitive's spinlock; the trait
/// itself is not thread-safe in isolation.
pub trait SuspendedThread: Default + Send {
    /// The threading system this slot belongs to.
    type System: ThreadSystem;

    /// True if a continuation is currently parked here.
    fn is_set(&self) -> bool;

    /// Suspend the current thread into this slot.  `f` runs after the context
    /// is fully saved and must release any spinlock protecting this slot.
    /// Once `f` returns, a notifier may legally consume the slot.
    fn wait_with<F>(&self, f: F)
    where
        F: FnOnce();

    /// Like [`wait_with`](Self::wait_with), but `f` may cancel the suspension
    /// by returning `false`, in which case the current thread resumes
    /// immediately and the slot stays empty.
    fn wait_with_cond<F>(&self, f: F)
    where
        F: FnOnce() -> bool;

    /// Push the parked thread to the scheduler (LIFO end).
    fn notify(&self);

    /// Switch directly to the parked thread; the current thread goes to the
    /// scheduler.
    fn enter(&self);

    /// Symmetric handoff: park the current thread here and switch to `next`.
    fn swap(&self, next: &Self);
}
