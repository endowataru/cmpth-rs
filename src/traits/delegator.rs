//! [`Delegator`] — queue-based lock for serialising hardware access.

use crate::traits::thread_system::ThreadSystem;

/// User-supplied consumer that the delegator executes on behalf of callers.
///
/// The consumer has exclusive access to a hardware resource (RDMA QP, epoll
/// fd, …). Callers that cannot acquire the delegator lock write their work
/// into a queue node; the consumer ULT drains the queue and calls `progress`
/// to poll for completions.
pub trait DelegatorConsumer<S: ThreadSystem>: Send + 'static {
    /// Per-call work descriptor written into the queue by a delegating caller.
    type Work: Send + Default;

    /// Execute one work item (called by whoever holds the delegator lock).
    /// Returns `(is_done, thread_to_wake)`.
    fn execute(&mut self, work: &mut Self::Work) -> (bool, Option<S::SuspendedThread>);

    /// Poll for completions (e.g. ibv_poll_cq).  Called while `is_active`.
    /// Returns a thread to wake if a completion was found.
    fn progress(&mut self) -> Option<S::SuspendedThread>;

    /// True while there are posted-but-not-completed operations.
    /// When false the consumer ULT suspends instead of spinning.
    fn is_active(&self) -> bool;
}

/// Delegator: a queue-based lock that serialises hardware access and batches
/// work items on behalf of callers that miss the lock.
pub trait Delegator<S: ThreadSystem, C: DelegatorConsumer<S>>:
    Sized + Send + Sync + 'static
{
    /// Start accepting delegations. Implementations may spawn the consumer
    /// ULT eagerly here or lazily on first use — see the implementation for
    /// which, and why (a `Self`-address-stability concern rules eager
    /// spawning out for `Delegator<S, C, Q>`).
    fn start(consumer: C) -> Self;

    /// Stop the consumer ULT (blocks until it exits).
    fn stop(self);

    /// Either execute `imm` inline (if the lock is free) or write work via
    /// `del` into the queue and suspend until the consumer executes it.
    ///
    /// `imm` — called with `&mut Consumer` when the caller wins the lock.
    ///   Returns `(is_done, Option<suspended_thread_to_wake_on_unlock>)`.
    /// `del` — called with `&mut C::Work` when delegating; fills in the work
    ///   and returns a reference to the `SuspendedThread` to park on.
    fn execute_or_delegate<Imm, Del>(&self, imm: Imm, del: Del)
    where
        Imm: FnOnce(&mut C) -> (bool, Option<S::SuspendedThread>),
        Del: FnOnce(&mut C::Work) -> &S::SuspendedThread;
}
