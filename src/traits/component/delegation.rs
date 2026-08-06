use crate::traits::system::stackful::ThreadSystem;

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
