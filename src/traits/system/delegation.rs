use crate::traits::component::delegation::DelegatorConsumer;
use crate::traits::system::stackful::ThreadSystem;

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
