use std::future::Future;
use std::pin::pin;
use std::task::Poll;

use crate::traits::component::delegation::DelegatorConsumer;
use crate::traits::component::stackful::Poller;
use crate::traits::component::tls::TlsSlot;
use crate::traits::system::TaskSystem;
use crate::traits::system::delegation::Delegator;
use crate::traits::system::sync::{StackfulBarrier, StackfulMutex};

/// Threading system interface bundle — swap the entire backend by changing
/// one type parameter.
pub trait ThreadSystem: TaskSystem {
    /// Drives a single `block_on` call; the customisation point for async
    /// integration.  See [`Poller`].
    type Poller: Poller;

    /// Block the current thread/ULT until `future` completes.
    ///
    /// On a ULT system this suspends only the calling ULT; the OS thread
    /// underneath keeps running other tasks.
    ///
    /// ```
    /// use cmpth::{DefaultStackfulOnlyTaskSystem, ScopedStackfulTaskSystem, ThreadSystem};
    ///
    /// DefaultStackfulOnlyTaskSystem::run(2, || {
    ///     let x = DefaultStackfulOnlyTaskSystem::block_on(async { 6 * 7 });
    ///     assert_eq!(x, 42);
    /// });
    /// ```
    ///
    /// The default implementation drives the future through [`Self::Poller`].
    fn block_on<F, T>(f: F) -> T
    where
        F: Future<Output = T> + Send,
        T: Send,
    {
        let pol = Self::Poller::new();
        let mut f = pin!(f);
        loop {
            match f.as_mut().poll(&mut pol.context()) {
                Poll::Ready(v) => return v,
                Poll::Pending => pol.wait(),
            }
        }
    }

    /// Yield the current thread/ULT so other tasks can run.
    fn yield_now();

    /// Spawn a new thread or ULT; returns a handle that can be joined.
    type JoinHandle<T: Send + 'static>: JoinHandleLike<T>;
    fn spawn<T, F>(f: F) -> Self::JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static;

    /// Mutex type for this system.
    type Mutex<T: Send>: StackfulMutex<T> + Send + Sync;

    /// Barrier type for this system.
    type Barrier: StackfulBarrier + Send + Sync;

    /// Parked-continuation handle for this system.
    type SuspendedThread: Send + Default;

    /// Delegator type for this system.
    type Delegator<C: DelegatorConsumer<Self>>: Delegator<Self, C>;

    /// Thread-specific storage slot: one `*mut T` per thread (or per ULT) of
    /// this system.  A nested scheduler stores its per-worker pointer here,
    /// which is why a single slot per level is enough — everything else is
    /// reached through the worker pointer.
    type ThreadSpecific<T: 'static>: TlsSlot<T>;
}

/// Common interface for join handles returned by [`ThreadSystem::spawn`].
pub trait JoinHandleLike<T: Send + 'static>: Send {
    fn join(self) -> T;
}
