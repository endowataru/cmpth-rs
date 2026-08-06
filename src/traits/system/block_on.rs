use std::future::Future;
use std::pin::pin;
use std::task::Poll;

use crate::traits::component::stackful::Poller;
use crate::traits::system::TaskSystem;

/// The `block_on` capability, split off from [`ThreadSystem`](crate::traits::system::stackful::ThreadSystem):
/// a system that only ever spawns/joins never needs a [`Poller`], so this
/// stays a separate trait rather than a member of `ThreadSystem` itself.
pub trait BlockOnSystem: TaskSystem {
    /// Drives a single `block_on` call; the customisation point for async
    /// integration.  See [`Poller`].
    type Poller: Poller;

    /// Block the current thread/ULT until `future` completes.
    ///
    /// On a ULT system this suspends only the calling ULT; the OS thread
    /// underneath keeps running other tasks.
    ///
    /// ```
    /// use cmpth::{BlockOnSystem, DefaultStackfulOnlyTaskSystem, StackfulBuilder, StackfulInitSystem};
    ///
    /// DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| {
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
}
