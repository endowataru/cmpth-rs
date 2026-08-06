use crate::traits::system::TaskSystem;

/// Threading system interface — spawn/join only.  Split off from what used
/// to be a six-capability bundle (see `block_on.rs`, `sync.rs`,
/// `suspend.rs`, `delegation.rs`, `nesting.rs` in this module for the rest)
/// so that code which only wants `spawn` isn't forced to also supply a
/// `Poller`, a `Mutex`, a `Barrier`, a `Delegator`, a TLS slot type, and a
/// parked-continuation type.
pub trait ThreadSystem: TaskSystem {
    /// Spawn a new thread or ULT; returns a handle that can be joined.
    type JoinHandle<T: Send + 'static>: JoinHandleLike<T>;
    fn spawn<T, F>(f: F) -> Self::JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static;

    /// Yield the current thread/ULT so other tasks can run.
    fn yield_now();
}

/// Common interface for join handles returned by [`ThreadSystem::spawn`].
pub trait JoinHandleLike<T: Send + 'static>: Send {
    fn join(self) -> T;
}
