pub mod block_on;
pub mod bundle;
pub mod delegation;
pub mod nesting;
pub mod scoped;
pub mod stackful;
pub mod stackless;
pub mod suspend;
pub mod sync;

/// Declares that a system provides an efficient (work-stealing) scheduler
/// as its execution model — the shared foundation both
/// [`ThreadSystem`](crate::traits::system::stackful::ThreadSystem) (spawn/join) and
/// the `scoped` family (`ScopedStackfulTaskSystem`/`ScopedStacklessTaskSystem`,
/// in [`crate::traits::system::scoped`]) build on: both assume the same efficient
/// scheduling underneath, just expose different capabilities on top of it.
pub trait TaskSystem: Sized + Send + Sync + 'static {
    /// This worker's own index among its `num_workers()` peers (stable for
    /// the lifetime of the calling task/thread). Not meaningful outside a
    /// managed worker pool — a system with no such pool (e.g. `OsSystem`,
    /// whose "workers" are just whatever OS threads happen to be running)
    /// always reports `0`.
    fn worker_num() -> usize;

    /// Number of parallel workers (OS threads or ULT worker threads).
    fn num_workers() -> usize;
}
