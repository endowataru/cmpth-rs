//! [`Barrier`] — thread-system-agnostic barrier.

/// Return value of [`Barrier::wait`], mirroring `std::sync::BarrierWaitResult`.
pub struct BarrierWaitResult {
    pub is_leader: bool,
}

impl BarrierWaitResult {
    pub fn is_leader(&self) -> bool { self.is_leader }
}

/// Barrier abstraction.
pub trait Barrier: Sized + Send + Sync {
    fn new(count: usize) -> Self;
    fn wait(&self) -> BarrierWaitResult;
}

/// Stackful/stackless-flavored barrier `wait`, same disambiguation pattern
/// as [`crate::traits::StackfulMutex`]/[`crate::traits::StacklessMutex`]: both
/// traits define a method literally named `wait`, resolved by which trait is
/// `use`d at the call site. Each carries its own `new`, same reasoning as
/// `StackfulMutex`/`StacklessMutex`.
pub trait StackfulBarrier: Sized + Send + Sync {
    fn new(count: usize) -> Self;
    fn wait(&self) -> BarrierWaitResult;
}

pub trait StacklessBarrier: Sized + Send + Sync {
    fn new(count: usize) -> Self;
    fn wait<'a>(&'a self) -> impl std::future::Future<Output = BarrierWaitResult> + Send + 'a;
}

/// A barrier usable from either calling convention — see
/// [`crate::traits::DualMutex`] for the same pattern applied to mutexes.
/// The interface owns the name here too: the concrete generic-over-N type
/// (`ult::sync::DualBarrier`) is re-exported under an alias
/// (`UltDualBarrier`) at the crate root to make room.
pub trait DualBarrier: Sized + Send + Sync + StackfulBarrier + StacklessBarrier {}

impl<M: StackfulBarrier + StacklessBarrier> DualBarrier for M {}
