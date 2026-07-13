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
