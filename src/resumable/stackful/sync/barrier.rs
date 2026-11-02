use std::collections::VecDeque;

use crate::spin::SpinLock;
use crate::traits::{BarrierWaitResult, Resumable, StackfulBarrier, StackfulResumable};
use crate::traits::{StackfulParking, SuspendableSystem};

// ---------------------------------------------------------------------------
// BarrierCore
// ---------------------------------------------------------------------------

pub struct BarrierState<S: StackfulParking> {
    pub(super) count: usize,
    pub(super) waiters: VecDeque<S::SuspendedThread>,
}

/// Raw barrier storage: this crate's own `SpinLock<BarrierState>` wait-queue
/// representation. Implementing this opts a type into [`StackfulBarrier`]
/// for free via the blanket impl below — the same two-tier relationship as
/// [`TaskDescCore`](crate::resumable::common::desc::TaskDescCore)/[`TaskDesc`](crate::resumable::common::desc::TaskDesc).
pub trait BarrierCore: Send + Sync + Sized {
    type System: StackfulParking;

    fn new_core(count: usize) -> Self;
    fn n(&self) -> usize;
    fn state(&self) -> &SpinLock<BarrierState<Self::System>>;
}

/// Blanket [`StackfulBarrier`] for any [`BarrierCore`]: the wait algorithm
/// lives here, not as a trait default on `BarrierCore`, so that trait stays
/// a pure accessor contract.
impl<M: BarrierCore> StackfulBarrier for M {
    fn new(count: usize) -> Self {
        M::new_core(count)
    }

    fn wait(&self) -> BarrierWaitResult {
        let mut s = self.state().lock();
        s.count += 1;
        if s.count == self.n() {
            s.count = 0;
            let sths: Vec<_> = s.waiters.drain(..).collect();
            drop(s);
            for sth in sths { sth.notify(); }
            return BarrierWaitResult { is_leader: true };
        }
        s.waiters.push_back(Default::default());
        let sth: *const <<Self as BarrierCore>::System as SuspendableSystem>::SuspendedThread = s.waiters.back().unwrap();
        unsafe { &*sth }.wait_with(move || drop(s));
        BarrierWaitResult { is_leader: false }
    }
}

// ---------------------------------------------------------------------------
// Barrier
// ---------------------------------------------------------------------------

pub struct Barrier<S: StackfulParking> {
    n: usize,
    state: SpinLock<BarrierState<S>>,
}

unsafe impl<S: StackfulParking> Send for Barrier<S> {}
unsafe impl<S: StackfulParking> Sync for Barrier<S> {}

impl<S: StackfulParking> Barrier<S> {
    pub fn new(n: usize) -> Self {
        assert!(n > 0);
        Barrier { n, state: SpinLock::new(BarrierState { count: 0, waiters: VecDeque::new() }) }
    }
}

impl<S: StackfulParking> BarrierCore for Barrier<S> {
    type System = S;
    fn new_core(count: usize) -> Self { Barrier::new(count) }
    fn n(&self) -> usize { self.n }
    fn state(&self) -> &SpinLock<BarrierState<S>> { &self.state }
}
