use std::collections::VecDeque;

use crate::spin::SpinLock;
use crate::traits::{Barrier as BarrierTrait, BarrierWaitResult, SuspendedThread};
use crate::ult::system::{UltSchedulerSystem, UltSystem};

// ---------------------------------------------------------------------------
// BarrierCore
// ---------------------------------------------------------------------------

pub struct BarrierState<S: UltSystem> {
    pub(super) count: usize,
    pub(super) waiters: VecDeque<S::SuspendedThread>,
}

pub trait BarrierCore: Send + Sync + Sized {
    type UltSystem: UltSystem;

    fn n(&self) -> usize;
    fn state(&self) -> &SpinLock<BarrierState<Self::UltSystem>>;

    fn wait_impl(&self) -> BarrierWaitResult {
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
        let sth: *const <Self::UltSystem as UltSchedulerSystem>::SuspendedThread = s.waiters.back().unwrap();
        unsafe { &*sth }.wait_with(move || drop(s));
        BarrierWaitResult { is_leader: false }
    }
}

// ---------------------------------------------------------------------------
// Barrier
// ---------------------------------------------------------------------------

pub struct Barrier<S: UltSystem> {
    n: usize,
    state: SpinLock<BarrierState<S>>,
}

unsafe impl<S: UltSystem> Send for Barrier<S> {}
unsafe impl<S: UltSystem> Sync for Barrier<S> {}

impl<S: UltSystem> Barrier<S> {
    pub fn new(n: usize) -> Self {
        assert!(n > 0);
        Barrier { n, state: SpinLock::new(BarrierState { count: 0, waiters: VecDeque::new() }) }
    }
}

impl<S: UltSystem> BarrierCore for Barrier<S> {
    type UltSystem = S;
    fn n(&self) -> usize { self.n }
    fn state(&self) -> &SpinLock<BarrierState<S>> { &self.state }
}

impl<S: UltSystem> BarrierTrait for Barrier<S> {
    fn new(count: usize) -> Self { Barrier::new(count) }
    fn wait(&self) -> BarrierWaitResult { self.wait_impl() }
}
