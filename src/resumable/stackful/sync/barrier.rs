use std::collections::VecDeque;

use crate::spin::SpinLock;
use crate::traits::{BarrierWaitResult, Resumable, StackfulBarrier, StackfulResumable};
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackful::system::StackfulSchedulerSystem;

// ---------------------------------------------------------------------------
// BarrierCore
// ---------------------------------------------------------------------------

pub struct BarrierState<S: StackfulSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    pub(super) count: usize,
    pub(super) waiters: VecDeque<S::SuspendedThread>,
}

/// Raw barrier storage: this crate's own `SpinLock<BarrierState>` wait-queue
/// representation. Implementing this opts a type into [`StackfulBarrier`]
/// for free via the blanket impl below — the same two-tier relationship as
/// [`TaskDescCore`](crate::resumable::common::desc::TaskDescCore)/[`TaskDesc`](crate::resumable::common::desc::TaskDesc).
pub trait BarrierCore: Send + Sync + Sized where <<Self as BarrierCore>::StackfulSchedulerSystem as SchedulerSystem>::Desc: StackfulTaskDesc {
    type StackfulSchedulerSystem: StackfulSchedulerSystem;

    fn new_core(count: usize) -> Self where <<Self as BarrierCore>::StackfulSchedulerSystem as SchedulerSystem>::Desc: StackfulTaskDesc;
    fn n(&self) -> usize;
    fn state(&self) -> &SpinLock<BarrierState<Self::StackfulSchedulerSystem>>;
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
        let sth: *const <<Self as BarrierCore>::StackfulSchedulerSystem as StackfulSchedulerSystem>::SuspendedThread = s.waiters.back().unwrap();
        unsafe { &*sth }.wait_with(move || drop(s));
        BarrierWaitResult { is_leader: false }
    }
}

// ---------------------------------------------------------------------------
// Barrier
// ---------------------------------------------------------------------------

pub struct Barrier<S: StackfulSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    n: usize,
    state: SpinLock<BarrierState<S>>,
}

unsafe impl<S: StackfulSchedulerSystem> Send for Barrier<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}
unsafe impl<S: StackfulSchedulerSystem> Sync for Barrier<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}

impl<S: StackfulSchedulerSystem> Barrier<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    pub fn new(n: usize) -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        assert!(n > 0);
        Barrier { n, state: SpinLock::new(BarrierState { count: 0, waiters: VecDeque::new() }) }
    }
}

impl<S: StackfulSchedulerSystem> BarrierCore for Barrier<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    type StackfulSchedulerSystem = S;
    fn new_core(count: usize) -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc { Barrier::new(count) }
    fn n(&self) -> usize where <S as SchedulerSystem>::Desc: StackfulTaskDesc { self.n }
    fn state(&self) -> &SpinLock<BarrierState<S>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc { &self.state }
}
