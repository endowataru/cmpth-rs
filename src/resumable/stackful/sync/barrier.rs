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

pub trait BarrierCore: Send + Sync + Sized where <<Self as BarrierCore>::StackfulSchedulerSystem as SchedulerSystem>::Desc: StackfulTaskDesc {
    type StackfulSchedulerSystem: StackfulSchedulerSystem;

    fn n(&self) -> usize;
    fn state(&self) -> &SpinLock<BarrierState<Self::StackfulSchedulerSystem>>;

    fn wait_impl(&self) -> BarrierWaitResult where <<Self as BarrierCore>::StackfulSchedulerSystem as SchedulerSystem>::Desc: StackfulTaskDesc {
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
        let sth: *const <Self::StackfulSchedulerSystem as StackfulSchedulerSystem>::SuspendedThread = s.waiters.back().unwrap();
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
    fn n(&self) -> usize where <S as SchedulerSystem>::Desc: StackfulTaskDesc { self.n }
    fn state(&self) -> &SpinLock<BarrierState<S>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc { &self.state }
}

impl<S: StackfulSchedulerSystem> StackfulBarrier for Barrier<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    fn new(count: usize) -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc { Barrier::new(count) }
    fn wait(&self) -> BarrierWaitResult where <S as SchedulerSystem>::Desc: StackfulTaskDesc { self.wait_impl() }
}
