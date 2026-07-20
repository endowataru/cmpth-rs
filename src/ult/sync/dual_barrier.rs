//! `DualBarrier<S, N>` — a prototype barrier generic over the wait-slot
//! flavor `N`, same idea as `uni_mutex.rs`. See
//! `docs/sync-async-unification.md`.

use std::collections::VecDeque;
use std::marker::PhantomData;

use crate::spin::SpinLock;
use crate::traits::{BarrierWaitResult, StackfulBarrier, StackfulResumable, StacklessBarrier, StacklessResumable};

struct DualBarrierState<N> {
    count: usize,
    waiters: VecDeque<N>,
}

pub struct DualBarrier<S, N> {
    n: usize,
    state: SpinLock<DualBarrierState<N>>,
    _marker: PhantomData<S>,
}

unsafe impl<S, N: Send> Send for DualBarrier<S, N> {}
unsafe impl<S, N: Send> Sync for DualBarrier<S, N> {}

impl<S, N> DualBarrier<S, N> {
    pub fn new(n: usize) -> Self {
        assert!(n > 0);
        DualBarrier {
            n,
            state: SpinLock::new(DualBarrierState { count: 0, waiters: VecDeque::new() }),
            _marker: PhantomData,
        }
    }
}

impl<S, N> StackfulBarrier for DualBarrier<S, N>
where
    S: Send + Sync + 'static,
    N: StackfulResumable<S> + Send + Sync + 'static,
{
    fn new(count: usize) -> Self {
        DualBarrier::new(count)
    }

    fn wait(&self) -> BarrierWaitResult {
        let mut s = self.state.lock();
        s.count += 1;
        if s.count == self.n {
            s.count = 0;
            let all: Vec<N> = s.waiters.drain(..).collect();
            drop(s);
            for w in all {
                w.notify();
            }
            return BarrierWaitResult { is_leader: true };
        }
        s.waiters.push_back(N::default());
        let w: *const N = s.waiters.back().unwrap();
        unsafe { &*w }.wait_with(move || drop(s));
        BarrierWaitResult { is_leader: false }
    }
}

impl<S, N> StacklessBarrier for DualBarrier<S, N>
where
    S: Send + Sync + 'static,
    N: StacklessResumable<S> + Send + Sync + 'static,
{
    fn new(count: usize) -> Self {
        DualBarrier::new(count)
    }

    async fn wait(&self) -> BarrierWaitResult {
        let mut s = self.state.lock();
        s.count += 1;
        if s.count == self.n {
            s.count = 0;
            let all: Vec<N> = s.waiters.drain(..).collect();
            drop(s);
            for w in all {
                w.notify();
            }
            return BarrierWaitResult { is_leader: true };
        }
        s.waiters.push_back(N::default());
        let w: *const N = s.waiters.back().unwrap();
        StacklessResumable::wait_with(unsafe { &*w }, move || drop(s)).await;
        BarrierWaitResult { is_leader: false }
    }
}
