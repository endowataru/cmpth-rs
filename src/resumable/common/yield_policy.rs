//! Yield policy: where a task that yields waits for its next turn.
//!
//! Work stealing moves work only to idle workers, so once every worker has
//! something to run, the tasks queued on one worker stay there: 24 tasks
//! that keep yielding on 6 workers can end up as five that each own a
//! worker and nineteen that share one.  [`LocalYield`] (the default) keeps
//! a yielding task on its worker's own deferred queue; [`GlobalYield`]
//! puts one yield in `EVERY` (chosen at random) on one FIFO queue the whole
//! pool takes from once its own queue is empty (with `EVERY = 1`, every
//! yield, as Go's `Gosched` does).  A worker then runs its current task for
//! about `EVERY` yields and swaps it for the oldest task of the shared
//! queue, so yielding tasks take turns on every worker.
//!
//! A policy may also set a turn budget ([`YieldPolicy::BUDGET`]): after
//! that many operations that did not suspend (see
//! [`StackfulResumable::consume_budget`](crate::traits::StackfulResumable::consume_budget)),
//! the running task yields, as Tokio's cooperative budget does.  Without
//! preemption, a task that keeps finding its locks free would otherwise
//! keep the tasks queued behind it waiting.
//!
//! Swap the implementation via [`crate::WorkerSystem::Yield`].

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::spin::SpinLock;

/// Where yielding tasks wait.  One value is shared by all workers of a
/// pool (it lives in `PoolCore`).
///
/// # Contract
///
/// * `requeue` is called by the worker a task yielded on, with a
///   pseudo-random value that worker draws per yield (`tick`).  Returning
///   `Some(t)` hands the task back: the worker defers it on its own queue.
///   Returning `None` means the policy kept it; the caller then announces
///   new work to the idle policy.
/// * `take` is called by any worker looking for work, after its own
///   queue.  It must be cheap when the policy keeps nothing.
pub trait YieldPolicy<T: Send>: Default + Send + Sync + 'static {
    /// Operations without a suspension a worker runs before its task must
    /// yield (0: no budget).
    const BUDGET: u32 = 0;

    /// `requeue` always hands the task back: the worker defers it without
    /// drawing `tick` or calling the policy.
    const LOCAL: bool = false;

    fn requeue(&self, tick: u32, t: T) -> Option<T>;

    fn take(&self) -> Option<T>;
}

/// A yielding task stays on its worker (its deferred queue).
#[derive(Default)]
pub struct LocalYield<const BUDGET: u32 = 0>;

impl<T: Send, const BUDGET: u32> YieldPolicy<T> for LocalYield<BUDGET> {
    const BUDGET: u32 = BUDGET;
    const LOCAL: bool = true;

    #[inline(always)]
    fn requeue(&self, _tick: u32, t: T) -> Option<T> {
        Some(t)
    }

    #[inline(always)]
    fn take(&self) -> Option<T> {
        None
    }
}

/// One yield in `EVERY` goes to the back of one queue shared by the pool;
/// the others stay on the worker.  `EVERY = 1` sends every
/// yield there, which costs a lock on one shared line per yield.
pub struct GlobalYield<T: Send, const EVERY: u32 = 1, const BUDGET: u32 = 0> {
    non_empty: AtomicBool,
    q: SpinLock<VecDeque<T>>,
}

impl<T: Send, const EVERY: u32, const BUDGET: u32> Default for GlobalYield<T, EVERY, BUDGET> {
    fn default() -> Self {
        GlobalYield { non_empty: AtomicBool::new(false), q: SpinLock::new(VecDeque::new()) }
    }
}

impl<T: Send + 'static, const EVERY: u32, const BUDGET: u32> YieldPolicy<T> for GlobalYield<T, EVERY, BUDGET> {
    const BUDGET: u32 = BUDGET;

    #[inline]
    fn requeue(&self, tick: u32, t: T) -> Option<T> {
        if EVERY > 1 && tick % EVERY != 0 {
            return Some(t);
        }
        self.q.lock().push_back(t);
        self.non_empty.store(true, Ordering::Release);
        None
    }

    #[inline]
    fn take(&self) -> Option<T> {
        if !self.non_empty.load(Ordering::Acquire) {
            return None;
        }
        let mut q = self.q.lock();
        let t = q.pop_front();
        if q.is_empty() {
            self.non_empty.store(false, Ordering::Relaxed);
        }
        t
    }
}
