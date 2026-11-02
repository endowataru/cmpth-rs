//! Yield policies ([`YieldPolicy`]).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::spin::SpinLock;

pub use crate::traits::component::yield_policy::YieldPolicy;

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
