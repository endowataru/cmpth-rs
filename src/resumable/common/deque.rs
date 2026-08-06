//! Worker deque policy.
//!
//! "Top" is the local (LIFO) end used by the owning worker; thieves steal
//! from the bottom.  Swap the implementation via [`crate::SchedulerSystem::Deque`].

use std::cell::UnsafeCell;
use std::collections::VecDeque;

use crate::spin::SpinLock;

/// Contract: `push_top`, `push_bottom` and `try_pop_top` are only called by
/// the worker that owns the deque; `try_steal_bottom` may be called from any
/// thread.
///
/// Generic over the element type `T` moved through the deque — neither
/// provided implementation ever inspects `T`, only stores/returns it. Every
/// concrete system today sets `T = SuspendedTaskToken<Self::Desc>` via
/// [`crate::SchedulerSystem::Item`].
pub trait WorkerDeque<T: Send>: Default + Send + Sync + 'static {
    fn push_top(&self, v: T);
    fn push_bottom(&self, v: T);
    fn try_pop_top(&self) -> Option<T>;
    /// Called from thief workers.
    fn try_steal_bottom(&self) -> Option<T>;
}

/// Default deque: lock-free Chase-Lev (crossbeam).  The owner pushes/pops the
/// LIFO end without contention; thieves steal the opposite end, so idle
/// workers polling for work do not slow the owner down.
///
/// Limitation: Chase-Lev has no owner-side "push bottom", so `push_bottom`
/// (used by `yield`) degrades to `push_top`; yielding still gives thieves a
/// steal window, but local FIFO fairness is approximated only.  Use
/// [`SpinDeque`] if exact yield ordering matters more than throughput.
pub struct CrossbeamDeque<T: Send> {
    /// Owner-only end (see the trait contract above).
    local: UnsafeCell<crossbeam_deque::Worker<T>>,
    stealer: crossbeam_deque::Stealer<T>,
}

unsafe impl<T: Send> Send for CrossbeamDeque<T> {}
// Safety: `local` is only touched by the owning worker (trait contract);
// `stealer` is thread-safe by construction.
unsafe impl<T: Send> Sync for CrossbeamDeque<T> {}

impl<T: Send> Default for CrossbeamDeque<T> {
    fn default() -> Self {
        let local = crossbeam_deque::Worker::new_lifo();
        let stealer = local.stealer();
        CrossbeamDeque { local: UnsafeCell::new(local), stealer }
    }
}

impl<T: Send + 'static> WorkerDeque<T> for CrossbeamDeque<T> {
    fn push_top(&self, c: T) {
        unsafe { &*self.local.get() }.push(c);
    }

    fn push_bottom(&self, c: T) {
        unsafe { &*self.local.get() }.push(c);
    }

    fn try_pop_top(&self) -> Option<T> {
        unsafe { &*self.local.get() }.pop()
    }

    fn try_steal_bottom(&self) -> Option<T> {
        loop {
            match self.stealer.steal() {
                crossbeam_deque::Steal::Success(c) => return Some(c),
                crossbeam_deque::Steal::Empty => return None,
                crossbeam_deque::Steal::Retry => std::hint::spin_loop(),
            }
        }
    }
}

/// Default deque: a spinlock-protected `VecDeque`.  Simple and correct;
/// replace with a lock-free Chase-Lev deque via the policy when profiling
/// says so.
pub struct SpinDeque<T: Send> {
    q: SpinLock<VecDeque<T>>,
}

impl<T: Send> Default for SpinDeque<T> {
    fn default() -> Self {
        SpinDeque { q: SpinLock::new(VecDeque::new()) }
    }
}

impl<T: Send + 'static> WorkerDeque<T> for SpinDeque<T> {
    fn push_top(&self, c: T) {
        self.q.lock().push_front(c);
    }

    fn push_bottom(&self, c: T) {
        self.q.lock().push_back(c);
    }

    fn try_pop_top(&self) -> Option<T> {
        self.q.lock().pop_front()
    }

    fn try_steal_bottom(&self) -> Option<T> {
        self.q.lock().pop_back()
    }
}
