//! Worker deque policy.
//!
//! "Top" is the local (LIFO) end used by the owning worker; thieves steal
//! from the bottom.  Swap the implementation via [`crate::SchedulerSystem::Deque`].

use std::cell::UnsafeCell;
use std::collections::VecDeque;

use crate::spin::SpinLock;
use crate::ult::desc::{SuspendedUlt, TaskDesc};

/// Contract: `push_top`, `push_bottom` and `try_pop_top` are only called by
/// the worker that owns the deque; `try_steal_bottom` may be called from any
/// thread.
///
/// Generic over the descriptor type `D` (see [`SuspendedUlt`]); every
/// concrete system today sets `D = BasicTaskDesc` via
/// [`crate::SchedulerSystem::Desc`].
pub trait WorkerDeque<D: TaskDesc>: Default + Send + Sync + 'static {
    fn push_top(&self, c: SuspendedUlt<D>);
    fn push_bottom(&self, c: SuspendedUlt<D>);
    fn try_pop_top(&self) -> Option<SuspendedUlt<D>>;
    /// Called from thief workers.
    fn try_steal_bottom(&self) -> Option<SuspendedUlt<D>>;
}

/// Default deque: lock-free Chase-Lev (crossbeam).  The owner pushes/pops the
/// LIFO end without contention; thieves steal the opposite end, so idle
/// workers polling for work do not slow the owner down.
///
/// Limitation: Chase-Lev has no owner-side "push bottom", so `push_bottom`
/// (used by `yield`) degrades to `push_top`; yielding still gives thieves a
/// steal window, but local FIFO fairness is approximated only.  Use
/// [`SpinDeque`] if exact yield ordering matters more than throughput.
pub struct CrossbeamDeque<D: TaskDesc> {
    /// Owner-only end (see the trait contract above).
    local: UnsafeCell<crossbeam_deque::Worker<SuspendedUlt<D>>>,
    stealer: crossbeam_deque::Stealer<SuspendedUlt<D>>,
}

unsafe impl<D: TaskDesc> Send for CrossbeamDeque<D> {}
// Safety: `local` is only touched by the owning worker (trait contract);
// `stealer` is thread-safe by construction.
unsafe impl<D: TaskDesc> Sync for CrossbeamDeque<D> {}

impl<D: TaskDesc> Default for CrossbeamDeque<D> {
    fn default() -> Self {
        let local = crossbeam_deque::Worker::new_lifo();
        let stealer = local.stealer();
        CrossbeamDeque { local: UnsafeCell::new(local), stealer }
    }
}

impl<D: TaskDesc> WorkerDeque<D> for CrossbeamDeque<D> {
    fn push_top(&self, c: SuspendedUlt<D>) {
        unsafe { &*self.local.get() }.push(c);
    }

    fn push_bottom(&self, c: SuspendedUlt<D>) {
        unsafe { &*self.local.get() }.push(c);
    }

    fn try_pop_top(&self) -> Option<SuspendedUlt<D>> {
        unsafe { &*self.local.get() }.pop()
    }

    fn try_steal_bottom(&self) -> Option<SuspendedUlt<D>> {
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
pub struct SpinDeque<D: TaskDesc> {
    q: SpinLock<VecDeque<SuspendedUlt<D>>>,
}

impl<D: TaskDesc> Default for SpinDeque<D> {
    fn default() -> Self {
        SpinDeque { q: SpinLock::new(VecDeque::new()) }
    }
}

impl<D: TaskDesc> WorkerDeque<D> for SpinDeque<D> {
    fn push_top(&self, c: SuspendedUlt<D>) {
        self.q.lock().push_front(c);
    }

    fn push_bottom(&self, c: SuspendedUlt<D>) {
        self.q.lock().push_back(c);
    }

    fn try_pop_top(&self) -> Option<SuspendedUlt<D>> {
        self.q.lock().pop_front()
    }

    fn try_steal_bottom(&self) -> Option<SuspendedUlt<D>> {
        self.q.lock().pop_back()
    }
}
