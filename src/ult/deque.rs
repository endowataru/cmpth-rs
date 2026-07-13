//! Worker deque policy.
//!
//! "Top" is the local (LIFO) end used by the owning worker; thieves steal
//! from the bottom.  Swap the implementation via [`crate::UltSchedulerSystem::Deque`].

use std::cell::UnsafeCell;
use std::collections::VecDeque;

use crate::spin::SpinLock;
use crate::ult::desc::SuspendedUlt;

/// Contract: `push_top`, `push_bottom` and `try_pop_top` are only called by
/// the worker that owns the deque; `try_steal_bottom` may be called from any
/// thread.
pub trait WorkerDeque: Default + Send + Sync + 'static {
    fn push_top(&self, c: SuspendedUlt);
    fn push_bottom(&self, c: SuspendedUlt);
    fn try_pop_top(&self) -> Option<SuspendedUlt>;
    /// Called from thief workers.
    fn try_steal_bottom(&self) -> Option<SuspendedUlt>;
}

/// Default deque: lock-free Chase-Lev (crossbeam).  The owner pushes/pops the
/// LIFO end without contention; thieves steal the opposite end, so idle
/// workers polling for work do not slow the owner down.
///
/// Limitation: Chase-Lev has no owner-side "push bottom", so `push_bottom`
/// (used by `yield`) degrades to `push_top`; yielding still gives thieves a
/// steal window, but local FIFO fairness is approximated only.  Use
/// [`SpinDeque`] if exact yield ordering matters more than throughput.
pub struct CrossbeamDeque {
    /// Owner-only end (see the trait contract above).
    local: UnsafeCell<crossbeam_deque::Worker<SuspendedUlt>>,
    stealer: crossbeam_deque::Stealer<SuspendedUlt>,
}

unsafe impl Send for CrossbeamDeque {}
// Safety: `local` is only touched by the owning worker (trait contract);
// `stealer` is thread-safe by construction.
unsafe impl Sync for CrossbeamDeque {}

impl Default for CrossbeamDeque {
    fn default() -> Self {
        let local = crossbeam_deque::Worker::new_lifo();
        let stealer = local.stealer();
        CrossbeamDeque { local: UnsafeCell::new(local), stealer }
    }
}

impl WorkerDeque for CrossbeamDeque {
    fn push_top(&self, c: SuspendedUlt) {
        unsafe { &*self.local.get() }.push(c);
    }

    fn push_bottom(&self, c: SuspendedUlt) {
        unsafe { &*self.local.get() }.push(c);
    }

    fn try_pop_top(&self) -> Option<SuspendedUlt> {
        unsafe { &*self.local.get() }.pop()
    }

    fn try_steal_bottom(&self) -> Option<SuspendedUlt> {
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
pub struct SpinDeque {
    q: SpinLock<VecDeque<SuspendedUlt>>,
}

impl Default for SpinDeque {
    fn default() -> Self {
        SpinDeque { q: SpinLock::new(VecDeque::new()) }
    }
}

impl WorkerDeque for SpinDeque {
    fn push_top(&self, c: SuspendedUlt) {
        self.q.lock().push_front(c);
    }

    fn push_bottom(&self, c: SuspendedUlt) {
        self.q.lock().push_back(c);
    }

    fn try_pop_top(&self) -> Option<SuspendedUlt> {
        self.q.lock().pop_front()
    }

    fn try_steal_bottom(&self) -> Option<SuspendedUlt> {
        self.q.lock().pop_back()
    }
}
