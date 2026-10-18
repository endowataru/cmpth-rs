//! Worker run-queue implementations of the
//! [`WorkerRunQueue`]
//! axis (the trait itself lives in `traits/`).
//!
//! Swap the implementation via [`crate::WorkerSystem::RunQueue`].

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::spin::SpinLock;

pub use crate::traits::component::run_queue::{RunQueueStealer, Steal, WorkerRunQueue};

// ---------------------------------------------------------------------------
// HybridRunQueue — default: lock-free Chase-Lev for `push`/`try_pop`, a
// spinlocked VecDeque (drainable by thieves too) for `defer`.
// ---------------------------------------------------------------------------

// Shared state for the `defer`-side queue: reachable from both the owner
// (`try_pop`) and thieves (`try_steal`) — if only the owner could drain it,
// deferred work would become unreachable whenever the owner blocks for a
// long time. Guarded by the same "atomic flag in front of a lock" pattern
// `StealPathQueue` (`external_queue.rs`) uses, so the common empty case
// never touches the lock.
struct Deferred<T: Send> {
    non_empty: AtomicBool,
    q: SpinLock<VecDeque<T>>,
}

impl<T: Send> Deferred<T> {
    fn push(&self, v: T) {
        self.q.lock().push_back(v);
        self.non_empty.store(true, Ordering::Release);
    }

    fn try_take(&self) -> Option<T> {
        if !self.non_empty.load(Ordering::Acquire) {
            return None;
        }
        let mut q = self.q.lock();
        let item = q.pop_front();
        if q.is_empty() {
            self.non_empty.store(false, Ordering::Relaxed);
        }
        item
    }
}

/// Default run queue: `push`/`try_pop` go through a lock-free Chase-Lev
/// deque (crossbeam) so the owner's hot path never contends with thieves;
/// `defer` goes through a spinlock-protected [`VecDeque`] instead, drained
/// by owner and thieves alike.
///
/// Ordering (matters): [`try_pop`](WorkerRunQueue::try_pop) checks `main`
/// first, then `deferred`; [`RunQueueStealer::try_steal`] checks `deferred`
/// first, then `main` — thieves prefer the queue the owner least wants (it
/// already gave that item up once, via `defer`), leaving `main` for the
/// owner's own locality.
///
/// `deferred`'s state (a spinlock-protected queue behind a non-empty flag,
/// same shape as
/// [`StealPathQueue`](crate::resumable::common::external_queue::StealPathQueue))
/// lives in a private `Deferred` type, `Arc`-shared with [`HybridStealer`]
/// so both the owner and thieves can reach it.
pub struct HybridRunQueue<T: Send> {
    /// Owner-only end (see the trait contract on [`WorkerRunQueue`]).
    main: UnsafeCell<crossbeam_deque::Worker<T>>,
    main_stealer: crossbeam_deque::Stealer<T>,
    deferred: Arc<Deferred<T>>,
}

unsafe impl<T: Send> Send for HybridRunQueue<T> {}
// Safety: `main` is only touched by the owning worker (trait contract);
// `main_stealer` is thread-safe by construction; `deferred` is internally
// synchronized.
unsafe impl<T: Send> Sync for HybridRunQueue<T> {}

impl<T: Send> Default for HybridRunQueue<T> {
    fn default() -> Self {
        let main = crossbeam_deque::Worker::new_lifo();
        let main_stealer = main.stealer();
        HybridRunQueue {
            main: UnsafeCell::new(main),
            main_stealer,
            deferred: Arc::new(Deferred { non_empty: AtomicBool::new(false), q: SpinLock::new(VecDeque::new()) }),
        }
    }
}

impl<T: Send + 'static> WorkerRunQueue<T> for HybridRunQueue<T> {
    type Stealer = HybridStealer<T>;

    fn push(&self, v: T) {
        unsafe { &*self.main.get() }.push(v);
    }

    fn defer(&self, v: T) {
        self.deferred.push(v);
    }

    fn try_pop(&self) -> Option<T> {
        if let Some(v) = unsafe { &*self.main.get() }.pop() {
            return Some(v);
        }
        self.deferred.try_take()
    }

    fn stealer(&self) -> Self::Stealer {
        HybridStealer { main: self.main_stealer.clone(), deferred: Arc::clone(&self.deferred) }
    }
}

/// Cloneable stealer handle for [`HybridRunQueue`]. Holds an `Arc`-shared
/// reference to the `deferred` queue (rather than borrowing the owning
/// `HybridRunQueue`) precisely so it can outlive any particular call and be
/// stored in a scheduler-wide stealer table, reached without going back
/// through the owning worker's own (otherwise `Cell`-guarded) state.
pub struct HybridStealer<T: Send> {
    main: crossbeam_deque::Stealer<T>,
    deferred: Arc<Deferred<T>>,
}

impl<T: Send> Clone for HybridStealer<T> {
    fn clone(&self) -> Self {
        HybridStealer { main: self.main.clone(), deferred: Arc::clone(&self.deferred) }
    }
}

impl<T: Send + 'static> RunQueueStealer<T> for HybridStealer<T> {
    fn try_steal(&self) -> Steal<T> {
        if let Some(v) = self.deferred.try_take() {
            return Steal::Success(v);
        }
        match self.main.steal() {
            crossbeam_deque::Steal::Success(v) => Steal::Success(v),
            crossbeam_deque::Steal::Empty => Steal::Empty,
            crossbeam_deque::Steal::Retry => Steal::Retry,
        }
    }
}

// ---------------------------------------------------------------------------
// SpinRunQueue — debug/deterministic implementation: a single VecDeque
// behind one spinlock, shared with the stealer via Arc.
// ---------------------------------------------------------------------------

/// Debug/deterministic run queue: a single spinlock-protected [`VecDeque`].
/// Being lock-based (not lock-free), it can satisfy the [`WorkerRunQueue`]
/// contract directly with one queue — `push` at the front, `defer` at the
/// back, `try_pop`/`try_steal` at opposite ends — with no need for
/// [`HybridRunQueue`]'s two-queue split: a single lock already gives exact
/// ordering at both ends, the property `HybridRunQueue` needs a second
/// queue to approximate without paying for a lock on the fast path.
///
/// Replace with a lock-free implementation via the policy when profiling
/// says so; use this one whenever exact ordering matters more than
/// throughput (e.g. deterministic tests).
pub struct SpinRunQueue<T: Send> {
    q: Arc<SpinLock<VecDeque<T>>>,
}

impl<T: Send> Default for SpinRunQueue<T> {
    fn default() -> Self {
        SpinRunQueue { q: Arc::new(SpinLock::new(VecDeque::new())) }
    }
}

impl<T: Send + 'static> WorkerRunQueue<T> for SpinRunQueue<T> {
    type Stealer = SpinStealer<T>;

    fn push(&self, v: T) {
        self.q.lock().push_front(v);
    }

    fn defer(&self, v: T) {
        self.q.lock().push_back(v);
    }

    fn try_pop(&self) -> Option<T> {
        self.q.lock().pop_front()
    }

    fn stealer(&self) -> Self::Stealer {
        SpinStealer { q: Arc::clone(&self.q) }
    }
}

/// Cloneable stealer handle for [`SpinRunQueue`] — see [`HybridStealer`] for
/// why this is `Arc`-backed rather than a borrow of the owning queue.
pub struct SpinStealer<T: Send> {
    q: Arc<SpinLock<VecDeque<T>>>,
}

impl<T: Send> Clone for SpinStealer<T> {
    fn clone(&self) -> Self {
        SpinStealer { q: Arc::clone(&self.q) }
    }
}

impl<T: Send + 'static> RunQueueStealer<T> for SpinStealer<T> {
    fn try_steal(&self) -> Steal<T> {
        match self.q.lock().pop_back() {
            Some(v) => Steal::Success(v),
            None => Steal::Empty,
        }
    }
}
