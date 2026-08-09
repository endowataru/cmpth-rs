//! Worker run-queue policy.
//!
//! **Not a deque**: no lock-free implementation can offer an owner-side push
//! at the *steal* end (Chase-Lev's steal index is advanced only by thieves,
//! via CAS), so a "deque" contract with a real `push_bottom` is a promise no
//! implementation here can actually keep — see [`WorkerRunQueue`]'s doc
//! comment. The interface below is defined by *scheduling intent* (`push` /
//! `defer`) rather than by position (`top` / `bottom`).
//!
//! Swap the implementation via [`crate::WorkerSystem::RunQueue`].

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::spin::SpinLock;

// ---------------------------------------------------------------------------
// Steal
// ---------------------------------------------------------------------------

/// Same shape as [`crossbeam_deque::Steal`], defined here so this axis's
/// contract does not expose crossbeam. `Retry` means "work exists but could
/// not be taken right now" (lost a race with another thief, or the owner) —
/// it must never be collapsed into `Empty`: a caller that treats a `Retry`
/// scan as "genuinely nothing to do" may back off (e.g. yield to the OS)
/// when work was actually available.
#[derive(Debug)]
pub enum Steal<T> {
    Empty,
    Success(T),
    Retry,
}

// ---------------------------------------------------------------------------
// WorkerRunQueue / RunQueueStealer
// ---------------------------------------------------------------------------

/// A queue of runnable items that other workers may steal from. Only the
/// owning worker holds this handle — `push`/`defer`/`try_pop` are only
/// called by the worker that owns the queue; stealing goes exclusively
/// through [`stealer`](Self::stealer)'s [`RunQueueStealer::try_steal`],
/// callable from any thread.
///
/// **Not a deque**: no single total order over items is promised. Each
/// operation is defined by *when the caller wants the item to run*, not by
/// where it is placed. This is a deliberate retreat from a `WorkerDeque`
/// contract this crate used to expose: that contract promised operations at
/// both ends of a deque, but no lock-free implementation can provide an
/// owner-side push at the *steal* end (Chase-Lev's steal index is advanced
/// by thieves via CAS only), so the old `push_bottom` silently degraded to
/// `push_top` — fair yielding was only approximated, and only by accident.
/// [`HybridRunQueue`] keeps a second, explicitly lock-based queue for
/// `defer`, so the ordering this trait promises is real, not aspirational.
///
/// Generic over the element type `T` moved through the queue — neither
/// provided implementation ever inspects `T`, only stores/returns it. Every
/// concrete system today sets `T = SuspendedTaskToken<Self::Desc>` via
/// [`crate::WorkerSystem::SuspendedToken`].
pub trait WorkerRunQueue<T: Send>: Send + Sync + 'static {
    type Stealer: RunQueueStealer<T>;

    /// Run next on this worker (a forked child, a woken task). Keeps
    /// locality; least likely to be stolen.
    fn push(&self, v: T);

    /// Run after work already queued here (`yield`). Most likely to be
    /// stolen.
    fn defer(&self, v: T);

    /// Take what this worker should run next.
    fn try_pop(&self) -> Option<T>;

    /// A cloneable handle other workers steal through.
    fn stealer(&self) -> Self::Stealer;
}

pub trait RunQueueStealer<T: Send>: Clone + Send + Sync + 'static {
    fn try_steal(&self) -> Steal<T>;
}

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
