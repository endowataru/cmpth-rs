//! External-queue trait and implementations for waking ULTs from outside the
//! scheduler (e.g. RDMA completion threads calling `Waker::wake()`).

use std::any::Any;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, Weak};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::traits::thread_system::ThreadSystem;
use crate::resumable::common::desc::SuspendedUlt;
use crate::resumable::common::scheduler::Scheduler;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::stackful::thread::{ErasedBody, fork_parent_first};
use crate::resumable::common::worker::{LocalQueue, UltWorker, Worker};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// How continuations pushed by external OS threads reach the ULT scheduler.
/// Base-level (`S: SchedulerSystem`): a stackless-only system still needs a
/// way for external OS threads to hand it work.
///
/// Two provided implementations:
///
/// * [`StealPathQueue`] (default) — workers drain the queue when their local
///   deque and steal attempts both fail; one atomic check added to the
///   steal-fail path.
/// * [`PollerUltQueue`] — a dedicated poller ULT drains the queue; zero
///   overhead on the steal path, but consumes one ULT stack. Inherently
///   stackful (it *is* a ULT), so only implemented for `S: StackfulSchedulerSystem`.
pub trait ExternalQueue<S: SchedulerSystem>: Default + Send + Sync + 'static {
    /// Push a continuation from an external (non-worker) OS thread.
    fn push(&self, cont: SuspendedUlt<S::Desc>);

    /// Drain one item from the queue in the worker steal-fail path.
    ///
    /// [`PollerUltQueue`] always returns `None`; the poller ULT handles
    /// delivery.
    fn try_pop(&self) -> Option<SuspendedUlt<S::Desc>>;

    /// Called once by [`crate::resumable::stackful::scheduler::run`] before workers start.
    ///
    /// May push setup tasks to worker 0's deque (e.g., a poller ULT).
    /// The default implementation is a no-op.
    fn on_start(&self, _scheduler: &Arc<Scheduler<S>>) {}
}

// ---------------------------------------------------------------------------
// StealPathQueue
// ---------------------------------------------------------------------------

/// External queue drained by workers in their steal-fail path.
///
/// `push()` is mutex-guarded and O(1).  `try_pop()` skips the lock entirely
/// when the queue is observed empty via an atomic flag (`Acquire` load),
/// keeping the steal-fail path fast in the common empty case.
pub struct StealPathQueue<D: crate::resumable::common::desc::TaskDesc> {
    non_empty: AtomicBool,
    inner: Mutex<Vec<SuspendedUlt<D>>>,
}

impl<D: crate::resumable::common::desc::TaskDesc> Default for StealPathQueue<D> {
    fn default() -> Self {
        StealPathQueue {
            non_empty: AtomicBool::new(false),
            inner: Mutex::new(Vec::new()),
        }
    }
}

impl<S: SchedulerSystem> ExternalQueue<S> for StealPathQueue<S::Desc> {
    fn push(&self, cont: SuspendedUlt<S::Desc>) {
        self.inner.lock().unwrap().push(cont);
        self.non_empty.store(true, Ordering::Release);
    }

    fn try_pop(&self) -> Option<SuspendedUlt<S::Desc>> {
        if !self.non_empty.load(Ordering::Acquire) {
            return None;
        }
        let mut q = self.inner.lock().unwrap();
        let item = q.pop();
        if q.is_empty() {
            self.non_empty.store(false, Ordering::Relaxed);
        }
        item
    }
}

// ---------------------------------------------------------------------------
// PollerUltQueue
// ---------------------------------------------------------------------------

/// External queue drained by a dedicated poller ULT.
///
/// `try_pop()` always returns `None` — the steal path is unaffected.
/// `on_start()` spawns a poller ULT that loops: drain the queue → yield →
/// repeat.  Each wake-up re-checks and forwards any pending continuations to
/// the worker's local deque.
pub struct PollerUltQueue<D: crate::resumable::common::desc::TaskDesc> {
    inner: Arc<Mutex<Vec<SuspendedUlt<D>>>>,
    _marker: PhantomData<D>,
}

impl<D: crate::resumable::common::desc::TaskDesc> Default for PollerUltQueue<D> {
    fn default() -> Self {
        PollerUltQueue { inner: Arc::new(Mutex::new(Vec::new())), _marker: PhantomData }
    }
}

impl<S: StackfulSchedulerSystem + ThreadSystem> ExternalQueue<S> for PollerUltQueue<S::Desc>
where
    S::Desc: crate::resumable::stackful::desc::StackfulTaskDesc + crate::resumable::common::desc::WakerTaskDesc,
{
    fn push(&self, cont: SuspendedUlt<S::Desc>) {
        self.inner.lock().unwrap().push(cont);
    }

    fn try_pop(&self) -> Option<SuspendedUlt<S::Desc>> {
        None
    }

    fn on_start(&self, scheduler: &Arc<Scheduler<S>>) {
        let inner = Arc::clone(&self.inner);
        // Weak avoids a reference cycle: Scheduler → deque → UltDesc → closure
        // → Scheduler.  When run() drops the last strong Arc, the Weak becomes
        // dead and the poller ULT exits on its next iteration.
        let sched_weak: Weak<Scheduler<S>> = Arc::downgrade(scheduler);
        let scheduler_ptr = Arc::as_ptr(scheduler) as *const ();

        let body: ErasedBody = Box::new(move || {
            loop {
                let pending: Vec<SuspendedUlt<S::Desc>> =
                    std::mem::take(&mut *inner.lock().unwrap());
                if let Some(wk) = UltWorker::<S>::current() {
                    for cont in pending {
                        wk.push_local_bottom(cont);
                    }
                }
                match sched_weak.upgrade() {
                    None => break,
                    Some(sched) => {
                        if sched.finished.load(Ordering::Acquire) {
                            break;
                        }
                    }
                }
                S::yield_now();
            }
            Box::new(()) as Box<dyn Any + Send>
        });

        let cont = fork_parent_first::<S>(body, scheduler_ptr);
        scheduler.workers[0].push_local_top(cont);
    }
}
