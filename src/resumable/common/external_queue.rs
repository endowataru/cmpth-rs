//! External-queue trait and implementations for waking ULTs from outside the
//! scheduler (e.g. RDMA completion threads calling `Waker::wake()`).

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::traits::stackful::ThreadSystem;
use crate::resumable::common::desc::{SuspendedTaskToken, TaskDescCore};
use crate::resumable::common::system::PoolSystem;
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::common::worker::{LocalQueue, UltWorker, WorkerOps};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Producer side of the external queue: all an off-pool thread needs to
/// hand a continuation back to a pool. Parameterized by the descriptor
/// type alone — deliberately narrower than [`ExternalQueue`], whose
/// consumer side (`try_pop`/`run_service`) needs the whole system. This is
/// what a task descriptor is allowed to see (see
/// [`HasExternalQueue`](crate::resumable::common::desc::HasExternalQueue)).
pub trait ExternalWakeQueue<D: TaskDescCore>: Send + Sync + 'static {
    /// Push a continuation from an external (non-worker) OS thread.
    fn push(&self, cont: SuspendedTaskToken<D>);
}

/// How continuations pushed by external OS threads reach the ULT scheduler.
/// Base-level (`S: PoolSystem`): a stackless-only system still needs a
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
///   Deliberately `Scheduler`-free: it needs `run_service` running as an
///   ordinary task, and asks for that declaratively ([`NEEDS_SERVICE`]) —
///   [`crate::resumable::stackful::init::init`] is what actually spawns and
///   joins it, once the caller is already running as a schedulable ULT.
///
/// [`NEEDS_SERVICE`]: ExternalQueue::NEEDS_SERVICE
pub trait ExternalQueue<S: PoolSystem>: ExternalWakeQueue<S::Desc> + Default + Send + Sync + 'static {
    /// Drain one item from the queue in the worker steal-fail path.
    ///
    /// [`PollerUltQueue`] always returns `None`; the poller ULT handles
    /// delivery.
    fn try_pop(&self) -> Option<SuspendedTaskToken<S::Desc>>;

    /// Whether this queue needs [`run_service`](Self::run_service) running
    /// as an ordinary task for the lifetime of the pool. Default: no
    /// service needed ([`StealPathQueue`]'s case).
    const NEEDS_SERVICE: bool = false;

    /// Service loop. Runs until [`stop_service`](Self::stop_service) asks it
    /// to return. Default: no-op (never called, since
    /// [`NEEDS_SERVICE`](Self::NEEDS_SERVICE) is `false`).
    fn run_service(&self) {}

    /// Ask a running [`run_service`](Self::run_service) call to return.
    /// Default: no-op.
    fn stop_service(&self) {}
}

// ---------------------------------------------------------------------------
// StealPathQueue
// ---------------------------------------------------------------------------

/// External queue drained by workers in their steal-fail path.
///
/// `push()` is mutex-guarded and O(1).  `try_pop()` skips the lock entirely
/// when the queue is observed empty via an atomic flag (`Acquire` load),
/// keeping the steal-fail path fast in the common empty case.
pub struct StealPathQueue<D: crate::resumable::common::desc::TaskDescCore> {
    non_empty: AtomicBool,
    inner: Mutex<Vec<SuspendedTaskToken<D>>>,
}

impl<D: crate::resumable::common::desc::TaskDescCore> Default for StealPathQueue<D> {
    fn default() -> Self {
        StealPathQueue {
            non_empty: AtomicBool::new(false),
            inner: Mutex::new(Vec::new()),
        }
    }
}

impl<D: crate::resumable::common::desc::TaskDescCore> ExternalWakeQueue<D> for StealPathQueue<D> {
    fn push(&self, cont: SuspendedTaskToken<D>) {
        self.inner.lock().unwrap().push(cont);
        self.non_empty.store(true, Ordering::Release);
    }
}

impl<S: PoolSystem> ExternalQueue<S> for StealPathQueue<S::Desc> {
    fn try_pop(&self) -> Option<SuspendedTaskToken<S::Desc>> {
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
/// `try_pop()` always returns `None` — the steal path is unaffected; the
/// only way anything moves out of `inner` is [`run_service`](Self::run_service)'s
/// loop, spawned as an ordinary task by
/// [`crate::resumable::stackful::init::init`] because
/// [`NEEDS_SERVICE`](ExternalQueue::NEEDS_SERVICE) is `true`. That task
/// drains the queue, forwards anything pending to the worker running it
/// (via [`LocalQueue::defer`]), then yields — until
/// [`stop_service`](Self::stop_service) sets `stop`, at which point one more
/// drain pass runs before the loop returns.
pub struct PollerUltQueue<D: crate::resumable::common::desc::TaskDescCore> {
    inner: Mutex<Vec<SuspendedTaskToken<D>>>,
    stop: AtomicBool,
}

impl<D: crate::resumable::common::desc::TaskDescCore> Default for PollerUltQueue<D> {
    fn default() -> Self {
        PollerUltQueue { inner: Mutex::new(Vec::new()), stop: AtomicBool::new(false) }
    }
}

impl<D: crate::resumable::common::desc::TaskDescCore> ExternalWakeQueue<D> for PollerUltQueue<D> {
    fn push(&self, cont: SuspendedTaskToken<D>) {
        self.inner.lock().unwrap().push(cont);
    }
}

impl<S: StackfulSchedulerSystem + ThreadSystem> ExternalQueue<S> for PollerUltQueue<S::Desc>
where
    S::Desc: crate::resumable::stackful::desc::StackfulTaskDesc,
{
    fn try_pop(&self) -> Option<SuspendedTaskToken<S::Desc>> {
        None
    }

    const NEEDS_SERVICE: bool = true;

    fn run_service(&self) {
        loop {
            let pending: Vec<SuspendedTaskToken<S::Desc>> =
                std::mem::take(&mut *self.inner.lock().unwrap());
            if let Some(wk) = UltWorker::<S>::current() {
                for cont in pending {
                    wk.defer(cont);
                }
            }
            // Check *after* draining (not before) so a `stop_service()` call
            // that races with a last-moment `push()` still gets one more
            // drain pass before this returns.
            if self.stop.load(Ordering::Acquire) {
                break;
            }
            S::yield_now();
        }
    }

    fn stop_service(&self) {
        self.stop.store(true, Ordering::Release);
    }
}
