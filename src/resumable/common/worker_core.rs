//! The queue layer every worker pool shares, whatever it schedules:
//! [`WorkerCore`] (one worker's run queue plus its steal loop), [`PoolCore`]
//! (the stealer table and shutdown flag one pool of such workers shares), and
//! the idle/dispatch loop that drives them (`idle_loop`/`try_run_one`).
//!
//! `UltWorker`/`Scheduler` (the descriptor-backed `resumable` flavors) embed
//! these; `scoped` uses `WorkerCore<ScopedSystem<M>>` directly as its
//! whole worker, since a stack-resident task needs nothing beyond the queue.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::resumable::common::deque::{RunQueueStealer, Steal, WorkerRunQueue};
use crate::resumable::common::external_queue::WakeHook;
use crate::resumable::common::idle::IdlePolicy;
use crate::resumable::common::lookup::CurrentLookup;
use crate::resumable::common::system::{RunnableItem, WorkerSystem};
use crate::resumable::common::worker::{LocalQueue, WorkerOps};
use crate::traits::stackful::SpawnableStackfulTaskSystem;

/// State shared by all workers of one pool: what a thief needs to reach a
/// victim, and when to stop.
pub struct PoolCore<S: WorkerSystem> {
    /// One cloneable stealer handle per worker, indexed like the pool's
    /// worker array. A thief reaches a victim's run queue exclusively through
    /// this table, never through the victim's worker struct: worker structs
    /// hold `Cell` fields whose `unsafe impl Sync` justification is "only the
    /// owning base thread touches these", so keeping thieves off them entirely
    /// turns that justification from a convention into something structural.
    pub(crate) stealers: Box<[<S::RunQueue as WorkerRunQueue<S::SuspendedToken>>::Stealer]>,
    pub(crate) finished: AtomicBool,
    pub(crate) idle: S::Idle,
}

impl<S: WorkerSystem> PoolCore<S> {
    /// Built from the worker cores in index order; call
    /// [`WorkerCore::bind`] on each once the returned value has its final
    /// address.
    pub(crate) fn new<'a>(cores: impl Iterator<Item = &'a WorkerCore<S>>) -> Self {
        PoolCore {
            stealers: cores.map(|c| c.deque.stealer()).collect(),
            finished: AtomicBool::new(false),
            idle: S::Idle::default(),
        }
    }

    /// Hook for an external queue to notify this pool's idle policy.
    pub(crate) fn wake_hook(&self) -> WakeHook {
        fn fire<S: WorkerSystem>(p: *const ()) {
            // SAFETY: see `WakeHook::new`'s contract; `p` is a `PoolCore<S>`.
            unsafe { &*(p as *const PoolCore<S>) }.idle.new_work();
        }
        // SAFETY: the pool is `Sync` and outlives the queue it is installed in.
        unsafe { WakeHook::new(self as *const Self as *const (), fire::<S>) }
    }

    /// Stop the pool: every worker leaves its idle loop, waking any that are
    /// parked.
    pub(crate) fn finish(&self) {
        self.finished.store(true, Ordering::Release);
        self.idle.shutdown();
    }
}

/// One worker's run queue, identity, and steal loop.
pub struct WorkerCore<S: WorkerSystem> {
    num: usize,
    pub(crate) deque: S::RunQueue,
    steal_seed: Cell<usize>,
    pool: Cell<*const PoolCore<S>>,
}

// SAFETY: `Cell` fields are only ever touched by the owning base thread (a
// worker struct is pinned to one thread for its lifetime); `deque` is
// internally synchronized; `pool` is read-only after `bind`.
unsafe impl<S: WorkerSystem> Send for WorkerCore<S> {}
unsafe impl<S: WorkerSystem> Sync for WorkerCore<S> {}

impl<S: WorkerSystem> WorkerCore<S> {
    pub(crate) fn new(num: usize) -> Self {
        WorkerCore {
            num,
            deque: S::RunQueue::default(),
            steal_seed: Cell::new(num.wrapping_mul(0x9E37_79B9).wrapping_add(1)),
            pool: Cell::new(std::ptr::null()),
        }
    }

    /// Called once per worker at pool construction, before any worker runs.
    pub(crate) fn bind(&self, pool: *const PoolCore<S>) {
        self.pool.set(pool);
    }

    #[inline]
    fn pool(&self) -> &PoolCore<S> {
        // SAFETY: set once via `bind` before this worker is ever driven; the
        // pool owns the OS threads that could still be running, so it
        // outlives every call through here.
        unsafe { &*self.pool.get() }
    }
}

impl<S: WorkerSystem> LocalQueue<S> for WorkerCore<S> {
    #[inline]
    fn push(&self, c: S::SuspendedToken) {
        self.deque.push(c);
        self.pool().idle.new_work();
    }

    #[inline]
    fn defer(&self, c: S::SuspendedToken) {
        self.deque.defer(c);
        self.pool().idle.new_work();
    }

    #[inline]
    fn try_pop(&self) -> Option<S::SuspendedToken> {
        self.deque.try_pop()
    }

    fn try_steal(&self) -> Steal<S::SuspendedToken> {
        let stealers = &self.pool().stealers;
        let n = stealers.len();
        if n <= 1 {
            return Steal::Empty;
        }
        let seed = self.steal_seed.get();
        self.steal_seed.set(seed.wrapping_add(1));
        let mut saw_retry = false;
        for i in 0..n {
            let victim = (seed + i) % n;
            if victim == self.num {
                continue;
            }
            match stealers[victim].try_steal() {
                Steal::Success(c) => return Steal::Success(c),
                Steal::Retry => saw_retry = true,
                Steal::Empty => {}
            }
        }
        if saw_retry { Steal::Retry } else { Steal::Empty }
    }

    #[inline]
    fn num(&self) -> usize {
        self.num
    }

    #[inline]
    fn num_workers(&self) -> usize {
        self.pool().stealers.len()
    }
}

/// A plain queue-only worker: `S::Worker` for systems with no per-worker
/// state beyond [`WorkerCore`] itself.
impl<S: WorkerSystem<Worker = WorkerCore<S>>> WorkerOps<S> for WorkerCore<S> {
    #[inline]
    fn current() -> Option<&'static Self> {
        <S::Lookup as CurrentLookup<S>>::current()
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Try to make progress once: pop our own local item, else steal. Returns
/// `false` if nothing was found anywhere right now.
#[inline]
pub(crate) fn try_run_one<S>(wk: &S::Worker) -> bool
where
    S: WorkerSystem<SuspendedToken: RunnableItem<S>>,
{
    if let Some(c) = wk.try_pop() {
        c.run_on(wk);
        return true;
    }
    if let Steal::Success(c) = wk.try_steal() {
        c.run_on(wk);
        return true;
    }
    false
}

/// The idle/dispatch loop every worker runs until `finished`: local pop,
/// then steal, then `extra` (a source outside the run queues — the external
/// queue for descriptor-backed pools, `|| None` otherwise), else hand control
/// to the pool's [`IdlePolicy`].
///
/// `try_steal` distinguishes "every victim was genuinely empty"
/// ([`Steal::Empty`]) from "some victim had work but it couldn't be taken
/// right now" ([`Steal::Retry`], e.g. lost a CAS race). The policy is told
/// which, because a `Retry` round must not count toward backing off: there is
/// known work nearby, so backing off (let alone parking) on its account would
/// be a real regression, not just noise.
#[inline]
pub(crate) fn idle_loop<S, X>(wk: &S::Worker, pool: &PoolCore<S>, mut extra: X)
where
    S: WorkerSystem<SuspendedToken: RunnableItem<S>>,
    X: FnMut() -> Option<S::SuspendedToken>,
{
    let mut ep = <S::Idle as IdlePolicy>::Episode::default();
    while !pool.finished.load(Ordering::Acquire) {
        if let Some(c) = wk.try_pop() {
            pool.idle.work_found(&mut ep);
            c.run_on(wk);
            continue;
        }
        let steal = wk.try_steal();
        if let Steal::Success(c) = steal {
            pool.idle.work_found(&mut ep);
            c.run_on(wk);
            continue;
        }
        if let Some(c) = extra() {
            pool.idle.work_found(&mut ep);
            c.run_on(wk);
            continue;
        }
        pool.idle.no_work_found(&mut ep, matches!(steal, Steal::Retry), &pool.finished, S::Base::yield_now);
    }
}
