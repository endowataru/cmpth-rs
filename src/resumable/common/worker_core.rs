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

use crate::resumable::common::deque::{Steal, WorkerRunQueue};
use crate::resumable::common::external_queue::WakeHook;
use crate::resumable::common::dispatch::{DispatchPolicy, Site, TaskSources};
use crate::resumable::common::idle::{IdleCx, IdlePolicy};
use crate::resumable::common::observe::{SchedObserver, StealOutcome};
use crate::resumable::common::worker_hooks::HooksSlot;
use crate::resumable::common::lookup::CurrentLookup;
use crate::resumable::common::steal::StealPolicy;
use crate::resumable::common::system::{RunnableItem, WorkerSystem};
use crate::resumable::common::worker::{LocalQueue, WorkerOps};

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
    pub(crate) hooks: S::Hooks,
}

impl<S: WorkerSystem> PoolCore<S> {
    /// Built from the worker cores in index order; call
    /// [`WorkerCore::bind`] on each once the returned value has its final
    /// address.
    pub(crate) fn new<'a>(cores: impl Iterator<Item = &'a WorkerCore<S>>) -> Self {
        let stealers: Box<[_]> = cores.map(|c| c.deque.stealer()).collect();
        let idle = S::Idle::default();
        idle.set_workers(stealers.len());
        PoolCore { stealers, finished: AtomicBool::new(false), idle, hooks: S::Hooks::default() }
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
        self.idle.shutdown::<S::Duties>();
    }
}

/// One worker's run queue, identity, and steal loop.
pub struct WorkerCore<S: WorkerSystem> {
    num: usize,
    pub(crate) deque: S::RunQueue,
    steal: <S::Steal as StealPolicy>::State,
    dispatch: <S::Dispatch as DispatchPolicy>::State,
    hooks: HooksSlot<S::Hooks>,
    pool: Cell<*const PoolCore<S>>,
}

// SAFETY: `Cell` fields, `steal` and `dispatch` are only ever touched by the
// owning base thread (a worker struct is pinned to one thread for its lifetime); `deque`
// is internally synchronized; `pool` is read-only after `bind`.
unsafe impl<S: WorkerSystem> Send for WorkerCore<S> {}
unsafe impl<S: WorkerSystem> Sync for WorkerCore<S> {}

impl<S: WorkerSystem> WorkerCore<S> {
    pub(crate) fn new(num: usize) -> Self {
        WorkerCore {
            num,
            deque: S::RunQueue::default(),
            steal: <S::Steal as StealPolicy>::new_state(num),
            dispatch: Default::default(),
            hooks: HooksSlot::default(),
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

    #[inline]
    fn hooks_slot(&self) -> &HooksSlot<S::Hooks> {
        &self.hooks
    }

    fn try_steal(&self) -> Steal<S::SuspendedToken> {
        let stealers = &self.pool().stealers;
        let n = stealers.len();
        if n <= 1 {
            return Steal::Empty;
        }
        let s = <S::Steal as StealPolicy>::try_steal(&self.steal, self.num, stealers);
        <S::Observe as SchedObserver>::steal(self.num, StealOutcome::of(&s));
        s
    }

    #[inline]
    fn dispatch_state(&self) -> &<S::Dispatch as DispatchPolicy>::State {
        &self.dispatch
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

/// A worker's task sources: its own queue, a steal round, and `external`
/// (a source outside the run queues — the external queue for
/// descriptor-backed pools, `|| None` otherwise).
pub(crate) struct Sources<'a, S: WorkerSystem, X> {
    pub(crate) wk: &'a S::Worker,
    pub(crate) external: X,
}

impl<S: WorkerSystem, X: Fn() -> Option<S::SuspendedToken>> TaskSources for Sources<'_, S, X> {
    type Item = S::SuspendedToken;

    #[inline(always)]
    fn local(&self) -> Option<S::SuspendedToken> {
        self.wk.try_pop()
    }

    #[inline(always)]
    fn steal(&self) -> Steal<S::SuspendedToken> {
        self.wk.try_steal()
    }

    #[inline(always)]
    fn external(&self) -> Option<S::SuspendedToken> {
        (self.external)()
    }
}

/// Pick the next task at `site` through the system's [`DispatchPolicy`],
/// reporting it to the observer.
#[inline(always)]
pub(crate) fn next_task<S: WorkerSystem, X: Fn() -> Option<S::SuspendedToken>>(
    wk: &S::Worker,
    external: X,
    site: Site,
) -> Steal<S::SuspendedToken> {
    let s = <S::Dispatch as DispatchPolicy>::next(wk.dispatch_state(), &Sources::<S, X> { wk, external }, site);
    if let Steal::Success(_) = s {
        <S::Observe as SchedObserver>::dispatch(wk.num(), site);
    }
    s
}

/// Try to make progress once at [`Site::Wait`] (a task waiting for a join
/// on its own stack). Returns `false` if nothing was found anywhere right
/// now.
#[inline]
pub(crate) fn try_run_one<S>(wk: &S::Worker) -> bool
where
    S: WorkerSystem<SuspendedToken: RunnableItem<S>>,
{
    match next_task::<S, _>(wk, || None, Site::Wait) {
        Steal::Success(c) => {
            c.run_on(wk);
            true
        }
        _ => false,
    }
}

/// What a waiting task does after `try_run_one` found nothing: spin, and
/// every 64 rounds run the system's duties (it cannot park: what it waits
/// for does not count as new work, so nothing would wake it).
#[inline]
pub(crate) fn wait_round<S: WorkerSystem>(rounds: &mut u32) {
    std::hint::spin_loop();
    *rounds = rounds.wrapping_add(1);
    if *rounds & 0x3F == 0 {
        if let Some(r) = <S::Duties as crate::resumable::common::duty::DutyDomain>::served() {
            r.run_idle();
        }
    }
}

/// The idle/dispatch loop every worker runs until `finished`: the next task
/// at [`Site::Loop`] through the system's [`DispatchPolicy`], else hand
/// control to the pool's [`IdlePolicy`].
///
/// A dispatch that found nothing distinguishes "every source was genuinely
/// empty" ([`Steal::Empty`]) from "some victim had work but it couldn't be
/// taken right now" ([`Steal::Retry`], e.g. lost a CAS race). The policy is
/// told which, because a `Retry` round must not count toward backing off:
/// there is known work nearby, so backing off (let alone parking) on its
/// account would be a real regression, not just noise.
#[inline]
pub(crate) fn idle_loop<S, X>(wk: &S::Worker, pool: &PoolCore<S>, external: X)
where
    S: WorkerSystem<SuspendedToken: RunnableItem<S>>,
    X: Fn() -> Option<S::SuspendedToken>,
{
    let me = wk.num();
    let mut ep = <S::Idle as IdlePolicy>::Episode::default();
    wk.hooks_slot().start(&pool.hooks, me, pool.stealers.len());
    while !pool.finished.load(Ordering::Acquire) {
        match next_task::<S, _>(wk, &external, Site::Loop) {
            Steal::Success(c) => {
                pool.idle.work_found(&mut ep, &IdleCx::<S>::new(me, false, &pool.finished));
                c.run_on(wk);
            }
            s => {
                <S::Observe as SchedObserver>::idle(me);
                pool.idle.no_work_found(&mut ep, &IdleCx::<S>::new(me, matches!(s, Steal::Retry), &pool.finished));
            }
        }
    }
    wk.hooks_slot().stop(&pool.hooks);
}
