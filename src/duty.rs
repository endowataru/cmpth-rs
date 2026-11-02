//! Duties: work that someone must check every so often because nothing
//! will announce its result.  Three kinds share one mechanism:
//!
//! - an obligation: a transport whose progress engine serves other
//!   processes' requests must run at least every `period`, whether or not a
//!   task waits for it ([`Duty::new`]);
//! - waits: tasks suspended until a completion only a poll can detect (a
//!   transport's requests) register as waiters ([`Duty::waiter`]), and
//!   while there is one the duty runs every `wait_period`
//!   ([`Duty::on_demand`] has no obligation at all, only waits);
//! - deadlines: a duty may ask to run at a given time
//!   ([`Duty::set_deadline`]); [`sleep`] is built on that.
//!
//! A duty is run, cheapest first, by:
//!
//! 1. whoever makes the progress anyway, reporting it with
//!    [`Duty::touch`] (the duty's `poll` does not run then);
//! 2. an idle worker, once the duty is half due (a deadline: once due), at
//!    the points where its idle policy already does something slow (yields
//!    its thread, or goes to sleep);
//! 3. the keeper, a thread asleep until the earliest due time, if nobody
//!    touched or ran the duty by then.
//!
//! A duty whose `poll` may block (it waits for something, or must not run
//! on a worker) is registered [`DutyPlace::Keeper`]: only the keeper runs
//! it.
//!
//! Busy workers never check, and neither does a worker between two tasks
//! (fork-join code hits the idle loop between tasks): the scheduler's paths
//! are unchanged up to the policy's yield, where a process without duties
//! pays one relaxed load.  A duty's `poll` returns whether it did
//! something; after [`IDLE_POLLS`] polls in a row that did nothing, the
//! period doubles up to the duty's maximum, and a poll that did something or
//! a touch resets it.  All polls run under one lock, so a dropped [`Duty`]
//! is never polled again.
//!
//! Duties live in a [`DutyRegistry`], with its own keeper.  [`Duty::new`]
//! and [`Duty::on_demand`] use the process-wide one
//! ([`DutyRegistry::global`]); a system's idle workers serve the registry
//! its [`DutyDomain`] names (`WorkerSystem::Duties`), so a pool can serve a
//! registry of its own ([`DutyRegistry::duty`]), or none ([`NoDuties`]).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

/// Polls in a row that did nothing before a duty's period doubles.
pub const IDLE_POLLS: u32 = 50;

/// "Never" as a time or a period.
const NEVER: u64 = u64::MAX;

/// How often an idle worker runs a duty that has waiters.
const EAGER_NS: u64 = 1000;

/// Where a duty's `poll` may run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DutyPlace {
    /// Any thread, including a worker's idle loop: `poll` must not block
    /// or suspend.
    Anywhere,
    /// The keeper thread only: `poll` may block it (briefly).
    Keeper,
}

struct Inner {
    home: &'static DutyRegistry,
    poll: Box<dyn Fn() -> bool + Send + Sync>,
    place: DutyPlace,
    /// Time (see `now_ns`) of the last poll or touch.
    last: AtomicU64,
    /// Current obligation period, between `min` and `max`; `NEVER` if none.
    period: AtomicU64,
    min: AtomicU64,
    max: AtomicU64,
    /// Period while `waiters != 0`.
    wait_period: AtomicU64,
    waiters: AtomicUsize,
    /// Requested time of the next poll; `NEVER` if none.
    deadline: AtomicU64,
    idle_polls: AtomicU32,
    keeper_polls: AtomicU64,
    worker_polls: AtomicU64,
}

impl Inner {
    fn effective_period(&self) -> u64 {
        let p = self.period.load(Ordering::Relaxed);
        if self.waiters.load(Ordering::Relaxed) != 0 {
            p.min(self.wait_period.load(Ordering::Relaxed))
        } else {
            p
        }
    }

    /// When the keeper must run it.
    fn due(&self) -> u64 {
        let last = self.last.load(Ordering::Relaxed);
        last.saturating_add(self.effective_period()).min(self.deadline.load(Ordering::Relaxed))
    }

    /// When an idle worker may run it: half due, or with waiters, almost
    /// at once (it has nothing better to do, and a waiter's completion
    /// found late delays its task by as much).
    fn half_due(&self) -> u64 {
        let last = self.last.load(Ordering::Relaxed);
        let gap = if self.waiters.load(Ordering::Relaxed) != 0 { EAGER_NS } else { self.effective_period() / 2 };
        last.saturating_add(gap).min(self.deadline.load(Ordering::Relaxed))
    }
}

/// A registered duty; dropping it unregisters it.
pub struct Duty(Arc<Inner>);

/// See [`Duty::toucher`].
#[derive(Clone)]
pub struct DutyToucher(Arc<Inner>);

impl DutyToucher {
    #[inline]
    pub fn touch(&self) {
        touch(&self.0);
    }

    /// See [`Duty::waiter`].
    pub fn waiter(&self) -> DutyWaiter {
        DutyWaiter::new(&self.0)
    }
}

/// A registered waiter of a duty (see [`Duty::waiter`]); dropping it
/// unregisters it.
pub struct DutyWaiter(Arc<Inner>);

impl DutyWaiter {
    fn new(i: &Arc<Inner>) -> DutyWaiter {
        i.home.waiters.fetch_add(1, Ordering::Relaxed);
        if i.waiters.fetch_add(1, Ordering::Relaxed) == 0 {
            replan(i);
        }
        DutyWaiter(Arc::clone(i))
    }
}

impl Drop for DutyWaiter {
    fn drop(&mut self) {
        self.0.waiters.fetch_sub(1, Ordering::Relaxed);
        self.0.home.waiters.fetch_sub(1, Ordering::Relaxed);
    }
}

#[inline]
fn touch(i: &Inner) {
    let now = now_ns();
    // Store only when it moves the due time noticeably: touches come from
    // every thread that calls into the resource.
    let grain = i.min.load(Ordering::Relaxed).min(i.wait_period.load(Ordering::Relaxed)) / 4;
    if now.wrapping_sub(i.last.load(Ordering::Relaxed)) > grain {
        i.last.store(now, Ordering::Relaxed);
        let min = i.min.load(Ordering::Relaxed);
        if i.period.load(Ordering::Relaxed) != min {
            i.period.store(min, Ordering::Relaxed);
            i.idle_polls.store(0, Ordering::Relaxed);
        }
    }
}

/// How often a duty's `poll` ran, by whom.
#[derive(Clone, Copy, Debug, Default)]
pub struct DutyStats {
    pub keeper_polls: u64,
    pub worker_polls: u64,
}

/// Settings of a [`DutyRegistry`].
#[derive(Clone, Copy, Debug)]
pub struct DutyConfig {
    /// How long an idle worker searches (running the duties) before it
    /// parks while some task waits on a duty (see
    /// [`DutyRegistry::wait_spin_ns`]).
    pub wait_spin: Duration,
    /// Run the keeper at background QoS (macOS; see the keeper).
    pub background_keeper: bool,
}

impl DutyConfig {
    /// 100 us of searching, keeper at normal QoS.
    pub const DEFAULT: DutyConfig = DutyConfig { wait_spin: Duration::from_micros(100), background_keeper: false };

    /// [`DEFAULT`](Self::DEFAULT), overridden by `CMPTH_WAIT_SPIN_US` and
    /// `CMPTH_DUTY_QOS=background`.
    pub fn from_env() -> DutyConfig {
        let mut c = DutyConfig::DEFAULT;
        if let Some(us) = std::env::var("CMPTH_WAIT_SPIN_US").ok().and_then(|v| v.parse::<u64>().ok()) {
            c.wait_spin = Duration::from_micros(us);
        }
        if std::env::var("CMPTH_DUTY_QOS").as_deref() == Ok("background") {
            c.background_keeper = true;
        }
        c
    }
}

/// A set of duties with one keeper thread, served by the idle workers of
/// the systems whose [`DutyDomain`] names it.  Make one a `static`:
///
/// ```
/// use std::time::Duration;
/// use cmpth::duty::{DutyConfig, DutyPlace, DutyRegistry};
///
/// static MINE: DutyRegistry = DutyRegistry::new(DutyConfig::DEFAULT);
/// let d = MINE.duty(Duration::from_millis(1), Duration::from_millis(8), DutyPlace::Keeper, || false);
/// drop(d);
/// ```
pub struct DutyRegistry {
    reg: Mutex<Vec<Arc<Inner>>>,
    keeper: OnceLock<std::thread::Thread>,
    /// Number of registered duties.
    active: AtomicUsize,
    /// Number of registered waiters, all duties together.
    waiters: AtomicUsize,
    /// Earliest time some duty is half due, as of its last poll (touches
    /// only make it later; a new waiter or deadline lowers it): idle
    /// workers take the lock only after it.
    half_due: AtomicU64,
    /// When the keeper plans to wake up.
    planned: AtomicU64,
    /// A parked worker keeps the [`DutyPlace::Anywhere`] duties' due times
    /// (see [`park_timeout`](Self::park_timeout)) ...
    timed: AtomicBool,
    /// ... and wakes up at this time.
    timed_until: AtomicU64,
    /// `None`: read from the environment on first use.
    init: Option<DutyConfig>,
    config: OnceLock<DutyConfig>,
}

static GLOBAL: DutyRegistry = DutyRegistry::with_init(None);

impl DutyRegistry {
    pub const fn new(config: DutyConfig) -> DutyRegistry {
        DutyRegistry::with_init(Some(config))
    }

    const fn with_init(init: Option<DutyConfig>) -> DutyRegistry {
        DutyRegistry {
            reg: Mutex::new(Vec::new()),
            keeper: OnceLock::new(),
            active: AtomicUsize::new(0),
            waiters: AtomicUsize::new(0),
            half_due: AtomicU64::new(0),
            planned: AtomicU64::new(NEVER),
            timed: AtomicBool::new(false),
            timed_until: AtomicU64::new(NEVER),
            init,
            config: OnceLock::new(),
        }
    }

    /// The process-wide registry: [`Duty::new`], [`Duty::on_demand`] and
    /// [`sleep`] use it, and so do systems whose `Duties` is
    /// [`GlobalDuties`].  Configured by [`DutyConfig::from_env`].
    pub fn global() -> &'static DutyRegistry {
        &GLOBAL
    }

    pub fn config(&self) -> &DutyConfig {
        self.config.get_or_init(|| self.init.unwrap_or_else(DutyConfig::from_env))
    }

    /// [`Duty::new`] in this registry.
    pub fn duty(
        &'static self,
        period: Duration,
        max_period: Duration,
        place: DutyPlace,
        poll: impl Fn() -> bool + Send + Sync + 'static,
    ) -> Duty {
        let min = ns(period).max(1000);
        self.register(min, ns(max_period).max(min), min, place, Box::new(poll))
    }

    /// [`Duty::on_demand`] in this registry.
    pub fn on_demand(&'static self, wait_period: Duration, place: DutyPlace, poll: impl Fn() -> bool + Send + Sync + 'static) -> Duty {
        self.register(NEVER, NEVER, ns(wait_period).max(1000), place, Box::new(poll))
    }
}

/// Which [`DutyRegistry`] a system's idle workers serve
/// (`WorkerSystem::Duties`).
pub trait DutyDomain: Send + Sync + 'static {
    /// `None`: the workers serve no duties, and the idle policies' duty
    /// calls compile away.
    fn registry() -> Option<&'static DutyRegistry>;
}

/// The process-wide registry ([`DutyRegistry::global`]).  The default.
pub struct GlobalDuties;

impl DutyDomain for GlobalDuties {
    #[inline(always)]
    fn registry() -> Option<&'static DutyRegistry> {
        Some(DutyRegistry::global())
    }
}

/// No duties: for a pool that must not run anyone's polls (one nested in
/// another that serves them, or a benchmark).
pub struct NoDuties;

impl DutyDomain for NoDuties {
    #[inline(always)]
    fn registry() -> Option<&'static DutyRegistry> {
        None
    }
}

static EPOCH: OnceLock<Instant> = OnceLock::new();

fn now_ns() -> u64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

fn ns(d: Duration) -> u64 {
    d.as_nanos().min(NEVER as u128 - 1) as u64
}

/// `i` may be due earlier than planned: tell idle workers and the keeper.
fn replan(i: &Inner) {
    let r = i.home;
    if i.place == DutyPlace::Anywhere {
        r.half_due.fetch_min(i.half_due(), Ordering::Relaxed);
    }
    // A parked worker's deadline only moves when it wakes: the keeper
    // covers anything earlier.  The fence pairs with the keeper's: either
    // it sees our due time or we see its plan.
    std::sync::atomic::fence(Ordering::SeqCst);
    if i.due() < r.planned.load(Ordering::Relaxed) {
        if let Some(k) = r.keeper.get() {
            k.unpark();
        }
    }
}

impl DutyRegistry {
    fn register(&'static self, min: u64, max: u64, wait: u64, place: DutyPlace, poll: Box<dyn Fn() -> bool + Send + Sync>) -> Duty {
        let inner = Arc::new(Inner {
            home: self,
            poll,
            place,
            last: AtomicU64::new(now_ns()),
            period: AtomicU64::new(min),
            min: AtomicU64::new(min),
            max: AtomicU64::new(max),
            wait_period: AtomicU64::new(wait),
            waiters: AtomicUsize::new(0),
            deadline: AtomicU64::new(NEVER),
            idle_polls: AtomicU32::new(0),
            keeper_polls: AtomicU64::new(0),
            worker_polls: AtomicU64::new(0),
        });
        self.reg.lock().unwrap().push(Arc::clone(&inner));
        self.active.fetch_add(1, Ordering::Relaxed);
        self.keeper.get_or_init(|| {
            std::thread::Builder::new().name("cmpth-duty-keeper".into()).spawn(move || self.keep()).unwrap().thread().clone()
        });
        replan(&inner);
        Duty(inner)
    }
}

impl Duty {
    /// Register `poll` in the process-wide registry, to run at least every
    /// `period` (backing off up to `max_period` while it finds nothing to
    /// do), and every `period` while it has waiters.
    pub fn new(
        period: Duration,
        max_period: Duration,
        place: DutyPlace,
        poll: impl Fn() -> bool + Send + Sync + 'static,
    ) -> Duty {
        DutyRegistry::global().duty(period, max_period, place, poll)
    }

    /// Register `poll` in the process-wide registry, to run only while it
    /// has waiters, every `wait_period`, or at its deadline.
    pub fn on_demand(wait_period: Duration, place: DutyPlace, poll: impl Fn() -> bool + Send + Sync + 'static) -> Duty {
        DutyRegistry::global().on_demand(wait_period, place, poll)
    }

    /// Change the obligation: at least every `period`, backing off up to
    /// `max_period`; `None`: only while there are waiters.
    pub fn set_period(&self, period: Option<(Duration, Duration)>) {
        let (min, max) = match period {
            Some((p, m)) => {
                let min = ns(p).max(1000);
                (min, ns(m).max(min))
            }
            None => (NEVER, NEVER),
        };
        let i = &self.0;
        i.min.store(min, Ordering::Relaxed);
        i.max.store(max, Ordering::Relaxed);
        i.period.store(min, Ordering::Relaxed);
        i.idle_polls.store(0, Ordering::Relaxed);
        replan(i);
    }

    /// Change the period while there are waiters.
    pub fn set_wait_period(&self, period: Duration) {
        self.0.wait_period.store(ns(period).max(1000), Ordering::Relaxed);
        replan(&self.0);
    }

    /// Ask for a poll at `at` (or none); the poll clears it, so a duty that
    /// wants the next one sets it again from its `poll`.
    pub fn set_deadline(&self, at: Option<Instant>) {
        set_deadline(&self.0, at);
    }

    /// Report that the progress this duty stands for just happened anyway.
    #[inline]
    pub fn touch(&self) {
        touch(&self.0);
    }

    /// Register a waiter: until it is dropped, the duty runs every
    /// `wait_period`.  Whoever detects the completion wakes the waiting
    /// task; the duty only makes sure someone looks.
    pub fn waiter(&self) -> DutyWaiter {
        DutyWaiter::new(&self.0)
    }

    /// A handle that only touches and registers waiters, for code that
    /// makes the progress but does not own the duty.  Touching an
    /// unregistered duty does nothing.
    pub fn toucher(&self) -> DutyToucher {
        DutyToucher(Arc::clone(&self.0))
    }

    pub fn stats(&self) -> DutyStats {
        DutyStats {
            keeper_polls: self.0.keeper_polls.load(Ordering::Relaxed),
            worker_polls: self.0.worker_polls.load(Ordering::Relaxed),
        }
    }
}

fn set_deadline(i: &Inner, at: Option<Instant>) {
    let t = at.map_or(NEVER, |at| ns(at.saturating_duration_since(*EPOCH.get_or_init(Instant::now))));
    i.deadline.store(t, Ordering::Relaxed);
    replan(i);
}

impl Drop for Duty {
    fn drop(&mut self) {
        let home = self.0.home;
        let mut reg = home.reg.lock().unwrap();
        reg.retain(|d| !Arc::ptr_eq(d, &self.0));
        home.active.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Run `d`'s poll (registry lock held) and update its period.
fn run(d: &Inner, by_keeper: bool) {
    d.deadline.store(NEVER, Ordering::Relaxed);
    let did = (d.poll)();
    d.last.store(now_ns(), Ordering::Relaxed);
    let (min, max) = (d.min.load(Ordering::Relaxed), d.max.load(Ordering::Relaxed));
    if did {
        d.period.store(min, Ordering::Relaxed);
        d.idle_polls.store(0, Ordering::Relaxed);
    } else if max > min {
        let n = d.idle_polls.load(Ordering::Relaxed) + 1;
        if n >= IDLE_POLLS {
            let p = d.period.load(Ordering::Relaxed);
            d.period.store(p.saturating_mul(2).min(max), Ordering::Relaxed);
            d.idle_polls.store(0, Ordering::Relaxed);
        } else {
            d.idle_polls.store(n, Ordering::Relaxed);
        }
    }
    let c = if by_keeper { &d.keeper_polls } else { &d.worker_polls };
    c.fetch_add(1, Ordering::Relaxed);
}

impl DutyRegistry {
    fn update_half_due(&self, duties: &[Arc<Inner>]) {
        let h = duties.iter().filter(|d| d.place == DutyPlace::Anywhere).map(|d| d.half_due()).min().unwrap_or(NEVER);
        self.half_due.store(h, Ordering::Relaxed);
    }

    /// The keeper thread.
    fn keep(&'static self) {
        // `background_keeper` (`CMPTH_DUTY_QOS=background`) runs the keeper
        // at background QoS.  macOS then coalesces its timers: a 20 us
        // period becomes polls milliseconds apart (100 ms: 11 polls instead
        // of about 400), so the period no longer means what it says.
        // Measured with a transport's progress as the duty (4 processes,
        // medians of 7): fork-join tasks that need the peers' progress
        // 0.64 s without the duty, 0.28 / 0.29 / 0.43 / 0.72 s at 20 / 100
        // / 200 / 500 us, 0.32 s at background and 20 us; a barrier-bound
        // loop 0.183 s without, 0.210 / 0.205 / 0.195 s at 20 / 100 / 500
        // us, 0.192 s at background.
        if self.config().background_keeper {
            sys::background();
        }
        loop {
            // Until the plan below is stored, any replan unparks us.  The
            // fence pairs with `replan`'s: either it sees this or we see the
            // due time it stored.
            self.planned.store(NEVER, Ordering::Relaxed);
            std::sync::atomic::fence(Ordering::SeqCst);
            let next = {
                let reg = self.reg.lock().unwrap();
                let now = now_ns();
                // Due times of the duties only the keeper runs, and of the
                // others, which a timed parked worker may cover.
                let (mut mine, mut any) = (NEVER, NEVER);
                for d in reg.iter() {
                    let mut due = d.due();
                    if now >= due {
                        run(d, true);
                        due = d.due();
                    }
                    match d.place {
                        DutyPlace::Keeper => mine = mine.min(due),
                        DutyPlace::Anywhere => any = any.min(due),
                    }
                }
                self.update_half_due(&reg);
                let until = self.timed_until.load(Ordering::Relaxed);
                if self.timed.load(Ordering::Acquire) && any >= until && until != NEVER {
                    // Covered: look again only if the worker fails to renew.
                    any = until.saturating_add(TIMED_SLACK);
                }
                let next = mine.min(any);
                self.planned.store(next, Ordering::Relaxed);
                next
            };
            // A replan between the store above and the park unparks us
            // first, and `park` then returns at once.
            if next == NEVER {
                std::thread::park();
            } else {
                std::thread::park_timeout(Duration::from_nanos(next.saturating_sub(now_ns()).max(1000)));
            }
        }
    }

    // --- for idle policies (through `IdleCx`) ----------------------------

    /// Run the duties that are half due.  Idle policies call it where they
    /// already do something slow (yield the thread, or before sleeping).
    /// Never inlined: inlined into `SpinIdle`'s loop, the rarely taken
    /// check alone made a one-worker fork-join fib 4.5 % slower.
    #[inline(never)]
    pub fn run_idle(&self) {
        if self.active.load(Ordering::Relaxed) != 0 {
            self.idle_slow();
        }
    }

    #[cold]
    fn idle_slow(&self) {
        let now = now_ns();
        if now < self.half_due.load(Ordering::Relaxed) {
            return;
        }
        let Ok(reg) = self.reg.try_lock() else { return };
        for d in reg.iter() {
            if d.place == DutyPlace::Anywhere && now >= d.half_due() {
                run(d, false);
            }
        }
        self.update_half_due(&reg);
    }

    /// How long an idle worker searches (running the duties) before it
    /// parks while some task waits on a duty: what it waits for may need
    /// this process's progress, or arrive soon.  0 without waiters, else
    /// [`DutyConfig::wait_spin`] (100 us by default).  Measured with a
    /// transport whose one-sided operations need the target's progress (4
    /// processes, medians of 7, time / energy): a barrier-bound loop 0.191
    /// s / 4.24 J spinning, 0.318 / 5.02 parked at once, 0.224 / 4.08 at
    /// 100 us, 0.223 / 4.12 at 500 us; a dynamically scheduled loop with
    /// idle workers 0.091 s / 8.6 J spinning, 0.092 / 4.4, 0.090 / 5.2,
    /// 0.091 / 5.7.
    #[inline(never)]
    pub fn wait_spin_ns(&self) -> u64 {
        if self.waiters.load(Ordering::Relaxed) == 0 {
            return 0;
        }
        ns(self.config().wait_spin)
    }

    /// Called by a worker about to park, `*timed` telling whether it
    /// already keeps the duties' due times: if no other parked worker
    /// does, this one does (`*timed` becomes true), and must park at most
    /// the returned time.  Waking up, it runs the due duties on its way
    /// back to sleep (the idle hook before it parks) and keeps the role; a
    /// task they wake runs on it directly.  It gives the role up
    /// ([`idle_done`](Self::idle_done)) when it finds work.  Meanwhile the
    /// keeper sleeps.
    #[inline(never)]
    pub fn park_timeout(&self, timed: &mut bool) -> Option<Duration> {
        if self.active.load(Ordering::Relaxed) == 0 {
            return None;
        }
        let Ok(reg) = self.reg.try_lock() else {
            // The keeper is busy with the registry: a holder keeps its last
            // wake-up time rather than sleeping without one.
            if !*timed {
                return None;
            }
            let until = self.timed_until.load(Ordering::Relaxed);
            return Some(Duration::from_nanos(until.saturating_sub(now_ns()).max(1000)));
        };
        let next = reg.iter().filter(|d| d.place == DutyPlace::Anywhere).map(|d| d.due()).min().unwrap_or(NEVER);
        drop(reg);
        if next == NEVER {
            if *timed {
                self.idle_done(timed);
            }
            return None;
        }
        if !*timed {
            // Taken, unless its holder stopped renewing it (a worker that
            // exited holding it never found work to give it up).
            let until = self.timed_until.load(Ordering::Relaxed);
            if self.timed.swap(true, Ordering::AcqRel) && (until == NEVER || now_ns() < until.saturating_add(TIMED_SLACK)) {
                return None;
            }
            *timed = true;
        } else {
            self.timed.store(true, Ordering::Release);
        }
        self.timed_until.store(next, Ordering::Relaxed);
        Some(Duration::from_nanos(next.saturating_sub(now_ns()).max(1000)))
    }

    /// A worker that kept the due times found work (see
    /// [`park_timeout`](Self::park_timeout)).
    #[cold]
    pub fn idle_done(&self, timed: &mut bool) {
        *timed = false;
        self.release_timed();
    }

    /// A pool shuts down: its workers exit without finding work, so one may
    /// hold the role.  If another pool's worker holds it, it takes it back
    /// when it next renews.
    pub fn release_timed(&self) {
        self.timed_until.store(NEVER, Ordering::Relaxed);
        self.timed.store(false, Ordering::Release);
        if let Some(k) = self.keeper.get() {
            k.unpark();
        }
    }
}

/// How long past a timed worker's wake-up time the keeper still trusts it
/// (it may have found work, or exited, without saying so yet).
const TIMED_SLACK: u64 = 1_000_000;

// ---------------------------------------------------------------------------
// Timers
// ---------------------------------------------------------------------------

struct TimerSlot {
    fired: AtomicBool,
    waker: Mutex<Option<Waker>>,
}

#[derive(Default)]
struct TimerQueue {
    heap: BinaryHeap<Reverse<(u64, u64)>>,
    slots: std::collections::HashMap<u64, Arc<TimerSlot>>,
}

struct Timers {
    /// Also guards the timer duty's deadline: both `fire_timers` and a new
    /// sleeper set it under this lock, so neither overwrites the other's.
    q: Mutex<TimerQueue>,
    seq: AtomicU64,
    duty: OnceLock<Duty>,
}

static TIMERS: OnceLock<Timers> = OnceLock::new();

fn timers() -> &'static Timers {
    let t = TIMERS.get_or_init(|| Timers {
        q: Mutex::new(TimerQueue::default()),
        seq: AtomicU64::new(0),
        duty: OnceLock::new(),
    });
    t.duty.get_or_init(|| Duty::on_demand(Duration::from_secs(3600), DutyPlace::Anywhere, fire_timers));
    t
}

/// The timer duty's poll: wake every sleeper whose time has come.
fn fire_timers() -> bool {
    let t = TIMERS.get().unwrap();
    let now = now_ns();
    let mut woken = Vec::new();
    {
        let mut q = t.q.lock().unwrap();
        while let Some(&Reverse((at, id))) = q.heap.peek() {
            if at > now {
                break;
            }
            q.heap.pop();
            if let Some(s) = q.slots.remove(&id) {
                s.fired.store(true, Ordering::Release);
                if let Some(w) = s.waker.lock().unwrap().take() {
                    woken.push(w);
                }
            }
        }
        if let Some(&Reverse((at, _))) = q.heap.peek() {
            // Under the registry lock: set the field directly (`run` has
            // cleared it); the caller recomputes the plan.
            t.duty.get().unwrap().0.deadline.fetch_min(at, Ordering::Relaxed);
        }
    }
    let did = !woken.is_empty();
    for w in woken {
        w.wake();
    }
    did
}

/// A future that resolves at `at` (see [`sleep`]).
pub struct Sleep {
    at: u64,
    slot: Option<(u64, Arc<TimerSlot>)>,
}

/// Sleep for `d`: on a ULT (through `block_on`) or in a stackless task,
/// the task suspends and its worker may run others or park; an idle worker
/// or the keeper wakes it.  Precision is the keeper's (tens of
/// microseconds), not better.
pub fn sleep(d: Duration) -> Sleep {
    Sleep { at: now_ns().saturating_add(ns(d)), slot: None }
}

impl Future for Sleep {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        match &this.slot {
            None => {
                if now_ns() >= this.at {
                    return Poll::Ready(());
                }
                let t = timers();
                let id = t.seq.fetch_add(1, Ordering::Relaxed);
                let slot = Arc::new(TimerSlot { fired: AtomicBool::new(false), waker: Mutex::new(Some(cx.waker().clone())) });
                let d = &t.duty.get().unwrap().0;
                let earlier = {
                    let mut q = t.q.lock().unwrap();
                    q.slots.insert(id, Arc::clone(&slot));
                    q.heap.push(Reverse((this.at, id)));
                    d.deadline.fetch_min(this.at, Ordering::Relaxed) > this.at
                };
                if earlier {
                    replan(d);
                }
                this.slot = Some((id, slot));
                Poll::Pending
            }
            Some((_, slot)) => {
                if slot.fired.load(Ordering::Acquire) {
                    return Poll::Ready(());
                }
                *slot.waker.lock().unwrap() = Some(cx.waker().clone());
                if slot.fired.load(Ordering::Acquire) {
                    return Poll::Ready(());
                }
                Poll::Pending
            }
        }
    }
}

impl Drop for Sleep {
    fn drop(&mut self) {
        if let Some((id, slot)) = self.slot.take() {
            // Never wake a task that stopped waiting: its waker may outlive
            // the task.  The heap entry fires harmlessly later.
            slot.waker.lock().unwrap().take();
            TIMERS.get().unwrap().q.lock().unwrap().slots.remove(&id);
        }
    }
}

#[cfg(target_os = "macos")]
mod sys {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos: u32, rel: i32) -> i32;
    }

    pub(super) fn background() {
        // QOS_CLASS_BACKGROUND
        unsafe { pthread_set_qos_class_self_np(0x09, 0) };
    }
}

#[cfg(not(target_os = "macos"))]
mod sys {
    pub(super) fn background() {}
}
