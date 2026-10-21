//! Duties: work a process owes to something outside its own tasks, which
//! must run at least every so often whether or not a task waits for it
//! (e.g. a transport whose progress engine serves other processes'
//! requests).
//!
//! A duty is registered with a period.  It is run, cheapest first, by:
//!
//! 1. whoever makes the progress anyway, reporting it with
//!    [`Duty::touch`] (the duty's `poll` does not run then);
//! 2. an idle worker, once the duty is half due, at the points where its
//!    idle policy already does something slow (yields its thread);
//! 3. the keeper, a thread asleep until the earliest deadline, if nobody
//!    touched or ran the duty within its period.
//!
//! A duty whose `poll` may block (it waits for something, or must not run
//! on a worker) is registered [`DutyPlace::Keeper`]: only the keeper runs
//! it.
//!
//! Busy workers never check, and neither does a worker between two tasks
//! (fork-join code hits the idle loop between tasks): the scheduler's paths
//! are unchanged up to the policy's yield, where a process without duties
//! pays one relaxed load.  A duty's
//! `poll` returns whether it
//! did something; after [`IDLE_POLLS`] polls in a row that did nothing, the
//! period doubles up to the duty's maximum, and a poll that did something or
//! a touch resets it.  All polls run under one lock, so a dropped [`Duty`]
//! is never polled again.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Polls in a row that did nothing before a duty's period doubles.
pub const IDLE_POLLS: u32 = 50;

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
    poll: Box<dyn Fn() -> bool + Send + Sync>,
    place: DutyPlace,
    /// Time (see `now_ns`) of the last poll or touch.
    last: AtomicU64,
    /// Current period; between `min` and `max`.
    period: AtomicU64,
    min: u64,
    max: u64,
    idle_polls: AtomicU32,
    keeper_polls: AtomicU64,
    worker_polls: AtomicU64,
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
}

#[inline]
fn touch(i: &Inner) {
    let now = now_ns();
    // Store only when it moves the deadline noticeably: touches come from
    // every thread that calls into the resource.
    if now.wrapping_sub(i.last.load(Ordering::Relaxed)) > i.min / 4 {
        i.last.store(now, Ordering::Relaxed);
        if i.period.load(Ordering::Relaxed) != i.min {
            i.period.store(i.min, Ordering::Relaxed);
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

struct Registry {
    duties: Vec<Arc<Inner>>,
    keeper: Option<std::thread::Thread>,
}

static REG: Mutex<Registry> = Mutex::new(Registry { duties: Vec::new(), keeper: None });
/// Number of registered duties.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);
/// Earliest time some duty is half due, as of its last poll (touches only
/// make it later): idle workers take the lock only after it.
static HALF_DUE: AtomicU64 = AtomicU64::new(0);

fn now_ns() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

impl Duty {
    /// Register `poll`, to run at least every `period` (backing off up to
    /// `max_period` while it finds nothing to do).
    pub fn new(
        period: Duration,
        max_period: Duration,
        place: DutyPlace,
        poll: impl Fn() -> bool + Send + Sync + 'static,
    ) -> Duty {
        let min = (period.as_nanos() as u64).max(1000);
        let inner = Arc::new(Inner {
            poll: Box::new(poll),
            place,
            last: AtomicU64::new(now_ns()),
            period: AtomicU64::new(min),
            min,
            max: (max_period.as_nanos() as u64).max(min),
            idle_polls: AtomicU32::new(0),
            keeper_polls: AtomicU64::new(0),
            worker_polls: AtomicU64::new(0),
        });
        let mut reg = REG.lock().unwrap();
        reg.duties.push(Arc::clone(&inner));
        ACTIVE.fetch_add(1, Ordering::Relaxed);
        HALF_DUE.store(0, Ordering::Relaxed);
        match &reg.keeper {
            Some(t) => t.unpark(),
            None => {
                let h = std::thread::Builder::new().name("cmpth-duty-keeper".into()).spawn(keeper).unwrap();
                reg.keeper = Some(h.thread().clone());
            }
        }
        Duty(inner)
    }

    /// Report that the progress this duty stands for just happened anyway.
    #[inline]
    pub fn touch(&self) {
        touch(&self.0);
    }

    /// A handle that only touches, for code that makes the progress but
    /// does not own the duty.  Touching an unregistered duty does nothing.
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

impl Drop for Duty {
    fn drop(&mut self) {
        let mut reg = REG.lock().unwrap();
        reg.duties.retain(|d| !Arc::ptr_eq(d, &self.0));
        ACTIVE.fetch_sub(1, Ordering::Relaxed);
        if let Some(t) = &reg.keeper {
            t.unpark();
        }
    }
}

/// Run `d`'s poll (registry lock held) and update its period.
fn run(d: &Inner, by_keeper: bool) {
    let did = (d.poll)();
    d.last.store(now_ns(), Ordering::Relaxed);
    if did {
        d.period.store(d.min, Ordering::Relaxed);
        d.idle_polls.store(0, Ordering::Relaxed);
    } else if d.max > d.min {
        let n = d.idle_polls.load(Ordering::Relaxed) + 1;
        if n >= IDLE_POLLS {
            let p = d.period.load(Ordering::Relaxed);
            d.period.store((p * 2).min(d.max), Ordering::Relaxed);
            d.idle_polls.store(0, Ordering::Relaxed);
        } else {
            d.idle_polls.store(n, Ordering::Relaxed);
        }
    }
    let c = if by_keeper { &d.keeper_polls } else { &d.worker_polls };
    c.fetch_add(1, Ordering::Relaxed);
}

fn update_half_due(duties: &[Arc<Inner>]) {
    let h = duties
        .iter()
        .filter(|d| d.place == DutyPlace::Anywhere)
        .map(|d| d.last.load(Ordering::Relaxed) + d.period.load(Ordering::Relaxed) / 2)
        .min()
        .unwrap_or(u64::MAX);
    HALF_DUE.store(h, Ordering::Relaxed);
}

fn keeper() {
    // `CMPTH_DUTY_QOS=background` runs the keeper at background QoS.
    // macOS then coalesces its timers: a 20 us period becomes polls
    // milliseconds apart (100 ms: 11 polls instead of about 400), so the
    // period no longer means what it says.  Measured with a transport's
    // progress as the duty (4 processes, medians of 7): fork-join tasks that
    // need the peers' progress 0.64 s without the duty, 0.28 / 0.29 / 0.43 /
    // 0.72 s at 20 / 100 / 200 / 500 us, 0.32 s at background and 20 us; a
    // barrier-bound loop 0.183 s without, 0.210 / 0.205 / 0.195 s at 20 /
    // 100 / 500 us, 0.192 s at background.
    if std::env::var("CMPTH_DUTY_QOS").as_deref() == Ok("background") {
        sys::background();
    }
    loop {
        let wait = {
            let mut reg = REG.lock().unwrap();
            if reg.duties.is_empty() {
                reg.keeper = None;
                return;
            }
            let now = now_ns();
            let mut next = u64::MAX;
            for d in &reg.duties {
                let mut due = d.last.load(Ordering::Relaxed) + d.period.load(Ordering::Relaxed);
                if now >= due {
                    run(d, true);
                    due = d.last.load(Ordering::Relaxed) + d.period.load(Ordering::Relaxed);
                }
                next = next.min(due);
            }
            update_half_due(&reg.duties);
            next.saturating_sub(now_ns())
        };
        std::thread::park_timeout(Duration::from_nanos(wait.max(1000)));
    }
}

/// Idle-policy hook (at its yields): run the duties that are half due.
/// Never inlined: inlined into `SpinIdle`'s loop, the rarely taken check
/// alone made a one-worker fork-join fib 4.5 % slower.
#[inline(never)]
pub(crate) fn idle_hook() {
    if ACTIVE.load(Ordering::Relaxed) != 0 {
        idle_slow();
    }
}

#[cold]
fn idle_slow() {
    let now = now_ns();
    if now < HALF_DUE.load(Ordering::Relaxed) {
        return;
    }
    let Ok(reg) = REG.try_lock() else { return };
    for d in &reg.duties {
        if d.place == DutyPlace::Anywhere && now >= d.last.load(Ordering::Relaxed) + d.period.load(Ordering::Relaxed) / 2 {
            run(d, false);
        }
    }
    update_half_due(&reg.duties);
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
