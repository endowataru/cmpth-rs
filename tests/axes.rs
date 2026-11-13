//! The worker axes: an observer sees the scheduler's events, a dispatch
//! policy is asked at the loop and at task switches, and idle workers run
//! the duties of their system's registry only.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

mod common;

use common::fib;
use cmpth::{DutyOptions, DutyDomain, DutyPlace, DutyRegistry, NoDuties};
use cmpth::{
    DispatchPolicy, JoinHandleLike, SchedObserver, Site, SpawnableStackfulTaskSystem, StackfulBuilder,
    StackfulInitSystem, StandardDispatch, Steal, StealOutcome, TaskSources, UltDualSystem,
};

// --- observer ---------------------------------------------------------------

static STEALS: AtomicU64 = AtomicU64::new(0);
static STOLEN: AtomicU64 = AtomicU64::new(0);
static LOOP: AtomicU64 = AtomicU64::new(0);
static SWITCH: AtomicU64 = AtomicU64::new(0);
static PARKS: AtomicU64 = AtomicU64::new(0);
static UNPARKS: AtomicU64 = AtomicU64::new(0);

pub struct Counting;

impl SchedObserver for Counting {
    fn steal(_me: usize, o: StealOutcome) {
        STEALS.fetch_add(1, Ordering::Relaxed);
        if o == StealOutcome::Success {
            STOLEN.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn dispatch(_me: usize, site: Site) {
        match site {
            Site::Loop => &LOOP,
            Site::Switch => &SWITCH,
            Site::Wait => return,
        }
        .fetch_add(1, Ordering::Relaxed);
    }
    fn park(_me: usize) {
        PARKS.fetch_add(1, Ordering::Relaxed);
    }
    fn unpark(_me: usize) {
        UNPARKS.fetch_add(1, Ordering::Relaxed);
    }
}

pub struct Observed;
cmpth::worker_tls_anchor!(Observed);
cmpth::impl_with_defaults! {
    impl cmpth::UltDualIdentity for Observed {
        type Idle = cmpth::SearchIdle;
        type Observe = Counting;
    }
}

#[test]
fn observer_sees_steals_dispatches_and_parks() {
    type S = UltDualSystem<Observed>;
    <S as StackfulInitSystem>::builder().workers(4).run(|| {
        // The child runs first and spins until its parent, left in the
        // queue, sets the flag: the parent goes on only once stolen, so at
        // least one steal succeeds however fast the machine is.
        let flag = Arc::new(AtomicBool::new(false));
        let f = Arc::clone(&flag);
        let h = S::spawn(move || {
            while !f.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
        });
        flag.store(true, Ordering::Release);
        JoinHandleLike::join(h);
        assert_eq!(fib::<S>(20), 6765);
        // Let the other workers run out of work and park.
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(fib::<S>(15), 610);
    });
    let (steals, stolen) = (STEALS.load(Ordering::Relaxed), STOLEN.load(Ordering::Relaxed));
    assert!(stolen > 0 && steals >= stolen, "{stolen} of {steals} steal rounds succeeded");
    assert!(LOOP.load(Ordering::Relaxed) > 0 && SWITCH.load(Ordering::Relaxed) > 0);
    let parks = PARKS.load(Ordering::Relaxed);
    assert!(parks > 0 && UNPARKS.load(Ordering::Relaxed) == parks, "{parks} parks");
}

// --- dispatch policy ----------------------------------------------------------

static ASKED: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// The standard order, stealing at task switches too, counting the sites.
pub struct StealOnSwitch;

impl DispatchPolicy for StealOnSwitch {
    type State = ();
    fn next<Q: TaskSources>(st: &(), q: &Q, site: Site) -> Steal<Q::Item> {
        ASKED[site as usize].fetch_add(1, Ordering::Relaxed);
        match site {
            Site::Switch => match StandardDispatch::next(st, q, Site::Switch) {
                Steal::Success(c) => Steal::Success(c),
                _ => q.steal(),
            },
            _ => StandardDispatch::next(st, q, site),
        }
    }
}

pub struct Dispatched;
cmpth::worker_tls_anchor!(Dispatched);
cmpth::impl_with_defaults! {
    impl cmpth::UltIdentity for Dispatched {
        type Dispatch = StealOnSwitch;
    }
}

#[test]
fn dispatch_policy_is_asked_at_loop_and_switch() {
    Dispatched::builder().workers(4).run(|| assert_eq!(fib::<Dispatched>(20), 6765));
    assert!(ASKED[Site::Loop as usize].load(Ordering::Relaxed) > 0);
    assert!(ASKED[Site::Switch as usize].load(Ordering::Relaxed) > 0);
}

// --- duty domains ---------------------------------------------------------------

static PRIVATE: DutyRegistry = DutyRegistry::new(DutyOptions::DEFAULT);

pub struct PrivateDuties;
impl DutyDomain for PrivateDuties {
    fn registry() -> &'static DutyRegistry {
        &PRIVATE
    }
}

pub struct ServesPrivate;
cmpth::worker_tls_anchor!(ServesPrivate);
cmpth::impl_with_defaults! {
    impl cmpth::UltIdentity for ServesPrivate {
        type Duties = PrivateDuties;
    }
}

pub struct ServesNone;
cmpth::worker_tls_anchor!(ServesNone);
cmpth::impl_with_defaults! {
    impl cmpth::UltIdentity for ServesNone {
        type Duties = NoDuties;
    }
}

/// Idle workers of `S` for 50 ms, then their share of `d`'s polls.
fn idle_polls<S: StackfulInitSystem>(d: &cmpth::resumable::common::duty::Duty) -> u64 {
    let before = d.stats().worker_polls;
    S::builder().workers(3).run(|| {
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_millis(50) {
            std::hint::spin_loop();
        }
    });
    d.stats().worker_polls - before
}

#[test]
fn idle_workers_serve_their_own_registry() {
    let n = Arc::new(AtomicUsize::new(0));
    let n2 = Arc::clone(&n);
    // Long obligation period, so the keeper leaves it to idle workers.
    let d = PRIVATE.duty(Duration::from_micros(200), Duration::from_micros(200), DutyPlace::Anywhere, move || {
        n2.fetch_add(1, Ordering::Relaxed);
        true
    });
    assert!(idle_polls::<ServesPrivate>(&d) > 0, "workers serving the registry never polled it");
    assert_eq!(idle_polls::<ServesNone>(&d), 0, "workers serving no registry polled it");
    assert!(n.load(Ordering::Relaxed) > 0);
}
