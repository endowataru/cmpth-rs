//! Duties run within their period without any worker, touches stand in
//! for polls, idle polls back the period off, and a dropped duty is never
//! polled again; waiters and deadlines bring polls forward, and `sleep`
//! suspends a ULT whose worker may park meanwhile.  One test function:
//! duties share one process-wide keeper.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use cmpth::duty::{Duty, DutyPlace};
use cmpth::{BlockOnSystem, ParkDualTaskSystem, SpawnableStackfulTaskSystem, StackfulBuilder, StackfulInitSystem};

#[test]
fn duties() {
    // Kept by the keeper alone: about one poll per period, 500 in 100 ms.
    // (Bounds are loose: on a loaded machine timers fire late.)
    let n = Arc::new(AtomicU64::new(0));
    let n2 = Arc::clone(&n);
    let d = Duty::new(Duration::from_micros(200), Duration::from_micros(200), DutyPlace::Anywhere, move || {
        n2.fetch_add(1, Ordering::Relaxed);
        true
    });
    std::thread::sleep(Duration::from_millis(100));
    let polls = n.load(Ordering::Relaxed);
    assert!((20..=600).contains(&polls), "{polls} polls in 100 ms at 200 us");

    // Touched more often than the period: the keeper stays out, but for a
    // touch that comes late (a sleep of 20 us may take over 200 us).
    let before = d.stats().keeper_polls;
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_millis(50) {
        d.touch();
        std::thread::sleep(Duration::from_micros(20));
    }
    let during = d.stats().keeper_polls - before;
    assert!(during <= 25, "{during} keeper polls while touched (250 untouched)");

    // Dropped: never polled again.
    drop(d);
    let after = n.load(Ordering::Relaxed);
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(n.load(Ordering::Relaxed), after);

    // Nothing to do: the period backs off to its maximum.
    let m = Arc::new(AtomicU64::new(0));
    let m2 = Arc::clone(&m);
    let d = Duty::new(Duration::from_micros(100), Duration::from_millis(10), DutyPlace::Keeper, move || {
        m2.fetch_add(1, Ordering::Relaxed);
        false
    });
    std::thread::sleep(Duration::from_millis(300));
    let polls = m.load(Ordering::Relaxed);
    // 50 at 100 us, 50 at 200 us, ...: about 280 polls in 300 ms, against
    // about 3000 without backoff.
    assert!(polls < 600, "{polls} polls with backoff");
    drop(d);

    // On demand: never polled without waiters, every wait period with one.
    let m = Arc::new(AtomicU64::new(0));
    let m2 = Arc::clone(&m);
    let d = Duty::on_demand(Duration::from_micros(200), DutyPlace::Anywhere, move || {
        m2.fetch_add(1, Ordering::Relaxed);
        false
    });
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(m.load(Ordering::Relaxed), 0, "polled without waiters");
    let w = d.waiter();
    std::thread::sleep(Duration::from_millis(50));
    drop(w);
    let polls = m.load(Ordering::Relaxed);
    assert!((10..=300).contains(&polls), "{polls} polls in 50 ms at 200 us with a waiter");
    std::thread::sleep(Duration::from_millis(5));
    let after = m.load(Ordering::Relaxed);
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(m.load(Ordering::Relaxed), after, "polled after the last waiter left");

    // A deadline: one poll, then none.
    let t0 = Instant::now();
    d.set_deadline(Some(t0 + Duration::from_millis(5)));
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(m.load(Ordering::Relaxed), after + 1, "polls for one deadline");
    drop(d);

    // `sleep` on parking workers: 64 ULTs on 4 workers sleep 2 ms each;
    // all wake, none early, and the run takes about 2 ms, not 64 x 2.
    let t0 = Instant::now();
    ParkDualTaskSystem::builder().workers(4).run(|| {
        let hs: Vec<_> = (0..64)
            .map(|_| {
                ParkDualTaskSystem::spawn(|| {
                    let t = Instant::now();
                    ParkDualTaskSystem::block_on(cmpth::duty::sleep(Duration::from_millis(2)));
                    t.elapsed()
                })
            })
            .collect();
        for h in hs {
            let e = h.join().unwrap();
            assert!(e >= Duration::from_millis(2), "woke after {e:?}");
        }
    });
    let e = t0.elapsed();
    assert!(e < Duration::from_millis(100), "64 sleeps of 2 ms took {e:?}");

    // All workers parked, one task waiting: a parked worker takes the timed
    // role and keeps the duty's due times.  (How many of the polls it keeps
    // depends on how late its timed wake-ups are: when one is over 1 ms
    // late, as on a loaded machine, the keeper takes over.)
    let d = Arc::new(Duty::on_demand(Duration::from_micros(200), DutyPlace::Anywhere, || false));
    let d2 = Arc::clone(&d);
    ParkDualTaskSystem::builder().workers(2).run(move || {
        let _w = d2.waiter();
        ParkDualTaskSystem::block_on(cmpth::duty::sleep(Duration::from_millis(50)));
    });
    let st = d.stats();
    assert!(st.worker_polls > 0 && st.worker_polls + st.keeper_polls > 25, "{st:?}");
}
