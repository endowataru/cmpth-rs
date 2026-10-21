//! Duties run within their period without any worker, touches stand in
//! for polls, idle polls back the period off, and a dropped duty is never
//! polled again.  One test function: duties share one process-wide keeper.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use cmpth::duty::{Duty, DutyPlace};

#[test]
fn duties() {
    // At background QoS macOS coalesces the keeper's timers to milliseconds.
    // SAFETY: set before the keeper starts; no other thread reads it.
    unsafe { std::env::set_var("CMPTH_DUTY_QOS", "default") };
    // Kept by the keeper alone: about one poll per period.
    let n = Arc::new(AtomicU64::new(0));
    let n2 = Arc::clone(&n);
    let d = Duty::new(Duration::from_micros(200), Duration::from_micros(200), DutyPlace::Anywhere, move || {
        n2.fetch_add(1, Ordering::Relaxed);
        true
    });
    std::thread::sleep(Duration::from_millis(100));
    let polls = n.load(Ordering::Relaxed);
    assert!((100..=600).contains(&polls), "{polls} polls in 100 ms at 200 us");

    // Touched more often than the period: the keeper stays out.
    let before = d.stats().keeper_polls;
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_millis(50) {
        d.touch();
        std::thread::sleep(Duration::from_micros(20));
    }
    let during = d.stats().keeper_polls - before;
    assert!(during <= 5, "{during} keeper polls while touched");

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
}
