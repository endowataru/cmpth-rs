//! Timer preemption: on one worker, a ULT that spins in a preemptible
//! region until another ULT of the same worker sets a flag finishes only
//! if it is preempted; the spinning code's registers survive each
//! preemption.
#![cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use cmpth::{DefaultDualTaskSystem as U, JoinHandleLike, SpawnableStackfulTaskSystem, StackfulBuilder, StackfulInitSystem};

fn with_watchdog(f: impl FnOnce()) {
    let done = Arc::new(AtomicBool::new(false));
    let d2 = Arc::clone(&done);
    std::thread::spawn(move || {
        let t0 = Instant::now();
        while !d2.load(Ordering::Relaxed) {
            if t0.elapsed() > Duration::from_secs(30) {
                eprintln!("preempt test hung");
                std::process::abort();
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    });
    f();
    done.store(true, Ordering::Relaxed);
}

#[test]
fn spinner_is_preempted_and_resumes_intact() {
    cmpth::resumable::stackful::preempt::enable(Duration::from_micros(500));
    with_watchdog(|| {
        <U as StackfulInitSystem>::builder().workers(1).run(|| {
            let flag = Arc::new(AtomicBool::new(false));
            let probes = Arc::new(AtomicU64::new(0));
            let f2 = Arc::clone(&flag);
            let spinner = U::spawn(move || {
                cmpth::resumable::stackful::preempt::preemptible::<U, _>(|| {
                    // Keeps live values in registers across preemptions.
                    let (mut a, mut b, mut x) = (1u64, 2u64, 0.5f64);
                    while !f2.load(Ordering::Relaxed) {
                        a = a.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        b = b.rotate_left(7) ^ a;
                        x = x * 1.000_000_1 + 1e-9;
                        assert_eq!(b, (b ^ a) ^ a);
                    }
                    std::hint::black_box((a, b, x));
                });
                cmpth::resumable::stackful::preempt::preemptions_here()
            });
            let (p2, f3) = (Arc::clone(&probes), Arc::clone(&flag));
            let probe = U::spawn(move || {
                // Runs only once the spinner is preempted (one worker).
                for _ in 0..20 {
                    p2.fetch_add(1, Ordering::Relaxed);
                    U::yield_now();
                }
                f3.store(true, Ordering::Relaxed);
            });
            JoinHandleLike::join(probe);
            let n = JoinHandleLike::join(spinner);
            assert!(n >= 1, "never preempted");
            assert_eq!(probes.load(Ordering::Relaxed), 20);
        })
    });
}
