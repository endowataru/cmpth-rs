//! `OsDelegator` from plain OS threads: callers that miss the lock delegate
//! their work and must be woken after the consumer ran it, and the
//! delegator must stay usable after `start` returns it by value.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cmpth::traits::Delegator as _;
use cmpth::{DelegationSystem, DelegatorConsumer, OsSystem};

#[derive(Default)]
struct AddWork {
    amount: u64,
    sth: <OsSystem as cmpth::SuspendableSystem>::SuspendedThread,
}

struct Counter {
    total: Arc<AtomicU64>,
}

impl DelegatorConsumer<OsSystem> for Counter {
    type Work = AddWork;

    fn execute(&mut self, work: &mut AddWork) -> (bool, Option<<OsSystem as cmpth::SuspendableSystem>::SuspendedThread>) {
        self.total.fetch_add(work.amount, Ordering::SeqCst);
        (true, Some(std::mem::take(&mut work.sth)))
    }

    fn progress(&mut self) -> Option<<OsSystem as cmpth::SuspendableSystem>::SuspendedThread> {
        None
    }

    fn is_active(&self) -> bool {
        false
    }
}

type Del = <OsSystem as DelegationSystem>::Delegator<Counter>;

/// Fail (not hang) if `f` takes longer than 30 s.
fn with_watchdog(f: impl FnOnce() + Send + 'static) {
    let done = Arc::new(AtomicBool::new(false));
    let done2 = Arc::clone(&done);
    let h = std::thread::spawn(move || {
        f();
        done2.store(true, Ordering::Release);
    });
    let start = Instant::now();
    while !done.load(Ordering::Acquire) {
        assert!(start.elapsed() < Duration::from_secs(30), "delegator hung");
        if h.is_finished() {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    h.join().unwrap();
}

#[test]
fn delegated_work_runs_and_wakes_its_caller() {
    with_watchdog(|| {
        let total = Arc::new(AtomicU64::new(0));
        // Moved into an Arc after `start`: the delegator must not depend on
        // the address it was built at.
        let del: Arc<Del> = Arc::new(Del::start(Counter { total: Arc::clone(&total) }));
        const THREADS: u64 = 8;
        const ITERS: u64 = 2000;
        let hs: Vec<_> = (0..THREADS)
            .map(|_| {
                let del = Arc::clone(&del);
                std::thread::spawn(move || {
                    for _ in 0..ITERS {
                        del.execute_or_delegate(
                            |c| {
                                c.total.fetch_add(1, Ordering::SeqCst);
                                (true, None)
                            },
                            |w: &mut AddWork| {
                                w.amount = 1;
                                &w.sth
                            },
                        );
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        assert_eq!(total.load(Ordering::SeqCst), THREADS * ITERS);
        Arc::try_unwrap(del).ok().expect("delegator still shared").stop();
    });
}
