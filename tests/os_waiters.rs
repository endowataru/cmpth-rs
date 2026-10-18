//! OS threads that are not workers waiting next to ULTs: on the system's
//! mutex, and on its delegator (both park on `DualResumable`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cmpth::traits::Delegator as DelegatorTrait;
use cmpth::traits::StackfulMutex;
use cmpth::{DefaultDualTaskSystem, DelegationSystem, DelegatorConsumer, DualResumable, StackfulSyncSystem};

mod common;
use common::*;

type S = DefaultDualTaskSystem;

#[test]
fn mutex_shared_by_ults_and_os_threads() {
    const ULTS: usize = 8;
    const OS: usize = 3;
    const N: u64 = 2000;
    run(2, || {
        let m: Arc<<S as StackfulSyncSystem>::Mutex<u64>> = Arc::new(StackfulMutex::new(0));
        let ults: Vec<_> = (0..ULTS)
            .map(|_| {
                let m = Arc::clone(&m);
                spawn(move || {
                    for _ in 0..N {
                        let mut g = m.lock();
                        *g += 1;
                        // A suspension while holding the lock.
                        if *g % 7 == 0 {
                            <S as cmpth::SpawnableStackfulTaskSystem>::yield_now();
                        }
                    }
                })
            })
            .collect();
        let os: Vec<_> = (0..OS)
            .map(|_| {
                let m = Arc::clone(&m);
                std::thread::spawn(move || {
                    for _ in 0..N {
                        *m.lock() += 1;
                    }
                })
            })
            .collect();
        for h in ults {
            h.join().unwrap();
        }
        for h in os {
            h.join().unwrap();
        }
        assert_eq!(*m.lock(), (ULTS + OS) as u64 * N);
    });
}

#[derive(Default)]
struct AddWork {
    amount: u64,
    sth: DualResumable<S>,
}

struct Counter {
    total: Arc<AtomicU64>,
}

impl DelegatorConsumer<S> for Counter {
    type Work = AddWork;
    fn execute(&mut self, work: &mut AddWork) -> (bool, Option<DualResumable<S>>) {
        self.total.fetch_add(work.amount, Ordering::SeqCst);
        (true, Some(std::mem::take(&mut work.sth)))
    }
    fn progress(&mut self) -> Option<DualResumable<S>> {
        None
    }
    fn is_active(&self) -> bool {
        false
    }
}

fn add(del: &<S as DelegationSystem>::Delegator<Counter>) {
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

#[test]
fn delegator_shared_by_ults_and_os_threads() {
    const ULTS: usize = 8;
    const OS: usize = 3;
    const N: u64 = 2000;
    run(2, || {
        let total = Arc::new(AtomicU64::new(0));
        let del: Arc<<S as DelegationSystem>::Delegator<Counter>> =
            Arc::new(DelegatorTrait::start(Counter { total: Arc::clone(&total) }));
        // Start the consumer ULT from a ULT.
        add(&del);
        let ults: Vec<_> = (0..ULTS)
            .map(|_| {
                let del = Arc::clone(&del);
                spawn(move || {
                    for _ in 0..N {
                        add(&del);
                    }
                })
            })
            .collect();
        let os: Vec<_> = (0..OS)
            .map(|_| {
                let del = Arc::clone(&del);
                std::thread::spawn(move || {
                    for _ in 0..N {
                        add(&del);
                    }
                })
            })
            .collect();
        for h in ults {
            h.join().unwrap();
        }
        for h in os {
            h.join().unwrap();
        }
        assert_eq!(total.load(Ordering::SeqCst), 1 + (ULTS + OS) as u64 * N);
        // Not stopped: see tests/delegator.rs.
        std::mem::forget(del);
    });
}
