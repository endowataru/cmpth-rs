//! OS threads that are not workers waiting next to ULTs: on the system's
//! mutex, and on its delegator (both park on `DualResumable`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cmpth::traits::Delegator as DelegatorTrait;
use cmpth::traits::StackfulMutex;
use std::pin::Pin;
use std::task::Poll;

use cmpth::{CallCx, DefaultDualTaskSystem, DelegationSystem, DelegatorConsumer, StackfulSyncSystem};

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

struct Counter {
    total: Arc<AtomicU64>,
}

impl DelegatorConsumer<S> for Counter {
    type Request<'a> = u64;
    type Response = ();
    fn execute(&mut self, amount: u64, _cx: &mut CallCx<'_, S, ()>) -> Poll<()> {
        self.total.fetch_add(amount, Ordering::SeqCst);
        Poll::Ready(())
    }
}

fn add(del: &<S as DelegationSystem>::Delegator<Counter>) {
    del.call(1);
}

#[test]
fn delegator_shared_by_ults_and_os_threads() {
    const ULTS: usize = 8;
    const OS: usize = 3;
    const N: u64 = 2000;
    run(2, || {
        let total = Arc::new(AtomicU64::new(0));
        let del: Arc<Pin<Box<<S as DelegationSystem>::Delegator<Counter>>>> =
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
        drop(Arc::try_unwrap(del).ok().expect("delegator still shared"));
    });
}

fn add_c(del: &cmpth::CombiningDelegator<S, Counter>) {
    del.call(1);
}

#[test]
fn combining_delegator_shared_by_ults_and_os_threads() {
    const ULTS: usize = 8;
    const OS: usize = 3;
    const N: u64 = 5000;
    // Before any worker pool exists: the holder runs inline.
    let total = Arc::new(AtomicU64::new(0));
    let del: Arc<Pin<Box<cmpth::CombiningDelegator<S, Counter>>>> =
        Arc::new(DelegatorTrait::start(Counter { total: Arc::clone(&total) }));
    add_c(&del);
    let d2 = Arc::clone(&del);
    run(2, move || {
        let ults: Vec<_> = (0..ULTS)
            .map(|_| {
                let del = Arc::clone(&d2);
                spawn(move || {
                    for _ in 0..N {
                        add_c(&del);
                    }
                })
            })
            .collect();
        let os: Vec<_> = (0..OS)
            .map(|_| {
                let del = Arc::clone(&d2);
                std::thread::spawn(move || {
                    for _ in 0..N {
                        add_c(&del);
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
    });
    add_c(&del);
    assert_eq!(total.load(Ordering::SeqCst), 2 + (ULTS + OS) as u64 * N);
}

/// A ULT woken by an OS thread runs even when its worker's other ULTs
/// only yield to each other (the external queue was drained only when a
/// worker idled).
#[test]
fn os_wake_reaches_a_worker_that_never_idles() {
    run(1, || {
        let m: Arc<<S as StackfulSyncSystem>::Mutex<u64>> = Arc::new(StackfulMutex::new(0));
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();
        let os = {
            let m = Arc::clone(&m);
            std::thread::spawn(move || {
                let g = m.lock();
                tx.send(()).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(50));
                drop(g); // wakes the ULT below from an OS thread
            })
        };
        rx.recv().unwrap();
        let waiter = {
            let (m, done) = (Arc::clone(&m), Arc::clone(&done));
            spawn(move || {
                *m.lock() += 1;
                done.store(true, Ordering::Release);
            })
        };
        let spinner = {
            let done = Arc::clone(&done);
            spawn(move || {
                while !done.load(Ordering::Acquire) {
                    <S as cmpth::SpawnableStackfulTaskSystem>::yield_now();
                }
            })
        };
        // The root also only yields.
        while !done.load(Ordering::Acquire) {
            <S as cmpth::SpawnableStackfulTaskSystem>::yield_now();
        }
        waiter.join().unwrap();
        spinner.join().unwrap();
        os.join().unwrap();
        assert_eq!(*m.lock(), 1);
    });
}
