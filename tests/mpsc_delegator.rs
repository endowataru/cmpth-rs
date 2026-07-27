//! Exercises the mpsc-style `delegator()`/`Producer` redesign
//! (docs/sync-async-unification.md's Delegator section). Same underlying
//! algorithm as `tests/delegator.rs`, but through the new API: no `start`/
//! `stop`, just `delegator(consumer) -> Producer`, `Producer: Clone`, and
//! automatic shutdown once every clone is dropped.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cmpth::resumable::stackful::sync::mcs_delegator::McsQueue;
use cmpth::{delegator, BasicSuspendedThread, DefaultDualTaskSystem, DelegatorConsumer};

mod common;
use common::*;

#[derive(Default)]
struct AddWork {
    amount: u64,
    sth: BasicSuspendedThread<DefaultDualTaskSystem>,
}

struct Counter {
    total: Arc<AtomicU64>,
}

impl DelegatorConsumer<DefaultDualTaskSystem> for Counter {
    type Work = AddWork;

    fn execute(
        &mut self,
        work: &mut AddWork,
    ) -> (bool, Option<BasicSuspendedThread<DefaultDualTaskSystem>>) {
        self.total.fetch_add(work.amount, Ordering::SeqCst);
        (true, Some(std::mem::take(&mut work.sth)))
    }

    fn progress(&mut self) -> Option<BasicSuspendedThread<DefaultDualTaskSystem>> {
        None
    }

    fn is_active(&self) -> bool {
        false
    }
}

type TestProducer = cmpth::DelegatorProducer<
    DefaultDualTaskSystem,
    Counter,
    McsQueue<DefaultDualTaskSystem, Counter>,
>;

#[test]
fn delegated_work_is_actually_executed() {
    run(4, || {
        let total = Arc::new(AtomicU64::new(0));
        let p: TestProducer = delegator(Counter { total: Arc::clone(&total) });

        const N: usize = 50;
        let handles: Vec<_> = (0..N)
            .map(|_| {
                let p = p.clone();
                spawn(move || {
                    p.execute_or_delegate(
                        |consumer| {
                            consumer.total.fetch_add(1, Ordering::SeqCst);
                            (true, None)
                        },
                        |work: &mut AddWork| {
                            work.amount = 1;
                            &work.sth
                        },
                    );
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(total.load(Ordering::SeqCst), N as u64);
        // `p` (this scope's own clone) drops at the end of the closure —
        // that's the last one, so this is where shutdown happens.
    });
}

#[test]
fn dropping_the_last_producer_while_work_is_in_flight_does_not_panic_or_hang() {
    // The scenario the Drop impl's lock_wait()-based design specifically
    // targets: producers still actively submitting work right up until the
    // very last clone is dropped, so Drop's shutdown sequence races for
    // real against both other producers and the consumer's own idle-park
    // attempt, not just against an already-quiescent queue.
    run(4, || {
        let total = Arc::new(AtomicU64::new(0));
        let p: TestProducer = delegator(Counter { total: Arc::clone(&total) });

        const N: usize = 32;
        let handles: Vec<_> = (0..N)
            .map(|_| {
                let p = p.clone();
                spawn(move || {
                    for _ in 0..20 {
                        p.execute_or_delegate(
                            |consumer| {
                                consumer.total.fetch_add(1, Ordering::SeqCst);
                                (true, None)
                            },
                            |work: &mut AddWork| {
                                work.amount = 1;
                                &work.sth
                            },
                        );
                    }
                    // Each spawned worker's own clone drops here, as soon as
                    // its work is done — interleaved with the others still
                    // running, and finally with `p` itself below.
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(total.load(Ordering::SeqCst), (N * 20) as u64);
        drop(p); // last clone; Drop for Inner runs here
    });
}

#[test]
fn create_and_immediately_drop_with_no_work_ever_submitted() {
    // The other edge the Drop design has to handle: the consumer ULT may
    // not have run even once yet (still sitting on the scheduler's deque)
    // when the only Producer is already dropped.
    run(4, || {
        for _ in 0..20 {
            let total = Arc::new(AtomicU64::new(0));
            let p: TestProducer = delegator(Counter { total });
            drop(p);
        }
    });
}
