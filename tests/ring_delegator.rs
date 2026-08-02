//! Exercises `RingBufDelegator` — until now, nothing in this crate's test
//! suite instantiated it at all, so a real indexing bug in
//! `RingBufQueue::try_follow_head` (checking the current head's own `ready`
//! flag instead of the successor's, the slot `set_next` actually writes)
//! went unnoticed: `try_follow_head` always reported "no successor yet",
//! silently falling back to the slower consumer-drain path on every
//! hand-off instead of ever taking the direct one. This test drives enough
//! concurrent delegation that the direct hand-off path is actually reached.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cmpth::resumable::stackful::sync::ring_delegator::RingBufDelegator;
use cmpth::traits::Delegator as DelegatorTrait;
use cmpth::{BasicStackfulOnlyResumable, DefaultDualTaskSystem, DelegatorConsumer};

mod common;
use common::*;

#[derive(Default)]
struct AddWork {
    amount: u64,
    sth: BasicStackfulOnlyResumable<DefaultDualTaskSystem>,
}

struct Counter {
    total: Arc<AtomicU64>,
}

impl DelegatorConsumer<DefaultDualTaskSystem> for Counter {
    type Work = AddWork;

    fn execute(
        &mut self,
        work: &mut AddWork,
    ) -> (bool, Option<BasicStackfulOnlyResumable<DefaultDualTaskSystem>>) {
        self.total.fetch_add(work.amount, Ordering::SeqCst);
        (true, Some(std::mem::take(&mut work.sth)))
    }

    fn progress(&mut self) -> Option<BasicStackfulOnlyResumable<DefaultDualTaskSystem>> {
        None
    }

    fn is_active(&self) -> bool {
        false
    }
}

#[test]
fn delegated_work_is_actually_executed() {
    run(4, || {
        let total = Arc::new(AtomicU64::new(0));
        let del: Arc<RingBufDelegator<DefaultDualTaskSystem, Counter>> =
            Arc::new(RingBufDelegator::start(Counter { total: Arc::clone(&total) }));

        const N: usize = 50;
        let handles: Vec<_> = (0..N)
            .map(|_| {
                let del = Arc::clone(&del);
                spawn(move || {
                    del.execute_or_delegate(
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

        // See tests/delegator.rs's matching comment: `stop(self)` isn't
        // safe to call here yet (consumer ULT holds a raw pointer into the
        // Delegator's current location). Leak instead of calling it.
        std::mem::forget(del);
    });
}
