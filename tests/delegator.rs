//! Exercises `Delegator`'s consumer-thread mode, which — until this session
//! — `start()` never actually activated (`consumer_th` stayed `None`
//! forever, so any caller that lost the lock race and delegated via `del`
//! would park permanently with nothing to ever wake it; see
//! docs/sync-async-unification.md's "Deliberately not done" section for how
//! this was found).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cmpth::traits::Delegator as DelegatorTrait;
use cmpth::{BasicStackfulOnlyResumable, DefaultDualTaskSystem, DelegatorConsumer, McsDelegator};

mod common;
use common::*;

/// Work item: an amount to add to the shared total, plus the slot the
/// delegating caller parks on so `execute()` can wake them when done. Per
/// `execute_or_delegate`'s contract, `del` fills in `Work` and returns a
/// reference to *this* field (not the queue node's own `sth`, which is only
/// used for plain lock-queueing) — the node's `sth.is_set()` stays false,
/// which is exactly the signal `consume()` uses to recognize genuinely
/// delegated work.
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
        // Extract the delegator's parked continuation by value (it can't be
        // cloned) so consume() can notify() it after this returns.
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
    // Directly exercises the branch execute_or_delegate never reached in any
    // existing test: a caller that loses the lock race, delegates via `del`,
    // and must be woken by the (now actually running) consumer ULT.
    run(4, || {
        let total = Arc::new(AtomicU64::new(0));
        let del: Arc<McsDelegator<DefaultDualTaskSystem, Counter>> =
            Arc::new(McsDelegator::start(Counter { total: Arc::clone(&total) }));

        const N: usize = 50;
        let handles: Vec<_> = (0..N)
            .map(|_| {
                let del = Arc::clone(&del);
                spawn(move || {
                    del.execute_or_delegate(
                        |consumer| {
                            // Immediate path: whoever wins the lock can just
                            // do the work itself instead of delegating.
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

        // Deliberately not calling `.stop()` here — see
        // docs/sync-async-unification.md's Delegator section. `stop(self)`
        // (shared with `OsDelegator` in os.rs) takes the Delegator *by
        // value*, but the consumer ULT holds a raw pointer to wherever it
        // lived when spawned. `Arc::try_unwrap(del).stop()` moves it out of
        // the Arc first, invalidating that pointer out from under the still
        // (briefly) running consumer ULT — a real, reproducible crash,
        // distinct from and found after fixing the three bugs this session
        // already fixed in the consumer-thread/queue protocol itself. Fixing
        // it needs an actual decision about `stop`'s ownership shape
        // (`&self` instead of `self`? Require the caller to never share the
        // Delegator after `start()`? Something else?) — not something to
        // guess at. The consumer ULT is simply leaked for this test (fine
        // for a short-lived test process); do not add a `.stop()` call back
        // here without first resolving that design question.
        std::mem::forget(del);
    });
}
