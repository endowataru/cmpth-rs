//! End-to-end tests for [`cmpth::DefaultStackfulOnlyTaskSystem`]:
//! `execute`'s dispatch is `execute_stackful` (always a real context switch,
//! no `poll_fn` tag check) rather than `execute_dual`. Nothing in these
//! tests calls `spawn_async` on this system — its dispatch never checks the
//! `poll_fn` tag, so an async task would be mis-handled if one ever landed
//! here — so this exercises exactly the branch-free path
//! `execute_stackful`/`pop_or_root_stackful` were built for.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cmpth::{DefaultStackfulOnlyTaskSystem, JoinHandleLike, ScopedStackfulTaskSystem, ThreadSystem};

#[test]
fn spawn_join_basic() {
    DefaultStackfulOnlyTaskSystem::run(2, || {
        let h = DefaultStackfulOnlyTaskSystem::spawn(|| 6 * 7);
        assert_eq!(JoinHandleLike::join(h), 42);
    });
}

#[test]
fn spawn_join_many_parallel() {
    let counter = Arc::new(AtomicU64::new(0));
    let counter2 = Arc::clone(&counter);
    DefaultStackfulOnlyTaskSystem::run(4, move || {
        let handles: Vec<_> = (0..200)
            .map(|i| {
                let counter = Arc::clone(&counter2);
                DefaultStackfulOnlyTaskSystem::spawn(move || {
                    counter.fetch_add(1, Ordering::Relaxed);
                    i * 2u64
                })
            })
            .collect();
        let mut sum = 0u64;
        for h in handles {
            sum += JoinHandleLike::join(h);
        }
        assert_eq!(sum, (0..200).map(|i| i * 2u64).sum::<u64>());
    });
    assert_eq!(counter.load(Ordering::Relaxed), 200);
}

#[test]
fn spawn_nested() {
    DefaultStackfulOnlyTaskSystem::run(2, || {
        let h = DefaultStackfulOnlyTaskSystem::spawn(|| {
            let inner = DefaultStackfulOnlyTaskSystem::spawn(|| 10);
            JoinHandleLike::join(inner) + 5
        });
        assert_eq!(JoinHandleLike::join(h), 15);
    });
}

#[test]
fn spawn_panic_propagates() {
    DefaultStackfulOnlyTaskSystem::run(1, || {
        let h = DefaultStackfulOnlyTaskSystem::spawn::<(), _>(|| panic!("boom"));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| h.join()));
        assert!(result.is_err() || result.unwrap().is_err());
    });
}

#[test]
fn yield_now_roundtrips() {
    DefaultStackfulOnlyTaskSystem::run(1, || {
        for _ in 0..1000 {
            DefaultStackfulOnlyTaskSystem::yield_now();
        }
    });
}
