//! Exercises `UltDualBarrier<S, N>` (docs/sync-async-unification.md) across all
//! three wait-slot flavors, including a dual instance where real ULTs and
//! `spawn_async` tasks arrive at the same barrier together.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use cmpth::traits::{BarrierWaitResult, StackfulBarrier, StacklessBarrier};
use cmpth::{BasicStackfulOnlyResumable, DefaultDualTaskSystem, SuspendedFuture, DualResumable, UltDualBarrier};

mod common;
use common::*;

#[test]
fn sync_only_flavor() {
    run(8, || {
        const N: usize = 8;
        let b: Arc<UltDualBarrier<DefaultDualTaskSystem, BasicStackfulOnlyResumable<DefaultDualTaskSystem>>> =
            Arc::new(UltDualBarrier::new(N));
        let before = Arc::new(AtomicUsize::new(0));
        let after = Arc::new(AtomicUsize::new(0));
        let leaders = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..N)
            .map(|_| {
                let b = Arc::clone(&b);
                let before = Arc::clone(&before);
                let after = Arc::clone(&after);
                let leaders = Arc::clone(&leaders);
                spawn(move || {
                    before.fetch_add(1, Ordering::SeqCst);
                    let r: BarrierWaitResult = StackfulBarrier::wait(&*b);
                    // Every waiter must observe all N arrivals before any one
                    // of them proceeds past the barrier.
                    assert_eq!(before.load(Ordering::SeqCst), N);
                    if r.is_leader() {
                        leaders.fetch_add(1, Ordering::SeqCst);
                    }
                    after.fetch_add(1, Ordering::SeqCst);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(after.load(Ordering::SeqCst), N);
        assert_eq!(leaders.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn async_only_flavor() {
    run(8, || {
        const N: usize = 8;
        let b: Arc<UltDualBarrier<DefaultDualTaskSystem, SuspendedFuture<DefaultDualTaskSystem>>> =
            Arc::new(UltDualBarrier::new(N));
        let before = Arc::new(AtomicUsize::new(0));
        let leaders = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..N)
            .map(|_| {
                let b = Arc::clone(&b);
                let before = Arc::clone(&before);
                let leaders = Arc::clone(&leaders);
                spawn_async(async move {
                    before.fetch_add(1, Ordering::SeqCst);
                    let r = StacklessBarrier::wait(&*b).await;
                    assert_eq!(before.load(Ordering::SeqCst), N);
                    if r.is_leader() {
                        leaders.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(leaders.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn dual_flavor_from_both_sync_and_async() {
    run(8, || {
        const NSYNC: usize = 4;
        const NASYNC: usize = 4;
        const N: usize = NSYNC + NASYNC;
        let b: Arc<UltDualBarrier<DefaultDualTaskSystem, DualResumable<DefaultDualTaskSystem>>> =
            Arc::new(UltDualBarrier::new(N));
        let before = Arc::new(AtomicUsize::new(0));
        let leaders = Arc::new(AtomicUsize::new(0));

        let sync_handles: Vec<_> = (0..NSYNC)
            .map(|_| {
                let b = Arc::clone(&b);
                let before = Arc::clone(&before);
                let leaders = Arc::clone(&leaders);
                spawn(move || {
                    before.fetch_add(1, Ordering::SeqCst);
                    let r = StackfulBarrier::wait(&*b);
                    assert_eq!(before.load(Ordering::SeqCst), N);
                    if r.is_leader() {
                        leaders.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        let async_handles: Vec<_> = (0..NASYNC)
            .map(|_| {
                let b = Arc::clone(&b);
                let before = Arc::clone(&before);
                let leaders = Arc::clone(&leaders);
                spawn_async(async move {
                    before.fetch_add(1, Ordering::SeqCst);
                    let r = StacklessBarrier::wait(&*b).await;
                    assert_eq!(before.load(Ordering::SeqCst), N);
                    if r.is_leader() {
                        leaders.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for h in sync_handles {
            h.join().unwrap();
        }
        for h in async_handles {
            h.join().unwrap();
        }
        assert_eq!(leaders.load(Ordering::SeqCst), 1);
    });
}
