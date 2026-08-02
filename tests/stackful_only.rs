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

// ---------------------------------------------------------------------------
// block_on — exercises ResumablePoller's actual park/wake path, not just the
// always-ready case (see `traits::stackful::ThreadSystem::block_on`'s
// doctest, which only ever polls `async { 6 * 7 }` once and never parks).
// ---------------------------------------------------------------------------

/// Future that yields exactly once before becoming ready, notifying itself
/// *during* the same `poll()` call that returns `Pending` — exercises
/// `ResumablePollerSlot`'s "wake raced in before park committed" cancel
/// path (`decide_park`'s NOTIFIED branch), the same race
/// `block_on_yield_once` (tests/integration.rs) checks for `DefaultDualTaskSystem`.
struct YieldOnce(bool);

impl std::future::Future for YieldOnce {
    type Output = u32;
    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<u32> {
        if self.0 {
            std::task::Poll::Ready(42)
        } else {
            self.0 = true;
            cx.waker().wake_by_ref(); // notify immediately; executor must handle the race
            std::task::Poll::Pending
        }
    }
}

#[test]
fn block_on_yield_once() {
    DefaultStackfulOnlyTaskSystem::run(2, || {
        let v = DefaultStackfulOnlyTaskSystem::block_on(YieldOnce(false));
        assert_eq!(v, 42);
    });
}

/// Future that is genuinely parked, then woken from a *different* ULT via a
/// cloned waker — exercises `ResumablePollerSlot`'s real PARKED ->
/// `ClaimedParked` -> `push_continuation` path (the `Arc`-backed clone must
/// stay valid across the hand-off to the waking ULT). Mirrors
/// `block_on_cross_ult_wake` (tests/integration.rs) for
/// `DefaultStackfulOnlyTaskSystem`.
#[test]
fn block_on_cross_ult_wake() {
    use std::sync::atomic::{AtomicBool, Ordering as Ord};
    use std::sync::Mutex;
    use std::task::Waker;

    struct WaitForWake {
        slot: Arc<Mutex<Option<Waker>>>,
        done: Arc<AtomicBool>,
    }
    impl std::future::Future for WaitForWake {
        type Output = ();
        fn poll(
            self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<()> {
            if self.done.load(Ord::Acquire) {
                return std::task::Poll::Ready(());
            }
            *self.slot.lock().unwrap() = Some(cx.waker().clone());
            std::task::Poll::Pending
        }
    }

    DefaultStackfulOnlyTaskSystem::run(2, || {
        let slot: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));
        let done: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

        let slot2 = Arc::clone(&slot);
        let done2 = Arc::clone(&done);
        let waker_h = DefaultStackfulOnlyTaskSystem::spawn(move || {
            loop {
                let w = slot2.lock().unwrap().take();
                if let Some(w) = w {
                    done2.store(true, Ord::Release); // must happen before wake()
                    w.wake();
                    break;
                }
                DefaultStackfulOnlyTaskSystem::yield_now();
            }
        });

        DefaultStackfulOnlyTaskSystem::block_on(WaitForWake { slot, done });
        JoinHandleLike::join(waker_h);
    });
}
