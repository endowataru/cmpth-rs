//! End-to-end tests for [`cmpth::DefaultStackfulOnlyTaskSystem`]:
//! `execute`'s dispatch is `execute_stackful` (always a real context switch,
//! no `poll_fn` tag check) rather than `execute_dual`. Nothing in these
//! tests calls `spawn_async` on this system — its dispatch never checks the
//! `poll_fn` tag, so an async task would be mis-handled if one ever landed
//! here — so this exercises exactly the branch-free path
//! `execute_stackful`/`pop_or_root_stackful` were built for.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cmpth::{
    BlockOnSystem, DefaultStackfulOnlyTaskSystem, JoinHandleLike, ScopedStackfulTaskSystem,
    StackfulBuilder, StackfulInitSystem, ThreadSystem,
};

#[test]
fn spawn_join_basic() {
    DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| {
        let h = DefaultStackfulOnlyTaskSystem::spawn(|| 6 * 7);
        assert_eq!(JoinHandleLike::join(h), 42);
    });
}

#[test]
fn spawn_join_many_parallel() {
    let counter = Arc::new(AtomicU64::new(0));
    let counter2 = Arc::clone(&counter);
    DefaultStackfulOnlyTaskSystem::builder().workers(4).run(move || {
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
    DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| {
        let h = DefaultStackfulOnlyTaskSystem::spawn(|| {
            let inner = DefaultStackfulOnlyTaskSystem::spawn(|| 10);
            JoinHandleLike::join(inner) + 5
        });
        assert_eq!(JoinHandleLike::join(h), 15);
    });
}

#[test]
fn spawn_panic_propagates() {
    DefaultStackfulOnlyTaskSystem::builder().workers(1).run(|| {
        let h = DefaultStackfulOnlyTaskSystem::spawn::<(), _>(|| panic!("boom"));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| h.join()));
        assert!(result.is_err() || result.unwrap().is_err());
    });
}

#[test]
fn yield_now_roundtrips() {
    DefaultStackfulOnlyTaskSystem::builder().workers(1).run(|| {
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
    DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| {
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

    DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| {
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

// ---------------------------------------------------------------------------
// Standalone init — StackfulBuilder::init, replacing bracketing `run`
// ---------------------------------------------------------------------------

/// `init()` returns with the caller already running as an ordinary ULT on
/// the pool: `spawn`/`join` work exactly as they do inside `run`, and
/// `parallel_call` — which panics ("called outside `run`" / no live pool)
/// when reached from plain, non-worker code — now works too. That's the
/// headline capability gain standalone init adds over the old bracketing
/// `run`: no closure to wrap arbitrary top-level (e.g. `main`) code in.
#[test]
fn standalone_init_then_spawn_join_and_parallel_call_work() {
    let guard = DefaultStackfulOnlyTaskSystem::builder().workers(2).init();

    let h = DefaultStackfulOnlyTaskSystem::spawn(|| 6 * 7);
    assert_eq!(JoinHandleLike::join(h), 42);

    let (a, b) = DefaultStackfulOnlyTaskSystem::parallel_call(|| 1 + 1, || 2 + 2);
    assert_eq!((a, b), (2, 4));

    // A handful of spawn/join rounds after parallel_call, to make sure the
    // pool is still in a sane state (root_cont/deque bookkeeping) after
    // mixing both capabilities on the same standalone-initialized pool.
    let handles: Vec<_> = (0..20).map(|i| DefaultStackfulOnlyTaskSystem::spawn(move || i * 2u64)).collect();
    let mut sum = 0u64;
    for h in handles {
        sum += JoinHandleLike::join(h);
    }
    assert_eq!(sum, (0..20u64).map(|i| i * 2).sum::<u64>());

    drop(guard);
}

/// `Builder::run` is defined in terms of `init`/`Drop`, wrapped in
/// `catch_unwind`/`resume_unwind` so a panic inside `f` never reaches the
/// guard's `Drop` while unwinding (which would abort — see that `Drop`
/// impl) — instead it's caught before `drop(guard)` runs, and re-raised
/// only afterward, on the caller's own thread. Confirms that whole
/// round-trip actually propagates the panic rather than swallowing it or
/// aborting.
#[test]
fn run_propagates_a_panic_in_f_to_the_caller() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| -> u64 {
            panic!("boom from run's root closure");
        });
    }));
    let err = result.expect_err("run should propagate the panic, not swallow it");
    let msg = err
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| err.downcast_ref::<String>().map(String::as_str))
        .unwrap_or_default();
    assert!(msg.contains("boom from run's root closure"), "unexpected panic payload: {msg:?}");

    // The pool from the panicking `run` call fully tore down (workers
    // joined, TLS cleared) before `run` re-raised -- an entirely new `run`
    // call works normally right after.
    DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| {
        assert_eq!(JoinHandleLike::join(DefaultStackfulOnlyTaskSystem::spawn(|| 1 + 1)), 2);
    });
}
