//! Exercises the `docs/sync-async-unification.md` prototype: `UltDualMutex<S, T, N>`
//! generic over `BasicSuspendedThread<S>` (sync-only), `SuspendedFuture<S>`
//! (async-only), and `SuspendedTask<S>` (dual).

use std::sync::Arc;

use cmpth::default::*;
use cmpth::traits::{StackfulMutex, StacklessMutex};
use cmpth::{BasicSuspendedThread, DualTaskSystem, SuspendedFuture, SuspendedTask, ThreadSystem, UltDualMutex};

#[test]
fn sync_only_flavor() {
    run(4, || {
        let m: Arc<UltDualMutex<DualTaskSystem, u64, BasicSuspendedThread<DualTaskSystem>>> =
            Arc::new(UltDualMutex::new(0));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let m = Arc::clone(&m);
                spawn(move || {
                    for _ in 0..50 {
                        *StackfulMutex::lock(&*m) += 1;
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(*StackfulMutex::lock(&*m), 400);
    });
}

#[test]
fn async_only_flavor() {
    // High contention (many tasks, few workers, many iterations each) is
    // deliberate: an earlier version of StacklessResumable::wait_with's default
    // impl registered the waker *after* publishing the MCS link instead of
    // before, opening a lost-wakeup race that a low-contention run could run
    // thousands of times without ever hitting. This shape reproduced an
    // hour-long hang within a handful of runs under `cargo test --all`'s
    // parallel scheduling; keep the contention high so a regression is
    // caught quickly again rather than passing by luck.
    run(4, || {
        let m: Arc<UltDualMutex<DualTaskSystem, u64, SuspendedFuture<DualTaskSystem>>> =
            Arc::new(UltDualMutex::new(0));
        let handles: Vec<_> = (0..32)
            .map(|_| {
                let m = Arc::clone(&m);
                spawn_async(async move {
                    for _ in 0..200 {
                        *StacklessMutex::lock(&*m).await += 1;
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let total = DualTaskSystem::block_on(async { *StacklessMutex::lock(&*m).await });
        assert_eq!(total, 32 * 200);
    });
}

#[test]
fn dual_flavor_from_both_sync_and_async() {
    // The key claim under test: one mutex instance, contended simultaneously
    // by real spawned ULTs (via StackfulMutex) and spawn_async tasks (via
    // StacklessMutex), stays correct. Fully-qualified calls throughout since
    // SuspendedTask implements both StackfulResumable and StacklessResumable, so
    // `.lock()` alone would be ambiguous if both traits were `use`d.
    run(4, || {
        let m: Arc<UltDualMutex<DualTaskSystem, u64, SuspendedTask<DualTaskSystem>>> =
            Arc::new(UltDualMutex::new(0));

        let sync_handles: Vec<_> = (0..16)
            .map(|_| {
                let m = Arc::clone(&m);
                spawn(move || {
                    for _ in 0..100 {
                        *StackfulMutex::lock(&*m) += 1;
                    }
                })
            })
            .collect();

        let async_handles: Vec<_> = (0..16)
            .map(|_| {
                let m = Arc::clone(&m);
                spawn_async(async move {
                    for _ in 0..100 {
                        *StacklessMutex::lock(&*m).await += 1;
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
        assert_eq!(*StackfulMutex::lock(&*m), 16 * 100 * 2);
    });
}

#[test]
fn sync_wait_from_inside_async_poll_panics_cleanly() {
    // Calling the StackfulResumable-flavored operation from inside a
    // spawn_async task's poll() must be caught, not silently corrupt the
    // worker's own dispatch-loop stack. This exercises the cur_task.is_root
    // guard added in place of an explicit OnUlt token
    // (docs/sync-async-unification.md). Like spawn_async_panic_propagates,
    // the panic is caught by the scheduler and surfaces via `join()`
    // returning `Err`, not as a test-thread unwind.
    use cmpth::traits::StackfulResumable;
    run(1, || {
        let h = spawn_async(async {
            let slot: SuspendedTask<DualTaskSystem> = Default::default();
            // Misuse: this must panic, not attempt a real context switch on
            // top of run_async_poll's shared stack.
            StackfulResumable::wait_with(&slot, || {});
        });
        let err = h.join().expect_err("expected the cur_task.is_root guard to panic");
        let msg = err
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| err.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("");
        assert!(msg.contains("outside a real ULT"), "unexpected panic message: {msg:?}");
    });
}
