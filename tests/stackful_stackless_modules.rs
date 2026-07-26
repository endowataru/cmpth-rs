//! Confirms `cmpth::traits::stackful::*` / `cmpth::traits::stackless::*`
//! actually unlock the flavored methods (`.lock()`, `.wait()`, `.is_set()`)
//! with no further imports needed — the whole point of the bulk-import
//! modules (docs/sync-async-unification.md).

use std::sync::Arc;

use cmpth::default::*;
use cmpth::{DualTaskSystem, SuspendedFuture, UltDualBarrier, UltDualMutex};

#[test]
fn stackful_module_unlocks_lock_and_wait() {
    use cmpth::traits::stackful::*;

    run(2, || {
        let m: UltDualMutex<DualTaskSystem, u64, cmpth::BasicSuspendedThread<DualTaskSystem>> =
            UltDualMutex::new(0);
        *m.lock() += 1;
        assert_eq!(*m.lock(), 1);

        let b: Arc<UltDualBarrier<DualTaskSystem, cmpth::BasicSuspendedThread<DualTaskSystem>>> =
            Arc::new(UltDualBarrier::new(1));
        let r = b.wait();
        assert!(r.is_leader());

        // StackfulSystem/ThreadSystem also in scope via the same bulk import.
        assert!(DualTaskSystem::num_workers() >= 1);
    });
}

#[test]
fn stackless_module_unlocks_lock_and_wait() {
    use cmpth::traits::stackless::*;

    run(2, || {
        let h = spawn_async(async {
            let m: UltDualMutex<DualTaskSystem, u64, SuspendedFuture<DualTaskSystem>> =
                UltDualMutex::new(0);
            *m.lock().await += 1;
            let v = *m.lock().await;

            let b: UltDualBarrier<DualTaskSystem, SuspendedFuture<DualTaskSystem>> =
                UltDualBarrier::new(1);
            let r = b.wait().await;
            assert!(r.is_leader());
            v
        });
        assert_eq!(h.join().unwrap(), 1);
    });
}
