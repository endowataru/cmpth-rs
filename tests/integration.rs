use cmpth::*;
// Sync traits are not at crate root (name would clash with the type aliases).
// Import them under aliases so that methods like lock(), wait(), notify_one()
// resolve correctly, and for explicit UFCS in generic tests.
use cmpth::traits::StackfulMutex;
use cmpth::traits::StackfulBarrier;

mod common;
use common::*;

// Shared static ULT-local slot used by the UltTls tests below.
// Key is lazily assigned once; each new scheduler run gives each ULT a fresh
// TLS map, so there is no cross-test interference.
static ULT_LOCAL: UltTls<DefaultDualTaskSystem, u64> = UltTls::new();

#[test]
fn create_and_join() {
    run(2, || {
        let h = spawn(|| 42u64);
        assert_eq!(h.join().unwrap(), 42);
    });
}

#[test]
fn many_threads() {
    run(4, || {
        let handles: Vec<_> = (0..100).map(|i| spawn(move || i * 2u64)).collect();
        let mut sum = 0u64;
        for h in handles {
            sum += h.join().unwrap();
        }
        assert_eq!(sum, (0..100u64).map(|i| i * 2).sum::<u64>());
    });
}

fn fib(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let h = spawn(move || fib(n - 1));
    let r2 = fib(n - 2);
    h.join().unwrap() + r2
}

#[test]
fn parallel_fib() {
    run(4, || {
        assert_eq!(fib(10), 55);
    });
}

#[test]
fn yield_roundtrip() {
    run(2, || {
        for _ in 0..100 {
            yield_now();
        }
    });
}

#[test]
fn mutex_stress() {
    run(4, || {
        use std::sync::Arc;
        let m = Arc::new(Mutex::new(0u64));
        let handles: Vec<_> = (0..100)
            .map(|_| {
                let m = Arc::clone(&m);
                spawn(move || {
                    for _ in 0..100 {
                        *m.lock() += 1;
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(*m.lock(), 10_000);
    });
}

#[test]
fn condvar_notify() {
    run(2, || {
        use std::sync::Arc;
        let pair = Arc::new((Mutex::new(false), Condvar::new()));
        let pair2 = Arc::clone(&pair);
        let h = spawn(move || {
            let (lock, cvar) = &*pair2;
            let mut ready = lock.lock();
            *ready = true;
            cvar.notify_one();
        });
        let (lock, cvar) = &*pair;
        let mut ready = lock.lock();
        while !*ready {
            ready = cvar.wait(ready);
        }
        drop(ready);
        h.join().unwrap();
    });
}

#[test]
fn barrier_sync() {
    run(4, || {
        use std::sync::Arc;
        let b = Arc::new(Barrier::new(10));
        let counter = Arc::new(Mutex::new(0u32));
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let b = Arc::clone(&b);
                let c = Arc::clone(&counter);
                spawn(move || {
                    *c.lock() += 1;
                    b.wait();
                    assert_eq!(*c.lock(), 10);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    });
}

#[test]
fn task_panic_propagates() {
    run(2, || {
        let h = spawn(|| panic!("boom"));
        assert!(h.join().is_err());
    });
}

#[test]
fn suspended_thread_cancel() {
    run(2, || {
        let sth = BasicStackfulOnlyResumable::<DefaultDualTaskSystem>::new();
        sth.wait_with_cond(|| false);
        assert!(!sth.is_set());
    });
}

// ---------------------------------------------------------------------------
// Nesting
// ---------------------------------------------------------------------------

#[test]
fn nested_spawn_join() {
    run(2, || {
        <DefaultNestedDualTaskSystem as StackfulInitSystem>::builder().workers(2).run(|| {
            let handles: Vec<_> = (0..50)
                .map(|i| <DefaultNestedDualTaskSystem as SpawnableStackfulTaskSystem>::spawn(move || i * 3u64))
                .collect();
            let mut sum = 0u64;
            for h in handles {
                sum += JoinHandleLike::join(h);
            }
            assert_eq!(sum, (0..50u64).map(|i| i * 3).sum::<u64>());
        });
    });
}

#[test]
fn nested_mutex() {
    run(2, || {
        <DefaultNestedDualTaskSystem as StackfulInitSystem>::builder().workers(2).run(|| {
            use std::sync::Arc;
            use cmpth::traits::StackfulMutex;
            type M = <DefaultNestedDualTaskSystem as StackfulSyncSystem>::Mutex<u64>;
            let m = Arc::new(<M as StackfulMutex<u64>>::new(0));
            let handles: Vec<_> = (0..20)
                .map(|_| {
                    let m = Arc::clone(&m);
                    <DefaultNestedDualTaskSystem as SpawnableStackfulTaskSystem>::spawn(move || {
                        for _ in 0..50 {
                            *m.lock() += 1;
                        }
                    })
                })
                .collect();
            for h in handles {
                JoinHandleLike::join(h);
            }
            assert_eq!(*m.lock(), 1000);
        });
    });
}

fn generic_workload<S: SpawnableStackfulTaskSystem + StackfulSyncSystem>() -> u64 {
    use std::sync::Arc;
    use cmpth::traits::StackfulMutex;
    let m = Arc::new(<S::Mutex<u64> as StackfulMutex<u64>>::new(0));
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let m = Arc::clone(&m);
            S::spawn(move || {
                for _ in 0..10 {
                    *m.lock() += 1;
                }
            })
        })
        .collect();
    for h in handles {
        JoinHandleLike::join(h);
    }
    *m.lock()
}

#[test]
fn generic_over_layers() {
    assert_eq!(generic_workload::<OsSystem>(), 80);
    run(2, || {
        assert_eq!(generic_workload::<DefaultDualTaskSystem>(), 80);
        <DefaultNestedDualTaskSystem as StackfulInitSystem>::builder().workers(2).run(|| {
            assert_eq!(generic_workload::<DefaultNestedDualTaskSystem>(), 80);
        });
    });
}

/// Nested standalone init: a second system ([`DefaultNestedDualTaskSystem`],
/// `Base = DefaultDualTaskSystem`) initialized *inside* the outer system's
/// pool, mirroring the existing nesting tests above (`nested_spawn_join`/
/// `nested_mutex`/`generic_over_layers`) but with `init()`/`Drop` instead of
/// bracketing `run`. The outer ULT calling `init` (and later dropping the
/// guard) is itself just an ordinary task on the *outer* pool, so this also
/// exercises `init`'s fork/suspend machinery running one level down from
/// the OS thread `run`'s own worker 0 would otherwise occupy.
#[test]
fn nested_standalone_init() {
    run(2, || {
        let guard = <DefaultNestedDualTaskSystem as StackfulInitSystem>::builder().workers(2).init();

        let handles: Vec<_> = (0..30)
            .map(|i| <DefaultNestedDualTaskSystem as SpawnableStackfulTaskSystem>::spawn(move || i * 3u64))
            .collect();
        let mut sum = 0u64;
        for h in handles {
            sum += JoinHandleLike::join(h);
        }
        assert_eq!(sum, (0..30u64).map(|i| i * 3).sum::<u64>());

        drop(guard);

        // The outer pool is unaffected by the inner one tearing down.
        assert_eq!(JoinHandleLike::join(spawn(|| 6 * 7)), 42);
    });
}

#[test]
fn detach_before_finish() {
    use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
    // Spawn a task, drop the handle (detach), verify the task still runs to
    // completion and its resources are freed (no leak, no crash).
    let done = Arc::new(AtomicBool::new(false));
    let done2 = Arc::clone(&done);
    run(2, move || {
        let h = spawn(move || {
            done2.store(true, Ordering::Release);
        });
        drop(h); // detach
        // Yield until the detached task finishes.
        while !done.load(Ordering::Acquire) {
            yield_now();
        }
    });
}

#[test]
fn detach_after_finish() {
    // Drop the handle after the task has already completed.
    run(2, || {
        let h = spawn(|| 99u64);
        // Let the child run to completion before dropping the handle.
        yield_now();
        drop(h); // task may already be finished; handle must not leak
    });
}

#[test]
fn mcs_mutex_basic() {
    use cmpth::McsMutex;
    run(4, || {
        let m = std::sync::Arc::new(McsMutex::<DefaultDualTaskSystem, u64>::new(0));
        let handles: Vec<_> = (0..8).map(|_| {
            let m = std::sync::Arc::clone(&m);
            spawn(move || { *m.lock() += 1; })
        }).collect();
        for h in handles { h.join().unwrap(); }
        assert_eq!(*m.lock(), 8);
    });
}

// ---------------------------------------------------------------------------
// block_on / async waker tests
// ---------------------------------------------------------------------------

/// Future that yields exactly once before becoming ready.
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
    // block_on should park, get woken (immediately by wake_by_ref), and re-poll to Ready.
    run(2, || {
        let v = DefaultDualTaskSystem::block_on(YieldOnce(false));
        assert_eq!(v, 42);
    });
}

#[test]
fn block_on_without_worker_busy_polls() {
    // No `current()` worker: block_on falls back to OsPoller's busy-poll,
    // which re-polls regardless of the waker.
    let v = OsSystem::block_on(YieldOnce(false));
    assert_eq!(v, 42);
}

#[test]
fn block_on_already_ready() {
    run(1, || {
        let v = DefaultDualTaskSystem::block_on(async { 99u32 });
        assert_eq!(v, 99);
    });
}

/// Future that is woken from another ULT via a cloned waker.
#[test]
fn block_on_cross_ult_wake() {
    use std::sync::atomic::{AtomicBool, Ordering as Ord};
    use std::sync::{Arc, Mutex};
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
            // Register waker and park.
            *self.slot.lock().unwrap() = Some(cx.waker().clone());
            std::task::Poll::Pending
        }
    }

    run(2, || {
        let slot: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));
        let done: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

        let slot2 = Arc::clone(&slot);
        let done2 = Arc::clone(&done);
        let waker_h = spawn(move || {
            loop {
                let w = slot2.lock().unwrap().take();
                if let Some(w) = w {
                    done2.store(true, Ord::Release); // must happen before wake()
                    w.wake();
                    break;
                }
                <DefaultDualTaskSystem as SpawnableStackfulTaskSystem>::yield_now();
            }
        });

        DefaultDualTaskSystem::block_on(WaitForWake { slot, done });
        waker_h.join().unwrap();
    });
}

/// `JoinHandle` as `Future`: await a spawned ULT from inside `block_on`.
#[test]
fn join_handle_as_future() {
    run(2, || {
        let v = DefaultDualTaskSystem::block_on(async {
            let h = spawn(|| 42u64);
            h.await
        });
        assert_eq!(v, 42);
    });
}

/// Drop a JoinHandle that had a waker registered (Future polled once, then dropped).
#[test]
fn join_handle_future_drop_mid_wait() {
    use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
    let done = Arc::new(AtomicBool::new(false));
    let done2 = Arc::clone(&done);
    let done3 = Arc::clone(&done);
    run(2, move || {
        let h = spawn(move || {
            yield_now(); // ensure child doesn't finish before first poll
            done2.store(true, Ordering::Release);
        });
        // Poll once (registers waker), then drop the future — should detach cleanly.
        DefaultDualTaskSystem::block_on(async {
            let mut h = std::pin::pin!(h);
            let _ = std::future::poll_fn(|cx| {
                // Drive one poll to register the async joiner, then return Ready
                // so block_on exits — the JoinHandle is still pending.
                let _ = h.as_mut().poll(cx);
                std::task::Poll::Ready(())
            }).await;
            // h drops here with an async waker registered
        });
        // Task must still complete and free itself.
        while !done3.load(Ordering::Acquire) {
            yield_now();
        }
    });
}

/// External OS thread wakes a parked ULT via `ExternalQueue`.
#[test]
fn block_on_external_thread_wake() {
    use std::sync::{Arc, Mutex};
    use std::task::Waker;

    struct WaitForExternalWake {
        slot: Arc<Mutex<Option<Waker>>>,
        ready: Arc<std::sync::atomic::AtomicBool>,
    }
    impl std::future::Future for WaitForExternalWake {
        type Output = u32;
        fn poll(
            self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<u32> {
            if self.ready.load(std::sync::atomic::Ordering::Acquire) {
                return std::task::Poll::Ready(7);
            }
            *self.slot.lock().unwrap() = Some(cx.waker().clone());
            std::task::Poll::Pending
        }
    }

    let slot: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));
    let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let slot2 = Arc::clone(&slot);
    let ready2 = Arc::clone(&ready);

    // Spawn an OS thread BEFORE run() so it has no scheduler affinity.
    let os_thread = std::thread::spawn(move || {
        // Wait until the ULT registers its waker.
        loop {
            let w = slot2.lock().unwrap().take();
            if let Some(w) = w {
                ready2.store(true, std::sync::atomic::Ordering::Release);
                w.wake();  // Called from outside the scheduler.
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    run(2, move || {
        let v = DefaultDualTaskSystem::block_on(WaitForExternalWake { slot, ready });
        assert_eq!(v, 7);
    });

    os_thread.join().unwrap();
}

// ---------------------------------------------------------------------------
// UltTls (ThreadSpecific) — per-ULT isolation
// ---------------------------------------------------------------------------

#[test]
fn ult_tls_per_ult_isolation() {
    // Each ULT stores a pointer to its own stack variable in the shared static
    // slot, yields repeatedly (allowing work-stealing to migrate the ULT across
    // OS threads), then reads the value back.  The value must be the ULT's own,
    // not that of any OS thread or sibling ULT.
    run(4, || {
        let handles: Vec<_> = (0u64..20).map(|i| {
            spawn(move || {
                let mut val: u64 = i;
                ULT_LOCAL.set(&mut val as *mut u64);
                for _ in 0..10 {
                    yield_now(); // may migrate to a different OS thread
                }
                let got = unsafe { *ULT_LOCAL.get() };
                assert_eq!(got, i, "ULT-local value corrupted after yield");
                ULT_LOCAL.set(std::ptr::null_mut());
            })
        }).collect();
        for h in handles { h.join().unwrap(); }
    });
}

// ---------------------------------------------------------------------------
// ReturnPool cross-worker return
// ---------------------------------------------------------------------------

#[test]
fn return_pool_cross_worker() {
    // Tasks yield once before returning so work-stealing can execute them on a
    // different worker than the one that allocated their descriptor.
    // ReturnPool must stage the descriptor in the allocating worker's remote
    // mailbox and flush it at the threshold.  A second phase re-spawns the same
    // count to exercise reuse of the returned descriptors.
    run(4, || {
        for phase in 0u64..2 {
            let handles: Vec<_> = (0u64..200).map(|i| {
                spawn(move || {
                    yield_now(); // encourage cross-worker migration before exit
                    phase * 1000 + i
                })
            }).collect();
            let sum: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
            let expected: u64 = (0..200).map(|i| phase * 1000 + i).sum();
            assert_eq!(sum, expected);
        }
    });
}

// ---------------------------------------------------------------------------
// spawn_async detach tests
// ---------------------------------------------------------------------------

#[test]
fn spawn_async_detach_before_finish() {
    // Drop the JoinHandle while the async task is still pending (first poll
    // not yet done).  The task must complete and free itself via the detach path.
    use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
    use std::future::poll_fn;

    let done = Arc::new(AtomicBool::new(false));
    let done2 = Arc::clone(&done);

    run(2, move || {
        let h = spawn_async(async move {
            // Yield once so the parent can drop the handle while we are pending.
            let mut yielded = false;
            poll_fn(|cx| {
                if yielded {
                    std::task::Poll::Ready(())
                } else {
                    yielded = true;
                    cx.waker().wake_by_ref();
                    std::task::Poll::Pending
                }
            }).await;
            done2.store(true, Ordering::Release);
        });
        drop(h); // detach: has_handle → false, joiner cleared
        while !done.load(Ordering::Acquire) {
            yield_now();
        }
    });
}

#[test]
fn spawn_async_detach_after_finish() {
    // Drop the JoinHandle after the async task has already set finished=true.
    // JoinHandle::drop must call result_drop and UltDesc::free (not the pool).
    use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

    let done = Arc::new(AtomicBool::new(false));
    let done2 = Arc::clone(&done);

    run(2, move || {
        let h = spawn_async(async move {
            done2.store(true, Ordering::Release);
            99u32
        });
        while !done.load(Ordering::Acquire) {
            yield_now();
        }
        drop(h); // finished=true branch in JoinHandle::drop
    });
}

// ---------------------------------------------------------------------------
// spawn_async tests
// ---------------------------------------------------------------------------

#[test]
fn spawn_async_immediate() {
    // Future that is immediately ready.
    run(2, || {
        let h = spawn_async(async { 99u64 });
        assert_eq!(h.join().unwrap(), 99);
    });
}

#[test]
fn spawn_async_yield() {
    // Future that yields once before completing.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    run(4, || {
        let flag = Arc::new(AtomicBool::new(false));
        let flag2 = Arc::clone(&flag);
        let h = spawn_async(async move {
            <DefaultDualTaskSystem as StacklessTaskSystem>::yield_now().await;
            flag2.store(true, Ordering::Release);
            42u32
        });
        assert_eq!(h.join().unwrap(), 42);
        assert!(flag.load(Ordering::Acquire));
    });
}

#[test]
fn spawn_async_many() {
    run(4, || {
        let handles: Vec<_> = (0u64..100).map(|i| spawn_async(async move { i * 3 })).collect();
        let sum: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(sum, (0..100u64).map(|i| i * 3).sum::<u64>());
    });
}

#[test]
fn spawn_async_nested() {
    // A spawn_async task itself spawning (and awaiting) a nested spawn_async
    // task, under execute_dual dispatch — the async-task mirror of
    // `nested_spawn_join`. Nested spawn must use the raw `StacklessTaskSystem`
    // trait method (not the blocking `spawn_async` helper, which calls
    // `block_on` and is only meant to be invoked from stackful ULT context).
    run(2, || {
        let h = spawn_async(async {
            let inner = <DefaultDualTaskSystem as StacklessTaskSystem>::spawn(|| async { 10u64 }).await;
            inner.await + 5
        });
        assert_eq!(h.join().unwrap(), 15);
    });
}

#[test]
fn spawn_async_join_handle_as_future() {
    // Await a spawn_async JoinHandle from within block_on.
    run(2, || {
        let h = spawn_async(async { 7u32 });
        let v = DefaultDualTaskSystem::block_on(h);
        assert_eq!(v, 7);
    });
}

#[test]
fn spawn_async_panic_propagates() {
    run(2, || {
        let h = spawn_async(async { panic!("async task panic") as u32 });
        assert!(h.join().is_err());
    });
}

/// Regression test: floating-point registers live across a suspension point.
///
/// AAPCS64 makes the lower halves of v8-v15 callee-saved, so a value the
/// compiler keeps in one of them must survive a context switch.  Each ULT
/// carries enough independent f64 state to occupy several registers, yields
/// repeatedly, and the result is compared against the same computation run
/// without any yielding.
#[test]
fn float_regs_survive_yield() {
    fn crunch(k: usize, yield_each_step: bool) -> f64 {
        let mut acc = [
            0.5 + k as f64,
            1.5 * (k + 1) as f64,
            2.25 + (k as f64) * 0.125,
            3.75 - (k as f64) * 0.0625,
            4.125 + (k as f64) * 2.0,
            5.0625 - (k as f64) * 0.5,
            6.03125 + (k as f64) * 0.25,
            7.015625 - (k as f64) * 0.125,
        ];
        for i in 0..500u64 {
            let x = (i as f64).mul_add(1.000001, 0.5);
            for (j, a) in acc.iter_mut().enumerate() {
                *a = a.mul_add(1.0000001, x * (j as f64 + 1.0) * 1e-9);
            }
            if yield_each_step {
                yield_now();
            }
        }
        acc.iter().sum()
    }

    let expected: Vec<f64> = (0..8).map(|k| crunch(k, false)).collect();
    run(2, move || {
        let handles: Vec<_> = (0..8)
            .map(|k| spawn(move || crunch(k, true)))
            .collect();
        for (k, h) in handles.into_iter().enumerate() {
            assert_eq!(h.join().unwrap(), expected[k], "ULT {k} float state corrupted");
        }
    });
}

// ---------------------------------------------------------------------------
// SpawnableStackfulTaskSystem implemented by hand (no UltIdentity blanket)
// ---------------------------------------------------------------------------

/// `UltIdentity`'s blanket impl is convenience, not architecture: everything
/// it generates can be written as a plain trait impl, as this does. The
/// only part that cannot be defaulted away on stable Rust is the
/// per-system TLS static (generic statics do not exist; a static inside a
/// default trait method would be shared across ALL systems, breaking
/// nested schedulers).
struct ManualSystem;

impl cmpth::PoolSystem for ManualSystem {
    type Desc  = DualTaskDesc<Self>;
    type ExternalQueue   = StealPathQueue<DualTaskDesc<Self>>;
    type Pool            = ReturnPool<DualTaskDesc<Self>, HeapStack>;
    // Unused: ManualSystem never calls spawn_async.
    type AsyncPool       = cmpth::resumable::common::pool::SimplePool<DualTaskDesc<Self>>;
    const ASYNC_POOL_SIZE: usize = 0;
    // Unused: ManualSystem never calls recurse.
    type RecursionPool   = cmpth::resumable::common::pool::ThresholdPool<cmpth::resumable::common::pool::BlockPool>;
}

impl cmpth::WorkerSystem for ManualSystem {
    type Base  = OsSystem;
    type SuspendedToken  = cmpth::SuspendedTaskToken<DualTaskDesc<Self>>;
    type Worker = UltWorker<Self>;
    type RunQueue = HybridRunQueue<cmpth::SuspendedTaskToken<DualTaskDesc<Self>>>;
    type Lookup          = TlsCurrent;

    fn worker_tls() -> &'static <OsSystem as cmpth::NestableSystem>::ThreadSpecific<UltWorker<Self>> {
        // The one thing a macro (or the user, as here) must write:
        // a distinct static per system, anchored in this fn body.
        static TLS: OsTls<UltWorker<ManualSystem>> =
            <OsTls<UltWorker<ManualSystem>> as TlsSlot<UltWorker<ManualSystem>>>::INIT;
        &TLS
    }
}

// `SchedulerSystem for ManualSystem` is no longer hand-written: it is
// blanket-derived (`cmpth::resumable::common::system`) from the
// `RunnableItem`/`ReclaimableDesc` impls for `DualTaskDesc<Self>`.

impl cmpth::StackfulWorkerSystem for ManualSystem {
    type Ctx   = NativeContext;
    type StackAlloc = HeapStack;
    const STACK_SIZE: usize = 64 * 1024;

    type SuspendedThread = BasicStackfulOnlyResumable<Self>;
}

impl SpawnableStackfulTaskSystem for ManualSystem {
    fn yield_now() {
        use cmpth::resumable::common::worker::WorkerOps;
        use cmpth::resumable::stackful::worker::StackfulWorker;
        match UltWorker::<Self>::current() {
            Some(wk) => { wk.yield_now(); }
            None => <OsSystem as SpawnableStackfulTaskSystem>::yield_now(),
        }
    }

    type JoinHandle<T: Send + 'static> = cmpth::resumable::common::thread::JoinHandle<Self, T>;

    fn spawn<T, F>(f: F) -> cmpth::resumable::common::thread::JoinHandle<Self, T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        cmpth::resumable::stackful::thread::spawn::<Self, T, F>(f)
    }
}

impl BlockOnSystem for ManualSystem {
    type Poller = cmpth::resumable::stackful::waker::UltPoller<Self>;
}

impl StackfulSyncSystem for ManualSystem {
    type Mutex<T: Send> = cmpth::McsMutex<Self, T>;
    type Barrier        = cmpth::resumable::stackful::sync::Barrier<Self>;
}

impl SuspendableSystem for ManualSystem {
    type SuspendedThread = BasicStackfulOnlyResumable<Self>;
}

impl DelegationSystem for ManualSystem {
    type Delegator<C: cmpth::DelegatorConsumer<Self>> =
        cmpth::resumable::stackful::sync::McsDelegator<Self, C>;
}

impl NestableSystem for ManualSystem {
    type ThreadSpecific<T: 'static> = cmpth::resumable::stackful::tls::UltTls<Self, T>;
}

#[test]
fn manual_impl_without_macro() {
    <ManualSystem as StackfulInitSystem>::builder().workers(2).run(|| {
        let h = <ManualSystem as SpawnableStackfulTaskSystem>::spawn(|| 6 * 7u64);
        assert_eq!(JoinHandleLike::join(h), 42);
    });
}

// ---------------------------------------------------------------------------
// PollerUltQueue — nothing else in the crate ever instantiates this
// `ExternalQueue` (no default system, no other test sets
// `type ExternalQueue = PollerUltQueue<..>`), so this is its only exercise.
// ---------------------------------------------------------------------------

/// Same shape as `ManualSystem`, `StackfulOnlyTaskDesc`-based like
/// `DefaultStackfulOnlyTaskSystem`, except `ExternalQueue` is
/// [`PollerUltQueue`] instead of the default [`StealPathQueue`].
struct PollerSystem;

impl cmpth::PoolSystem for PollerSystem {
    type Desc  = StackfulOnlyTaskDesc<Self>;
    type ExternalQueue   = PollerUltQueue<StackfulOnlyTaskDesc<Self>>;
    type Pool            = ReturnPool<StackfulOnlyTaskDesc<Self>, HeapStack>;
    // Unused: PollerSystem never calls spawn_async.
    type AsyncPool       = cmpth::resumable::common::pool::SimplePool<StackfulOnlyTaskDesc<Self>>;
    const ASYNC_POOL_SIZE: usize = 0;
    // Unused: PollerSystem never calls recurse.
    type RecursionPool   = cmpth::resumable::common::pool::ThresholdPool<cmpth::resumable::common::pool::BlockPool>;
}

impl cmpth::WorkerSystem for PollerSystem {
    type Base  = OsSystem;
    type SuspendedToken  = cmpth::SuspendedTaskToken<StackfulOnlyTaskDesc<Self>>;
    type Worker = UltWorker<Self>;
    type RunQueue = HybridRunQueue<cmpth::SuspendedTaskToken<StackfulOnlyTaskDesc<Self>>>;
    type Lookup          = TlsCurrent;

    fn worker_tls() -> &'static <OsSystem as cmpth::NestableSystem>::ThreadSpecific<UltWorker<Self>> {
        static TLS: OsTls<UltWorker<PollerSystem>> =
            <OsTls<UltWorker<PollerSystem>> as TlsSlot<UltWorker<PollerSystem>>>::INIT;
        &TLS
    }
}

// `SchedulerSystem for PollerSystem` is likewise blanket-derived, from the
// `RunnableItem`/`ReclaimableDesc` impls for `StackfulOnlyTaskDesc<Self>`.

impl cmpth::StackfulWorkerSystem for PollerSystem {
    type Ctx   = NativeContext;
    type StackAlloc = HeapStack;
    const STACK_SIZE: usize = 64 * 1024;

    type SuspendedThread = BasicStackfulOnlyResumable<Self>;
}

impl SpawnableStackfulTaskSystem for PollerSystem {
    fn yield_now() {
        use cmpth::resumable::common::worker::WorkerOps;
        use cmpth::resumable::stackful::worker::StackfulWorker;
        match UltWorker::<Self>::current() {
            Some(wk) => { wk.yield_now(); }
            None => <OsSystem as SpawnableStackfulTaskSystem>::yield_now(),
        }
    }

    type JoinHandle<T: Send + 'static> = cmpth::resumable::common::thread::JoinHandle<Self, T>;

    fn spawn<T, F>(f: F) -> cmpth::resumable::common::thread::JoinHandle<Self, T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        cmpth::resumable::stackful::thread::spawn::<Self, T, F>(f)
    }
}

impl BlockOnSystem for PollerSystem {
    // `StackfulOnlyTaskDesc` doesn't implement `WakerTaskDescCore` (only
    // `DualTaskDesc`/`StacklessOnlyTaskDesc` do), so `UltPoller` (as
    // `ManualSystem` above uses) isn't available here — `ResumablePoller`
    // is the stackful-only-descriptor poller, same as
    // `UltIdentity`'s own blanket `BlockOnSystem` impl uses.
    type Poller = cmpth::resumable::stackful::waker::ResumablePoller<Self>;
}

impl StackfulSyncSystem for PollerSystem {
    type Mutex<T: Send> = cmpth::McsMutex<Self, T>;
    type Barrier        = cmpth::resumable::stackful::sync::Barrier<Self>;
}

impl SuspendableSystem for PollerSystem {
    type SuspendedThread = BasicStackfulOnlyResumable<Self>;
}

impl DelegationSystem for PollerSystem {
    type Delegator<C: cmpth::DelegatorConsumer<Self>> =
        cmpth::resumable::stackful::sync::McsDelegator<Self, C>;
}

impl NestableSystem for PollerSystem {
    type ThreadSpecific<T: 'static> = cmpth::resumable::stackful::tls::UltTls<Self, T>;
}

/// External OS thread wakes a ULT parked in `block_on`, on a system whose
/// `ExternalQueue` is `PollerUltQueue` — mirrors `block_on_external_thread_wake`
/// above (same `WaitForExternalWake` future, same "OS thread spawned before
/// `run()` so it has no scheduler affinity" setup), but the delivery
/// mechanism underneath is entirely different.
///
/// Whether this actually distinguishes poller-delivery from steal-path
/// delivery: yes, structurally, not just by observation. `PollerUltQueue::
/// try_pop` (see `src/resumable/common/external_queue.rs`) unconditionally
/// returns `None` — a worker's steal-fail path can *never* observe or drain
/// anything sitting in a `PollerUltQueue`, unlike `StealPathQueue` where
/// that path is the only consumer. The sole way anything ever leaves a
/// `PollerUltQueue` is `run_service`'s loop calling `UltWorker::defer` —
/// and `run_service` only ever runs if `init` actually spawned it
/// (`NEEDS_SERVICE`) and keeps running until `StackfulInit::drop` calls
/// `stop_service`/joins it. So if the spawn/join wiring in `init.rs` were
/// missing, wrong, or the spawned task never actually reached
/// `run_service`, the `w.wake()` call below would push into a queue with
/// no reader: the parked ULT would never be resumed, `block_on` would never
/// return, and this test would hang rather than pass. That hang (not a
/// silent pass) is what proves the poller — and only the poller — is what
/// delivers the wake on this system, whereas the exact same test body
/// against `DefaultDualTaskSystem`/`ManualSystem`
/// (`StealPathQueue`-backed) passes for a structurally different reason
/// (a worker's own steal-fail `try_pop()`).
#[test]
fn poller_ult_queue_external_thread_wake() {
    use std::sync::{Arc, Mutex};
    use std::task::Waker;

    struct WaitForExternalWake {
        slot: Arc<Mutex<Option<Waker>>>,
        ready: Arc<std::sync::atomic::AtomicBool>,
    }
    impl std::future::Future for WaitForExternalWake {
        type Output = u32;
        fn poll(
            self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<u32> {
            if self.ready.load(std::sync::atomic::Ordering::Acquire) {
                return std::task::Poll::Ready(7);
            }
            *self.slot.lock().unwrap() = Some(cx.waker().clone());
            std::task::Poll::Pending
        }
    }

    let slot: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));
    let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let slot2 = Arc::clone(&slot);
    let ready2 = Arc::clone(&ready);

    // Spawn an OS thread BEFORE run() so it has no scheduler affinity —
    // `UltWorker::<PollerSystem>::current()` is `None` on it, which is
    // exactly what routes `w.wake()` through `push_continuation`'s
    // external-queue branch (`src/resumable/common/waker.rs`) instead of
    // straight onto some worker's own deque.
    let os_thread = std::thread::spawn(move || {
        loop {
            let w = slot2.lock().unwrap().take();
            if let Some(w) = w {
                ready2.store(true, std::sync::atomic::Ordering::Release);
                w.wake(); // Called from outside the scheduler.
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    <PollerSystem as StackfulInitSystem>::builder().workers(2).run(|| {
        let v = PollerSystem::block_on(WaitForExternalWake { slot, ready });
        assert_eq!(v, 7);
    });

    os_thread.join().unwrap();
}

// ---------------------------------------------------------------------------
// parallel_call on DefaultDualTaskSystem — the make_context-backed blanket
// (`resumable::stackful::thread::parallel_call`,
// `docs/scoped-ult-promotion.md` §9.8.4) is generic over any
// `S: SpawnableStackfulTaskSystem + StackfulSchedulerSystem`, which
// `DefaultDualTaskSystem` already satisfies — so it applies here with zero
// code changes, riding on `DualTaskDesc`'s existing `TaskDispatch::Ctx`
// path exactly as predicted (§9.8.7). Nothing previously exercised this:
// every other `parallel_call` test in the crate uses a stackful-only
// system. Mirrors `tests/stackful_only.rs`'s own suite.
// ---------------------------------------------------------------------------

/// Forces the branch to be stolen — see `tests/stackful_only.rs`'s test of
/// the same name for why this is the only way to actually exercise
/// `branch_entry` being switched into for the first time (here, additionally,
/// through dual's tag-checked dispatch instead of stackful-only's untagged
/// one).
#[test]
fn dual_parallel_call_stolen_branch_actually_runs() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    run(2, || {
        let b_ran = Arc::new(AtomicBool::new(false));
        let b_ran2 = Arc::clone(&b_ran);
        let (a, b) = <DefaultDualTaskSystem as ScopedStackfulTaskSystem>::parallel_call(
            move || {
                let mut spins: u64 = 0;
                while !b_ran2.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                    spins += 1;
                    assert!(spins < 200_000_000, "b never ran -- steal did not happen");
                }
                1u64
            },
            move || {
                b_ran.store(true, Ordering::Release);
                2u64
            },
        );
        assert_eq!((a, b), (1, 2));
    });
}

#[test]
fn dual_parallel_call_unstolen_branch_panic_propagates() {
    run(1, || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            <DefaultDualTaskSystem as ScopedStackfulTaskSystem>::parallel_call(|| 1u64, || -> u64 { panic!("boom-b-inline") })
        }));
        assert!(result.is_err(), "un-stolen branch panic should propagate through parallel_call");
    });
}

#[test]
fn dual_parallel_call_stolen_branch_panic_propagates() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    run(2, || {
        let b_ran = Arc::new(AtomicBool::new(false));
        let b_ran2 = Arc::clone(&b_ran);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            <DefaultDualTaskSystem as ScopedStackfulTaskSystem>::parallel_call(
                move || {
                    let mut spins: u64 = 0;
                    while !b_ran2.load(Ordering::Acquire) {
                        std::hint::spin_loop();
                        spins += 1;
                        assert!(spins < 200_000_000, "b never ran -- steal did not happen");
                    }
                    1u64
                },
                move || -> u64 {
                    b_ran.store(true, Ordering::Release);
                    panic!("boom-b-stolen");
                },
            )
        }));
        assert!(result.is_err(), "stolen branch panic should propagate through parallel_call");
    });
}

/// Regression guard mirroring `tests/stackful_only.rs`'s test of the same
/// name (that one caught a real bug: `parallel_call` using a worker
/// reference captured before a migrating `a` ran). `a` here recurses
/// through nested `parallel_call`s of its own, so any inner branch that
/// gets stolen and joined migrates the calling ULT to a different worker
/// before this call's own `a` returns.
fn dual_parallel_fib(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let (a, b) = <DefaultDualTaskSystem as ScopedStackfulTaskSystem>::parallel_call(move || dual_parallel_fib(n - 1), move || dual_parallel_fib(n - 2));
    a + b
}

#[test]
fn dual_parallel_call_nested_recursive_nothing_corrupts_across_worker_migration() {
    for _ in 0..30 {
        run(4, || {
            assert_eq!(dual_parallel_fib(24), 46_368);
        });
    }
}

/// Mirrors `tests/stackful_only.rs`'s descriptor-recycling test: a single
/// worker, so every branch takes the un-stolen fast path and immediately
/// recycles its descriptor through the general pool, alternating closure
/// shapes to catch anything wrongly carried over between calls — here
/// additionally through `DualTaskDesc`'s `TaskDispatch::Ctx` union arm,
/// which every recycled descriptor gets re-pinned to via `commit_as_ctx`.
#[test]
fn dual_parallel_call_recycled_descriptor_correct_across_different_closure_types() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    run(1, || {
        for round in 0..50u64 {
            let (a, b) = <DefaultDualTaskSystem as ScopedStackfulTaskSystem>::parallel_call(move || round + 1, move || round + 2);
            assert_eq!((a, b), (round + 1, round + 2));

            let v = vec![round as u32, round as u32 + 1, round as u32 + 2];
            let x = round;
            let y = round * 2;
            let (a2, b2) = <DefaultDualTaskSystem as ScopedStackfulTaskSystem>::parallel_call(
                move || v.iter().sum::<u32>(),
                move || (x + y) as u32,
            );
            assert_eq!(a2, (round as u32) + (round as u32 + 1) + (round as u32 + 2));
            assert_eq!(b2, (round + round * 2) as u32);

            let touched = Arc::new(AtomicBool::new(false));
            let touched2 = Arc::clone(&touched);
            let ((), rc) = <DefaultDualTaskSystem as ScopedStackfulTaskSystem>::parallel_call(
                move || { touched2.store(true, Ordering::Relaxed); },
                move || round,
            );
            assert!(touched.load(Ordering::Relaxed));
            assert_eq!(rc, round);
        }
    });
}

/// Mirrors `tests/stackful_only.rs`'s large-closure test: a big capture
/// just pushes `exec_top` (derived from `f_ptr`) further down the stack.
#[test]
fn dual_parallel_call_large_closure_works() {
    run(2, || {
        let big = [7u64; 40]; // 320 bytes of captured state
        let (a, b) = <DefaultDualTaskSystem as ScopedStackfulTaskSystem>::parallel_call(
            || 1u64,
            move || big.iter().sum::<u64>(),
        );
        assert_eq!((a, b), (1, 7 * 40));
    });
}

// ---------------------------------------------------------------------------
// ScopedStacklessTaskSystem::parallel_call on DefaultDualTaskSystem — the
// recurse(a)/push-and-maybe-direct-poll(b) blanket
// (`resumable::stackless::system::ScopedStacklessTaskSystem::parallel_call`,
// `docs/scoped-ult-promotion.md`) is generic over any
// `S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>`, which
// `DefaultDualTaskSystem` already satisfies — applies here with zero code
// changes. Mirrors `tests/stackless_only.rs`'s own suite (the async
// counterpart to the `dual_parallel_call_*` stackful tests above).
// ---------------------------------------------------------------------------

// Small extension trait so panic tests can catch a panic across an
// `.await` point — same shape as `tests/stackless_only.rs`'s own
// `AwaitCatch`/`AwaitCatchFuture` (not shared across the two test binaries,
// each is a separate compilation unit).
trait DualAwaitCatch: std::future::Future + Sized {
    fn await_catch(self) -> DualAwaitCatchFuture<Self> {
        DualAwaitCatchFuture(self)
    }
}
impl<F: std::future::Future> DualAwaitCatch for F {}

struct DualAwaitCatchFuture<F>(F);

impl<F: std::future::Future> std::future::Future for DualAwaitCatchFuture<F> {
    type Output = Result<F::Output, ()>;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| inner.poll(cx))) {
            Ok(std::task::Poll::Ready(v)) => std::task::Poll::Ready(Ok(v)),
            Ok(std::task::Poll::Pending) => std::task::Poll::Pending,
            Err(_) => std::task::Poll::Ready(Err(())),
        }
    }
}

#[test]
fn dual_async_parallel_call_basic() {
    <DefaultDualTaskSystem as StacklessInitSystem>::builder().workers(1).run_async(async {
        let (a, b) = <DefaultDualTaskSystem as ScopedStacklessTaskSystem>::parallel_call(|| async { 1 + 1 }, || async { 2 + 2 }).await;
        assert_eq!((a, b), (2, 4));
    });
}

/// Mirrors `tests/stackless_only.rs::parallel_call_stolen_branch_actually_runs`.
#[test]
fn dual_async_parallel_call_stolen_branch_actually_runs() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    <DefaultDualTaskSystem as StacklessInitSystem>::builder().workers(2).run_async(async {
        let b_ran = Arc::new(AtomicBool::new(false));
        let b_ran2 = Arc::clone(&b_ran);
        let (a, b) = <DefaultDualTaskSystem as ScopedStacklessTaskSystem>::parallel_call(
            move || async move {
                let mut spins: u64 = 0;
                while !b_ran2.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                    spins += 1;
                    assert!(spins < 200_000_000, "b never ran -- steal did not happen");
                }
                1u64
            },
            move || async move {
                b_ran.store(true, Ordering::Release);
                2u64
            },
        )
        .await;
        assert_eq!((a, b), (1, 2));
    });
}

/// Mirrors `tests/stackless_only.rs::parallel_call_unstolen_branch_panic_propagates`.
#[test]
fn dual_async_parallel_call_unstolen_branch_panic_propagates() {
    <DefaultDualTaskSystem as StacklessInitSystem>::builder().workers(1).run_async(async {
        let result = <DefaultDualTaskSystem as ScopedStacklessTaskSystem>::parallel_call(
            || async { 1u64 },
            || async {
                panic!("boom-b-inline");
                #[allow(unreachable_code)]
                0u64
            },
        )
        .await_catch()
        .await;
        assert!(result.is_err(), "un-stolen branch panic should propagate through parallel_call");
    });
}

/// Mirrors `tests/stackless_only.rs::parallel_call_stolen_branch_panic_propagates`.
#[test]
fn dual_async_parallel_call_stolen_branch_panic_propagates() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    <DefaultDualTaskSystem as StacklessInitSystem>::builder().workers(2).run_async(async {
        let b_ran = Arc::new(AtomicBool::new(false));
        let b_ran2 = Arc::clone(&b_ran);
        let result = <DefaultDualTaskSystem as ScopedStacklessTaskSystem>::parallel_call(
            move || async move {
                let mut spins: u64 = 0;
                while !b_ran2.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                    spins += 1;
                    assert!(spins < 200_000_000, "b never ran -- steal did not happen");
                }
                1u64
            },
            move || async move {
                b_ran.store(true, Ordering::Release);
                panic!("boom-b-stolen");
                #[allow(unreachable_code)]
                2u64
            },
        )
        .await_catch()
        .await;
        assert!(result.is_err(), "stolen branch panic should propagate through parallel_call");
    });
}

/// E0733 regression + nested-recursion/worker-migration stress — this is
/// the exact test shape that caught a real bug during development (stale
/// `wk` used after `recurse(mk_a).await`, which can resume this poll chain
/// on a different worker's OS thread with no native-stack continuity to
/// rely on, unlike stackful's analogous migration case).
fn dual_parallel_fib_async(n: u64) -> impl std::future::Future<Output = u64> + Send {
    async move {
        if n <= 1 {
            return n;
        }
        let (a, b) = <DefaultDualTaskSystem as ScopedStacklessTaskSystem>::parallel_call(
            move || dual_parallel_fib_async(n - 1),
            move || dual_parallel_fib_async(n - 2),
        )
        .await;
        a + b
    }
}

#[test]
fn dual_async_parallel_call_nested_recursive_e0733_regression_and_stress() {
    for _ in 0..30 {
        <DefaultDualTaskSystem as StacklessInitSystem>::builder().workers(4).run_async(async {
            assert_eq!(dual_parallel_fib_async(20).await, 6_765);
        });
    }
}
