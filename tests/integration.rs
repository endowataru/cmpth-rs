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
        DefaultNestedDualTaskSystem::run(2, || {
            let handles: Vec<_> = (0..50)
                .map(|i| <DefaultNestedDualTaskSystem as ThreadSystem>::spawn(move || i * 3u64))
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
        DefaultNestedDualTaskSystem::run(2, || {
            use std::sync::Arc;
            use cmpth::traits::StackfulMutex;
            type M = <DefaultNestedDualTaskSystem as ThreadSystem>::Mutex<u64>;
            let m = Arc::new(<M as StackfulMutex<u64>>::new(0));
            let handles: Vec<_> = (0..20)
                .map(|_| {
                    let m = Arc::clone(&m);
                    <DefaultNestedDualTaskSystem as ThreadSystem>::spawn(move || {
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

fn generic_workload<S: ThreadSystem>() -> u64 {
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
        DefaultNestedDualTaskSystem::run(2, || {
            assert_eq!(generic_workload::<DefaultNestedDualTaskSystem>(), 80);
        });
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
                <DefaultDualTaskSystem as ThreadSystem>::yield_now();
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
// ThreadSystem implemented by hand (no UltIdentity blanket)
// ---------------------------------------------------------------------------

/// `UltIdentity`'s blanket impl is convenience, not architecture: everything
/// it generates can be written as a plain trait impl, as this does. The
/// only part that cannot be defaulted away on stable Rust is the
/// per-system TLS static (generic statics do not exist; a static inside a
/// default trait method would be shared across ALL systems, breaking
/// nested schedulers).
struct ManualSystem;

impl cmpth::SchedulerSystem for ManualSystem {
    type Base  = OsSystem;
    type Desc  = BasicTaskDesc;
    type Deque = CrossbeamDeque<BasicTaskDesc>;
    type ExternalQueue   = StealPathQueue<BasicTaskDesc>;
    type Pool            = ReturnPool<BasicTaskDesc, HeapStack>;
    // Unused: ManualSystem never calls spawn_async.
    type AsyncPool       = cmpth::resumable::common::pool::SimplePool<BasicTaskDesc>;
    const ASYNC_POOL_SIZE: usize = 0;
    // Unused: ManualSystem never calls recurse.
    type RecursionPool   = cmpth::resumable::common::pool::ThresholdPool<cmpth::resumable::common::pool::BlockPool>;
    type Lookup          = TlsCurrent;

    fn worker_tls() -> &'static <OsSystem as cmpth::ThreadSystem>::ThreadSpecific<UltWorker<Self>> {
        // The one thing a macro (or the user, as here) must write:
        // a distinct static per system, anchored in this fn body.
        static TLS: OsTls<UltWorker<ManualSystem>> =
            <OsTls<UltWorker<ManualSystem>> as TlsSlot<UltWorker<ManualSystem>>>::INIT;
        &TLS
    }

    // Stackful-only: no poll_fn tag check, see `execute_stackful`'s doc comment.
    fn execute(wk: &UltWorker<Self>, cont: cmpth::SuspendedTaskToken<BasicTaskDesc>) {
        cmpth::resumable::stackful::worker::execute_stackful(wk, cont)
    }

    fn free_finished_desc(wk: &UltWorker<Self>, desc: *mut BasicTaskDesc) {
        cmpth::resumable::stackful::worker::free_finished_desc_stackful(wk, desc)
    }
}

impl cmpth::StackfulSchedulerSystem for ManualSystem {
    type Ctx   = NativeContext;
    type StackAlloc = HeapStack;
    const STACK_SIZE: usize = 64 * 1024;

    type SuspendedThread = BasicStackfulOnlyResumable<Self>;
}

impl ThreadSystem for ManualSystem {
    type Poller = cmpth::resumable::stackful::waker::UltPoller<Self>;

    fn yield_now() {
        use cmpth::resumable::common::worker::Worker;
        use cmpth::resumable::stackful::worker::StackfulWorker;
        match UltWorker::<Self>::current() {
            Some(wk) => { wk.yield_now(); }
            None => <OsSystem as ThreadSystem>::yield_now(),
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

    type Mutex<T: Send> = cmpth::McsMutex<Self, T>;
    type Barrier        = cmpth::resumable::stackful::sync::Barrier<Self>;
    type SuspendedThread = BasicStackfulOnlyResumable<Self>;
    type Delegator<C: cmpth::DelegatorConsumer<Self>> =
        cmpth::resumable::stackful::sync::McsDelegator<Self, C>;
    type ThreadSpecific<T: 'static> = cmpth::resumable::stackful::tls::UltTls<Self, T>;
}

#[test]
fn manual_impl_without_macro() {
    ManualSystem::run(2, || {
        let h = <ManualSystem as ThreadSystem>::spawn(|| 6 * 7u64);
        assert_eq!(JoinHandleLike::join(h), 42);
    });
}
