//! `ParkIdle` end-to-end: workers that have gone to sleep must wake for every
//! source of new work (local push, spawn from a running task, external-thread
//! wake) and for shutdown. A lost wakeup shows up as a hang, so every test
//! runs under a watchdog that turns a hang into a failure.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Waker;
use std::time::{Duration, Instant};

use cmpth::{
    JoinHandleLike, OsSystem, ParkIdle, SpawnableStackfulTaskSystem, StackfulBuilder, StacklessBuilder,
    StackfulInitSystem, StacklessInitSystem, UltAsyncIdentity, UltAsyncSystem, UltIdentity,
};

pub struct ParkStackful;

impl UltIdentity for ParkStackful {
    type Base = OsSystem;
    type Ctx = cmpth::NativeContext;
    type Desc = cmpth::StackfulOnlyTaskDesc<Self>;
    type RunQueue = cmpth::HybridRunQueue<cmpth::SuspendedTaskToken<cmpth::StackfulOnlyTaskDesc<Self>>>;
    type Alloc = cmpth::HeapStack;
    type Lookup = cmpth::TlsCurrent;
    type Idle = ParkIdle;

    fn worker_tls_anchor() -> &'static <OsSystem as cmpth::NestableSystem>::ThreadSpecific<cmpth::UltWorker<Self>> {
        static A: cmpth::TlsAnchor = cmpth::TlsAnchor::new();
        cmpth::TlsSlot::from_anchor(&A)
    }
}

pub struct ParkAsyncMarker;

impl UltAsyncIdentity for ParkAsyncMarker {
    type Base = OsSystem;
    type Desc = cmpth::StacklessOnlyTaskDesc<UltAsyncSystem<Self>>;
    type RunQueue = cmpth::HybridRunQueue<cmpth::SuspendedTaskToken<cmpth::StacklessOnlyTaskDesc<UltAsyncSystem<Self>>>>;
    type Lookup = cmpth::InlineTlsCurrent;
    type Idle = ParkIdle;

    fn worker_tls_anchor() -> &'static <OsSystem as cmpth::NestableSystem>::ThreadSpecific<cmpth::UltWorker<UltAsyncSystem<Self>>> {
        static A: cmpth::TlsAnchor = cmpth::TlsAnchor::new();
        cmpth::TlsSlot::from_anchor(&A)
    }
}

type ParkAsync = UltAsyncSystem<ParkAsyncMarker>;

/// Run `f` on a helper thread; fail (not hang) if it takes longer than 30 s.
fn with_watchdog(f: impl FnOnce() + Send + 'static) {
    let done = Arc::new(AtomicBool::new(false));
    let done2 = Arc::clone(&done);
    let h = std::thread::spawn(move || {
        f();
        done2.store(true, Ordering::Release);
    });
    let start = Instant::now();
    while !done.load(Ordering::Acquire) {
        assert!(start.elapsed() < Duration::from_secs(30), "lost wakeup: pool hung");
        if h.is_finished() {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    h.join().unwrap();
}

fn fib(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let h = ParkStackful::spawn(move || fib(n - 1));
    let b = fib(n - 2);
    JoinHandleLike::join(h) + b
}

#[test]
fn spawn_join_tree() {
    with_watchdog(|| {
        ParkStackful::builder().workers(4).run(|| assert_eq!(fib(20), 6765));
    });
}

/// Workers register, sleep, and must all wake for a burst of spawns issued by
/// the only running task.
#[test]
fn wake_after_all_workers_parked() {
    with_watchdog(|| {
        for _ in 0..5 {
            let counter = Arc::new(AtomicU64::new(0));
            let counter2 = Arc::clone(&counter);
            ParkStackful::builder().workers(4).run(move || {
                std::thread::sleep(Duration::from_millis(50));
                let hs: Vec<_> = (0..64)
                    .map(|_| {
                        let c = Arc::clone(&counter2);
                        ParkStackful::spawn(move || {
                            std::thread::sleep(Duration::from_micros(200));
                            c.fetch_add(1, Ordering::Relaxed);
                        })
                    })
                    .collect();
                for h in hs {
                    JoinHandleLike::join(h);
                }
            });
            assert_eq!(counter.load(Ordering::Relaxed), 64);
        }
    });
}

/// Repeated idle/burst cycles: the window between "worker registers" and
/// "worker sleeps" is what the fences protect, so hammer it.
#[test]
fn burst_idle_cycles() {
    with_watchdog(|| {
        ParkStackful::builder().workers(4).run(|| {
            for round in 0..200u64 {
                let hs: Vec<_> = (0..4).map(|i| ParkStackful::spawn(move || round + i)).collect();
                let sum: u64 = hs.into_iter().map(JoinHandleLike::join).sum();
                assert_eq!(sum, 4 * round + 6);
                if round % 20 == 0 {
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
        });
    });
}

/// An OS thread outside the pool wakes a task while every worker is parked:
/// only the external-queue wake hook can get this running again.
#[test]
fn external_thread_wake_reaches_parked_workers() {
    struct WaitForExternalWake {
        slot: Arc<Mutex<Option<Waker>>>,
        ready: Arc<AtomicBool>,
    }
    impl std::future::Future for WaitForExternalWake {
        type Output = u32;
        fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<u32> {
            if self.ready.load(Ordering::Acquire) {
                return std::task::Poll::Ready(7);
            }
            *self.slot.lock().unwrap() = Some(cx.waker().clone());
            std::task::Poll::Pending
        }
    }

    with_watchdog(|| {
        let slot: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));
        let ready = Arc::new(AtomicBool::new(false));
        let (slot2, ready2) = (Arc::clone(&slot), Arc::clone(&ready));

        let os_thread = std::thread::spawn(move || {
            loop {
                let w = slot2.lock().unwrap().take();
                if let Some(w) = w {
                    // Long enough for every worker to spin, yield, and park.
                    std::thread::sleep(Duration::from_millis(100));
                    ready2.store(true, Ordering::Release);
                    w.wake();
                    return;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        });

        let out = Arc::new(AtomicU64::new(0));
        let out2 = Arc::clone(&out);
        ParkAsync::builder().workers(3).run_async(async move {
            out2.store(WaitForExternalWake { slot, ready }.await as u64, Ordering::Release);
        });
        os_thread.join().unwrap();
        assert_eq!(out.load(Ordering::Acquire), 7);
    });
}

/// Shutdown must release workers that are parked, including on a pool whose
/// root task finishes immediately.
#[test]
fn shutdown_releases_parked_workers() {
    with_watchdog(|| {
        for _ in 0..10 {
            ParkStackful::builder().workers(4).run(|| {
                std::thread::sleep(Duration::from_millis(30));
            });
        }
    });
}
