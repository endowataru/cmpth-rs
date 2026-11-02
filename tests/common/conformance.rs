//! Checks that a system behaves as its traits promise, for systems built
//! from replaced parts (an [`IdlePolicy`](cmpth::IdlePolicy), a
//! [`StealPolicy`](cmpth::StealPolicy), a
//! [`DispatchPolicy`](cmpth::DispatchPolicy), a whole scheduler) as much as
//! for the standard ones.  Each check panics on a wrong result and on a
//! hang (a lost wakeup shows up as one): it runs under a watchdog that
//! fails it after [`TIMEOUT`].
//!
//! The checks spawn their own pools (`builder().workers(n).run`), so call
//! them outside any pool of the same system.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use super::fib;

use cmpth::traits::{StackfulBarrier, StackfulMutex};
use cmpth::{
    JoinHandleLike, Resumable, ScopedStackfulTaskSystem, SpawnableStackfulTaskSystem,
    StackfulBuilder, StackfulInitSystem, StackfulParking, StackfulResumable, StackfulSyncSystem,
    StacklessBuilder, StacklessInitSystem, StacklessTaskSystem,
};

/// How long one check may take before it counts as hung.
pub const TIMEOUT: Duration = Duration::from_secs(60);

/// Run `f` on its own thread; panic with `name` if it panics or takes
/// longer than [`TIMEOUT`] (the thread is then left behind).
fn watchdog(name: &str, f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = mpsc::channel();
    let h = std::thread::Builder::new()
        .name(format!("cmpth-conformance-{name}"))
        .spawn(move || {
            f();
            let _ = tx.send(());
        })
        .unwrap();
    match rx.recv_timeout(TIMEOUT) {
        Ok(()) => h.join().unwrap(),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            if let Err(e) = h.join() {
                std::panic::resume_unwind(e);
            }
        }
        Err(mpsc::RecvTimeoutError::Timeout) => panic!("conformance::{name}: no result after {TIMEOUT:?} (hung)"),
    }
}

/// `spawn`/`join` and `yield_now` on `workers` workers (and on one):
/// a fork-join tree, many independent tasks, tasks spawned after the
/// workers went idle, and tasks that only yield to each other.
pub fn stackful<S>(workers: usize)
where
    S: StackfulInitSystem + SpawnableStackfulTaskSystem,
{
    for n in [1, workers] {
        watchdog("stackful", move || {
            S::builder().workers(n).run(move || {
                assert_eq!(fib::<S>(20), 6765, "fork-join result");

                let hs: Vec<_> = (0..10_000u64).map(|i| S::spawn(move || i * 2)).collect();
                let sum: u64 = hs.into_iter().map(JoinHandleLike::join).sum();
                assert_eq!(sum, 10_000 * 9_999, "independent tasks");

                // Let the other workers run out of work (and park, if they
                // do), then hand them work again, several times.
                for _ in 0..20 {
                    std::thread::sleep(Duration::from_millis(2));
                    let hs: Vec<_> = (0..64u64).map(|i| S::spawn(move || fib::<S>(10) + i)).collect();
                    let sum: u64 = hs.into_iter().map(JoinHandleLike::join).sum();
                    assert_eq!(sum, 64 * 55 + 63 * 32, "tasks spawned after idling");
                }

                // Tasks that yield until all have started: fair enough
                // that none is left behind.
                let started = Arc::new(AtomicUsize::new(0));
                let k = 4 * n;
                let hs: Vec<_> = (0..k)
                    .map(|_| {
                        let started = Arc::clone(&started);
                        S::spawn(move || {
                            started.fetch_add(1, Ordering::SeqCst);
                            while started.load(Ordering::SeqCst) < k {
                                S::yield_now();
                            }
                        })
                    })
                    .collect();
                hs.into_iter().for_each(JoinHandleLike::join);
            });
        });
    }
}

/// Parking a task on [`SuspendableSystem::SuspendedThread`](cmpth::SuspendableSystem::SuspendedThread)
/// and waking it from another task, many times (a lost wakeup hangs).
pub fn parking<S>(workers: usize)
where
    S: StackfulInitSystem + StackfulParking,
{
    park_and_wake::<S>("parking", workers, false);
}

/// [`parking`], woken from an OS thread outside the pool instead: for
/// systems whose parked-thread handle allows that (the dual systems'
/// [`DualResumable`](cmpth::DualResumable); a stackful-only system's
/// handle can only be notified from its workers).
pub fn parking_from_os<S>(workers: usize)
where
    S: StackfulInitSystem + StackfulParking,
{
    park_and_wake::<S>("parking_from_os", workers, true);
}

fn park_and_wake<S>(name: &'static str, workers: usize, os: bool)
where
    S: StackfulInitSystem + StackfulParking,
{
    /// The slot lives on the waiting task's stack until it is woken.
    struct Ptr<T>(*const T);
    unsafe impl<T> Send for Ptr<T> {}

    fn wake<T: Resumable<S>, S>(slot: Ptr<T>, ready: Arc<AtomicBool>, yield_now: fn(), late: bool) {
        // SAFETY: the waiter does not return (and drop the slot) before
        // `notify`.
        let slot = unsafe { &*slot.0 };
        while !ready.load(Ordering::Acquire) || !slot.is_set() {
            yield_now();
        }
        if late {
            // Long enough for idle workers to park: the wakeup must then
            // reach one of them.
            std::thread::sleep(Duration::from_millis(3));
        }
        slot.notify();
    }

    watchdog(name, move || {
        S::builder().workers(workers).run(move || {
            for round in 0..200 {
                let late = round % 10 == 0;
                let slot = S::SuspendedThread::default();
                let ready = Arc::new(AtomicBool::new(false));
                let (p, r) = (Ptr(&slot as *const S::SuspendedThread), Arc::clone(&ready));
                // A waking task yields: it may run on the waiter's own
                // worker.
                let (thread, task) = if os {
                    (Some(std::thread::spawn(move || wake::<_, S>(p, r, std::thread::yield_now, late))), None)
                } else {
                    (None, Some(S::spawn(move || wake::<_, S>(p, r, S::yield_now, late))))
                };
                slot.wait_with(|| ready.store(true, Ordering::Release));
                if let Some(t) = thread {
                    t.join().unwrap();
                }
                if let Some(t) = task {
                    JoinHandleLike::join(t);
                }
            }
        });
    });
}

/// The system's [`StackfulSyncSystem`] mutex and barrier under contention.
pub fn sync<S>(workers: usize)
where
    S: StackfulInitSystem + SpawnableStackfulTaskSystem + StackfulSyncSystem,
{
    watchdog("sync", move || {
        S::builder().workers(workers).run(move || {
            let m = Arc::new(<S::Mutex<u64> as StackfulMutex<u64>>::new(0));
            let hs: Vec<_> = (0..8 * workers)
                .map(|_| {
                    let m = Arc::clone(&m);
                    S::spawn(move || {
                        for _ in 0..1000 {
                            *m.lock() += 1;
                        }
                    })
                })
                .collect();
            hs.into_iter().for_each(JoinHandleLike::join);
            assert_eq!(*m.lock(), 8 * workers as u64 * 1000, "mutex lost an increment");

            let k = 2 * workers;
            let b = Arc::new(<S::Barrier as StackfulBarrier>::new(k));
            let phase = Arc::new(AtomicUsize::new(0));
            let hs: Vec<_> = (0..k)
                .map(|_| {
                    let (b, phase) = (Arc::clone(&b), Arc::clone(&phase));
                    S::spawn(move || {
                        let mut leaders = 0;
                        for round in 0..50 {
                            phase.fetch_add(1, Ordering::SeqCst);
                            if b.wait().is_leader {
                                leaders += 1;
                            }
                            assert!(phase.load(Ordering::SeqCst) >= (round + 1) * k, "barrier let a task through early");
                            b.wait();
                        }
                        leaders
                    })
                })
                .collect();
            let leaders: usize = hs.into_iter().map(JoinHandleLike::join).sum();
            assert_eq!(leaders, 50, "one leader per barrier round");
        });
    });
}

/// `parallel_call` on a `scoped` system: a fork-join tree, nested.
pub fn scoped<S>(workers: usize)
where
    S: StackfulInitSystem + ScopedStackfulTaskSystem,
{
    fn pfib<S: ScopedStackfulTaskSystem>(n: u64) -> u64 {
        if n < 2 {
            return n;
        }
        let (a, b) = S::parallel_call(move || pfib::<S>(n - 1), move || pfib::<S>(n - 2));
        a + b
    }
    for n in [1, workers] {
        watchdog("scoped", move || {
            let r = S::builder().workers(n).run(|| {
                let mut s = 0;
                for _ in 0..20 {
                    std::thread::sleep(Duration::from_millis(1));
                    s += pfib::<S>(18);
                }
                s
            });
            assert_eq!(r, 20 * 2584, "parallel_call result");
        });
    }
}

/// Stackless `spawn` on a stackless system: a fork-join tree and many
/// independent tasks.
pub fn stackless<S>(workers: usize)
where
    S: StacklessInitSystem + StacklessTaskSystem,
{
    fn afib<S: StacklessTaskSystem>(n: u64) -> Pin<Box<dyn Future<Output = u64> + Send>> {
        Box::pin(async move {
            if n < 2 {
                return n;
            }
            let h = S::spawn(move || afib::<S>(n - 1)).await;
            let b = afib::<S>(n - 2).await;
            h.await + b
        })
    }
    for n in [1, workers] {
        watchdog("stackless", move || {
            S::builder().workers(n).run_async(async {
                assert_eq!(afib::<S>(18).await, 2584, "fork-join result");
                let mut hs = Vec::new();
                for i in 0..5_000u64 {
                    hs.push(S::spawn(move || async move { i * 2 }).await);
                }
                let mut sum = 0;
                for h in hs {
                    sum += h.await;
                }
                assert_eq!(sum, 5_000 * 4_999, "independent tasks");
            });
        });
    }
}
