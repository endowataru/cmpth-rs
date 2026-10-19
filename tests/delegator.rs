//! Every delegator against the same consumers: calls that finish at once
//! (mutual exclusion and responses), calls the consumer defers and
//! completes from `progress`, closures through `FnConsumer`, `with_lock`
//! mixed with calls, and dropping the delegator afterwards.  On ULTs
//! (`McsDelegator`, `CombiningDelegator`) and OS threads (`OsDelegator`,
//! `CombiningDelegator`).

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::Poll;
use std::time::{Duration, Instant};

use cmpth::traits::{Delegator, DelegatorRun};
use cmpth::{
    CallCx, CombiningDelegator, Completion, DefaultDualTaskSystem, DelegationSystem, DelegatorConsumer, FnConsumer,
    JoinHandleLike, OsSystem, Ready, SpawnableStackfulTaskSystem, SuspendableSystem,
};

mod common;
use common::*;

type U = DefaultDualTaskSystem;

/// Adds the request to a total that only the delegator protects; responds
/// with the new total.
#[derive(Default)]
struct Counter {
    total: u64,
}

impl<S: SuspendableSystem> DelegatorConsumer<S> for Counter {
    type Request<'a> = u64;
    type Response = u64;
    fn execute(&mut self, req: u64, _cx: &mut CallCx<'_, S, u64>) -> Poll<u64> {
        // Not atomic: a lost update shows a broken lock.
        let t = self.total;
        std::hint::black_box(&t);
        self.total = t + req;
        Poll::Ready(self.total)
    }
}

/// Defers every call; `progress` completes the oldest with twice its
/// request.
struct Gate<S: SuspendableSystem> {
    pending: VecDeque<(u64, Completion<S, u64>)>,
    completed: u64,
}

impl<S: SuspendableSystem> Default for Gate<S> {
    fn default() -> Self {
        Gate { pending: VecDeque::new(), completed: 0 }
    }
}

impl<S: SuspendableSystem> DelegatorConsumer<S> for Gate<S> {
    type Request<'a> = u64;
    type Response = u64;
    fn execute(&mut self, req: u64, cx: &mut CallCx<'_, S, u64>) -> Poll<u64> {
        self.pending.push_back((req, cx.defer()));
        Poll::Pending
    }
    fn progress(&mut self) -> Option<Ready<S>> {
        let (req, c) = self.pending.pop_front()?;
        self.completed += 1;
        Some(c.complete(req * 2))
    }
    fn is_active(&self) -> bool {
        !self.pending.is_empty()
    }
}

fn with_watchdog<R>(f: impl FnOnce() -> R) -> R {
    let done = Arc::new(AtomicBool::new(false));
    let d2 = Arc::clone(&done);
    std::thread::spawn(move || {
        let t0 = Instant::now();
        while !d2.load(Ordering::Relaxed) {
            if t0.elapsed() > Duration::from_secs(60) {
                eprintln!("delegator test hung");
                std::process::abort();
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    });
    let r = f();
    done.store(true, Ordering::Relaxed);
    r
}

/// `threads` threads of `S` each run `body(i)` `n` times on a shared
/// delegator, which is dropped at the end.
fn hammer<S, C, D>(threads: u64, n: u64, consumer: C, body: impl Fn(&D, u64) + Send + Sync + Copy + 'static) -> Pin<Box<D>>
where
    S: SpawnableStackfulTaskSystem + SuspendableSystem,
    C: DelegatorConsumer<S>,
    D: Delegator<S, C>,
{
    let del: Arc<Pin<Box<D>>> = Arc::new(D::start(consumer));
    let hs: Vec<_> = (0..threads)
        .map(|t| {
            let del = Arc::clone(&del);
            S::spawn(move || {
                for i in 0..n {
                    body(&del, t * n + i);
                }
            })
        })
        .collect();
    for h in hs {
        h.join();
    }
    Arc::try_unwrap(del).ok().expect("delegator still shared")
}

fn counter_case<S: SpawnableStackfulTaskSystem + SuspendableSystem, D: Delegator<S, Counter>>() {
    const T: u64 = 8;
    const N: u64 = 3000;
    let del = hammer::<S, Counter, D>(T, N, Counter::default(), |d, _| {
        let r = d.call(1);
        assert!(r >= 1);
    });
    assert_eq!(del.with_lock(|c| c.total), T * N);
    drop(del);
}

fn gate_case<S: SpawnableStackfulTaskSystem + SuspendableSystem, D: Delegator<S, Gate<S>>>() {
    const T: u64 = 6;
    const N: u64 = 1000;
    let del = hammer::<S, Gate<S>, D>(T, N, Gate::default(), |d, i| {
        assert_eq!(d.call(i), i * 2);
    });
    assert_eq!(del.with_lock(|g| (g.completed, g.pending.len())), (T * N, 0));
    drop(del);
}

fn fn_case<S: SpawnableStackfulTaskSystem + SuspendableSystem, D: Delegator<S, FnConsumer>>() {
    const T: u64 = 8;
    const N: u64 = 3000;
    struct Shared(std::cell::UnsafeCell<u64>);
    unsafe impl Sync for Shared {}
    // Per case (tests run in parallel); `&'static` keeps `body` `Copy`.
    let total: &'static Shared = Box::leak(Box::new(Shared(std::cell::UnsafeCell::new(0))));
    let del = hammer::<S, FnConsumer, D>(T, N, FnConsumer, |d, i| {
        let r = d.run(|| {
            let p = total.0.get();
            unsafe { *p += 1 };
            i + 1
        });
        assert_eq!(r, i + 1);
    });
    assert_eq!(del.run(|| unsafe { *total.0.get() }), T * N);
}

fn lock_case<S: SpawnableStackfulTaskSystem + SuspendableSystem, D: Delegator<S, Counter>>() {
    const T: u64 = 8;
    const N: u64 = 2000;
    let del = hammer::<S, Counter, D>(T, N, Counter::default(), |d, i| {
        if i % 3 == 0 {
            d.with_lock(|c| {
                let t = c.total;
                std::hint::black_box(&t);
                c.total = t + 1;
            });
        } else {
            d.call(1);
        }
    });
    assert_eq!(del.with_lock(|c| c.total), T * N);
}

type Mcs<C> = <U as DelegationSystem>::Delegator<C>;
type Os<C> = <OsSystem as DelegationSystem>::Delegator<C>;

#[test]
fn mcs_on_ults_counter() {
    with_watchdog(|| run(4, counter_case::<U, Mcs<Counter>>));
}

#[test]
fn mcs_on_ults_gate() {
    with_watchdog(|| run(4, gate_case::<U, Mcs<Gate<U>>>));
}

#[test]
fn mcs_on_ults_fn() {
    with_watchdog(|| run(4, fn_case::<U, Mcs<FnConsumer>>));
}

#[test]
fn mcs_on_ults_lock() {
    with_watchdog(|| run(4, lock_case::<U, Mcs<Counter>>));
}

type Ring<C> = cmpth::resumable::stackful::sync::RingBufDelegator<U, C>;

#[test]
fn ring_on_ults_counter() {
    with_watchdog(|| run(4, counter_case::<U, Ring<Counter>>));
}

#[test]
fn ring_on_ults_gate() {
    with_watchdog(|| run(4, gate_case::<U, Ring<Gate<U>>>));
}

#[test]
fn ring_on_ults_fn() {
    with_watchdog(|| run(4, fn_case::<U, Ring<FnConsumer>>));
}

#[test]
fn ring_on_ults_lock() {
    with_watchdog(|| run(4, lock_case::<U, Ring<Counter>>));
}

#[test]
fn combining_on_ults() {
    with_watchdog(|| {
        run(4, || {
            counter_case::<U, CombiningDelegator<U, Counter>>();
            gate_case::<U, CombiningDelegator<U, Gate<U>>>();
            fn_case::<U, CombiningDelegator<U, FnConsumer>>();
            lock_case::<U, CombiningDelegator<U, Counter>>();
        })
    });
}

#[test]
fn os_on_os_threads() {
    with_watchdog(|| {
        counter_case::<OsSystem, Os<Counter>>();
        gate_case::<OsSystem, Os<Gate<OsSystem>>>();
        fn_case::<OsSystem, Os<FnConsumer>>();
        lock_case::<OsSystem, Os<Counter>>();
    });
}

#[test]
fn combining_on_os_threads() {
    with_watchdog(|| {
        counter_case::<OsSystem, CombiningDelegator<OsSystem, Counter>>();
        gate_case::<OsSystem, CombiningDelegator<OsSystem, Gate<OsSystem>>>();
        fn_case::<OsSystem, CombiningDelegator<OsSystem, FnConsumer>>();
        lock_case::<OsSystem, CombiningDelegator<OsSystem, Counter>>();
    });
}

/// A request may borrow from the caller's frame.
struct Sum;
impl<S: SuspendableSystem> DelegatorConsumer<S> for Sum {
    type Request<'a> = &'a [u64];
    type Response = u64;
    fn execute(&mut self, req: &[u64], _cx: &mut CallCx<'_, S, u64>) -> Poll<u64> {
        Poll::Ready(req.iter().sum())
    }
}

#[test]
fn borrowed_requests() {
    with_watchdog(|| {
        run(4, || {
            let del = hammer::<U, Sum, CombiningDelegator<U, Sum>>(8, 2000, Sum, |d, i| {
                let v = [i, 1, 2];
                assert_eq!(d.call(&v), i + 3);
            });
            drop(del);
            let del = hammer::<U, Sum, Mcs<Sum>>(8, 2000, Sum, |d, i| {
                let v = vec![i; 3];
                assert_eq!(d.call(&v), i * 3);
            });
            drop(del);
        })
    });
}

