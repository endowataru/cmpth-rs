//! Scratch: the cost of one delegated call, run the way a closure-running
//! client uses a delegator (a consumer that runs whatever closure the
//! caller hands it).  Prints ns per call for each delegator, uncontended
//! and contended, on ULTs and on OS threads.
//!
//! usage: delegator_micro [calls-per-thread]

use std::cell::{Cell, UnsafeCell};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;

use cmpth::traits::Delegator as DelegatorTrait;
use cmpth::StackfulBuilder;
use cmpth::{
    CombiningDelegator, DefaultDualTaskSystem, DelegationSystem, DelegatorConsumer, OsSystem, SpawnableStackfulTaskSystem,
    StackfulInitSystem, SuspendableSystem,
};

// ---------------------------------------------------------------------------
// A closure-running consumer (the shape of the client this mirrors)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Job(*mut (dyn FnMut() + 'static));
unsafe impl Send for Job {}

struct FnWork<S: SuspendableSystem> {
    job: Option<Job>,
    sth: S::SuspendedThread,
}
impl<S: SuspendableSystem> Default for FnWork<S> {
    fn default() -> Self {
        FnWork { job: None, sth: Default::default() }
    }
}

struct FnConsumer<S>(PhantomData<fn() -> S>);
impl<S: SuspendableSystem + 'static> DelegatorConsumer<S> for FnConsumer<S> {
    type Work = FnWork<S>;
    fn execute(&mut self, work: &mut FnWork<S>) -> (bool, Option<S::SuspendedThread>) {
        if let Some(j) = work.job.take() {
            unsafe { (*j.0)() };
        }
        (true, Some(std::mem::take(&mut work.sth)))
    }
    fn progress(&mut self) -> Option<S::SuspendedThread> {
        None
    }
    fn is_active(&self) -> bool {
        false
    }
}

#[inline(always)]
fn delegate<S: SuspendableSystem + 'static, D: DelegatorTrait<S, FnConsumer<S>>, R>(del: &D, f: impl FnOnce() -> R) -> R {
    let mut f = Some(f);
    let mut out = None;
    let mut run = || out = Some((f.take().unwrap())());
    let job: *mut (dyn FnMut() + '_) = &mut run;
    let job: *mut (dyn FnMut() + 'static) = unsafe { std::mem::transmute(job) };
    let cell = Cell::new(Some(Job(job)));
    del.execute_or_delegate(
        |_c| {
            if let Some(j) = cell.take() {
                unsafe { (*j.0)() };
            }
            (true, None)
        },
        |w: &mut FnWork<S>| {
            w.job = cell.take();
            &w.sth
        },
    );
    drop(run);
    out.unwrap()
}

fn new_del<S: SuspendableSystem + 'static, D: DelegatorTrait<S, FnConsumer<S>>>() -> Arc<D> {
    Arc::new(D::start(FnConsumer(PhantomData)))
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct Shared(UnsafeCell<u64>);
unsafe impl Sync for Shared {}

/// `threads` threads of `S` each make `n` calls; returns ns per call.
fn measure<S, D>(threads: usize, n: u64) -> f64
where
    S: SpawnableStackfulTaskSystem + SuspendableSystem + 'static,
    D: DelegatorTrait<S, FnConsumer<S>>,
{
    let del: Arc<D> = new_del::<S, D>();
    let total = Arc::new(Shared(UnsafeCell::new(0)));
    // Warm up (also starts a lazily spawned consumer).
    delegate(&*del, || unsafe { *total.0.get() += 1 });
    let t0 = Instant::now();
    let hs: Vec<_> = (0..threads)
        .map(|_| {
            let (del, total) = (Arc::clone(&del), Arc::clone(&total));
            S::spawn(move || {
                for _ in 0..n {
                    delegate(&*del, || unsafe { *total.0.get() += 1 });
                }
            })
        })
        .collect();
    for h in hs {
        cmpth::JoinHandleLike::join(h);
    }
    let dt = t0.elapsed();
    assert_eq!(unsafe { *total.0.get() }, 1 + threads as u64 * n);
    // A consumer ULT/thread may still reference it: never stopped (as in
    // the tests).
    std::mem::forget(del);
    dt.as_nanos() as f64 / (threads as u64 * n) as f64
}

type U = DefaultDualTaskSystem;
type UMcs = <U as DelegationSystem>::Delegator<FnConsumer<U>>;
type UComb = CombiningDelegator<U, FnConsumer<U>>;
type OOs = <OsSystem as DelegationSystem>::Delegator<FnConsumer<OsSystem>>;
type OComb = CombiningDelegator<OsSystem, FnConsumer<OsSystem>>;

fn main() {
    let n: u64 = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(200_000);
    for &(w, t) in &[(1usize, 1usize), (4, 4), (4, 16)] {
        let (mcs, comb) = <U as StackfulInitSystem>::builder()
            .workers(w)
            .run(move || (measure::<U, UMcs>(t, n), measure::<U, UComb>(t, n)));
        println!("ult w{w} t{t}  mcs {mcs:8.1}  combining {comb:8.1} ns/call");
    }
    for &t in &[1usize, 4] {
        let os = measure::<OsSystem, OOs>(t, n / 4);
        let comb = measure::<OsSystem, OComb>(t, n);
        println!("os  t{t}      os  {os:8.1}  combining {comb:8.1} ns/call");
    }
}
