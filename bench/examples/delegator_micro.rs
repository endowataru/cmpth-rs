//! Scratch: the cost of one delegated call, run the way a closure-running
//! client uses a delegator (a consumer that runs whatever closure the
//! caller hands it).  Prints ns per call for each delegator, uncontended
//! and contended, on ULTs and on OS threads.
//!
//! usage: delegator_micro [calls-per-thread]

use std::cell::UnsafeCell;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use cmpth::traits::{Delegator, DelegatorRun};
use cmpth::StackfulBuilder;
use cmpth::{
    CombiningDelegator, DefaultDualTaskSystem, DelegationSystem, FnConsumer, OsSystem, SpawnableStackfulTaskSystem,
    StackfulInitSystem, SuspendableSystem,
};

#[inline(always)]
fn delegate<S: SuspendableSystem, D: Delegator<S, FnConsumer>, R: Send>(del: &D, f: impl FnOnce() -> R + Send) -> R {
    del.run(f)
}

fn new_del<S: SuspendableSystem, D: Delegator<S, FnConsumer>>() -> Arc<Pin<Box<D>>> {
    Arc::new(D::start(FnConsumer))
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
    D: Delegator<S, FnConsumer>,
{
    let del: Arc<Pin<Box<D>>> = new_del::<S, D>();
    let total = Arc::new(Shared(UnsafeCell::new(0)));
    // Warm up (also starts a lazily spawned consumer).
    delegate(&**del, || unsafe { *total.0.get() += 1 });
    let t0 = Instant::now();
    let hs: Vec<_> = (0..threads)
        .map(|_| {
            let (del, total) = (Arc::clone(&del), Arc::clone(&total));
            S::spawn(move || {
                let d: &D = &del;
                for _ in 0..n {
                    delegate(d, || unsafe { *total.0.get() += 1 });
                }
            })
        })
        .collect();
    for h in hs {
        cmpth::JoinHandleLike::join(h);
    }
    let dt = t0.elapsed();
    assert_eq!(unsafe { *total.0.get() }, 1 + threads as u64 * n);
    drop(del);
    dt.as_nanos() as f64 / (threads as u64 * n) as f64
}

type U = DefaultDualTaskSystem;
type UMcs = <U as DelegationSystem>::Delegator<FnConsumer>;
type UComb = CombiningDelegator<U, FnConsumer>;
type OOs = <OsSystem as DelegationSystem>::Delegator<FnConsumer>;
type OComb = CombiningDelegator<OsSystem, FnConsumer>;

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
