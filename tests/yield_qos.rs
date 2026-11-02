//! Yield policies: tasks that keep yielding all finish and take turns on
//! every worker with `GlobalYield`, and a turn budget keeps a delegator
//! correct while it makes callers that never wait yield.  QoS policies:
//! systems with each policy run and pass the conformance checks.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

mod common;

use common::conformance;
use cmpth::traits::Delegator;
use cmpth::{
    CombiningDelegator, DualTaskDesc, FnConsumer, GlobalYield, JoinHandleLike, LocalYield, SpawnableStackfulTaskSystem,
    StackfulBuilder, StackfulInitSystem, SuspendedTaskToken, UltDualSystem,
};

type Tok<M> = SuspendedTaskToken<DualTaskDesc<UltDualSystem<M>>>;

macro_rules! dual {
    ($name:ident { $($body:tt)* }) => {
        pub struct $name;
        cmpth::worker_tls_anchor!($name);
        cmpth::impl_with_defaults! { impl cmpth::UltDualIdentity for $name { $($body)* } }
    };
}

dual!(Fair { type Yield = GlobalYield<Tok<Fair>>; });
dual!(Fair61 { type Yield = GlobalYield<Tok<Fair61>, 61>; });
dual!(Budget { type Yield = LocalYield<16>; });
dual!(FairBudget { type Yield = GlobalYield<Tok<FairBudget>, 61, 16>; });
dual!(Background { type Qos = cmpth::BackgroundQos; type Idle = cmpth::SearchIdle; });
dual!(Demote { type Qos = cmpth::DemoteIdleQos; });
dual!(Split { type Qos = cmpth::SplitQos<2>; type Idle = cmpth::SearchIdle; });

/// `t` tasks yield `n` times each on `w` workers; all must finish.
fn yields_finish<S: SpawnableStackfulTaskSystem + StackfulInitSystem>(w: usize, t: usize, n: u64) {
    let total = <S as StackfulInitSystem>::builder().workers(w).run(move || {
        let hs: Vec<_> = (0..t)
            .map(|_| {
                S::spawn(move || {
                    for _ in 0..n {
                        S::yield_now();
                    }
                    n
                })
            })
            .collect();
        hs.into_iter().map(JoinHandleLike::join).sum::<u64>()
    });
    assert_eq!(total, t as u64 * n);
}

#[test]
fn global_yield_finishes() {
    yields_finish::<UltDualSystem<Fair>>(4, 24, 2_000);
    yields_finish::<UltDualSystem<Fair61>>(4, 24, 2_000);
    yields_finish::<UltDualSystem<Fair>>(1, 8, 2_000);
}

/// 4 tasks per worker, each busy for a few microseconds between yields,
/// for 300 ms: the slowest must not fall far behind the fastest.  With the
/// local policy, 3 of the 4 workers end up with one task each and the 4th
/// with the other 13, which gives min/max of about 1/13.
#[test]
fn global_yield_takes_turns() {
    type S = UltDualSystem<Fair61>;
    let (w, t) = (4usize, 16usize);
    let counts = <S as StackfulInitSystem>::builder().workers(w).run(move || {
        let stop = Arc::new(AtomicBool::new(false));
        let hs: Vec<_> = (0..t)
            .map(|_| {
                let stop = Arc::clone(&stop);
                <S as SpawnableStackfulTaskSystem>::spawn(move || {
                    let mut n = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        let t0 = Instant::now();
                        while t0.elapsed() < Duration::from_micros(5) {}
                        <S as SpawnableStackfulTaskSystem>::yield_now();
                        n += 1;
                    }
                    n
                })
            })
            .collect();
        // Not `sleep` here: that would block worker 0 with tasks queued on it.
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            stop.store(true, Ordering::Relaxed);
        });
        hs.into_iter().map(JoinHandleLike::join).collect::<Vec<u64>>()
    });
    let (min, max) = (*counts.iter().min().unwrap(), *counts.iter().max().unwrap());
    assert!(min * 3 > max, "unfair turns: {counts:?}");
}

fn budget_case<S: SpawnableStackfulTaskSystem + StackfulInitSystem + cmpth::SuspendableSystem>()
where
    S::SuspendedThread: cmpth::traits::StackfulResumable<S>,
{
    let total = <S as StackfulInitSystem>::builder().workers(4).run(|| {
        let del = Arc::new(CombiningDelegator::<S, FnConsumer>::start(FnConsumer));
        let sum = Arc::new(AtomicU64::new(0));
        let hs: Vec<_> = (0..16)
            .map(|_| {
                let (del, sum) = (Arc::clone(&del), Arc::clone(&sum));
                S::spawn(move || {
                    for _ in 0..3_000 {
                        let s = Arc::clone(&sum);
                        del.call(&mut || {
                            // Not atomic as a whole: a lost update shows a broken lock.
                            let v = s.load(Ordering::Relaxed);
                            s.store(v + 1, Ordering::Relaxed);
                        });
                    }
                })
            })
            .collect();
        for h in hs {
            JoinHandleLike::join(h);
        }
        sum.load(Ordering::Relaxed)
    });
    assert_eq!(total, 16 * 3_000);
}

#[test]
fn budget_keeps_delegator_correct() {
    budget_case::<UltDualSystem<Budget>>();
    budget_case::<UltDualSystem<FairBudget>>();
}

#[test]
fn policies_pass_conformance() {
    fn all<S>()
    where
        S: StackfulInitSystem + cmpth::StackfulParking + cmpth::StackfulSyncSystem,
    {
        conformance::stackful::<S>(4);
        conformance::parking::<S>(4);
        conformance::parking_from_os::<S>(4);
        conformance::sync::<S>(4);
    }
    all::<UltDualSystem<Fair>>();
    all::<UltDualSystem<FairBudget>>();
    all::<UltDualSystem<Background>>();
    all::<UltDualSystem<Demote>>();
    all::<UltDualSystem<Split>>();
}
