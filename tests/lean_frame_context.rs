//! End-to-end coverage for [`cmpth::LeanFrameContext`], the alternative
//! `ContextPolicy` that declares `x21`-`x28`/`r14`-`r15` as clobbers instead
//! of folding them into the ctx frame (see the module docs on
//! `cmpth::resumable::stackful::context` for the tradeoff). This mirrors
//! `DefaultStackfulOnlyTaskSystem` exactly except for `type Ctx`, to prove
//! the alternative policy is a genuine drop-in: same trait, same call
//! sites, only the assembly backing `S::Ctx` differs.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cmpth::{
    JoinHandleLike, LeanFrameContext, OsSystem, ScopedStackfulTaskSystem, SpawnableStackfulTaskSystem,
    StackfulBuilder, StackfulInitSystem, UltIdentity,
};

pub struct LeanContextSystem;

impl UltIdentity for LeanContextSystem {
    type Base = OsSystem;
    type Ctx = LeanFrameContext;
    type Desc = cmpth::resumable::stackful::desc::StackfulOnlyTaskDesc<Self>;
    type RunQueue = cmpth::HybridRunQueue<
        cmpth::resumable::common::desc::SuspendedTaskToken<
            cmpth::resumable::stackful::desc::StackfulOnlyTaskDesc<Self>,
        >,
    >;
    type Alloc = cmpth::resumable::common::stack::HeapStack;
    type Lookup = cmpth::resumable::common::lookup::TlsCurrent;

    fn worker_tls_anchor() -> &'static <OsSystem as cmpth::NestableSystem>::ThreadSpecific<
        cmpth::resumable::common::worker::UltWorker<Self>,
    > {
        static A: cmpth::traits::component::tls::TlsAnchor = cmpth::traits::component::tls::TlsAnchor::new();
        cmpth::traits::component::tls::TlsSlot::from_anchor(&A)
    }
}

#[test]
fn spawn_join_basic() {
    LeanContextSystem::builder().workers(2).run(|| {
        let h = LeanContextSystem::spawn(|| 6 * 7);
        assert_eq!(JoinHandleLike::join(h), 42);
    });
}

#[test]
fn spawn_join_many_parallel() {
    let counter = Arc::new(AtomicU64::new(0));
    let counter2 = Arc::clone(&counter);
    LeanContextSystem::builder().workers(4).run(move || {
        let handles: Vec<_> = (0..200)
            .map(|i| {
                let counter3 = Arc::clone(&counter2);
                LeanContextSystem::spawn(move || {
                    counter3.fetch_add(1, Ordering::Relaxed);
                    i * 2u64
                })
            })
            .collect();
        let mut sum = 0u64;
        for h in handles {
            sum += JoinHandleLike::join(h);
        }
        assert_eq!(sum, (0..200).map(|i| i * 2u64).sum::<u64>());
    });
    assert_eq!(counter.load(Ordering::Relaxed), 200);
}

#[test]
fn spawn_nested() {
    LeanContextSystem::builder().workers(2).run(|| {
        let h = LeanContextSystem::spawn(|| {
            let inner = LeanContextSystem::spawn(|| 10);
            JoinHandleLike::join(inner) + 5
        });
        assert_eq!(JoinHandleLike::join(h), 15);
    });
}

#[test]
fn yield_now_roundtrips() {
    LeanContextSystem::builder().workers(1).run(|| {
        for _ in 0..1000 {
            LeanContextSystem::yield_now();
        }
    });
}

fn parallel_fib(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let (a, b) = LeanContextSystem::parallel_call(
        move || parallel_fib(n - 1),
        move || parallel_fib(n - 2),
    );
    a + b
}

#[test]
fn parallel_call_recursive() {
    LeanContextSystem::builder().workers(4).run(|| {
        assert_eq!(parallel_fib(20), 6765);
    });
}

/// Forces the branch to be stolen — only passes if a `LeanFrameContext`-built
/// context can actually be switched into for the first time on a different
/// worker (the same `make_context`/`branch_entry` machinery
/// `parallel_call_stolen_branch_actually_runs` in `tests/stackful_only.rs`
/// covers for `NativeContext`).
#[test]
fn parallel_call_stolen_branch_actually_runs() {
    LeanContextSystem::builder().workers(2).run(|| {
        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ran2 = Arc::clone(&ran);
        let (_, b) = LeanContextSystem::parallel_call(
            move || {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while !ran2.load(Ordering::Relaxed) {
                    assert!(std::time::Instant::now() < deadline, "b never ran -- steal path broken");
                    LeanContextSystem::yield_now();
                }
                1u64
            },
            move || {
                ran.store(true, Ordering::Relaxed);
                2u64
            },
        );
        assert_eq!(b, 2);
    });
}

/// Same regression this crate already runs for `NativeContext`
/// (`tests/integration.rs::float_regs_survive_yield`): AAPCS64 makes the
/// lower halves of `v8`-`v15` callee-saved regardless of which
/// `ContextPolicy` is in play, so both policies must protect them the same
/// way (both declare them `lateout`).
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
                LeanContextSystem::yield_now();
            }
        }
        acc.iter().sum()
    }

    let expected: Vec<f64> = (0..8).map(|k| crunch(k, false)).collect();
    LeanContextSystem::builder().workers(2).run(move || {
        let handles: Vec<_> = (0..8)
            .map(|k| LeanContextSystem::spawn(move || crunch(k, true)))
            .collect();
        for (k, h) in handles.into_iter().enumerate() {
            assert_eq!(JoinHandleLike::join(h), expected[k], "ULT {k} float state corrupted");
        }
    });
}
