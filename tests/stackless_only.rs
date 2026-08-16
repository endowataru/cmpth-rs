//! End-to-end tests for [`cmpth::DefaultStacklessOnlyTaskSystem`]
//! (`UltAsyncIdentity`): no `Ctx`/`StackAlloc`/`StackfulSchedulerSystem` at
//! all, `execute`'s dispatch is `execute_async` (always poll, no `poll_fn`
//! tag check) instead of `execute_dual`/`execute_stackful`. Only
//! `StacklessTaskSystem`'s `run_async`/`spawn`/`recurse`/`.await` are
//! reachable here — there is no `spawn` (stackful), no `.join()` (blocking),
//! no `block_on`: none of that is expressible without `StackfulSchedulerSystem`.

use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use cmpth::{
    DefaultStacklessOnlyTaskSystem, ScopedStacklessTaskSystem, StacklessBuilder,
    StacklessInitSystem, StacklessTaskSystem,
};

#[test]
fn spawn_async_await_basic() {
    DefaultStacklessOnlyTaskSystem::builder().workers(2).run_async(async {
        let h = DefaultStacklessOnlyTaskSystem::spawn(|| async { 6 * 7 }).await;
        assert_eq!(h.await, 42);
    });
}

#[test]
fn spawn_async_many_parallel() {
    let counter = Arc::new(AtomicU64::new(0));
    DefaultStacklessOnlyTaskSystem::builder().workers(4).run_async(async move {
        let counter = Arc::clone(&counter);
        let mut handles = Vec::with_capacity(200);
        for i in 0..200u64 {
            let counter = Arc::clone(&counter);
            let h = DefaultStacklessOnlyTaskSystem::spawn(move || async move {
                counter.fetch_add(1, Ordering::Relaxed);
                i * 2u64
            })
            .await;
            handles.push(h);
        }
        let mut sum = 0u64;
        for h in handles {
            sum += h.await;
        }
        assert_eq!(sum, (0..200).map(|i| i * 2u64).sum::<u64>());
        assert_eq!(counter.load(Ordering::Relaxed), 200);
    });
}

#[test]
fn spawn_async_nested() {
    DefaultStacklessOnlyTaskSystem::builder().workers(2).run_async(async {
        let h = DefaultStacklessOnlyTaskSystem::spawn(|| async {
            let inner = DefaultStacklessOnlyTaskSystem::spawn(|| async { 10 }).await;
            inner.await + 5
        })
        .await;
        assert_eq!(h.await, 15);
    });
}

#[test]
fn spawn_async_panic_propagates_via_await() {
    DefaultStacklessOnlyTaskSystem::builder().workers(1).run_async(async {
        let h = DefaultStacklessOnlyTaskSystem::spawn::<(), _, _>(|| async { panic!("boom") }).await;
        let result = std::panic::AssertUnwindSafe(h.await_catch())
            .0
            .await;
        assert!(result.is_err());
    });
}

#[test]
fn spawn_async_detach_before_finish() {
    // Drop the SpawnHandle while the task is still pending (first poll not
    // yet done). The task must complete and free itself via the detach path
    // — the execute_async mirror of `integration.rs`'s
    // `spawn_async_detach_before_finish` (which exercises the same path
    // under execute_dual).
    use std::future::poll_fn;

    let done = Arc::new(AtomicBool::new(false));
    let done2 = Arc::clone(&done);

    DefaultStacklessOnlyTaskSystem::builder().workers(2).run_async(async move {
        let h = DefaultStacklessOnlyTaskSystem::spawn(move || async move {
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
            })
            .await;
            done2.store(true, Ordering::Release);
        })
        .await;
        drop(h); // detach: has_handle → false, joiner cleared
        while !done.load(Ordering::Acquire) {
            DefaultStacklessOnlyTaskSystem::yield_now().await;
        }
    });
}

#[test]
fn spawn_async_detach_after_finish() {
    // Drop the SpawnHandle after the task has already set finished=true.
    let done = Arc::new(AtomicBool::new(false));
    let done2 = Arc::clone(&done);

    DefaultStacklessOnlyTaskSystem::builder().workers(2).run_async(async move {
        let h = DefaultStacklessOnlyTaskSystem::spawn(move || async move {
            done2.store(true, Ordering::Release);
            99u32
        })
        .await;
        while !done.load(Ordering::Acquire) {
            DefaultStacklessOnlyTaskSystem::yield_now().await;
        }
        drop(h); // finished=true branch
    });
}

// Small extension trait so the panic test can catch the panic across an
// `.await` point without pulling in a futures-util dependency just for this.
trait AwaitCatch: std::future::Future + Sized {
    fn await_catch(self) -> AwaitCatchFuture<Self> {
        AwaitCatchFuture(self)
    }
}
impl<F: std::future::Future> AwaitCatch for F {}

struct AwaitCatchFuture<F>(F);

impl<F: std::future::Future> std::future::Future for AwaitCatchFuture<F> {
    type Output = Result<F::Output, ()>;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // JoinHandle's own panic-resuming happens inside its `poll` (it calls
        // `std::panic::resume_unwind`), so catching it here requires
        // `catch_unwind` around the poll call itself.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| inner.poll(cx))) {
            Ok(std::task::Poll::Ready(v)) => std::task::Poll::Ready(Ok(v)),
            Ok(std::task::Poll::Pending) => std::task::Poll::Pending,
            Err(_) => std::task::Poll::Ready(Err(())),
        }
    }
}

#[test]
fn run_async_root_future_runs_to_completion() {
    let done = Arc::new(AtomicU64::new(0));
    let done2 = Arc::clone(&done);
    DefaultStacklessOnlyTaskSystem::builder().workers(3).run_async(async move {
        done2.store(1, Ordering::Release);
    });
    assert_eq!(done.load(Ordering::Acquire), 1);
}

#[test]
fn async_task_system_yield_now() {
    let flag = Arc::new(AtomicU64::new(0));
    let flag2 = Arc::clone(&flag);
    DefaultStacklessOnlyTaskSystem::builder().workers(2).run_async(async move {
        let h = DefaultStacklessOnlyTaskSystem::spawn(move || async move {
            flag2.store(1, Ordering::Release);
        })
        .await;
        while flag.load(Ordering::Acquire) == 0 {
            DefaultStacklessOnlyTaskSystem::yield_now().await;
        }
        h.await;
    });
}

/// `yield_now().await` must let work that is *already queued* on this
/// worker run before the yielding task resumes -- the stackless counterpart
/// of `stackful_only.rs::yield_now_lets_already_queued_work_run_first`.
///
/// Unlike `spawn`'s stackful counterpart, `spawn_async` never context
/// switches into the child: it just registers a descriptor and pushes it
/// (`spawn_now`, `resumable/stackless/thread.rs`) while the spawning task's
/// own poll keeps running synchronously past the first `.await`
/// (`SpawnAction::poll` is unconditionally `Ready`). So — unlike the
/// stackful test, which needs two queued items to be observable because
/// `suspend_to_sched` pops the next continuation *before* its callback
/// re-queues the yielding task — a single queued sibling here would
/// already discriminate fair (`defer`) from unfair (`push`): whichever one
/// is used, it competes directly against what's already sitting in the run
/// queue at the moment `yield_now`'s `Pending` arm runs. Two siblings are
/// used anyway, both to mirror the stackful test's shape and to confirm the
/// *relative* order between them survives untouched (LIFO `push`: the
/// second spawned runs first).
///
/// Single worker, so nothing here can run except via this worker's own run
/// queue -- no stealing to muddy the ordering.
#[test]
fn yield_now_lets_already_queued_work_run_first() {
    let order = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let rec = {
        let o = Arc::clone(&order);
        move |v: u8| o.lock().unwrap().push(v)
    };

    let r = rec.clone();
    DefaultStacklessOnlyTaskSystem::builder().workers(1).run_async(async move {
        let r1 = r.clone();
        let r2 = r.clone();

        // Pushed LIFO: c1 first (ends up underneath), c2 second (ends up on
        // top). Both `.await`s here only drive `SpawnAction` to `Ready`
        // (registration), not the child body itself -- neither child has
        // run yet.
        let h1 = DefaultStacklessOnlyTaskSystem::spawn(move || async move {
            r1(3);
        })
        .await;
        let h2 = DefaultStacklessOnlyTaskSystem::spawn(move || async move {
            r2(2);
        })
        .await;

        r(1);
        DefaultStacklessOnlyTaskSystem::yield_now().await;
        r(4);

        h1.await;
        h2.await;
    });

    assert_eq!(
        *order.lock().unwrap(),
        vec![1, 2, 3, 4],
        "yield_now re-queued the yielding task ahead of work already waiting"
    );
}

// ---------------------------------------------------------------------------
// parallel_call (ScopedStacklessTaskSystem) — `a` via `recurse`, `b` pushed
// and popped back by identity, direct-polled if un-stolen
// (`resumable::stackless::system::ScopedStacklessTaskSystem::parallel_call`,
// `docs/scoped-ult-promotion.md`). Untested anywhere in the crate before
// this session.
// ---------------------------------------------------------------------------

#[test]
fn parallel_call_basic() {
    DefaultStacklessOnlyTaskSystem::builder().workers(1).run_async(async {
        let (a, b) = DefaultStacklessOnlyTaskSystem::parallel_call(
            || async { 1 + 1 },
            || async { 2 + 2 },
        )
        .await;
        assert_eq!((a, b), (2, 4));
    });
}

/// Forces `b` to be stolen: `a` busy-spins with *no* `.await` inside the
/// loop (so it never yields control away — its single `poll()` call just
/// runs synchronously until `b_ran` is set), while an idle second worker's
/// own steal loop (a separate OS thread, needs nothing from this one) picks
/// up `b`. Mirrors `tests/stackful_only.rs`'s test of the same name: this
/// can only complete via a real steal, so it hangs (bounded, fails loudly)
/// rather than silently passing if stealing is broken.
#[test]
fn parallel_call_stolen_branch_actually_runs() {
    DefaultStacklessOnlyTaskSystem::builder().workers(2).run_async(async {
        let b_ran = Arc::new(AtomicBool::new(false));
        let b_ran2 = Arc::clone(&b_ran);
        let (a, b) = DefaultStacklessOnlyTaskSystem::parallel_call(
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

/// Un-stolen `b` panics: single worker, so `b` can never be stolen — forces
/// the direct-poll fast path (`BranchPoll`), which has no `catch_unwind` of
/// its own and must propagate the panic by unwinding normally.
#[test]
fn parallel_call_unstolen_branch_panic_propagates() {
    DefaultStacklessOnlyTaskSystem::builder().workers(1).run_async(async {
        let result = DefaultStacklessOnlyTaskSystem::parallel_call(
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

/// Stolen `b` panics: same forced-steal shape as
/// `parallel_call_stolen_branch_actually_runs`, but `b` panics right after
/// signalling `a`. Must propagate through the existing `poll_spawned_task`/
/// `JoinHandle` machinery, exactly like `spawn_async_panic_propagates_via_await`
/// already does for a plain spawned task.
#[test]
fn parallel_call_stolen_branch_panic_propagates() {
    DefaultStacklessOnlyTaskSystem::builder().workers(2).run_async(async {
        let b_ran = Arc::new(AtomicBool::new(false));
        let b_ran2 = Arc::clone(&b_ran);
        let result = DefaultStacklessOnlyTaskSystem::parallel_call(
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

/// E0733 regression check: `a` is wrapped via `recurse` specifically so a
/// genuinely self-recursive `Fa`/`Fb` (this function calling itself through
/// `parallel_call`) doesn't hit the infinite-size cycle a bare `mk().await`
/// would. This is also a nested-recursion stress test: many iterations,
/// multiple workers, so `try_pop`'s identity check runs against real
/// concurrent steals repeatedly, not just once.
fn parallel_fib_async(n: u64) -> impl Future<Output = u64> + Send {
    async move {
        if n <= 1 {
            return n;
        }
        let (a, b) = DefaultStacklessOnlyTaskSystem::parallel_call(
            move || parallel_fib_async(n - 1),
            move || parallel_fib_async(n - 2),
        )
        .await;
        a + b
    }
}

#[test]
fn parallel_call_nested_recursive_e0733_regression_and_stress() {
    for _ in 0..30 {
        DefaultStacklessOnlyTaskSystem::builder().workers(4).run_async(async {
            assert_eq!(parallel_fib_async(20).await, 6_765);
        });
    }
}

// ---------------------------------------------------------------------------
// parallel_call's warm-reuse cache for `b`'s pool-backed descriptor
// (`UltWorker::branch_warm_async`, `docs/scoped-ult-promotion.md` §9.12):
// un-stolen branches keep their pool slot and get reused across calls,
// skipping `alloc_async_task`.
// ---------------------------------------------------------------------------

/// Single worker, so every branch below takes the un-stolen fast path — the
/// only path that populates or reads back from the warm cache. Alternates
/// several genuinely different closure/future shapes back to back: each
/// later call should be popping a descriptor whose storage was last written
/// for a *different* `Fb` than the one about to run.
#[test]
fn parallel_call_warm_cache_reused_correctly_across_different_closure_types() {
    DefaultStacklessOnlyTaskSystem::builder().workers(1).run_async(async {
        for round in 0..50u64 {
            let (a, b) = DefaultStacklessOnlyTaskSystem::parallel_call(
                move || async move { round + 1 },
                move || async move { round + 2 },
            )
            .await;
            assert_eq!((a, b), (round + 1, round + 2));

            let v = vec![round as u32, round as u32 + 1, round as u32 + 2];
            let x = round;
            let y = round * 2;
            let (a2, b2) = DefaultStacklessOnlyTaskSystem::parallel_call(
                move || async move { v.iter().sum::<u32>() },
                move || async move { (x + y) as u32 },
            )
            .await;
            assert_eq!(a2, (round as u32) + (round as u32 + 1) + (round as u32 + 2));
            assert_eq!(b2, (round + round * 2) as u32);

            let touched = Arc::new(AtomicBool::new(false));
            let touched2 = Arc::clone(&touched);
            let ((), rc) = DefaultStacklessOnlyTaskSystem::parallel_call(
                move || async move { touched2.store(true, Ordering::Relaxed); },
                move || async move { round },
            )
            .await;
            assert!(touched.load(Ordering::Relaxed));
            assert_eq!(rc, round);
        }
    });
}

/// A closure/future whose captured data exceeds `ASYNC_POOL_SIZE` (512
/// bytes): exercises the oversized-fallback path (dynamic per-call
/// allocation, never warm-cached — exactly what every `parallel_call` did
/// before this cache existed).
#[test]
fn parallel_call_oversized_closure_falls_back_correctly() {
    DefaultStacklessOnlyTaskSystem::builder().workers(2).run_async(async {
        let big = [7u64; 80]; // 640 bytes, well over ASYNC_POOL_SIZE
        let (a, b) = DefaultStacklessOnlyTaskSystem::parallel_call(
            || async { 1u64 },
            move || async move { big.iter().sum::<u64>() },
        )
        .await;
        assert_eq!((a, b), (1, 7 * 80));
    });
}
