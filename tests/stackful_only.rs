//! End-to-end tests for [`cmpth::DefaultStackfulOnlyTaskSystem`]:
//! `execute`'s dispatch is `execute_stackful` (always a real context switch,
//! no `poll_fn` tag check) rather than `execute_dual`. Nothing in these
//! tests calls `spawn_async` on this system — its dispatch never checks the
//! `poll_fn` tag, so an async task would be mis-handled if one ever landed
//! here — so this exercises exactly the branch-free path
//! `execute_stackful`/`pop_or_root_stackful` were built for.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use cmpth::{
    BlockOnSystem, DefaultStackfulOnlyTaskSystem, JoinHandleLike, ScopedStackfulTaskSystem,
    StackfulBuilder, StackfulInitSystem, SpawnableStackfulTaskSystem,
};

#[test]
fn spawn_join_basic() {
    DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| {
        let h = DefaultStackfulOnlyTaskSystem::spawn(|| 6 * 7);
        assert_eq!(JoinHandleLike::join(h), 42);
    });
}

#[test]
fn spawn_join_many_parallel() {
    let counter = Arc::new(AtomicU64::new(0));
    let counter2 = Arc::clone(&counter);
    DefaultStackfulOnlyTaskSystem::builder().workers(4).run(move || {
        let handles: Vec<_> = (0..200)
            .map(|i| {
                let counter = Arc::clone(&counter2);
                DefaultStackfulOnlyTaskSystem::spawn(move || {
                    counter.fetch_add(1, Ordering::Relaxed);
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
    DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| {
        let h = DefaultStackfulOnlyTaskSystem::spawn(|| {
            let inner = DefaultStackfulOnlyTaskSystem::spawn(|| 10);
            JoinHandleLike::join(inner) + 5
        });
        assert_eq!(JoinHandleLike::join(h), 15);
    });
}

#[test]
fn spawn_panic_propagates() {
    DefaultStackfulOnlyTaskSystem::builder().workers(1).run(|| {
        let h = DefaultStackfulOnlyTaskSystem::spawn::<(), _>(|| panic!("boom"));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| h.join()));
        assert!(result.is_err() || result.unwrap().is_err());
    });
}

#[test]
fn yield_now_roundtrips() {
    DefaultStackfulOnlyTaskSystem::builder().workers(1).run(|| {
        for _ in 0..1000 {
            DefaultStackfulOnlyTaskSystem::yield_now();
        }
    });
}

// ---------------------------------------------------------------------------
// parallel_call — the make_context-backed fork-parent-first path
// (`docs/scoped-ult-promotion.md` §9.8.4). `standalone_init_then_spawn_join_and_parallel_call_work`
// above already covers the un-stolen path under normal conditions; these
// exercise the two things nothing else does: `branch_entry` actually being
// switched into for the first time on a *different* worker (`make_context`
// built the context correctly), and panic propagation through each of the
// two very differently-shaped completion routes (plain inline call vs. the
// full `JoinHandle`/`join_state` protocol).
// ---------------------------------------------------------------------------

/// Forces the branch to be stolen: `a` spins until `b` has actually run,
/// which — since `parallel_call` only pops `b` back locally *after* `a`
/// returns — is only possible if the idle second worker steals and runs it
/// first. This can only pass by actually switching into a
/// `make_context`-built context for the first time (`branch_entry`); if
/// that machinery were broken, this hangs (bounded, so it fails loudly
/// instead of wedging the test run) rather than silently passing.
#[test]
fn parallel_call_stolen_branch_actually_runs() {
    DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| {
        let b_ran = Arc::new(AtomicBool::new(false));
        let b_ran2 = Arc::clone(&b_ran);
        let (ra, rb) = DefaultStackfulOnlyTaskSystem::parallel_call(
            move || {
                let mut spins: u64 = 0;
                while !b_ran2.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                    spins += 1;
                    assert!(spins < 200_000_000, "b never ran -- steal did not happen (or branch_entry is broken)");
                }
                1u64
            },
            move || {
                b_ran.store(true, Ordering::Release);
                2u64
            },
        );
        assert_eq!((ra, rb), (1, 2));
    });
}

/// Un-stolen `b` panics: only one worker, so `b` can never be stolen —
/// forces the plain-inline-call fast path, which propagates the panic
/// directly (no `catch_unwind` wrapper there, matching `scoped`'s own
/// inline fast path).
#[test]
fn parallel_call_unstolen_branch_panic_propagates() {
    DefaultStackfulOnlyTaskSystem::builder().workers(1).run(|| {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            DefaultStackfulOnlyTaskSystem::parallel_call(|| 1u64, || -> u64 { panic!("boom-b-inline") })
        }));
        assert!(result.is_err(), "un-stolen branch panic should propagate through parallel_call");
    });
}

/// Stolen `b` panics: same forced-steal shape as
/// `parallel_call_stolen_branch_actually_runs`, but `b` panics right after
/// signalling `a`. The panic must propagate through `branch_entry`'s
/// `catch_unwind` -> `exit_with_result` -> `JoinHandle::join`'s
/// `resume_unwind`, exactly like `spawn`+`join` already does
/// (`spawn_panic_propagates`) — this is the first test exercising that same
/// route reached via `parallel_call` instead.
#[test]
fn parallel_call_stolen_branch_panic_propagates() {
    DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| {
        let b_ran = Arc::new(AtomicBool::new(false));
        let b_ran2 = Arc::clone(&b_ran);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            DefaultStackfulOnlyTaskSystem::parallel_call(
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

/// Regression guard: `parallel_call` must re-derive its worker after `a`
/// returns, not keep using the one captured before `a` ran. `a` here
/// recurses through nested `parallel_call`s of its own, so any inner branch
/// that gets stolen and joined migrates the calling ULT to a different
/// worker before this call's own `a` returns. The bug this catches (found
/// via `cargo bench --bench fib`, not by any of the tests above — they only
/// ever used a non-recursive `a`, which can never migrate): using the stale
/// worker afterward manipulates a *different* worker's deque/pool from a
/// different OS thread, corrupting shared queue state and eventually
/// producing a "double-resume" (`SIGSEGV` in release, a `debug_assert!` in
/// debug) once some other, unrelated task happens to reuse the same
/// descriptor. A single call with a trivial `a` cannot exercise this — the
/// bug needs real recursive depth and enough workers that a steal actually
/// happens, repeated enough times to hit it reliably.
fn parallel_fib(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let (a, b) = DefaultStackfulOnlyTaskSystem::parallel_call(move || parallel_fib(n - 1), move || parallel_fib(n - 2));
    a + b
}

#[test]
fn parallel_call_nested_recursive_nothing_corrupts_across_worker_migration() {
    for _ in 0..30 {
        DefaultStackfulOnlyTaskSystem::builder().workers(4).run(|| {
            assert_eq!(parallel_fib(24), 46_368);
        });
    }
}

// ---------------------------------------------------------------------------
// parallel_call's warm-reuse cache (`BranchWarmPool`,
// `docs/scoped-ult-promotion.md` §9.10): un-stolen branches keep their
// `ctx`/fixed-offset header valid and get reused across calls, skipping
// `ContextPolicy::make_context`. These tests are the ones that can actually
// distinguish "correct because it happens to rebuild everything every time"
// from "correct even when reusing a stale-looking descriptor".
// ---------------------------------------------------------------------------

/// A single worker, so every branch below takes the un-stolen fast path —
/// the *only* path that ever populates or reads back from the warm cache.
/// Alternates several genuinely different closure shapes (captured byte
/// count, alignment, and result type all differ) back to back on the same
/// worker: each later call should be popping a descriptor whose `ctx` was
/// built — and whose header region was last written — for a *different*
/// type than the one it's about to run. Wrong results here would mean the
/// fixed dispatch-slot/`f_ptr`/`result_ptr` offsets aren't actually
/// type-independent the way the design assumes.
#[test]
fn parallel_call_warm_cache_reused_correctly_across_different_closure_types() {
    DefaultStackfulOnlyTaskSystem::builder().workers(1).run(|| {
        for round in 0..50u64 {
            // Round shape 1: tiny closures, u64 result.
            let (a, b) = DefaultStackfulOnlyTaskSystem::parallel_call(move || round + 1, move || round + 2);
            assert_eq!((a, b), (round + 1, round + 2));

            // Round shape 2: a `Vec` + several scalars captured, `u32` result
            // — a different size, alignment, and result type than shape 1.
            let v = vec![round as u32, round as u32 + 1, round as u32 + 2];
            let x = round;
            let y = round * 2;
            let (a2, b2) = DefaultStackfulOnlyTaskSystem::parallel_call(
                move || v.iter().sum::<u32>(),
                move || (x + y) as u32,
            );
            assert_eq!(a2, (round as u32) + (round as u32 + 1) + (round as u32 + 2));
            assert_eq!(b2, (round + round * 2) as u32);

            // Round shape 3: unit result, to exercise a zero-sized `Rb`.
            let touched = Arc::new(AtomicBool::new(false));
            let touched2 = Arc::clone(&touched);
            let ((), rc) = DefaultStackfulOnlyTaskSystem::parallel_call(
                move || { touched2.store(true, Ordering::Relaxed); },
                move || round,
            );
            assert!(touched.load(Ordering::Relaxed));
            assert_eq!(rc, round);
        }
    });
}

/// A closure whose captured data exceeds `BRANCH_HEADER_BUDGET`: exercises
/// the oversized-fallback path (dynamic `exec_top`, never warm-cached —
/// exactly what every `parallel_call` did before this cache existed).
/// Correctness only, since this path is deliberately never sped up.
#[test]
fn parallel_call_oversized_closure_falls_back_correctly() {
    DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| {
        let big = [7u64; 40]; // 320 bytes, well over the fixed budget
        let (a, b) = DefaultStackfulOnlyTaskSystem::parallel_call(
            || 1u64,
            move || big.iter().sum::<u64>(),
        );
        assert_eq!((a, b), (1, 7 * 40));
    });
}

// ---------------------------------------------------------------------------
// block_on — exercises ResumablePoller's actual park/wake path, not just the
// always-ready case (see `traits::stackful::SpawnableStackfulTaskSystem::block_on`'s
// doctest, which only ever polls `async { 6 * 7 }` once and never parks).
// ---------------------------------------------------------------------------

/// Future that yields exactly once before becoming ready, notifying itself
/// *during* the same `poll()` call that returns `Pending` — exercises
/// `ResumablePollerSlot`'s "wake raced in before park committed" cancel
/// path (`decide_park`'s NOTIFIED branch), the same race
/// `block_on_yield_once` (tests/integration.rs) checks for `DefaultDualTaskSystem`.
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
    DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| {
        let v = DefaultStackfulOnlyTaskSystem::block_on(YieldOnce(false));
        assert_eq!(v, 42);
    });
}

/// Future that is genuinely parked, then woken from a *different* ULT via a
/// cloned waker — exercises `ResumablePollerSlot`'s real PARKED ->
/// `ClaimedParked` -> `push_continuation` path (the `Arc`-backed clone must
/// stay valid across the hand-off to the waking ULT). Mirrors
/// `block_on_cross_ult_wake` (tests/integration.rs) for
/// `DefaultStackfulOnlyTaskSystem`.
#[test]
fn block_on_cross_ult_wake() {
    use std::sync::atomic::{AtomicBool, Ordering as Ord};
    use std::sync::Mutex;
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
            *self.slot.lock().unwrap() = Some(cx.waker().clone());
            std::task::Poll::Pending
        }
    }

    DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| {
        let slot: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));
        let done: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

        let slot2 = Arc::clone(&slot);
        let done2 = Arc::clone(&done);
        let waker_h = DefaultStackfulOnlyTaskSystem::spawn(move || {
            loop {
                let w = slot2.lock().unwrap().take();
                if let Some(w) = w {
                    done2.store(true, Ord::Release); // must happen before wake()
                    w.wake();
                    break;
                }
                DefaultStackfulOnlyTaskSystem::yield_now();
            }
        });

        DefaultStackfulOnlyTaskSystem::block_on(WaitForWake { slot, done });
        JoinHandleLike::join(waker_h);
    });
}

// ---------------------------------------------------------------------------
// stack_size — StackfulBuilder::stack_size
// ---------------------------------------------------------------------------

/// Touches a 200 KiB on-stack array — well above the system's default 64
/// KiB `STACK_SIZE` (a stack that size couldn't even fit this array's own
/// storage) but comfortably under the 512 KiB this test configures via
/// `StackfulBuilder::stack_size`. `black_box` on the mutable reference (to
/// stop the write from being proven dead) and on the final sum (to stop the
/// read from being proven constant) keep the optimizer from eliding the
/// array altogether — without them there would be no real stack use to
/// exercise, even at `opt-level=0`. If `stack_size` were silently ignored
/// (falling back to the 64 KiB default), spawning this task would overrun
/// the ULT's guard page and abort/crash rather than pass.
#[inline(never)]
fn consume_stack() -> u64 {
    let mut buf = [0u8; 200 * 1024];
    std::hint::black_box(&mut buf);
    buf[0] = 1;
    buf[buf.len() - 1] = 41;
    std::hint::black_box(buf[0] as u64 + buf[buf.len() - 1] as u64)
}

#[test]
fn stack_size_configurable_via_builder() {
    const CONFIGURED_STACK_SIZE: usize = 512 * 1024;
    DefaultStackfulOnlyTaskSystem::builder()
        .workers(2)
        .stack_size(CONFIGURED_STACK_SIZE)
        .run(|| {
            let h = DefaultStackfulOnlyTaskSystem::spawn(consume_stack);
            assert_eq!(JoinHandleLike::join(h), 42);
        });
}

// ---------------------------------------------------------------------------
// Standalone init — StackfulBuilder::init, replacing bracketing `run`
// ---------------------------------------------------------------------------

/// `init()` returns with the caller already running as an ordinary ULT on
/// the pool: `spawn`/`join` work exactly as they do inside `run`, and
/// `parallel_call` — which panics ("called outside `run`" / no live pool)
/// when reached from plain, non-worker code — now works too. That's the
/// headline capability gain standalone init adds over the old bracketing
/// `run`: no closure to wrap arbitrary top-level (e.g. `main`) code in.
#[test]
fn standalone_init_then_spawn_join_and_parallel_call_work() {
    let guard = DefaultStackfulOnlyTaskSystem::builder().workers(2).init();

    let h = DefaultStackfulOnlyTaskSystem::spawn(|| 6 * 7);
    assert_eq!(JoinHandleLike::join(h), 42);

    let (a, b) = DefaultStackfulOnlyTaskSystem::parallel_call(|| 1 + 1, || 2 + 2);
    assert_eq!((a, b), (2, 4));

    // A handful of spawn/join rounds after parallel_call, to make sure the
    // pool is still in a sane state (root_cont/deque bookkeeping) after
    // mixing both capabilities on the same standalone-initialized pool.
    let handles: Vec<_> = (0..20).map(|i| DefaultStackfulOnlyTaskSystem::spawn(move || i * 2u64)).collect();
    let mut sum = 0u64;
    for h in handles {
        sum += JoinHandleLike::join(h);
    }
    assert_eq!(sum, (0..20u64).map(|i| i * 2).sum::<u64>());

    drop(guard);
}

/// `Builder::run` is defined in terms of `init`/`Drop`, wrapped in
/// `catch_unwind`/`resume_unwind` so a panic inside `f` never reaches the
/// guard's `Drop` while unwinding (which would abort — see that `Drop`
/// impl) — instead it's caught before `drop(guard)` runs, and re-raised
/// only afterward, on the caller's own thread. Confirms that whole
/// round-trip actually propagates the panic rather than swallowing it or
/// aborting.
#[test]
fn run_propagates_a_panic_in_f_to_the_caller() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| -> u64 {
            panic!("boom from run's root closure");
        });
    }));
    let err = result.expect_err("run should propagate the panic, not swallow it");
    let msg = err
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| err.downcast_ref::<String>().map(String::as_str))
        .unwrap_or_default();
    assert!(msg.contains("boom from run's root closure"), "unexpected panic payload: {msg:?}");

    // The pool from the panicking `run` call fully tore down (workers
    // joined, TLS cleared) before `run` re-raised -- an entirely new `run`
    // call works normally right after.
    DefaultStackfulOnlyTaskSystem::builder().workers(2).run(|| {
        assert_eq!(JoinHandleLike::join(DefaultStackfulOnlyTaskSystem::spawn(|| 1 + 1)), 2);
    });
}

/// `yield_now()` must let work that is *already queued* on this worker run
/// before the yielding task resumes.
///
/// Needs **two** queued items to be observable. `suspend_to_sched` pops the
/// next continuation *before* its callback pushes the yielding task
/// (`resumable/stackful/worker.rs`), so with a single queued item both a
/// LIFO and a FIFO re-queue behave identically. The degradation only shows
/// up once something else is still queued behind the item that was popped.
///
/// `spawn` is child-first: it switches into the child and the child
/// publishes the *parent's* continuation. Two nested spawns therefore leave
/// two continuations queued (`c1` on top of `main`) while `c2` runs:
///
/// - fair   -> c2 yields; c1 resumes; then `main` (queued first) ; then c2
/// - unfair -> c2 yields; c1 resumes; then **c2 jumps back ahead of main**
#[test]
fn yield_now_lets_already_queued_work_run_first() {
    let order = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let rec = {
        let o = Arc::clone(&order);
        move |v: u8| o.lock().unwrap().push(v)
    };

    let r = rec.clone();
    DefaultStackfulOnlyTaskSystem::builder().workers(1).run(move || {
        let r1 = r.clone();
        let h1 = DefaultStackfulOnlyTaskSystem::spawn(move || {
            let r2 = r1.clone();
            let h2 = DefaultStackfulOnlyTaskSystem::spawn(move || {
                r2(1);
                DefaultStackfulOnlyTaskSystem::yield_now();
                r2(4);
            });
            r1(2);
            JoinHandleLike::join(h2);
        });
        r(3);
        JoinHandleLike::join(h1);
    });

    assert_eq!(
        *order.lock().unwrap(),
        vec![1, 2, 3, 4],
        "yield_now re-queued the yielding task ahead of work already waiting"
    );
}
