//! Scratch: isolates the cost of the `ContextPolicy` primitives themselves.
//!
//! Three shapes, all on a **single worker** so nothing can be stolen and the
//! measured cost is the switch machinery rather than the scheduler's
//! load balancing:
//!
//! - `pingpong` — two ULTs yielding to each other: one full
//!   `swap_context` per reported switch.
//! - `pcall` — `parallel_call` with empty bodies: the fork-parent-first path,
//!   which never actually switches but does run `make_context` (cold) or the
//!   warm-cache reuse (hot). High register pressure, so this is the shape
//!   where inlining the switch primitives is expected to pay off the most
//!   (`docs/scoped-ult-promotion.md` §9.14.3, experiment 3).
//! - `spawnjoin` — `spawn` + `join` of an empty task: child-first fork, i.e.
//!   `save_context` on the way in and `restore_context`/`swap_context` on the
//!   way out.
//!
//! Generic over the system type throughout; the single concrete choice is in
//! `main`.

use cmpth::{
    JoinHandleLike, ScopedStackfulTaskSystem, SpawnableStackfulTaskSystem, StackfulBuilder,
    StackfulInitSystem,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Two ULTs on one worker, `iters` yields each way.
fn pingpong<S>(iters: u64) -> f64
where
    S: SpawnableStackfulTaskSystem + StackfulInitSystem,
{
    let done = Arc::new(AtomicU64::new(0));
    let done2 = Arc::clone(&done);
    let elapsed = Arc::new(AtomicU64::new(0));
    let elapsed2 = Arc::clone(&elapsed);
    S::builder().workers(1).run(move || {
        let h = S::spawn(move || {
            while done2.load(Ordering::Relaxed) == 0 {
                S::yield_now();
            }
        });
        // Let the child reach its first yield before timing.
        S::yield_now();
        let start = Instant::now();
        for _ in 0..iters {
            S::yield_now();
        }
        elapsed2.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        done.store(1, Ordering::Relaxed);
        h.join();
    });
    // Each loop iteration is one switch out and one switch back.
    elapsed.load(Ordering::Relaxed) as f64 / (2 * iters) as f64
}

/// `parallel_call` with empty bodies on one worker.
fn pcall<S>(iters: u64) -> f64
where
    S: ScopedStackfulTaskSystem + StackfulInitSystem,
{
    let elapsed = Arc::new(AtomicU64::new(0));
    let elapsed2 = Arc::clone(&elapsed);
    S::builder().workers(1).run(move || {
        let start = Instant::now();
        for _ in 0..iters {
            let (a, b) = S::parallel_call(|| std::hint::black_box(0u64), || std::hint::black_box(0u64));
            std::hint::black_box((a, b));
        }
        elapsed2.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    });
    elapsed.load(Ordering::Relaxed) as f64 / iters as f64
}

/// `spawn` + `join` of an empty task on one worker.
fn spawnjoin<S>(iters: u64) -> f64
where
    S: SpawnableStackfulTaskSystem + StackfulInitSystem,
{
    let elapsed = Arc::new(AtomicU64::new(0));
    let elapsed2 = Arc::clone(&elapsed);
    S::builder().workers(1).run(move || {
        let start = Instant::now();
        for _ in 0..iters {
            let h = S::spawn(|| std::hint::black_box(0u64));
            std::hint::black_box(h.join());
        }
        elapsed2.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    });
    elapsed.load(Ordering::Relaxed) as f64 / iters as f64
}

fn main() {
    type S = cmpth::DefaultStackfulOnlyTaskSystem;

    let iters: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2_000_000);
    let reps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    println!("{:>12}  {:>14}  {:>14}  {:>14}", "rep", "pingpong", "pcall", "spawnjoin");
    let mut best = (f64::MAX, f64::MAX, f64::MAX);
    for r in 0..reps {
        let p = pingpong::<S>(iters);
        let c = pcall::<S>(iters);
        let s = spawnjoin::<S>(iters);
        best = (best.0.min(p), best.1.min(c), best.2.min(s));
        println!("{r:>12}  {p:>11.2} ns  {c:>11.2} ns  {s:>11.2} ns");
    }
    println!(
        "{:>12}  {:>11.2} ns  {:>11.2} ns  {:>11.2} ns",
        "best", best.0, best.1, best.2
    );
}
