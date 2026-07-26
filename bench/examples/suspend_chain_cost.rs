//! Scratch: isolates the cost of a *genuine* suspend+resume through N
//! nested `.await` layers (pure `std::future::Future` dispatch, no cmpth
//! scheduler involved) and compares it against the cost of a real stackful
//! ULT context switch (cmpth's `yield_now`, ping-ponging between two ULTs).
//!
//! Not part of the criterion suite — a one-off measurement for a specific
//! question: does suspending through a deep `.await` chain get more
//! expensive than a stackful context switch as the chain gets deeper?

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Instant;

fn block_on_busy(mut fut: Pin<&mut dyn Future<Output = ()>>) {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => continue,
        }
    }
}

/// `depth` layers of plain `async move { inner.await }` wrapping a single
/// genuine suspend point (`cmpth::future::yield_now`: Pending once, wakes
/// itself, Ready next). Built at runtime (not via a recursive `async fn`,
/// so no E0733) via repeated boxing — each layer is its own, non-recursive
/// anonymous type.
fn build_chain(depth: usize) -> Pin<Box<dyn Future<Output = ()>>> {
    let mut fut: Pin<Box<dyn Future<Output = ()>>> = Box::pin(cmpth::future::yield_now());
    for _ in 0..depth {
        fut = Box::pin(async move { fut.await });
    }
    fut
}

fn measure_build_only(depth: usize, iters: u64) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        let fut = build_chain(depth);
        std::hint::black_box(&fut);
    }
    start.elapsed().as_nanos() as f64 / iters as f64
}

fn measure_build_and_run(depth: usize, iters: u64) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        let mut fut = build_chain(depth);
        block_on_busy(fut.as_mut());
    }
    start.elapsed().as_nanos() as f64 / iters as f64
}

/// Real stackful ULT context switch cost: two ULTs ping-ponging via
/// `ThreadSystem::yield_now`, `iters` round trips total.
fn measure_stackful_switch(iters: u64) -> f64 {
    use cmpth::ThreadSystem;
    let counter = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    cmpth::default::run(2, move || {
        let counter2 = Arc::clone(&counter);
        let h = cmpth::default::spawn(move || {
            while counter2.load(Ordering::Relaxed) < iters {
                cmpth::DefaultDualTaskSystem::yield_now();
            }
        });
        while counter.load(Ordering::Relaxed) < iters {
            counter.fetch_add(1, Ordering::Relaxed);
            cmpth::DefaultDualTaskSystem::yield_now();
        }
        h.join().unwrap();
    });
    start.elapsed().as_nanos() as f64 / iters as f64
}

fn main() {
    let iters: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);

    let switch_ns = measure_stackful_switch(iters.min(2_000_000));
    println!("stackful context switch (yield_now ping-pong): {switch_ns:.2} ns/switch\n");

    println!("{:>7}  {:>12}  {:>12}  {:>12}", "depth", "build-only", "build+run", "run-only(~)");
    for &depth in &[0usize, 1, 5, 10, 50, 100, 500, 1000, 5000] {
        let n = if depth >= 1000 { iters / 20 } else { iters };
        let build = measure_build_only(depth, n);
        let total = measure_build_and_run(depth, n);
        println!("{depth:>7}  {build:>9.2} ns  {total:>9.2} ns  {:>9.2} ns", total - build);
    }
}
