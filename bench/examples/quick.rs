//! Quick spawn/join overhead microbenchmark (fast iteration during tuning).
//!
//! Compares the default configuration (heap stacks + TLS lookup) against
//! the arena configuration (mmap arena stacks + sp-based lookup).
//!
//! ```sh
//! cargo run --release -p cmpth-bench --example quick
//! ```

use std::time::Instant;

use cmpth::ult::system::UltSystem;
use cmpth::{JoinHandleLike, ThreadSystem};

cmpth::ult_system! {
    /// Arena stacks + sp-based worker lookup.
    pub struct ArenaSys {
        base:        cmpth::OsSystem,
        context:     cmpth::NativeContext,
        deque:       cmpth::CrossbeamDeque,
        stack_size:  64 * 1024,
        stack_alloc: cmpth::ArenaStack,
        lookup:      cmpth::SpCurrent,
    }
}

cmpth::ult_system! {
    /// Control: arena stacks but classic TLS lookup — separates the cost of
    /// the stack allocator from the cost of the lookup.
    pub struct ArenaTlsSys {
        base:        cmpth::OsSystem,
        context:     cmpth::NativeContext,
        deque:       cmpth::CrossbeamDeque,
        stack_size:  64 * 1024,
        stack_alloc: cmpth::ArenaStack,
        lookup:      cmpth::TlsCurrent,
    }
}

fn fib<S: ThreadSystem>(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let h = S::spawn(move || fib::<S>(n - 1));
    let r2 = fib::<S>(n - 2);
    JoinHandleLike::join(h) + r2
}

fn bench_one<S: UltSystem>(label: &'static str) {
    S::run(1, move || {
        // Warm up the task pool and caches.
        assert_eq!(fib::<S>(20), 6765);

        // --- spawn + join round-trip (leaf task); min of 5 rounds ---
        let n = 500_000u64;
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            for i in 0..n {
                let h = S::spawn(move || i);
                std::hint::black_box(JoinHandleLike::join(h));
            }
            best = best.min(t.elapsed().as_nanos() as f64 / n as f64);
        }
        println!("[{label}] spawn+join : {best:6.1} ns/pair (min of 5)");

        // --- yield ping-pong: isolates switch + deque cost ---
        let yields = 500_000u64;
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            let h1 = S::spawn(move || {
                for _ in 0..yields {
                    S::yield_now();
                }
            });
            let h2 = S::spawn(move || {
                for _ in 0..yields {
                    S::yield_now();
                }
            });
            JoinHandleLike::join(h1);
            JoinHandleLike::join(h2);
            best = best.min(t.elapsed().as_nanos() as f64 / (2 * yields) as f64);
        }
        println!("[{label}] yield      : {best:6.1} ns/yield (min of 5)");

        // --- fib(34): 9,227,464 spawn/join pairs per run; min of 5 ---
        const PAIRS: f64 = 9_227_464.0;
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            assert_eq!(fib::<S>(34), 5_702_887);
            best = best.min(t.elapsed().as_secs_f64());
        }
        let ms_per_run = best * 1e3;
        let ns_per_pair = best * 1e9 / PAIRS;
        println!("[{label}] fib(34)    : {ms_per_run:6.2} ms/run  ({ns_per_pair:.1} ns/pair, min of 5)");
    });
}

fn main() {
    bench_one::<cmpth::DefaultUltSystem>("heap+tls ");
    bench_one::<ArenaTlsSys>("arena+tls");
    bench_one::<ArenaSys>("arena+sp ");
}
