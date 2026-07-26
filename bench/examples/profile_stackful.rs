//! Scratch binary for sampling profilers — stackful-only counterpart to
//! `profile_stackless`, for apples-to-apples comparison of where time goes.

use cmpth_bench::BenchSystem;
use std::sync::{Arc, Mutex};

fn main() {
    let n: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(32);
    let workers: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    loop {
        let result = Arc::new(Mutex::new(0u64));
        let result2 = Arc::clone(&result);
        cmpth_bench::StackfulOnlyBench::run(workers, move || {
            *result2.lock().unwrap() = cmpth_bench::fib::<cmpth_bench::StackfulOnlyBench>(n);
        });
        let r = *result.lock().unwrap();
        std::hint::black_box(r);
    }
}
