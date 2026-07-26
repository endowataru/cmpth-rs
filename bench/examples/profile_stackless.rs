//! Scratch binary for sampling profilers (macOS `sample`, `perf`, etc.) —
//! runs stackless-only fib in a loop for long enough to gather a useful
//! trace. Not part of the criterion suite.

fn main() {
    let n: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(32);
    let workers: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    loop {
        let r = cmpth_bench::run_fib_async::<cmpth_bench::AsyncOnlySystem>(workers, n);
        std::hint::black_box(r);
    }
}
