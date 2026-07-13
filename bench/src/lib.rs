//! Benchmark harness for cmpth and compatible threading libraries.
//!
//! # Usage
//!
//! Benchmark functions are generic over [`BenchSystem`].  To compare
//! libraries, pass a different system type — everything else stays the same.
//!
//! ```ignore
//! fn my_bench<S: BenchSystem>(c: &mut Criterion, label: &str, workers: usize) {
//!     c.bench_function(format!("{label}/fib34/{workers}w"), |b| {
//!         b.iter(|| S::run(workers, || assert_eq!(fib::<S>(34), 5_702_887)));
//!     });
//! }
//! // in bench file:
//! my_bench::<CmpthBench>(c, "cmpth", 4);
//! my_bench::<RayonBench>(c, "rayon", 4);
//! ```
//!
//! # par_join and deadlock safety
//!
//! Recursive fork-join programs deadlock in OS-thread pools when all threads
//! block waiting for tasks that are still in the queue.  [`BenchSystem::par_join`]
//! avoids this: ULT systems (cmpth, MassiveThreads) suspend the ULT while
//! keeping the OS thread free; Rayon overrides `par_join` with `rayon::join`
//! which actively work-steals while waiting.
//!
//! [`OsThreadBench`] does NOT override `par_join` and therefore deadlocks for
//! recursive benchmarks.  Use it only in `spawn_overhead` (non-recursive).

use cmpth::JoinHandleLike as _;

// ---------------------------------------------------------------------------
// BenchSystem trait
// ---------------------------------------------------------------------------

/// Minimal interface needed to run the benchmark suite against a threading
/// library.  One concrete type per library.
pub trait BenchSystem: Send + Sync + 'static {
    /// The join handle type returned by [`spawn`](BenchSystem::spawn).
    type JoinHandle<T: Send + 'static>: cmpth::JoinHandleLike<T>;

    /// Start the threading system with `num_workers` OS threads and run `f`
    /// as the root task.  Blocks until `f` and all tasks spawned by `f` finish.
    fn run(num_workers: usize, f: impl FnOnce() + Send + 'static);

    /// Spawn a task and return a handle that can be joined.
    fn spawn<T: Send + 'static>(
        f: impl FnOnce() -> T + Send + 'static,
    ) -> Self::JoinHandle<T>;

    /// Run `a` and `b` in parallel and return both results.
    ///
    /// The default implementation spawns `a` as a task, runs `b` on the
    /// current thread, then joins `a`.  This is correct for ULT systems but
    /// deadlocks for OS-thread pools under recursive use; those must override.
    fn par_join<RA, RB>(
        a: impl FnOnce() -> RA + Send + 'static,
        b: impl FnOnce() -> RB + Send + 'static,
    ) -> (RA, RB)
    where
        RA: Send + 'static,
        RB: Send + 'static,
    {
        let h = Self::spawn(a);
        let rb = b();
        (h.join(), rb)
    }
}

// ---------------------------------------------------------------------------
// CmpthBench — cmpth DefaultUltSystem
// ---------------------------------------------------------------------------

pub struct CmpthBench;

impl BenchSystem for CmpthBench {
    type JoinHandle<T: Send + 'static> =
        <cmpth::DefaultUltSystem as cmpth::ThreadSystem>::JoinHandle<T>;

    fn run(num_workers: usize, f: impl FnOnce() + Send + 'static) {
        use cmpth::UltSystem as _;
        cmpth::DefaultUltSystem::run(num_workers, f);
    }

    fn spawn<T: Send + 'static>(
        f: impl FnOnce() -> T + Send + 'static,
    ) -> Self::JoinHandle<T> {
        use cmpth::ThreadSystem as _;
        cmpth::DefaultUltSystem::spawn(f)
    }
}

// ---------------------------------------------------------------------------
// OsThreadBench — raw std::thread (non-recursive benchmarks only)
// ---------------------------------------------------------------------------

/// OS-thread baseline.  `num_workers` is ignored; each `spawn` creates a new
/// OS thread.  **Do not use for recursive fork-join benchmarks** (fib, nqueens)
/// because `par_join` blocks the OS thread, which deadlocks under recursion.
pub struct OsThreadBench;

pub struct OsJoinHandle<T>(std::thread::JoinHandle<T>);

impl<T: Send + 'static> cmpth::JoinHandleLike<T> for OsJoinHandle<T> {
    fn join(self) -> T {
        self.0.join().unwrap_or_else(|e| std::panic::resume_unwind(e))
    }
}

impl BenchSystem for OsThreadBench {
    type JoinHandle<T: Send + 'static> = OsJoinHandle<T>;

    fn run(_num_workers: usize, f: impl FnOnce() + Send + 'static) {
        f();
    }

    fn spawn<T: Send + 'static>(
        f: impl FnOnce() -> T + Send + 'static,
    ) -> OsJoinHandle<T> {
        OsJoinHandle(std::thread::spawn(f))
    }
}

// ---------------------------------------------------------------------------
// RayonBench — Rayon work-stealing thread pool
// ---------------------------------------------------------------------------

pub struct RayonBench;

pub struct RayonJoinHandle<T>(std::sync::mpsc::Receiver<std::thread::Result<T>>);

impl<T: Send + 'static> cmpth::JoinHandleLike<T> for RayonJoinHandle<T> {
    fn join(self) -> T {
        self.0
            .recv()
            .expect("rayon task sender dropped")
            .unwrap_or_else(|e| std::panic::resume_unwind(e))
    }
}

impl BenchSystem for RayonBench {
    type JoinHandle<T: Send + 'static> = RayonJoinHandle<T>;

    fn run(num_workers: usize, f: impl FnOnce() + Send + 'static) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .build()
            .expect("rayon pool build failed")
            .install(f);
    }

    fn spawn<T: Send + 'static>(
        f: impl FnOnce() -> T + Send + 'static,
    ) -> RayonJoinHandle<T> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        rayon::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            let _ = tx.send(result);
        });
        RayonJoinHandle(rx)
    }

    // Override the default: use rayon::join which work-steals while waiting,
    // avoiding the deadlock that blocking join causes under recursive fork-join.
    fn par_join<RA, RB>(
        a: impl FnOnce() -> RA + Send + 'static,
        b: impl FnOnce() -> RB + Send + 'static,
    ) -> (RA, RB)
    where
        RA: Send + 'static,
        RB: Send + 'static,
    {
        rayon::join(a, b)
    }
}

// ---------------------------------------------------------------------------
// MythBench — MassiveThreads (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "massivethreads")]
pub mod myth;
#[cfg(feature = "massivethreads")]
pub use myth::MythBench;

// ---------------------------------------------------------------------------
// MayBench — may stackful coroutines (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "may")]
pub mod may_bench;
#[cfg(feature = "may")]
pub use may_bench::MayBench;

// ---------------------------------------------------------------------------
// TokioBench — Tokio async runtime (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "tokio-rt")]
pub mod tokio_bench;
#[cfg(feature = "tokio-rt")]
pub use tokio_bench::TokioBench;

// ---------------------------------------------------------------------------
// ArgobotsBench — Argobots ULT (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "argobots")]
pub mod argobots;
#[cfg(feature = "argobots")]
pub use argobots::ArgobotsBench;

// ---------------------------------------------------------------------------
// Generic benchmark functions
// ---------------------------------------------------------------------------

/// Parallel Fibonacci via binary fork-join.
pub fn fib<S: BenchSystem>(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let (r1, r2) = S::par_join(move || fib::<S>(n - 1), move || fib::<S>(n - 2));
    r1 + r2
}

/// Count N-Queens solutions for an n×n board.
///
/// Parallel variant: candidates at each row are split into two halves and
/// processed with [`BenchSystem::par_join`] down to `par_depth` levels.
pub fn nqueens<S: BenchSystem>(placed: Vec<u32>, n: u32, par_depth: usize) -> u32 {
    let row = placed.len() as u32;
    if row == n {
        return 1;
    }
    let candidates: Vec<u32> = (0..n)
        .filter(|&col| {
            placed.iter().enumerate().all(|(r, &c)| {
                c != col
                    && (r as i32 - row as i32).unsigned_abs()
                        != (c as i32 - col as i32).unsigned_abs()
            })
        })
        .collect();
    nqueens_sum::<S>(candidates, placed, n, par_depth)
}

fn nqueens_sum<S: BenchSystem>(
    candidates: Vec<u32>,
    placed: Vec<u32>,
    n: u32,
    par_depth: usize,
) -> u32 {
    match candidates.len() {
        0 => 0,
        // Single candidate or serial cutoff: no spawning.
        1 => {
            let mut next = placed;
            next.push(candidates[0]);
            nqueens::<S>(next, n, par_depth.saturating_sub(1))
        }
        _ if par_depth == 0 => candidates
            .into_iter()
            .map(|col| {
                let mut next = placed.clone();
                next.push(col);
                nqueens::<S>(next, n, 0)
            })
            .sum(),
        // Binary split of the candidate list.
        _ => {
            let mid = candidates.len() / 2;
            let right = candidates[mid..].to_vec();
            let left = candidates[..mid].to_vec();
            let placed_r = placed.clone();
            let (l, r) = S::par_join(
                move || nqueens_sum::<S>(left, placed, n, par_depth - 1),
                move || nqueens_sum::<S>(right, placed_r, n, par_depth - 1),
            );
            l + r
        }
    }
}
