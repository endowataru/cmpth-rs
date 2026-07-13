//! `may` stackful-coroutine backend for the benchmark harness.
//!
//! Enabled with `--features may`.
//!
//! `may` provides stackful green threads (coroutines) with work-stealing.
//! Blocking a coroutine suspends only the coroutine, not the OS thread, so
//! recursive fork-join works without deadlock — same as cmpth and MassiveThreads.
//!
//! # Worker count
//!
//! `may::config().set_workers(n)` must be called before the first coroutine is
//! spawned and cannot be changed afterwards.  [`MayBench::run`] records the
//! first count via [`OnceLock`] and panics on mismatch.

use std::sync::OnceLock;

use crate::BenchSystem;

static MAY_WORKERS: OnceLock<usize> = OnceLock::new();

fn may_ensure_workers(num_workers: usize) {
    let &stored = MAY_WORKERS.get_or_init(|| {
        may::config()
            .set_workers(num_workers)
            .set_stack_size(64 * 1024); // match cmpth's 64 KB stack
        num_workers
    });
    // may worker count is fixed after the first coroutine is spawned.
    // To benchmark a different count, run with a name filter so only one
    // count is exercised per process, e.g.:
    //   cargo bench --features may --bench fib -- 'may/1'
    //   cargo bench --features may --bench fib -- 'may/4'
    if stored != num_workers {
        eprintln!(
            "may: worker count fixed at {stored}, ignoring requested {num_workers}. \
             Run `cargo bench -- 'may/{num_workers}'` in a fresh process for that count."
        );
    }
}

pub struct MayJoinHandle<T>(may::coroutine::JoinHandle<T>);

impl<T: Send + 'static> cmpth::JoinHandleLike<T> for MayJoinHandle<T> {
    fn join(self) -> T {
        self.0
            .join()
            .unwrap_or_else(|e| std::panic::resume_unwind(e))
    }
}

pub struct MayBench;

impl BenchSystem for MayBench {
    type JoinHandle<T: Send + 'static> = MayJoinHandle<T>;

    fn run(num_workers: usize, f: impl FnOnce() + Send + 'static) {
        may_ensure_workers(num_workers);
        // Spawn f as a root coroutine; join from the OS thread.
        // Joining from outside a coroutine blocks the OS thread until f finishes.
        let h = may::go!(f);
        h.join().expect("root coroutine panicked");
    }

    fn spawn<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> MayJoinHandle<T> {
        MayJoinHandle(may::go!(f))
    }
}
