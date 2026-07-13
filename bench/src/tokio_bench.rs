//! Tokio async-runtime backend for the benchmark harness.
//!
//! Enabled with `--features tokio-rt`.
//!
//! # Join mechanism
//!
//! `BenchSystem::join` must block the calling thread until the spawned task
//! finishes.  `Handle::block_on` inside `block_in_place` re-enters the runtime
//! context and deadlocks under recursive fork-join (the nested block_on
//! prevents the replacement worker from waking us up correctly).
//!
//! Instead we use `std::thread::park / unpark`: the spawned async task stores
//! its result in an `Arc<Mutex<Option<T>>>` and calls `thread::unpark` on the
//! waiting thread.  `block_in_place` signals Tokio to spawn a replacement
//! worker, so the pool stays live while this thread sleeps.
//!
//! # Worker count
//!
//! A fresh `Runtime` is created on each `run` call, so any worker count works.
//! Recursive `par_join` deadlocks when the pool is exhausted (all N workers
//! blocked waiting for tasks no one runs).  Use at least 2 workers for
//! recursive benchmarks; the bench files enforce this.

use std::sync::{Arc, Mutex};

use crate::BenchSystem;

// ---------------------------------------------------------------------------
// JoinHandle — park/unpark based synchronisation
// ---------------------------------------------------------------------------

pub struct TokioJoinHandle<T> {
    slot:    Arc<Mutex<Option<std::thread::Result<T>>>>,
    waiter:  std::thread::Thread,
    _inner:  tokio::task::JoinHandle<()>,
}

impl<T: Send + 'static> cmpth::JoinHandleLike<T> for TokioJoinHandle<T> {
    fn join(self) -> T {
        // Move this OS thread out of the async worker pool so Tokio spawns a
        // replacement.  The replacement runs the queued task which eventually
        // unparks us.
        tokio::task::block_in_place(|| {
            loop {
                if let Some(result) = self.slot.lock().unwrap().take() {
                    return result.unwrap_or_else(|e| std::panic::resume_unwind(e));
                }
                std::thread::park();
            }
        })
    }
}

// ---------------------------------------------------------------------------
// TokioBench
// ---------------------------------------------------------------------------

pub struct TokioBench;

impl BenchSystem for TokioBench {
    type JoinHandle<T: Send + 'static> = TokioJoinHandle<T>;

    fn run(num_workers: usize, f: impl FnOnce() + Send + 'static) {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_workers)
            .build()
            .expect("tokio runtime build failed")
            .block_on(async { f() });
    }

    fn spawn<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> TokioJoinHandle<T> {
        let slot: Arc<Mutex<Option<std::thread::Result<T>>>> = Arc::new(Mutex::new(None));
        let slot2 = slot.clone();
        let waiter = std::thread::current();
        let waiter2 = waiter.clone();
        let inner = tokio::spawn(async move {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            *slot2.lock().unwrap() = Some(result);
            waiter2.unpark();
        });
        TokioJoinHandle { slot, waiter, _inner: inner }
    }
}
