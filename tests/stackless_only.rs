//! End-to-end tests for a pure stackless-only system (`ult_async_system!`):
//! no `Ctx`/`StackAlloc`/`UltSchedulerSystem` at all, `execute`'s dispatch is
//! `execute_async` (always poll, no `poll_fn` tag check) instead of
//! `execute_dual`/`execute_stackful`. Only `AsyncTaskSystem`'s
//! `run_async`/`spawn`/`recurse`/`.await` are reachable here — there is no
//! `spawn` (stackful), no `.join()` (blocking), no `block_on`: none of that
//! is expressible without `UltSchedulerSystem`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cmpth::AsyncTaskSystem;

cmpth::ult_async_system! {
    struct AsyncOnlySystem {
        base:  cmpth::OsSystem,
        deque: cmpth::CrossbeamDeque<cmpth::BasicTaskDesc>,
    }
}

#[test]
fn spawn_async_await_basic() {
    AsyncOnlySystem::run_async(2, async {
        let h = AsyncOnlySystem::spawn(|| async { 6 * 7 }).await;
        assert_eq!(h.await, 42);
    });
}

#[test]
fn spawn_async_many_parallel() {
    let counter = Arc::new(AtomicU64::new(0));
    AsyncOnlySystem::run_async(4, async move {
        let counter = Arc::clone(&counter);
        let mut handles = Vec::with_capacity(200);
        for i in 0..200u64 {
            let counter = Arc::clone(&counter);
            let h = AsyncOnlySystem::spawn(move || async move {
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
    AsyncOnlySystem::run_async(2, async {
        let h = AsyncOnlySystem::spawn(|| async {
            let inner = AsyncOnlySystem::spawn(|| async { 10 }).await;
            inner.await + 5
        })
        .await;
        assert_eq!(h.await, 15);
    });
}

#[test]
fn spawn_async_panic_propagates_via_await() {
    AsyncOnlySystem::run_async(1, async {
        let h = AsyncOnlySystem::spawn::<(), _, _>(|| async { panic!("boom") }).await;
        let result = std::panic::AssertUnwindSafe(h.await_catch())
            .0
            .await;
        assert!(result.is_err());
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
    AsyncOnlySystem::run_async(3, async move {
        done2.store(1, Ordering::Release);
    });
    assert_eq!(done.load(Ordering::Acquire), 1);
}

#[test]
fn async_task_system_yield_now() {
    let flag = Arc::new(AtomicU64::new(0));
    let flag2 = Arc::clone(&flag);
    AsyncOnlySystem::run_async(2, async move {
        let h = AsyncOnlySystem::spawn(move || async move {
            flag2.store(1, Ordering::Release);
        })
        .await;
        while flag.load(Ordering::Acquire) == 0 {
            AsyncOnlySystem::yield_now().await;
        }
        h.await;
    });
}
