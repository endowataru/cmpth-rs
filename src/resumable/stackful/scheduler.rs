//! [`run`] — the stackful scheduler entry point. See
//! [`crate::resumable::common::scheduler`] for the shared
//! [`Scheduler`](crate::resumable::common::scheduler::Scheduler) struct and
//! worker idle loop this drives.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::traits::thread_system::{JoinHandleLike, ThreadSystem, TlsSlot};
use crate::resumable::common::external_queue::ExternalQueue;
use crate::resumable::common::scheduler::{recursion_pool_threshold, worker_loop, Scheduler};
use crate::resumable::common::pool::{DescPool, DynamicPool};
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::stackful::thread::fork_parent_first;
use crate::resumable::common::worker::{LocalQueue, UltWorker, Worker};

/// Start `num_workers` workers on the base system, run `root` as the first
/// task, and return when `root` completes and all workers have shut down.
pub fn run<S, F>(num_workers: usize, root: F)
where
    S: StackfulSchedulerSystem,
    S::Desc: crate::resumable::stackful::desc::StackfulTaskDesc + crate::resumable::common::desc::WakerTaskDesc,
    F: FnOnce() + Send + 'static,
{
    assert!(num_workers >= 1, "need at least one worker");
    assert!(
        UltWorker::<S>::current().is_none(),
        "cmpth: nested run() of the same system on one thread"
    );

    // Resolve this system's TLS slot index now, single-threaded, before any
    // worker OS thread starts — see `TlsSlot::warm_up`.
    S::worker_tls().warm_up();

    let workers: Box<[UltWorker<S>]> = (0..num_workers).map(UltWorker::new).collect();
    let shared = Arc::new(Scheduler {
        workers,
        finished: std::sync::atomic::AtomicBool::new(false),
        external_queue: S::ExternalQueue::default(),
        task_pool: S::Pool::new_pool(num_workers, S::STACK_SIZE),
        async_task_pool: S::AsyncPool::new_pool(num_workers, S::ASYNC_POOL_SIZE),
        recursion_pool: S::RecursionPool::new(num_workers, recursion_pool_threshold::<S>()),
    });
    for w in shared.workers.iter() {
        w.shared.set(Arc::as_ptr(&shared));
    }

    shared.external_queue.on_start(&shared);

    let shared2 = Arc::clone(&shared);
    let scheduler_ptr = Arc::as_ptr(&shared) as *const ();
    let root_cont = fork_parent_first::<S>(Box::new(move || {
        root();
        shared2.finished.store(true, Ordering::Release);
        Box::new(()) as Box<dyn Any + Send>
    }), scheduler_ptr);
    shared.workers[0].push_local_top(root_cont);

    let handles: Vec<_> = (1..num_workers)
        .map(|i| {
            let shared = Arc::clone(&shared);
            S::Base::spawn(move || worker_loop(&shared.workers[i]))
        })
        .collect();

    worker_loop(&shared.workers[0]);

    for h in handles {
        h.join();
    }
}

/// [`run`], but returns `root`'s result — the
/// [`ScopedStackfulTaskSystem::run`](crate::traits::scoped::ScopedStackfulTaskSystem::run)
/// trait method's actual body.
///
/// `run`'s root task is detached (no `JoinHandle`), and a detached task's
/// descriptor is freed by the exit path the moment it finishes — there's no
/// safe way to read a result back out of it afterward. Rather than
/// complicate that already-delicate detach/exit protocol just to keep one
/// result alive, this wraps `root` in a plain `Arc<Mutex<Option<R>>>`
/// side-channel instead: negligible cost for something called once per
/// program (this is the top-level entry point, not a hot path).
pub fn run_with_result<S, F, R>(num_workers: usize, root: F) -> R
where
    S: StackfulSchedulerSystem,
    S::Desc: crate::resumable::stackful::desc::StackfulTaskDesc + crate::resumable::common::desc::WakerTaskDesc,
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let result = Arc::new(std::sync::Mutex::new(None));
    let result2 = Arc::clone(&result);
    run::<S, _>(num_workers, move || {
        let r = root();
        *result2.lock().unwrap() = Some(r);
    });
    result.lock().unwrap().take().expect("cmpth: run's root task did not complete")
}
