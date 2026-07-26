//! [`run_async`] — the stackless scheduler entry point. See
//! [`crate::resumable::common::scheduler`] for the shared
//! [`Scheduler`](crate::resumable::common::scheduler::Scheduler) struct and
//! worker idle loop this drives.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::traits::common::TlsSlot;
use crate::traits::stackful::{JoinHandleLike, ThreadSystem};
use crate::resumable::common::external_queue::ExternalQueue;
use crate::resumable::common::scheduler::{recursion_pool_threshold, worker_loop, Scheduler};
use crate::resumable::stackless::desc::AsyncTaskDesc;
use crate::resumable::common::pool::{DescPool, DynamicPool};
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackless::thread::fork_async_parent_first;
use crate::resumable::common::worker::{LocalQueue, UltWorker, Worker};

/// Start `num_workers` workers on the base system and run `root` (a
/// `Future`) as the first task, entirely without stackful ULT machinery —
/// the stackless-only counterpart to
/// [`stackful::scheduler::run`](crate::resumable::stackful::scheduler::run).
///
/// `root` is pushed via `fork_async_parent_first` rather than
/// `fork_parent_first`: there is no `Ctx`/`StackAlloc` to build a real
/// stack or context from (this function only requires `S: SchedulerSystem`,
/// not `S: StackfulSchedulerSystem`), and no current worker exists yet to call
/// `spawn_async` through. The worker dispatch loop is reused unchanged from
/// `run` — `Worker::execute` already dispatches through
/// [`SchedulerSystem::execute`], so a stackless-only system's override
/// (always poll, never switch) is exercised automatically, with no separate
/// dispatch loop needed here.
pub fn run_async<S, F>(num_workers: usize, root: F)
where
    S: SchedulerSystem,
    S::Desc: AsyncTaskDesc,
    F: std::future::Future<Output = ()> + Send + 'static,
{
    assert!(num_workers >= 1, "need at least one worker");
    assert!(
        UltWorker::<S>::current().is_none(),
        "cmpth: nested run_async() of the same system on one thread"
    );

    // Resolve this system's TLS slot index now, single-threaded, before any
    // worker OS thread starts — see `TlsSlot::warm_up`.
    S::worker_tls().warm_up();

    let workers: Box<[UltWorker<S>]> = (0..num_workers).map(UltWorker::new).collect();
    let shared = Arc::new(Scheduler {
        workers,
        finished: std::sync::atomic::AtomicBool::new(false),
        external_queue: S::ExternalQueue::default(),
        // task_pool is never touched on a pure stackless-only system (no
        // `spawn`), so its configured size is irrelevant.
        task_pool: S::Pool::new_pool(num_workers, 0),
        async_task_pool: S::AsyncPool::new_pool(num_workers, S::ASYNC_POOL_SIZE),
        recursion_pool: S::RecursionPool::new(num_workers, recursion_pool_threshold::<S>()),
    });
    for w in shared.workers.iter() {
        w.shared.set(Arc::as_ptr(&shared));
    }

    shared.external_queue.on_start(&shared);

    let shared2 = Arc::clone(&shared);
    let scheduler_ptr = Arc::as_ptr(&shared) as *const ();
    let root_cont = fork_async_parent_first::<S, _>(
        async move {
            root.await;
            shared2.finished.store(true, Ordering::Release);
        },
        scheduler_ptr,
    );
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
