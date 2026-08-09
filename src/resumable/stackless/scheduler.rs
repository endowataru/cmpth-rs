//! [`run_async`] — the stackless scheduler entry point. See
//! [`crate::resumable::common::scheduler`] for the shared
//! [`Scheduler`] struct and
//! worker idle loop this drives.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::traits::common::TlsSlot;
use crate::traits::stackful::{JoinHandleLike, ThreadSystem};
use crate::resumable::common::deque::WorkerRunQueue;
use crate::resumable::common::scheduler::{recursion_pool_threshold, worker_loop, Scheduler};
use crate::resumable::common::pool::{DescPool, DynamicPool};
use crate::resumable::stackless::system::StacklessSchedulerSystem;
use crate::resumable::stackless::thread::fork_async_parent_first;
use crate::resumable::common::worker::{LocalQueue, UltWorker, WorkerOps};

/// Start `num_workers` workers on the base system and run `root` (a
/// `Future`) as the first task, entirely without stackful ULT machinery —
/// the stackless-only counterpart to the old bracketing
/// `stackful::scheduler::run` (now
/// [`stackful::init::init`](crate::resumable::stackful::init::init) +
/// [`StackfulBuilder::run`](crate::traits::system::stackful::StackfulBuilder::run)).
/// Stackless has no standalone-init counterpart — see
/// [`StacklessInitSystem`](crate::traits::system::stackless::StacklessInitSystem)'s
/// doc comment for why.
///
/// `root` is pushed via `fork_async_parent_first` rather than
/// `fork_parent_first`: there is no `Ctx`/`StackAlloc` to build a real
/// stack or context from (this function only requires `S: SchedulerSystem`,
/// not `S: StackfulSchedulerSystem`), and no current worker exists yet to call
/// `spawn_async` through. The worker dispatch loop is reused unchanged from
/// `run` — a popped item's own
/// [`RunnableItem::run_on`](crate::resumable::common::system::RunnableItem::run_on)
/// impl already dispatches correctly, so a stackless-only system's
/// `RunnableItem` impl (always poll, never switch) is exercised
/// automatically, with no separate dispatch loop needed here.
pub fn run_async<S, F>(num_workers: usize, root: F)
where
    S: StacklessSchedulerSystem + crate::resumable::common::system::WorkerSystem<Worker = UltWorker<S>>,
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
    let stealers = workers.iter().map(|w| w.deque.stealer()).collect();
    let shared = Arc::new(Scheduler {
        workers,
        stealers,
        finished: std::sync::atomic::AtomicBool::new(false),
        external_queue: S::ExternalQueue::default(),
        // Meaningless-but-harmless here, same as the `task_pool` line right
        // below: a pure stackless-only system has no real stacks (no
        // `spawn`), so there is no `STACK_SIZE` const to read and no
        // configured value to thread through — see `Scheduler::stack_size`'s
        // own doc comment.
        stack_size: 0,
        // task_pool is never touched on a pure stackless-only system (no
        // `spawn`), so its configured size is irrelevant.
        task_pool: S::Pool::new_pool(num_workers, 0),
        async_task_pool: S::AsyncPool::new_pool(num_workers, S::ASYNC_POOL_SIZE),
        recursion_pool: S::RecursionPool::new(num_workers, recursion_pool_threshold::<S>()),
    });
    for w in shared.workers.iter() {
        w.bind_scheduler(Arc::as_ptr(&shared));
    }

    let shared2 = Arc::clone(&shared);
    let external_queue_ptr = &shared.external_queue as *const _;
    let root_cont = fork_async_parent_first::<S, _>(
        async move {
            root.await;
            shared2.finished.store(true, Ordering::Release);
        },
        external_queue_ptr,
    );
    shared.workers[0].push(root_cont.into());

    let handles: Vec<_> = (1..num_workers)
        .map(|i| {
            let shared = Arc::clone(&shared);
            S::Base::spawn(move || worker_loop(&shared.workers[i], &shared))
        })
        .collect();

    worker_loop(&shared.workers[0], &shared);

    for h in handles {
        h.join();
    }
}
