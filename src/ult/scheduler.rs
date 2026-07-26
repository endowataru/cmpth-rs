//! Scheduler: worker set, main loop, and `run` entry point.

use std::alloc::Layout;
use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::traits::thread_system::{JoinHandleLike, TlsSlot, ThreadSystem};
use crate::ult::desc::AsyncTaskDesc;
use crate::ult::external_queue::ExternalQueue;
use crate::ult::pool::{DescPool, DynamicPool};
use crate::ult::system::{SchedulerSystem, UltSchedulerSystem};
use crate::ult::thread::{fork_async_parent_first, fork_parent_first};
use crate::ult::worker::{LocalQueue, UltWorker, Worker};

/// State shared by all workers of one scheduler instance. Base-level
/// (`S: SchedulerSystem`): shared by stackful-only, dual, and (eventually)
/// stackless-only systems alike — only [`run`] (the stackful entry point)
/// needs the stackful extension.
pub struct Scheduler<S: SchedulerSystem> {
    pub(crate) workers: Box<[UltWorker<S>]>,
    pub(crate) finished: AtomicBool,
    pub(crate) external_queue: S::ExternalQueue,
    pub(crate) task_pool: S::Pool,
    /// Separate from `task_pool`: `spawn_async` needs a much smaller fixed
    /// slot size than a ULT stack, and a dual system needs both live at once
    /// (see [`SchedulerSystem::AsyncPool`]).
    pub(crate) async_task_pool: S::AsyncPool,
    /// Pool backing [`crate::ult::thread::recurse`] — see
    /// [`SchedulerSystem::RecursionPool`] for why this needs none of
    /// [`task_pool`](Self::task_pool)'s `TaskDesc`/stealing-specific
    /// construction, just the same fixed-slot free-list mechanism.
    pub(crate) recursion_pool: S::RecursionPool,
}

/// Threshold `S::RecursionPool` is configured with — reuses
/// [`SchedulerSystem::ASYNC_POOL_SIZE`] (both are "small `Future` storage"
/// budgets) rather than adding a second, near-duplicate per-system
/// constant; align 16 covers realistic recursive-`async fn` frames without
/// needing its own knob either.
fn recursion_pool_threshold<S: SchedulerSystem>() -> Layout {
    Layout::from_size_align(S::ASYNC_POOL_SIZE, 16)
        .expect("cmpth: ASYNC_POOL_SIZE not a valid Layout size for align 16")
}

unsafe impl<S: SchedulerSystem> Send for Scheduler<S> {}
unsafe impl<S: SchedulerSystem> Sync for Scheduler<S> {}

/// Start `num_workers` workers on the base system, run `root` as the first
/// task, and return when `root` completes and all workers have shut down.
pub fn run<S, F>(num_workers: usize, root: F)
where
    S: UltSchedulerSystem,
    S::Desc: crate::ult::desc::StackfulTaskDesc + crate::ult::desc::WakerTaskDesc,
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
        finished: AtomicBool::new(false),
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

/// Start `num_workers` workers on the base system and run `root` (a
/// `Future`) as the first task, entirely without stackful ULT machinery —
/// the stackless-only counterpart to [`run`].
///
/// `root` is pushed via `fork_async_parent_first` rather than
/// `fork_parent_first`: there is no `Ctx`/`StackAlloc` to build a real
/// stack or context from (this function only requires `S: SchedulerSystem`,
/// not `S: UltSchedulerSystem`), and no current worker exists yet to call
/// `spawn_async` through. The worker dispatch loop is reused unchanged from
/// [`run`] — `Worker::execute` already dispatches through
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
        finished: AtomicBool::new(false),
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

pub(crate) fn worker_loop<S: SchedulerSystem>(wk: &UltWorker<S>) {
    S::worker_tls().set(wk as *const UltWorker<S> as *mut UltWorker<S>);
    wk.cur_task.set(wk.root_desc() as *const _ as *mut _);

    let shared = wk.shared();
    let mut idle_rounds = 0u32;
    while !shared.finished.load(Ordering::Acquire) {
        if let Some(c) = wk.pop_local()
            .or_else(|| wk.try_steal())
            .or_else(|| shared.external_queue.try_pop())
        {
            wk.execute(c);
            idle_rounds = 0;
            continue;
        }
        std::hint::spin_loop();
        idle_rounds += 1;
        if idle_rounds & 0x3F == 0 {
            S::Base::yield_now();
        }
    }

    S::worker_tls().set(std::ptr::null_mut());
}
