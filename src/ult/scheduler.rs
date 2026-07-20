//! Scheduler: worker set, main loop, and `run` entry point.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::traits::thread_system::{JoinHandleLike, TlsSlot, ThreadSystem};
use crate::ult::external_queue::ExternalQueue;
use crate::ult::pool::DescPool;
use crate::ult::system::UltSchedulerSystem;
use crate::ult::thread::fork_parent_first;
use crate::ult::worker::{LocalQueue, UltWorker, Worker};

/// State shared by all workers of one scheduler instance.
pub struct Scheduler<S: UltSchedulerSystem> {
    pub(crate) workers: Box<[UltWorker<S>]>,
    pub(crate) finished: AtomicBool,
    pub(crate) external_queue: S::ExternalQueue,
    pub(crate) task_pool: S::Pool,
}

unsafe impl<S: UltSchedulerSystem> Send for Scheduler<S> {}
unsafe impl<S: UltSchedulerSystem> Sync for Scheduler<S> {}

/// Start `num_workers` workers on the base system, run `root` as the first
/// task, and return when `root` completes and all workers have shut down.
pub fn run<S, F>(num_workers: usize, root: F)
where
    S: UltSchedulerSystem,
    F: FnOnce() + Send + 'static,
{
    assert!(num_workers >= 1, "need at least one worker");
    assert!(
        UltWorker::<S>::current().is_none(),
        "cmpth: nested run() of the same system on one thread"
    );

    let workers: Box<[UltWorker<S>]> = (0..num_workers).map(UltWorker::new).collect();
    let shared = Arc::new(Scheduler {
        workers,
        finished: AtomicBool::new(false),
        external_queue: S::ExternalQueue::default(),
        task_pool: S::Pool::new_pool(num_workers, S::STACK_SIZE),
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

fn worker_loop<S: UltSchedulerSystem>(wk: &UltWorker<S>) {
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
