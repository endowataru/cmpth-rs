//! [`Scheduler`]: worker set shared by every flavor, plus the worker idle
//! loop. Flavor-specific entry points live in
//! [`stackful::init::init`](crate::resumable::stackful::init::init) and
//! [`stackless::scheduler::run_async`](crate::resumable::stackless::scheduler::run_async).

use std::alloc::Layout;
use std::sync::atomic::Ordering;

use crate::traits::common::TlsSlot;
use crate::traits::stackful::ThreadSystem;
use crate::resumable::common::external_queue::ExternalQueue;
use crate::resumable::common::system::{DescScheduler, SchedulerSystem};
use crate::resumable::common::worker::{LocalQueue, UltWorker, WorkerOps};

/// State shared by all workers of one scheduler instance. Base-level
/// (`S: SchedulerSystem`): shared by stackful-only, dual, and (eventually)
/// stackless-only systems alike — only
/// [`init`](crate::resumable::stackful::init::init) (the stackful entry
/// point) needs the stackful extension.
pub struct Scheduler<S: SchedulerSystem> {
    pub(crate) workers: Box<[UltWorker<S>]>,
    pub(crate) finished: std::sync::atomic::AtomicBool,
    pub(crate) external_queue: S::ExternalQueue,
    pub(crate) task_pool: S::Pool,
    /// Separate from `task_pool`: `spawn_async` needs a much smaller fixed
    /// slot size than a ULT stack, and a dual system needs both live at once
    /// (see [`SchedulerSystem::AsyncPool`]).
    pub(crate) async_task_pool: S::AsyncPool,
    /// Pool backing [`crate::resumable::stackless::thread::recurse`] — see
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
pub(crate) fn recursion_pool_threshold<S: SchedulerSystem>() -> Layout {
    Layout::from_size_align(S::ASYNC_POOL_SIZE, 16)
        .expect("cmpth: ASYNC_POOL_SIZE not a valid Layout size for align 16")
}

unsafe impl<S: SchedulerSystem> Send for Scheduler<S> {}
unsafe impl<S: SchedulerSystem> Sync for Scheduler<S> {}

pub(crate) fn worker_loop<S>(wk: &UltWorker<S>)
where
    S: DescScheduler,
{
    S::worker_tls().set(wk as *const UltWorker<S> as *mut UltWorker<S>);
    // SAFETY: `root_desc` is embedded by value in `UltWorker` and only ever
    // reachable through `wk.root_desc() -> &S::Desc` (never a token) until
    // this very call, which runs once per worker at startup — this is the
    // first token ever constructed for it, so exclusivity is trivial.
    let root_task = unsafe {
        crate::resumable::common::desc::RunningTaskToken::from_raw(
            wk.root_desc() as *const _ as *mut _,
        )
    };
    wk.set_cur_task(root_task);

    worker_idle_loop(wk);

    S::worker_tls().set(std::ptr::null_mut());
}

/// The idle/dispatch loop itself, factored out of [`worker_loop`] so the
/// stackful initializer (`resumable::stackful::init`) can drive it on a
/// bootstrap it sets up differently: TLS is already pointed at `wk` and
/// `cur_task` is already populated (by the child-first fork's switch shim,
/// not by the two lines above) before that caller ever reaches this
/// function, and TLS teardown happens later too (on whichever OS thread
/// ends up running the initializer's `Drop`, not necessarily this one) — so
/// neither belongs inside this shared core.
pub(crate) fn worker_idle_loop<S>(wk: &UltWorker<S>)
where
    S: DescScheduler,
{
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
}
