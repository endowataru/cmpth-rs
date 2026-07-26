//! [`ScopedTaskSystem`] — the concrete marker type implementing
//! [`ScopedStackfulTaskSystem`]/[`ScopedStacklessTaskSystem`].
//!
//! Unlike `resumable`'s systems, there is no pluggable backend axis here (no
//! `Base`/`Deque`/`Pool` choice — both engines always use `crossbeam_deque`
//! and always own their worker threads directly), so one concrete type
//! implementing both traits is enough; no `_system!` macro is needed.

use std::future::Future;

use crate::traits::{ScopedStackfulTaskSystem, ScopedStacklessTaskSystem, TaskSystem};

use super::{async_engine, sync_engine};

/// The concrete [`ScopedStackfulTaskSystem`]/[`ScopedStacklessTaskSystem`]
/// implementation. Zero-sized — all state lives in the worker pool spun up
/// by [`run`](ScopedStackfulTaskSystem::run)/[`run_async`](ScopedStacklessTaskSystem::run_async)
/// for the duration of that call.
pub struct ScopedTaskSystem;

impl TaskSystem for ScopedTaskSystem {
    /// Checks both engines' thread-locals (only one is ever populated at a
    /// time — `run`/`run_async` never overlap in the same call tree) and
    /// falls back to `0`/`1` outside either, matching
    /// `resumable`'s `TaskSystem` blanket's `UltWorker::current() ==
    /// None` fallback.
    fn worker_num() -> usize {
        sync_engine::current_worker_num().or_else(async_engine::current_worker_num).unwrap_or(0)
    }

    fn num_workers() -> usize {
        sync_engine::current_num_workers().or_else(async_engine::current_num_workers).unwrap_or(1)
    }
}

impl ScopedStackfulTaskSystem for ScopedTaskSystem {
    fn run<F, R>(num_workers: usize, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        sync_engine::run(num_workers, f)
    }

    fn parallel_call<Fa, Fb, Ra, Rb>(a: Fa, b: Fb) -> (Ra, Rb)
    where
        Fa: FnOnce() -> Ra + Send + 'static,
        Fb: FnOnce() -> Rb + Send + 'static,
        Ra: Send + 'static,
        Rb: Send + 'static,
    {
        sync_engine::parallel_call(a, b)
    }
}

impl ScopedStacklessTaskSystem for ScopedTaskSystem {
    fn run_async<F>(num_workers: usize, root: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        async_engine::run_async(num_workers, root)
    }

    fn parallel_call<Fa, Fb, Ra, Rb, MkA, MkB>(mk_a: MkA, mk_b: MkB) -> impl Future<Output = (Ra, Rb)> + Send
    where
        MkA: FnOnce() -> Fa,
        MkB: FnOnce() -> Fb,
        Fa: Future<Output = Ra> + Send + 'static,
        Fb: Future<Output = Rb> + Send + 'static,
        Ra: Send + 'static,
        Rb: Send + 'static,
    {
        async_engine::parallel_call(mk_a, mk_b)
    }
}
