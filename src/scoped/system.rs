//! [`ParallelInvokeSystem`] — the concrete marker type implementing
//! [`StackfulParallelInvoke`]/[`StacklessParallelInvoke`].
//!
//! Unlike `ult`'s systems, there is no pluggable backend axis here (no
//! `Base`/`Deque`/`Pool` choice — both engines always use `crossbeam_deque`
//! and always own their worker threads directly), so one concrete type
//! implementing both traits is enough; no `_system!` macro is needed.

use std::future::Future;

use crate::traits::{StackfulParallelInvoke, StacklessParallelInvoke};

use super::{async_engine, sync_engine};

/// The concrete [`StackfulParallelInvoke`]/[`StacklessParallelInvoke`]
/// implementation. Zero-sized — all state lives in the worker pool spun up
/// by [`run`](StackfulParallelInvoke::run)/[`run_async`](StacklessParallelInvoke::run_async)
/// for the duration of that call.
pub struct ParallelInvokeSystem;

impl StackfulParallelInvoke for ParallelInvokeSystem {
    fn run<F, R>(num_workers: usize, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        sync_engine::run(num_workers, f)
    }

    fn parallel_invoke<Fa, Fb, Ra, Rb>(a: Fa, b: Fb) -> (Ra, Rb)
    where
        Fa: FnOnce() -> Ra + Send,
        Fb: FnOnce() -> Rb + Send,
        Ra: Send,
        Rb: Send,
    {
        sync_engine::parallel_invoke(a, b)
    }
}

impl StacklessParallelInvoke for ParallelInvokeSystem {
    fn run_async<F>(num_workers: usize, root: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        async_engine::run_async(num_workers, root)
    }

    fn parallel_invoke<Fa, Fb, Ra, Rb, MkA, MkB>(mk_a: MkA, mk_b: MkB) -> impl Future<Output = (Ra, Rb)> + Send
    where
        MkA: FnOnce() -> Fa,
        MkB: FnOnce() -> Fb,
        Fa: Future<Output = Ra> + Send + 'static,
        Fb: Future<Output = Rb> + Send + 'static,
        Ra: Send + 'static,
        Rb: Send + 'static,
    {
        async_engine::parallel_invoke(mk_a, mk_b)
    }
}
