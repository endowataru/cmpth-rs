//! [`ScopedTaskSystem`] — the concrete marker type implementing
//! [`ScopedStackfulTaskSystem`]/[`ScopedStacklessTaskSystem`].
//!
//! Unlike `resumable`'s systems, there is no pluggable backend axis here (no
//! `Base`/`Deque`/`Pool` choice — both engines always use `crossbeam_deque`
//! and always own their worker threads directly), so one concrete type
//! implementing both traits is enough; no `_system!` macro is needed.

use std::future::Future;

use crate::traits::{
    ScopedStackfulTaskSystem, ScopedStacklessTaskSystem, StackfulBuilder, StackfulInitSystem,
};

use super::sync_engine::SyncInit;
use super::{async_engine, sync_engine};

/// The concrete [`ScopedStackfulTaskSystem`]/[`ScopedStacklessTaskSystem`]
/// implementation. Zero-sized — all state lives in the worker pool spun up
/// by [`StackfulBuilder::run`]/[`StackfulBuilder::init`] (backed by
/// `sync_engine`) for the duration of that call/guard.
pub struct ScopedTaskSystem;

// `TaskSystem for ScopedTaskSystem` is no longer hand-written here: it is
// blanket-derived (`resumable::common::system`) now that `ScopedTaskSystem`
// is a genuine `WorkerSystem`, exactly like every `resumable`-engine
// flavor. `S::Worker::current()` (i.e. `ScopedWorker::current()`) reads the
// same shared `worker_tls()` slot both `sync_engine` and `async_engine` set
// while driving a worker (only one of the two is ever active on a given OS
// thread at once — `run`/`run_async` never overlap in the same call tree),
// so this needs no special-casing to check two separate thread-locals the
// way the old hand-written impl did.

impl ScopedStackfulTaskSystem for ScopedTaskSystem {
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

/// Builder for [`ScopedTaskSystem`]'s [`StackfulInitSystem`]. Overrides the
/// default [`StackfulBuilder::run`] with a direct call into
/// [`sync_engine::run`] rather than going through `init`/`Drop`: this
/// engine has no context-switch/panic-across-`Drop` hazard to guard against
/// (see [`SyncInit`]'s own doc comment), so there is nothing the default's
/// `catch_unwind` wrapping buys here that a plain `std::thread::spawn`-style
/// panic-propagates-through-`join` doesn't already give for free — and
/// going direct also lets the un-stolen root task stay exactly the
/// single-call shape it always was, with no extra `Arc<Mutex<Option<R>>>`
/// result side-channel.
pub struct ScopedBuilder {
    num_workers: Option<usize>,
}

impl StackfulBuilder<ScopedTaskSystem> for ScopedBuilder {
    fn workers(mut self, n: usize) -> Self {
        self.num_workers = Some(n);
        self
    }

    /// No-op: `ScopedTaskSystem` has no ULT/context-switch concept at all
    /// (see this module's doc comment) — every task runs directly on an OS
    /// worker thread, so there is no per-task stack to size. Accepted only
    /// to satisfy [`StackfulBuilder`]'s uniform interface.
    fn stack_size(self, _bytes: usize) -> Self {
        self
    }

    fn init(self) -> SyncInit {
        sync_engine::init(self.num_workers.unwrap_or_else(crate::os::available_parallelism))
    }

    fn run<F, R>(self, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        sync_engine::run(self.num_workers.unwrap_or_else(crate::os::available_parallelism), f)
    }
}

impl StackfulInitSystem for ScopedTaskSystem {
    type Builder = ScopedBuilder;
    type Init = SyncInit;

    fn builder() -> Self::Builder {
        ScopedBuilder { num_workers: None }
    }
}

impl ScopedStacklessTaskSystem for ScopedTaskSystem {
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
