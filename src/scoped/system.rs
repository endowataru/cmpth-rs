//! [`ScopedIdentity`]/[`ScopedSystem`] — the config-trait + generic-wrapper
//! pair implementing [`ScopedStackfulTaskSystem`]/[`ScopedStacklessTaskSystem`],
//! and [`ScopedTaskSystem`], the default instantiation most callers want.
//!
//! Unlike `resumable`'s systems, most of the backend axis is not pluggable
//! here (no `Base`/`Deque`/`Pool` choice — both engines always use
//! `crossbeam_deque` and always own their worker threads directly: see
//! `super`'s module doc comment for why that's a deliberate fit for this
//! engine's "no separately pooled task descriptor" design, not an
//! oversight). `Idle` is the one axis this engine does expose, and it's
//! exposed the same way every `resumable` flavor exposes its own axes: as
//! an associated type on a config trait a marker type implements
//! ([`ScopedIdentity`], mirroring
//! [`UltAsyncIdentity`](crate::resumable::stackless::system::UltAsyncIdentity)'s
//! shape), not as a bare generic parameter on the system itself — idle
//! behaviour is entirely a [`PoolCore`](crate::resumable::common::worker_core::PoolCore)
//! property, orthogonal to how a branch itself is represented, but *how
//! you pick it* should look the same regardless of which flavor you're
//! configuring.

use std::future::Future;
use std::marker::PhantomData;

use crate::resumable::common::idle::{IdlePolicy, SpinIdle};
use crate::traits::{
    ScopedStackfulTaskSystem, ScopedStacklessTaskSystem, StackfulBuilder, StackfulInitSystem,
};

use super::sync_engine::SyncInit;
use super::{async_engine, sync_engine};

/// Assembles a complete `scoped` system from a single associated type —
/// the config-trait every `resumable` flavor uses
/// ([`UltIdentity`](crate::resumable::stackful::system::UltIdentity),
/// [`UltAsyncIdentity`](crate::resumable::stackless::system::UltAsyncIdentity)),
/// just with one member instead of several: `scoped` has no `Base`/`Desc`/
/// `RunQueue`/`Lookup` choice to make (see this module's doc comment), so
/// `Idle` is the whole bundle. Implement this for your own marker type and
/// use [`ScopedSystem<M>`] as the actual system (the thing you call
/// `builder()`/`parallel_call` on) — exactly the
/// [`UltAsyncIdentity`](crate::resumable::stackless::system::UltAsyncIdentity)/
/// [`UltAsyncSystem`](crate::resumable::stackless::system::UltAsyncSystem)
/// shape, for the same reason: a bare blanket `impl<M: ScopedIdentity> ... for M`
/// would be one more blanket impl on bare `M` for Rust's coherence checker
/// to worry about overlapping with `UltIdentity`'s own, so this wrapper
/// targets a genuinely different type instead.
///
/// ```
/// use cmpth::{ScopedStackfulTaskSystem, StackfulBuilder, StackfulInitSystem};
///
/// pub struct MyScopedMarker;
///
/// impl cmpth::ScopedIdentity for MyScopedMarker {
///     type Idle = cmpth::ParkIdle;
/// }
///
/// type MyScopedSystem = cmpth::ScopedSystem<MyScopedMarker>;
///
/// fn fib<S: ScopedStackfulTaskSystem>(n: u64) -> u64 {
///     if n <= 1 { return n; }
///     let (a, b) = S::parallel_call(move || fib::<S>(n - 1), move || fib::<S>(n - 2));
///     a + b
/// }
///
/// let r = MyScopedSystem::builder().workers(2).run(move || fib::<MyScopedSystem>(20));
/// assert_eq!(r, 6765);
/// ```
pub trait ScopedIdentity: Sized + Send + Sync + 'static {
    /// Idle/wakeup policy — see [`crate::IdlePolicy`].
    type Idle: IdlePolicy;
}

/// The default [`ScopedIdentity`]: [`SpinIdle`], same as every other
/// flavor's own unconfigured default. Backs the [`ScopedTaskSystem`] alias;
/// not meant to be named directly — implement [`ScopedIdentity`] on your
/// own marker type instead if you want a different `Idle`.
pub struct DefaultScopedMarker;

impl ScopedIdentity for DefaultScopedMarker {
    type Idle = SpinIdle;
}

/// The generic `scoped` system — `M`'s [`ScopedIdentity::Idle`] is this
/// system's [`WorkerSystem::Idle`](crate::WorkerSystem::Idle). Zero-sized —
/// all other state lives in the worker pool spun up by
/// [`StackfulBuilder::run`]/[`StackfulBuilder::init`] (backed by
/// `sync_engine`) for the duration of that call/guard.
pub struct ScopedSystem<M: ScopedIdentity> {
    _marker: PhantomData<fn() -> M>,
}

/// The default `scoped` system ([`SpinIdle`]) — what every caller not
/// naming its own [`ScopedIdentity`] marker wants. A plain alias, not a
/// distinct type: [`ScopedSystem<DefaultScopedMarker>`].
pub type ScopedTaskSystem = ScopedSystem<DefaultScopedMarker>;

// `TaskSystem for ScopedSystem<M>` is no longer hand-written here: it is
// blanket-derived (`resumable::common::system`) now that `ScopedSystem<M>`
// is a genuine `WorkerSystem`, exactly like every `resumable`-engine
// flavor. `S::Worker::current()` (i.e. `ScopedWorker::<M>::current()`) reads
// the same shared `worker_tls()` slot both `sync_engine` and `async_engine`
// set while driving a worker (only one of the two is ever active on a given
// OS thread at once — `run`/`run_async` never overlap in the same call
// tree), so this needs no special-casing to check two separate
// thread-locals the way the old hand-written impl did.

impl<M: ScopedIdentity> ScopedStackfulTaskSystem for ScopedSystem<M> {
    fn parallel_call<Fa, Fb, Ra, Rb>(a: Fa, b: Fb) -> (Ra, Rb)
    where
        Fa: FnOnce() -> Ra + Send + 'static,
        Fb: FnOnce() -> Rb + Send + 'static,
        Ra: Send + 'static,
        Rb: Send + 'static,
    {
        sync_engine::parallel_call::<M, _, _, _, _>(a, b)
    }
}

/// Builder for [`ScopedSystem`]'s [`StackfulInitSystem`]. Overrides the
/// default [`StackfulBuilder::run`] with a direct call into
/// [`sync_engine::run`] rather than going through `init`/`Drop`: this
/// engine has no context-switch/panic-across-`Drop` hazard to guard against
/// (see [`SyncInit`]'s own doc comment), so there is nothing the default's
/// `catch_unwind` wrapping buys here that a plain `std::thread::spawn`-style
/// panic-propagates-through-`join` doesn't already give for free — and
/// going direct also lets the un-stolen root task stay exactly the
/// single-call shape it always was, with no extra `Arc<Mutex<Option<R>>>`
/// result side-channel.
pub struct ScopedBuilder<M: ScopedIdentity> {
    num_workers: Option<usize>,
    _marker: PhantomData<fn() -> M>,
}

impl<M: ScopedIdentity> StackfulBuilder<ScopedSystem<M>> for ScopedBuilder<M> {
    fn workers(mut self, n: usize) -> Self {
        self.num_workers = Some(n);
        self
    }

    /// No-op: `ScopedSystem` has no ULT/context-switch concept at all (see
    /// this module's doc comment) — every task runs directly on an OS
    /// worker thread, so there is no per-task stack to size. Accepted only
    /// to satisfy [`StackfulBuilder`]'s uniform interface.
    fn stack_size(self, _bytes: usize) -> Self {
        self
    }

    fn init(self) -> SyncInit<M> {
        sync_engine::init::<M>(self.num_workers.unwrap_or_else(crate::os::available_parallelism))
    }

    fn run<F, R>(self, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        sync_engine::run::<M, F, R>(self.num_workers.unwrap_or_else(crate::os::available_parallelism), f)
    }
}

impl<M: ScopedIdentity> StackfulInitSystem for ScopedSystem<M> {
    type Builder = ScopedBuilder<M>;
    type Init = SyncInit<M>;

    fn builder() -> Self::Builder {
        ScopedBuilder { num_workers: None, _marker: PhantomData }
    }
}

impl<M: ScopedIdentity> ScopedStacklessTaskSystem for ScopedSystem<M> {
    fn parallel_call<Fa, Fb, Ra, Rb, MkA, MkB>(mk_a: MkA, mk_b: MkB) -> impl Future<Output = (Ra, Rb)> + Send
    where
        MkA: FnOnce() -> Fa,
        MkB: FnOnce() -> Fb,
        Fa: Future<Output = Ra> + Send + 'static,
        Fb: Future<Output = Rb> + Send + 'static,
        Ra: Send + 'static,
        Rb: Send + 'static,
    {
        async_engine::parallel_call::<M, _, _, _, _, _, _>(mk_a, mk_b)
    }
}
