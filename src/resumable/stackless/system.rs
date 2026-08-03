//! The blanket [`StacklessTaskSystem`]/[`ScopedStacklessTaskSystem`] impls
//! for every async-capable [`SchedulerSystem`], and the [`UltAsyncIdentity`]
//! config trait that assembles a stackless-only system.
//!
//! The trait declarations themselves live in [`crate::traits::stackless`]/
//! [`crate::traits::scoped`] (pure interface, no `resumable`-layer types in
//! their own signatures); this module only supplies the bodies, which is
//! where naming `SchedulerSystem` and concrete resumable types
//! (`JoinHandle`, `spawn_async`, `recurse`, `run_async`) is fine.

use std::future::Future;
use std::marker::PhantomData;

use crate::traits::stackful::ThreadSystem;
use crate::resumable::common::deque::WorkerDeque;
use crate::resumable::common::lookup::CurrentLookup;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::worker::UltWorker;
use crate::resumable::stackless::desc::AsyncTaskDesc;
use crate::traits::scoped::ScopedStacklessTaskSystem;

// `StacklessTaskSystem` now lives in `crate::traits::stackless` —
// re-exported below for callers that still spell out
// `resumable::stackless::system::StacklessTaskSystem`.
pub use crate::traits::stackless::StacklessTaskSystem;

impl<S: SchedulerSystem> StacklessTaskSystem for S
where
    S::Desc: AsyncTaskDesc,
{
    type SpawnHandle<T: Send + 'static> = crate::resumable::common::thread::JoinHandle<S, T>;

    fn spawn<T, F, Mk>(mk: Mk) -> impl Future<Output = Self::SpawnHandle<T>> + Send
    where
        F: Future<Output = T> + Send + 'static,
        Mk: FnOnce() -> F + Send + 'static,
        T: Send + 'static,
    {
        crate::resumable::stackless::thread::spawn_async::<Self, T, F, Mk>(mk)
    }

    fn recurse<F, Mk>(mk: Mk) -> impl Future<Output = F::Output> + Send
    where
        F: Future + Send,
        Mk: FnOnce() -> F,
    {
        crate::resumable::stackless::thread::recurse::<Self, F, Mk>(mk)
    }
}

/// `parallel_call` for a system that already has `StacklessTaskSystem`'s
/// `spawn`: strictly cheaper capability, satisfied trivially by spawning
/// one branch and awaiting the other inline — same relationship as
/// `resumable::stackful::system`'s `ScopedStackfulTaskSystem` blanket. Calls
/// the same `spawn_async` free function `StacklessTaskSystem::spawn`'s
/// blanket does directly (rather than going through `S::spawn`) so this
/// impl doesn't need a `StacklessTaskSystem` bound of its own — the two
/// blankets are independent, both satisfied by the same `SchedulerSystem +
/// AsyncTaskDesc` condition.
impl<S: SchedulerSystem> ScopedStacklessTaskSystem for S
where
    S::Desc: AsyncTaskDesc,
{
    fn run_async<F>(num_workers: usize, root: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        crate::resumable::stackless::scheduler::run_async::<Self, F>(num_workers, root)
    }

    async fn parallel_call<Fa, Fb, Ra, Rb, MkA, MkB>(mk_a: MkA, mk_b: MkB) -> (Ra, Rb)
    where
        MkA: FnOnce() -> Fa + Send + 'static,
        MkB: FnOnce() -> Fb + Send + 'static,
        Fa: Future<Output = Ra> + Send + 'static,
        Fb: Future<Output = Rb> + Send + 'static,
        Ra: Send + 'static,
        Rb: Send + 'static,
    {
        let h = crate::resumable::stackless::thread::spawn_async::<Self, Ra, Fa, MkA>(mk_a).await;
        let rb = mk_b().await;
        (h.await, rb)
    }
}

// ---------------------------------------------------------------------------
// UltAsyncIdentity
// ---------------------------------------------------------------------------

/// Assembles a complete **stackless-only** ULT system from a handful of
/// associated types — the config-trait replacement for what used to be the
/// `ult_async_system!` macro. Implement this for your own marker type and
/// use [`UltAsyncSystem<M>`] as the actual system (the thing you call
/// `run_async`/`spawn` on).
///
/// A bare blanket `impl<M: UltAsyncIdentity> SchedulerSystem for M` (mirroring
/// [`UltIdentity`](crate::resumable::stackful::system::UltIdentity)'s
/// bare-`M` shape) would conflict under Rust's coherence rules with
/// `UltIdentity`'s own bare-`M` blanket impl — the compiler can't prove no
/// type ever implements both traits, even though in practice none would.
/// The [`UltAsyncSystem<M>`] wrapper sidesteps this by targeting a
/// genuinely different type (verified directly against a real downstream
/// crate: both flavors coexist and resolve to distinct per-marker
/// `worker_tls` statics through the wrapper). See
/// [`UltIdentity`](crate::resumable::stackful::system::UltIdentity)'s doc comment for why a config trait is used at all
/// instead of a generic struct callers would type-alias (Rust's orphan
/// rules forbid implementing a foreign trait for a type alias of a foreign
/// generic struct) — [`UltAsyncSystem<M>`] itself is only ever *named*, never
/// implemented against, by downstream code, so it doesn't reintroduce that
/// problem.
///
/// Unlike `UltIdentity`, only implies [`SchedulerSystem`] — never
/// [`StackfulSchedulerSystem`](crate::resumable::stackful::system::StackfulSchedulerSystem),
/// so it never names a context-switch policy or stack allocator, because it
/// has none. Its only entry points are
/// [`crate::resumable::stackless::scheduler::run_async`] (run) and
/// [`crate::resumable::stackless::thread::spawn_async`] (spawn); there is no
/// `spawn`, no `block_on`, no `ThreadSystem` impl at all for it (that
/// requires stackful capability this system deliberately doesn't have).
///
/// `Worker::execute`'s dispatch is [`crate::resumable::stackful::worker::execute_stackful`]
/// -shaped in spirit but for polling instead of switching: it always polls,
/// with no `poll_fn`-tag branch, because every task on this system is one.
///
/// `ASYNC_POOL_SIZE` defaults to 512. [`InlineTlsCurrent`](crate::resumable::stackless::lookup::InlineTlsCurrent)
/// is the natural `Lookup` choice — sound specifically because this system
/// never implements `StackfulSchedulerSystem` and so never does a real
/// context switch (see that type's doc comment for the hazard that would
/// otherwise apply) — but it isn't defaulted here, matching `UltIdentity`:
/// associated types can't carry defaults on stable Rust.
///
/// ```
/// use cmpth::SuspendedTaskToken;
/// use cmpth::{ScopedStacklessTaskSystem, StacklessTaskSystem, ThreadSystem};
///
/// pub struct MyAsyncMarker;
///
/// impl cmpth::UltAsyncIdentity for MyAsyncMarker {
///     type Base = cmpth::OsSystem;
///     type Desc = cmpth::StacklessOnlyTaskDesc;
///     type Deque = cmpth::CrossbeamDeque<cmpth::StacklessOnlyTaskDesc>;
///     type Lookup = cmpth::InlineTlsCurrent;
///
///     fn worker_tls_anchor() -> &'static <cmpth::OsSystem as ThreadSystem>::ThreadSpecific<cmpth::UltWorker<cmpth::UltAsyncSystem<Self>>> {
///         static A: cmpth::TlsAnchor = cmpth::TlsAnchor::new();
///         cmpth::TlsSlot::from_anchor(&A)
///     }
/// }
///
/// type MyAsyncSystem = cmpth::UltAsyncSystem<MyAsyncMarker>;
///
/// MyAsyncSystem::run_async(2, async {
///     let h = MyAsyncSystem::spawn(|| async { 6 * 7 }).await;
///     assert_eq!(h.await, 42);
/// });
/// ```
pub trait UltAsyncIdentity: Sized + Send + Sync + 'static {
    /// The threading system this scheduler runs on.
    type Base: ThreadSystem;

    /// Task descriptor type. Most implementors want
    /// [`StacklessOnlyTaskDesc`](crate::resumable::stackless::desc::StacklessOnlyTaskDesc)
    /// (no unused `ctx` slot); a system that also needs stackful `spawn`/
    /// dual capability on the same tasks wants
    /// [`DualTaskDesc`](crate::resumable::dual::desc::DualTaskDesc)
    /// instead.
    type Desc: crate::resumable::common::desc::TaskDescAlloc + AsyncTaskDesc;

    /// Work-stealing deque implementation.
    type Deque: WorkerDeque<Self::Desc>;

    /// Fixed slot size for the `spawn_async` descriptor pool.
    const ASYNC_POOL_SIZE: usize = 512;

    /// Current-worker lookup policy.
    type Lookup: CurrentLookup<UltAsyncSystem<Self>>
    where
        UltAsyncSystem<Self>: SchedulerSystem;

    /// The per-system TLS anchor backing [`SchedulerSystem::worker_tls`].
    /// Named in terms of [`UltAsyncSystem<Self>`] — the actual final
    /// system type — not bare `Self`, since `Self` here is just the config
    /// marker; see this trait's own doc comment for why.
    fn worker_tls_anchor() -> &'static <<Self as UltAsyncIdentity>::Base as ThreadSystem>::ThreadSpecific<UltWorker<UltAsyncSystem<Self>>>
    where
        UltAsyncSystem<Self>: SchedulerSystem;
}

/// The actual stackless-only system type: call `run_async`/`spawn` on
/// `UltAsyncSystem<M>`, not on `M` itself. See [`UltAsyncIdentity`]'s doc
/// comment for why `M` alone can't directly implement `SchedulerSystem`.
pub struct UltAsyncSystem<M: UltAsyncIdentity> {
    _marker: PhantomData<fn() -> M>,
}

impl<M: UltAsyncIdentity> SchedulerSystem for UltAsyncSystem<M> {
    type Base  = M::Base;
    type Desc  = M::Desc;
    type Deque = M::Deque;
    type ExternalQueue = crate::resumable::common::external_queue::StealPathQueue<M::Desc>;
    // Never actually allocated through: this flavor has no `spawn`, only
    // `spawn_async` (which goes through AsyncPool below). SimplePool is the
    // cheapest DescPool to instantiate for a type that's never used.
    type Pool = crate::resumable::common::pool::SimplePool<M::Desc>;
    const ASYNC_POOL_SIZE: usize = <M as UltAsyncIdentity>::ASYNC_POOL_SIZE;
    type AsyncPool = crate::resumable::common::pool::ReturnPool<M::Desc, crate::resumable::stackless::stack::AsyncArenaStack>;
    type RecursionPool = crate::resumable::common::pool::ThresholdPool<crate::resumable::common::pool::BlockPool>;
    type Lookup = <M as UltAsyncIdentity>::Lookup;

    fn worker_tls() -> &'static <M::Base as ThreadSystem>::ThreadSpecific<UltWorker<Self>> {
        <M as UltAsyncIdentity>::worker_tls_anchor()
    }

    // Stackless-only: always poll, never switch — no poll_fn tag check,
    // because every task on this system is a poll_fn task.
    fn execute(wk: &UltWorker<Self>, cont: crate::resumable::common::desc::SuspendedTaskToken<M::Desc>) {
        crate::resumable::stackless::worker::execute_async(wk, cont)
    }

    fn free_finished_desc(wk: &UltWorker<Self>, desc: *mut M::Desc) {
        unsafe { crate::resumable::stackless::worker::free_finished_desc_async(wk, desc) }
    }
}
