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

use crate::traits::stackful::{NestableSystem, ThreadSystem};
use crate::resumable::common::deque::WorkerRunQueue;
use crate::resumable::common::lookup::CurrentLookup;
use crate::resumable::common::system::{PoolSystem, SchedulerSystem, WorkerSystem};
use crate::resumable::common::worker::{UltWorker, WorkerOps};
use crate::resumable::stackless::desc::AsyncTaskDesc;
use crate::traits::scoped::ScopedStacklessTaskSystem;

// `StacklessTaskSystem` now lives in `crate::traits::stackless` —
// re-exported below for callers that still spell out
// `resumable::stackless::system::StacklessTaskSystem`.
pub use crate::traits::stackless::StacklessTaskSystem;

impl<S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>> StacklessTaskSystem for S {
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

    fn yield_now() -> impl Future<Output = ()> {
        let mut yielded = false;
        std::future::poll_fn(move |cx| {
            if yielded {
                return std::task::Poll::Ready(());
            }
            yielded = true;
            // Fair only when there is a `run_async_poll` frame to actually
            // consume the flag: `polling_async` non-null is precisely "this
            // worker is synchronously driving me right now" (see that
            // field's doc comment on `UltWorker`). Outside that -- e.g.
            // `yield_now().await` reached from inside a ULT's `block_on`,
            // where no `run_async_poll` frame exists on this call chain --
            // setting the flag would just leave it there for some later,
            // unrelated task's poll to observe, so fall back to the plain
            // self-wake this method always did.
            if let Some(wk) = UltWorker::<S>::current() {
                if !wk.polling_async.get().is_null() {
                    wk.yield_requested.set(true);
                }
            }
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        })
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
impl<S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>> ScopedStacklessTaskSystem for S {
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

/// [`StacklessBuilder`](crate::traits::stackless::StacklessBuilder)
/// implementation shared by every `resumable`-backed stackless system
/// (blanket-derived just below, for any `S: StacklessSchedulerSystem`).
pub struct StacklessBuilderImpl<S> {
    num_workers: Option<usize>,
    _marker: PhantomData<fn() -> S>,
}

impl<S> StacklessBuilderImpl<S> {
    fn new() -> Self {
        StacklessBuilderImpl { num_workers: None, _marker: PhantomData }
    }
}

impl<S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>> crate::traits::stackless::StacklessBuilder<S> for StacklessBuilderImpl<S> {
    fn workers(mut self, n: usize) -> Self {
        self.num_workers = Some(n);
        self
    }

    fn run_async<F>(self, root: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let num_workers = self.num_workers.unwrap_or_else(crate::os::available_parallelism);
        crate::resumable::stackless::scheduler::run_async::<S, F>(num_workers, root)
    }
}

/// Replaces the old `ScopedStacklessTaskSystem::run_async` — same blanket
/// condition as `ScopedStacklessTaskSystem` itself just above.
impl<S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>> crate::traits::stackless::StacklessInitSystem for S {
    type Builder = StacklessBuilderImpl<Self>;

    fn builder() -> Self::Builder {
        StacklessBuilderImpl::new()
    }
}

/// Capability trait mirroring
/// [`StackfulSchedulerSystem`](crate::resumable::stackful::system::StackfulSchedulerSystem)'s
/// role on the stackless side: blanket-derived for any `SchedulerSystem`
/// whose descriptor is `AsyncTaskDesc`, purely so that
/// `HasExternalQueue<Self::Desc, Queue = Self::ExternalQueue>` (see that
/// trait's doc comment for why this specific nested-supertrait-bound shape
/// is needed) reaches every stackless leaf function (`spawn_now`,
/// `fork_async_parent_first`, `try_wake_async`) merely bounded `S:
/// StacklessSchedulerSystem`, with no need to restate it. Unlike
/// `StackfulSchedulerSystem`, this trait has no associated types of its own
/// to assemble — it exists solely as a fold point for this bound.
pub trait StacklessSchedulerSystem:
    SchedulerSystem
    + PoolSystem<
        Desc: AsyncTaskDesc
                  + crate::resumable::common::desc::TaskDescCore<
                      Owned: crate::resumable::common::desc::HasExternalQueue<Self::Desc, Queue = Self::ExternalQueue>,
                  >,
    >
{
}

impl<S: SchedulerSystem> StacklessSchedulerSystem for S
where
    S::Desc: AsyncTaskDesc,
    <S::Desc as crate::resumable::common::desc::TaskDescCore>::Owned:
        crate::resumable::common::desc::HasExternalQueue<S::Desc, Queue = S::ExternalQueue>,
{
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
/// This system's `RunnableItem` impl (`resumable::stackless::worker`) is
/// shaped in spirit like the stackful-only one but for polling instead of
/// switching: it always polls, with no `poll_fn`-tag branch, because every
/// task on this system is one.
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
/// use cmpth::{NestableSystem, StacklessBuilder, StacklessInitSystem, StacklessTaskSystem, ThreadSystem};
///
/// pub struct MyAsyncMarker;
///
/// impl cmpth::UltAsyncIdentity for MyAsyncMarker {
///     type Base = cmpth::OsSystem;
///     type Desc = cmpth::StacklessOnlyTaskDesc<cmpth::UltAsyncSystem<Self>>;
///     type RunQueue = cmpth::HybridRunQueue<cmpth::SuspendedTaskToken<cmpth::StacklessOnlyTaskDesc<cmpth::UltAsyncSystem<Self>>>>;
///     type Lookup = cmpth::InlineTlsCurrent;
///
///     fn worker_tls_anchor() -> &'static <cmpth::OsSystem as NestableSystem>::ThreadSpecific<cmpth::UltWorker<cmpth::UltAsyncSystem<Self>>> {
///         static A: cmpth::TlsAnchor = cmpth::TlsAnchor::new();
///         cmpth::TlsSlot::from_anchor(&A)
///     }
/// }
///
/// type MyAsyncSystem = cmpth::UltAsyncSystem<MyAsyncMarker>;
///
/// MyAsyncSystem::builder().workers(2).run_async(async {
///     let h = MyAsyncSystem::spawn(|| async { 6 * 7 }).await;
///     assert_eq!(h.await, 42);
/// });
/// ```
pub trait UltAsyncIdentity: Sized + Send + Sync + 'static {
    /// The threading system this scheduler runs on.
    type Base: ThreadSystem + NestableSystem;

    /// Task descriptor type. Most implementors want
    /// [`StacklessOnlyTaskDesc<UltAsyncSystem<Self>>`](crate::resumable::stackless::desc::StacklessOnlyTaskDesc)
    /// (no unused `ctx` slot); a system that also needs stackful `spawn`/
    /// dual capability on the same tasks wants
    /// [`DualTaskDesc<UltAsyncSystem<Self>>`](crate::resumable::dual::desc::DualTaskDesc)
    /// instead. `where UltAsyncSystem<Self>: PoolSystem` — the lowest rung
    /// that makes the bound below well-formed (`Owned: HasExternalQueue<_,
    /// Queue = <UltAsyncSystem<Self> as PoolSystem>::ExternalQueue>` needs
    /// that projection nameable). Deliberately **not**
    /// `UltAsyncSystem<Self>: SchedulerSystem`: `SchedulerSystem` is now
    /// blanket-derived from `Desc`'s own identity (`Desc:
    /// ReclaimableDesc<_>`, `Item: RunnableItem<_>` — see `common::system`),
    /// so gating `Desc` itself on `SchedulerSystem` would be circular
    /// (`Desc` needs `SchedulerSystem` needs `Desc`, verified as a real
    /// `E0275` overflow before this bound was lowered). `PoolSystem` is
    /// safe because [`PoolSystem for UltAsyncSystem<M>`](struct@UltAsyncSystem)'s
    /// impl below is unconditional — it never routes back through
    /// `SchedulerSystem`.
    type Desc: crate::resumable::common::desc::TaskDescAlloc
        + AsyncTaskDesc
        + crate::resumable::common::desc::TaskDescCore<
            Owned: crate::resumable::common::desc::HasExternalQueue<
                Self::Desc,
                Queue = <UltAsyncSystem<Self> as PoolSystem>::ExternalQueue,
            >,
        >
    where
        UltAsyncSystem<Self>: PoolSystem;

    /// Work-stealing run queue implementation.
    type RunQueue: WorkerRunQueue<crate::resumable::common::desc::SuspendedTaskToken<Self::Desc>> + Default;

    /// Fixed slot size for the `spawn_async` descriptor pool.
    const ASYNC_POOL_SIZE: usize = 512;

    /// Current-worker lookup policy. `where UltAsyncSystem<Self>:
    /// WorkerSystem` — the lowest rung that makes `CurrentLookup<_>`
    /// well-formed (that trait's own declaration is `CurrentLookup<S:
    /// WorkerSystem>`); same not-`SchedulerSystem` reasoning as
    /// [`Desc`](Self::Desc). `WorkerSystem for UltAsyncSystem<M>`'s impl
    /// below is likewise unconditional.
    type Lookup: CurrentLookup<UltAsyncSystem<Self>>
    where
        UltAsyncSystem<Self>: WorkerSystem;

    /// The per-system TLS anchor backing [`WorkerSystem::worker_tls`].
    /// Named in terms of [`UltAsyncSystem<Self>`] — the actual final
    /// system type — not bare `Self`, since `Self` here is just the config
    /// marker; see this trait's own doc comment for why. `where
    /// UltAsyncSystem<Self>: WorkerSystem` — `UltWorker<_>` itself requires
    /// it structurally (`UltWorker<S: WorkerSystem>`); same
    /// not-`SchedulerSystem` reasoning as [`Desc`](Self::Desc).
    fn worker_tls_anchor() -> &'static <<Self as UltAsyncIdentity>::Base as NestableSystem>::ThreadSpecific<UltWorker<UltAsyncSystem<Self>>>
    where
        UltAsyncSystem<Self>: WorkerSystem;
}

/// The actual stackless-only system type: call `run_async`/`spawn` on
/// `UltAsyncSystem<M>`, not on `M` itself. See [`UltAsyncIdentity`]'s doc
/// comment for why `M` alone can't directly implement `SchedulerSystem`.
pub struct UltAsyncSystem<M: UltAsyncIdentity> {
    _marker: PhantomData<fn() -> M>,
}

impl<M: UltAsyncIdentity> PoolSystem for UltAsyncSystem<M> {
    type Desc  = M::Desc;
    type ExternalQueue = crate::resumable::common::external_queue::StealPathQueue<M::Desc>;
    // Never actually allocated through: this flavor has no `spawn`, only
    // `spawn_async` (which goes through AsyncPool below). SimplePool is the
    // cheapest DescPool to instantiate for a type that's never used.
    type Pool = crate::resumable::common::pool::SimplePool<M::Desc>;
    const ASYNC_POOL_SIZE: usize = <M as UltAsyncIdentity>::ASYNC_POOL_SIZE;
    type AsyncPool = crate::resumable::common::pool::ReturnPool<M::Desc, crate::resumable::common::stack::HeapStack>;
    type RecursionPool = crate::resumable::common::pool::ThresholdPool<crate::resumable::common::pool::BlockPool>;
}

impl<M: UltAsyncIdentity> WorkerSystem for UltAsyncSystem<M> {
    type Base  = M::Base;
    type SuspendedToken  = crate::resumable::common::desc::SuspendedTaskToken<M::Desc>;
    type Worker = UltWorker<Self>;
    type RunQueue = M::RunQueue;
    type Lookup = <M as UltAsyncIdentity>::Lookup;

    fn worker_tls() -> &'static <M::Base as NestableSystem>::ThreadSpecific<UltWorker<Self>> {
        <M as UltAsyncIdentity>::worker_tls_anchor()
    }
}

// `SchedulerSystem for UltAsyncSystem<M>` is no longer hand-written here: it
// is blanket-derived (`common::system`) once `Item: RunnableItem<_>` and
// `Desc: ReclaimableDesc<_>` hold, which they do for any real `Desc` choice
// (`StacklessOnlyTaskDesc<Self>` or `DualTaskDesc<Self>` — see those types'
// `RunnableItem`/`ReclaimableDesc` impls in `stackless::worker`/`dual::worker`).
