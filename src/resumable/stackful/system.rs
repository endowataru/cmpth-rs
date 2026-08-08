//! [`StackfulSchedulerSystem`] — extends
//! [`SchedulerSystem`]
//! with real-stack context-switch capability, and the [`UltIdentity`]
//! config trait that assembles a complete stackful-only system from a
//! handful of associated types.
//!
//! # Nesting
//!
//! Every `UltIdentity` implementor is a full `ThreadSystem`, so naming a
//! second one as `Base` stacks one ULT scheduler on top of another without
//! any extra boilerplate:
//!
//! ```ignore
//! pub struct DefaultDualTaskSystem;
//! impl cmpth::UltIdentity for DefaultDualTaskSystem { type Base = cmpth::OsSystem; ... }
//!
//! pub struct DefaultNestedDualTaskSystem;
//! impl cmpth::UltIdentity for DefaultNestedDualTaskSystem { type Base = DefaultDualTaskSystem; ... }
//! ```

use crate::traits::stackful::{
    BlockOnSystem, ContextPolicy, DelegationSystem, NestableSystem, StackfulSyncSystem,
    SuspendableSystem, ThreadSystem,
};
use crate::resumable::common::deque::WorkerRunQueue;
use crate::resumable::common::lookup::CurrentLookup;
use crate::resumable::common::system::{DescScheduler, PoolSystem, SchedulerSystem, WorkerSystem};
use crate::resumable::common::desc::{HasExternalQueue, SuspendedTaskToken, TaskDescCore};
use crate::resumable::common::stack::StackAlloc;
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::stackful::suspended::StackfulOnlyResumableCore;
use crate::resumable::common::worker::UltWorker;

// `StackfulTaskSystem` now lives in `crate::traits::stackful` — re-exported
// below for callers that still spell out
// `resumable::stackful::system::StackfulTaskSystem`.
pub use crate::traits::stackful::StackfulTaskSystem;

/// Extends [`WorkerSystem`] with real-stack context-switch machinery:
/// context-switch policy, stack allocator, stack size, and the
/// stackful parked-continuation type. Split out from [`SchedulerSystem`]'s
/// dispatch axis (`execute`/`free_finished_desc`) so this axis — "can this
/// system run real ULTs" — can be named independently; [`StackfulSchedulerSystem`]
/// below folds the two back together for callers that need both.
///
/// "This system can run real ULTs" is checked at [`StackfulSchedulerSystem`]
/// (via its `DescScheduler<Desc: StackfulTaskDesc + ...>` nest), not here.
///
/// Deliberately does **not** nest ANY bound on `WorkerSystem::Desc` here —
/// not `StackfulTaskDesc`, and not `TaskDescCore<Owned:
/// HasExternalQueue<Self::Desc, Queue = Self::ExternalQueue>>` (which an
/// earlier version of this trait nested alongside it). Nesting *any* bound
/// on `Desc` here is fine as long as `Desc` stays abstract — it is then
/// just an implied bound nobody has to discharge — but the moment any impl
/// pins `Desc` to a concrete type — exactly what the `RunnableItem`/
/// `ReclaimableDesc` impls in `stackful::worker`/`dual::worker` do — every
/// unrelated fact this crate needs about that same concrete `Desc`
/// (`HasExternalQueue`, but empirically also `HasPollFn` reached via
/// `AsyncTaskDesc` for the dual flavor, and almost certainly anything else
/// projected through the same `WorkerSystem::Desc` slot) stops normalizing
/// — not just the nested bound itself. Verified directly: removing only the
/// `HasExternalQueue` nest left a real `E0277` on `HasPollFn` for the dual
/// `RunnableItem` impl, which disappeared only once `StackfulTaskDesc` was
/// removed too. No amount of restating either bound as a `where`-clause
/// rescues it — the clause is not assumed while its own well-formedness is
/// checked. So this trait now carries no opinion at all about `Desc`'s
/// shape; every real requirement on it lives on [`StackfulSchedulerSystem`]
/// instead — the fold trait, never named directly by an impl that must
/// itself stay below `SchedulerSystem`.
pub trait StackfulWorkerSystem: WorkerSystem
{
    /// Context-switch implementation.
    type Ctx: ContextPolicy;

    /// Stack allocation policy for this system.
    type StackAlloc: crate::resumable::common::stack::StackAlloc;

    /// Stack size for each ULT (in bytes).
    const STACK_SIZE: usize;

    /// Parked-continuation type for this system. `where Self::Desc:
    /// StackfulTaskDesc` lives on this one associated type's own
    /// declaration, not on the trait's supertrait bound list: unlike a
    /// supertrait-list nest, a `where`-clause here doesn't propagate as an
    /// implied bound to callers merely bounded `S: StackfulWorkerSystem`
    /// (deliberately — see this trait's own doc comment for why nesting it
    /// unconditionally there is exactly the poisoning problem), but it
    /// keeps this associated type itself well-formed to declare, and every
    /// real caller that actually names `Self::SuspendedThread` already
    /// carries `StackfulTaskDesc` independently (via
    /// [`StackfulSchedulerSystem`] or a concrete `Desc` pin).
    type SuspendedThread: StackfulOnlyResumableCore<StackfulWorkerSystem = Self>
    where
        Self::Desc: StackfulTaskDesc;

    /// Resolve what a suspending/exiting ULT switches into when its local
    /// deque is empty: the worker's own root (scheduler-loop) continuation.
    ///
    /// `where Self: DescScheduler`: not part of this trait's own supertrait
    /// bound (`Self` is only known to be `WorkerSystem` inside
    /// `StackfulWorkerSystem` itself — the whole reason dispatch was split
    /// off into [`SchedulerSystem`] in the first place), but `UltWorker<Self>`
    /// requires `SchedulerSystem` structurally (`UltWorker<S: SchedulerSystem>`)
    /// and [`pop_or_root_stackful`](crate::resumable::stackful::worker::pop_or_root_stackful)'s
    /// `SuspendedTaskToken<Self::Desc>` return type requires the `Item`
    /// pinning `DescScheduler` provides (`wk.deque.try_pop()` hands back
    /// `Self::Item`), so both live on the one method that actually needs
    /// them. Every real implementor satisfies it (see
    /// [`StackfulSchedulerSystem`]'s doc comment).
    ///
    /// Default: [`crate::resumable::stackful::worker::pop_or_root_stackful`] — correct
    /// whenever `Self::Desc` isn't also `AsyncTaskDesc` (stackful-only),
    /// since every popped item is then guaranteed to be a real, switchable
    /// continuation. Dual configs override with
    /// [`crate::resumable::dual::worker::pop_or_root_dual`], which requeues an async
    /// task popped off the top instead of trying to switch into it.
    fn pop_or_root(wk: &UltWorker<Self>) -> SuspendedTaskToken<Self::Desc>
    where
        Self: DescScheduler,
    {
        crate::resumable::stackful::worker::pop_or_root_stackful(wk)
    }
}

/// The fold point: a [`SchedulerSystem`] (dispatch) that is also a
/// [`StackfulWorkerSystem`] (real-stack capability) whose scheduling unit is
/// this crate's own task descriptor/worker ([`DescScheduler`]) — i.e. every
/// system built on the `resumable` engine that can run real ULTs.
///
/// Blanket-derived (not implemented directly by concrete systems anymore):
/// every member that used to live here now lives on [`SchedulerSystem`] or
/// [`StackfulWorkerSystem`], so once a concrete system implements both of
/// those (plus [`DescScheduler`], itself blanket-derived whenever `Item`/
/// `Worker` have their expected shapes), it gets `StackfulSchedulerSystem`
/// for free.
///
/// `Desc: StackfulTaskDesc` and `Owned: HasExternalQueue<Self::Desc, Queue =
/// Self::ExternalQueue>` are nested directly in the supertrait bound list
/// (`DescScheduler<Desc: ...>`), not a separate `where`-clause — that's
/// what lets every function merely bounded `S: StackfulSchedulerSystem` get
/// both for free, with no need to restate either. A `where`-clause form
/// (`SchedulerSystem where Self::Desc: ...`) does *not* propagate this way
/// (verified empirically, both for a `where`-clause on this trait's own
/// declaration and for one on `SchedulerSystem::Desc`'s declaration in a
/// different trait) — only associated-type bounds nested in a supertrait's
/// own bound list are treated as real implied bounds. [`DescScheduler`]
/// itself is one such supertrait, folding in `Item`/`Worker` once so this
/// trait doesn't have to restate them.
///
/// Both `StackfulTaskDesc` and `HasExternalQueue` live here now, not on
/// [`StackfulWorkerSystem`] (see that trait's doc comment for why nesting
/// *any* bound on `Desc` there breaks unrelated obligations on the same
/// concrete descriptor once one is pinned): nesting them here, on the fold
/// trait, means they only ever normalize against a concrete `Desc` at a
/// point where `Desc`'s well-formedness is already independently
/// established (via `DescScheduler`/`StackfulWorkerSystem` each
/// individually holding), never while an impl's *own* bounds are still
/// being checked.
pub trait StackfulSchedulerSystem:
    SchedulerSystem
    + StackfulWorkerSystem
    + DescScheduler<Desc: StackfulTaskDesc + TaskDescCore<Owned: HasExternalQueue<Self::Desc, Queue = Self::ExternalQueue>>>
{
}

impl<
    S: SchedulerSystem
        + StackfulWorkerSystem
        + DescScheduler<Desc: StackfulTaskDesc + TaskDescCore<Owned: HasExternalQueue<S::Desc, Queue = S::ExternalQueue>>>,
> StackfulSchedulerSystem for S
{
}

// ---------------------------------------------------------------------------
// Blanket ScopedStackfulTaskSystem/StackfulTaskSystem for every ThreadSystem
// ---------------------------------------------------------------------------

/// `parallel_call`'s "nothing outlives this call" constraint is strictly
/// stricter than `spawn`/`join`'s (a spawned task may outlive the caller),
/// so anything with `ThreadSystem` capability trivially satisfies it too —
/// spawn `a`, run `b` inline, join. Same shape as
/// `bench/src/lib.rs`'s `BenchSystem::par_join` default body, which this
/// predates and mirrors.
impl<S: ThreadSystem + StackfulSchedulerSystem> crate::traits::scoped::ScopedStackfulTaskSystem for S
where
    S::Desc: StackfulTaskDesc,
{
    fn parallel_call<Fa, Fb, Ra, Rb>(a: Fa, b: Fb) -> (Ra, Rb)
    where
        Fa: FnOnce() -> Ra + Send + 'static,
        Fb: FnOnce() -> Rb + Send + 'static,
        Ra: Send + 'static,
        Rb: Send + 'static,
    {
        let h = <S as ThreadSystem>::spawn(a);
        let rb = b();
        (crate::traits::stackful::JoinHandleLike::join(h), rb)
    }
}

/// The bracketing/standalone-init entry point (`Builder::run`/`Builder::init`),
/// replacing what used to be `ScopedStackfulTaskSystem::run` — same blanket
/// condition as that trait's own impl just above, since both ultimately
/// need the same "real ULTs on a real stack" capability.
impl<S: ThreadSystem + StackfulSchedulerSystem> crate::traits::stackful::StackfulInitSystem for S
where
    S::Desc: StackfulTaskDesc,
{
    type Builder = crate::resumable::stackful::init::StackfulBuilderImpl<Self>;
    type Init = crate::resumable::stackful::init::StackfulInit<Self>;

    fn builder() -> Self::Builder {
        crate::resumable::stackful::init::StackfulBuilderImpl::new()
    }
}

/// Empty bundle: `ThreadSystem` is implemented directly (via `UltIdentity`'s
/// blanket impl or by hand); `ScopedStackfulTaskSystem` is blanket-derived
/// from it just above. This impl just ties the four bounds together as one.
impl<
    S: crate::traits::scoped::ScopedStackfulTaskSystem + ThreadSystem + StackfulSyncSystem + BlockOnSystem,
> crate::traits::stackful::StackfulTaskSystem for S
{
}

// ---------------------------------------------------------------------------
// UltIdentity
// ---------------------------------------------------------------------------

/// Assembles a complete stackful-only ULT system from a handful of
/// associated types — the config-trait replacement for what used to be the
/// `ult_system!` macro. Implement this for your own marker type and a
/// blanket `SchedulerSystem`/`StackfulSchedulerSystem`/`ThreadSystem` impl
/// covers the rest.
///
/// Not a generic struct (`UltSystem<Base, Ctx, ...>`) that callers would
/// type-alias: Rust's orphan rules forbid implementing a foreign trait
/// (`SchedulerIdentity`-shaped) for a foreign type (a type alias for a
/// `cmpth`-defined generic struct is still `cmpth`'s type, not the
/// caller's) — verified directly against a real downstream crate. A
/// config trait sidesteps this: the caller's own marker type is what
/// implements `UltIdentity`, and the blanket impls below (written inside
/// `cmpth`, where they're allowed to name `cmpth`'s own traits freely) are
/// what extend it with `SchedulerSystem`/etc.
///
/// `Lookup`/`worker_tls_anchor` are the two members that must be resolved
/// through `Self`, not a free type parameter of some other type: `Lookup`
/// exposing itself as a blanket-impl condition on an unrelated generic
/// parameter would create a self-referential trait-resolution cycle
/// through [`CurrentLookup`]'s own blanket impl
/// (`impl<S: WorkerSystem> CurrentLookup<S> for TlsCurrent`) — proving this
/// trait's own `where Self: WorkerSystem` gate (see [`Lookup`](Self::Lookup))
/// would require re-entering that very impl if `Lookup` weren't pinned to
/// `Self` directly. `worker_tls_anchor`'s `static` has the same requirement
/// for an unrelated reason: a `static` declared inside a generic function
/// body is one shared instance across every monomorphization, not one per
/// instantiation (also verified directly) — every implementor needs its
/// own `static`, anchored by its own function body.
///
/// ```
/// use cmpth::{ThreadSystem, NestableSystem, StackfulBuilder, StackfulInitSystem, JoinHandleLike};
///
/// pub struct MySystem;
///
/// impl cmpth::UltIdentity for MySystem {
///     type Base = cmpth::OsSystem;
///     type Ctx = cmpth::NativeContext;
///     type Desc = cmpth::StackfulOnlyTaskDesc<Self>;
///     type RunQueue = cmpth::HybridRunQueue<cmpth::SuspendedTaskToken<cmpth::StackfulOnlyTaskDesc<Self>>>;
///     type Alloc = cmpth::HeapStack;
///     type Lookup = cmpth::TlsCurrent;
///
///     fn worker_tls_anchor() -> &'static <cmpth::OsSystem as NestableSystem>::ThreadSpecific<cmpth::UltWorker<Self>> {
///         static A: cmpth::TlsAnchor = cmpth::TlsAnchor::new();
///         cmpth::TlsSlot::from_anchor(&A)
///     }
/// }
///
/// MySystem::builder().workers(2).run(|| {
///     let h = MySystem::spawn(|| 42);
///     assert_eq!(JoinHandleLike::join(h), 42);
/// });
/// ```
///
/// `STACK_SIZE` defaults to 64 KiB; override it like any other associated
/// const.
pub trait UltIdentity: Sized + Send + Sync + 'static {
    /// The threading system this scheduler runs on.
    type Base: ThreadSystem + NestableSystem;

    /// Context-switch implementation.
    type Ctx: ContextPolicy;

    /// Task descriptor type. Most implementors want
    /// [`StackfulOnlyTaskDesc<Self>`](crate::resumable::stackful::desc::StackfulOnlyTaskDesc)
    /// (no unused `poll_fn` slot); a system that also needs `spawn_async`/
    /// dual capability on the same tasks wants
    /// [`DualTaskDesc<Self>`](crate::resumable::dual::desc::DualTaskDesc)
    /// instead. `where Self: PoolSystem` — the lowest rung that makes the
    /// bound below well-formed (`Owned: HasExternalQueue<Self::Desc, Queue =
    /// Self::ExternalQueue>` needs `Self::ExternalQueue`, a `PoolSystem`
    /// associated type). Deliberately **not** `Self: SchedulerSystem`:
    /// `SchedulerSystem` is now blanket-derived from `Desc`'s own identity
    /// (`Desc: ReclaimableDesc<Self>`, `Item: RunnableItem<Self>` — see
    /// `common::system`), so gating `Desc` itself on `SchedulerSystem` would
    /// be circular (`Desc` needs `SchedulerSystem` needs `Desc`, verified as
    /// a real `E0275` overflow before this bound was lowered). `PoolSystem`
    /// is safe because [`PoolSystem for M`](trait@UltIdentity)'s impl below
    /// is unconditional — it never routes back through `SchedulerSystem`.
    type Desc: crate::resumable::common::desc::TaskDescAlloc
        + StackfulTaskDesc
        + TaskDescCore<Owned: HasExternalQueue<<Self as UltIdentity>::Desc, Queue = Self::ExternalQueue>>
    where
        Self: PoolSystem;

    /// Work-stealing run queue implementation.
    type RunQueue: WorkerRunQueue<SuspendedTaskToken<Self::Desc>> + Default;

    /// Stack allocation policy.
    type Alloc: StackAlloc;

    /// Stack size for each ULT (in bytes).
    const STACK_SIZE: usize = 64 * 1024;

    /// Current-worker lookup policy. `where Self: WorkerSystem` — the lowest
    /// rung that makes `CurrentLookup<Self>` well-formed (that trait's own
    /// declaration is `CurrentLookup<S: WorkerSystem>`); same
    /// not-`SchedulerSystem` reasoning as [`Desc`](Self::Desc). `WorkerSystem
    /// for M`'s impl below is likewise unconditional.
    type Lookup: CurrentLookup<Self>
    where
        Self: WorkerSystem;

    /// The per-system TLS anchor backing [`WorkerSystem::worker_tls`].
    /// `where Self: WorkerSystem` — `UltWorker<Self>` itself requires it
    /// structurally (`UltWorker<S: WorkerSystem>`); same not-`SchedulerSystem`
    /// reasoning as [`Desc`](Self::Desc).
    fn worker_tls_anchor() -> &'static <<Self as UltIdentity>::Base as NestableSystem>::ThreadSpecific<UltWorker<Self>>
    where
        Self: WorkerSystem;
}

impl<M: UltIdentity> PoolSystem for M {
    type Desc  = M::Desc;
    type ExternalQueue = crate::resumable::common::external_queue::StealPathQueue<M::Desc>;
    type Pool          = crate::resumable::common::pool::ReturnPool<M::Desc, M::Alloc>;
    // Never actually allocated through: nothing calls spawn_async on a
    // stackful-only UltIdentity system (StacklessTaskSystem's blanket
    // impl still applies whenever M::Desc: AsyncTaskDesc, e.g. for
    // DualTaskDesc, but the capability just goes unused here). Mirrors
    // UltAsyncIdentity's unused `Pool` in the other direction.
    type AsyncPool = crate::resumable::common::pool::SimplePool<M::Desc>;
    const ASYNC_POOL_SIZE: usize = 0;
    // Never actually taken from: nothing calls `recurse` on a
    // stackful-only UltIdentity system either. Mirrors `AsyncPool` above.
    type RecursionPool = crate::resumable::common::pool::ThresholdPool<crate::resumable::common::pool::BlockPool>;
}

impl<M: UltIdentity> WorkerSystem for M {
    type Base  = M::Base;
    type Item  = SuspendedTaskToken<M::Desc>;
    type Worker = UltWorker<Self>;
    type RunQueue = M::RunQueue;
    type Lookup = <M as UltIdentity>::Lookup;

    fn worker_tls() -> &'static <M::Base as NestableSystem>::ThreadSpecific<UltWorker<Self>> {
        <M as UltIdentity>::worker_tls_anchor()
    }
}

// `SchedulerSystem for M` is no longer hand-written here: it is
// blanket-derived (`common::system`) once `M::Item: RunnableItem<M>` and
// `M::Desc: ReclaimableDesc<M>` hold, which they do for any real `Desc`
// choice (`StackfulOnlyTaskDesc<Self>` or `DualTaskDesc<Self>` — see those
// types' `RunnableItem`/`ReclaimableDesc` impls in `stackful::worker`/
// `dual::worker`).

impl<M: UltIdentity> StackfulWorkerSystem for M
where
    <M as PoolSystem>::Desc: StackfulTaskDesc,
    <<M as PoolSystem>::Desc as TaskDescCore>::Owned:
        HasExternalQueue<<M as PoolSystem>::Desc, Queue = <M as PoolSystem>::ExternalQueue>,
{
    type Ctx = M::Ctx;
    type StackAlloc = M::Alloc;
    const STACK_SIZE: usize = <M as UltIdentity>::STACK_SIZE;

    type SuspendedThread = crate::resumable::stackful::suspended::BasicStackfulOnlyResumable<Self>;
}

impl<M: UltIdentity + StackfulSchedulerSystem> ThreadSystem for M
where
    <M as PoolSystem>::Desc: StackfulTaskDesc,
{
    fn yield_now() {
        use crate::resumable::common::worker::WorkerOps;
        use crate::resumable::stackful::worker::StackfulWorker;
        match UltWorker::<Self>::current() {
            Some(wk) => { wk.yield_now(); }
            None => <<M as UltIdentity>::Base as ThreadSystem>::yield_now(),
        }
    }

    type JoinHandle<T: Send + 'static> = crate::resumable::common::thread::JoinHandle<Self, T>;

    fn spawn<T, F>(f: F) -> crate::resumable::common::thread::JoinHandle<Self, T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        crate::resumable::stackful::thread::spawn::<Self, T, F>(f)
    }
}

impl<M: UltIdentity + StackfulSchedulerSystem> BlockOnSystem for M
where
    <M as PoolSystem>::Desc: StackfulTaskDesc,
{
    type Poller = crate::resumable::stackful::waker::ResumablePoller<Self>;
}

impl<M: UltIdentity + StackfulSchedulerSystem> StackfulSyncSystem for M
where
    <M as PoolSystem>::Desc: StackfulTaskDesc,
{
    type Mutex<T: Send>  = crate::resumable::common::sync::DualMutex<Self, T, crate::resumable::stackful::suspended::BasicStackfulOnlyResumable<Self>>;
    type Barrier         = crate::resumable::common::sync::DualBarrier<Self, crate::resumable::stackful::suspended::BasicStackfulOnlyResumable<Self>>;
}

impl<M: UltIdentity + StackfulSchedulerSystem> SuspendableSystem for M
where
    <M as PoolSystem>::Desc: StackfulTaskDesc,
{
    type SuspendedThread = crate::resumable::stackful::suspended::BasicStackfulOnlyResumable<Self>;
}

impl<M: UltIdentity + StackfulSchedulerSystem> DelegationSystem for M
where
    <M as PoolSystem>::Desc: StackfulTaskDesc,
{
    type Delegator<C: crate::traits::stackful::DelegatorConsumer<Self>> =
        crate::resumable::stackful::sync::McsDelegator<Self, C>;
}

impl<M: UltIdentity + StackfulSchedulerSystem> NestableSystem for M
where
    <M as PoolSystem>::Desc: StackfulTaskDesc,
{
    type ThreadSpecific<T: 'static> = crate::resumable::stackful::tls::UltTls<Self, T>;
}
