//! [`StackfulSchedulerSystem`] — extends
//! [`SchedulerSystem`]
//! with real-stack context-switch capability, and the [`UltIdentity`]
//! config trait that assembles a complete stackful-only system from a
//! handful of associated types.
//!
//! # Nesting
//!
//! Every `UltIdentity` implementor is a full `SpawnableStackfulTaskSystem`, so naming a
//! second one as `Base` stacks one ULT scheduler on top of another without
//! any extra boilerplate:
//!
//! ```ignore
//! pub struct Outer;
//! impl cmpth::UltIdentity for Outer { type Base = cmpth::OsSystem; ... }
//!
//! pub struct Inner;
//! impl cmpth::UltIdentity for Inner { type Base = Outer; ... }
//! ```
//!
//! [`DefaultNestedDualTaskSystem`](crate::DefaultNestedDualTaskSystem) is
//! built the same way through [`UltDualIdentity`](crate::UltDualIdentity).

use crate::traits::stackful::{
    BlockOnSystem, ContextPolicy, DelegationSystem, NestableSystem, StackfulSyncSystem,
    SuspendableSystem, SpawnableStackfulTaskSystem,
};
use crate::traits::{TlsSlot, WorkerTlsAnchor};
use crate::resumable::common::deque::WorkerRunQueue;
use crate::resumable::common::lookup::CurrentLookup;
use crate::resumable::common::system::{PoolSystem, SchedulerSystem, WorkerSystem};
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
/// (via its `PoolSystem<Desc: StackfulTaskDesc + ...>` nest), not here.
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
        Self: PoolSystem,
        Self::Desc: StackfulTaskDesc;

    /// What a suspending/exiting ULT switches into: the dispatch policy's
    /// pick at [`Site::Switch`](crate::Site::Switch), else the worker's own
    /// root (scheduler-loop) continuation.
    ///
    /// Default: [`crate::resumable::stackful::worker::next_or_root_stackful`] —
    /// correct whenever `Self::Desc` isn't also `AsyncTaskDesc`
    /// (stackful-only), since every task is then a real, switchable
    /// continuation. Dual systems override with
    /// [`crate::resumable::dual::worker::next_or_root_dual`], which puts a
    /// stackless task back instead of trying to switch into it.
    #[inline]
    fn next_or_root(wk: &UltWorker<Self>) -> SuspendedTaskToken<Self::Desc>
    where
        Self: PoolSystem,
        SuspendedTaskToken<Self::Desc>: From<Self::SuspendedToken>,
    {
        crate::resumable::stackful::worker::next_or_root_stackful(wk)
    }
}

/// The fold point: a [`SchedulerSystem`] (dispatch) that is also a
/// [`StackfulWorkerSystem`] (real-stack capability) whose worker can
/// actually drive real ULTs — i.e. every system built on the `resumable`
/// engine that can run real ULTs.
///
/// Blanket-derived (not implemented directly by concrete systems anymore):
/// every member that used to live here now lives on [`SchedulerSystem`] or
/// [`StackfulWorkerSystem`], so once a concrete system implements both of
/// those, plus `Worker`/`Desc` having their expected *capabilities* (not
/// identities — see below), it gets `StackfulSchedulerSystem` for free.
///
/// No `DescScheduler` anywhere in this fold, unlike earlier revisions:
/// `Worker: StackfulWorker<Self>` is a **capability** bound, not an
/// identity pin (`Worker = UltWorker<Self>`) — `UltWorker<S>` is the only
/// thing that implements `WorkerOps`/`ContextSwitcher`/`StackfulLocalQueue`
/// today, but nothing here has to say so. `Desc: StackfulTaskDesc +
/// TaskDescCore<Owned: HasExternalQueue<...>>` is likewise a bound, not an
/// identity. The one remaining `Worker = UltWorker<Self>` identity pin this
/// crate needs lives nowhere near this trait: it's local to
/// [`crate::resumable::stackful::init::init`] and its immediate callers —
/// the actual "construct the concrete `Scheduler<S>`" boundary, the one
/// place that's legitimately allowed to name `UltWorker<S>` directly. Every
/// generic function merely bounded `S: StackfulSchedulerSystem` never needs
/// that identity: it only ever asks `S::Worker` to implement a capability.
///
/// `Desc: StackfulTaskDesc` and `Owned: HasExternalQueue<Self::Desc, Queue =
/// Self::ExternalQueue>` are nested directly in the supertrait bound list
/// (`PoolSystem<Desc: ...>`), not a separate `where`-clause — that's
/// what lets every function merely bounded `S: StackfulSchedulerSystem` get
/// both for free, with no need to restate either. A `where`-clause form
/// (`SchedulerSystem where Self::Desc: ...`) does *not* propagate this way
/// (verified empirically, both for a `where`-clause on this trait's own
/// declaration and for one on `SchedulerSystem::Desc`'s declaration in a
/// different trait) — only associated-type bounds nested in a supertrait's
/// own bound list are treated as real implied bounds.
///
/// Both `StackfulTaskDesc` and `HasExternalQueue` live here now, not on
/// [`StackfulWorkerSystem`] (see that trait's doc comment for why nesting
/// *any* bound on `Desc` there breaks unrelated obligations on the same
/// concrete descriptor once one is pinned): nesting them here, on the fold
/// trait, means they only ever normalize against a concrete `Desc` at a
/// point where `Desc`'s well-formedness is already independently
/// established, never while an impl's *own* bounds are still being
/// checked.
///
/// `Worker: StackfulWorker<Self>` had to be nested here rather than folded
/// through a separate marker trait: `StackfulWorker<S>`'s own declaration
/// used to require `S: StackfulSchedulerSystem`, which is mutually
/// recursive with this trait nesting `Worker: StackfulWorker<Self>` on
/// itself — proving `UltWorker<X>: StackfulWorker<X>` needed `X:
/// StackfulSchedulerSystem`, which needed `X::Worker: StackfulWorker<X>`
/// again, an infinite loop (`overflow evaluating the requirement`).
/// Lowering `StackfulWorker<S>`'s own bound to `S: StackfulWorkerSystem`
/// (all its default method bodies only ever need `WorkerOps`/
/// `ContextSwitcher`/`StackfulLocalQueue`, never dispatch) broke the cycle.
pub trait StackfulSchedulerSystem:
    SchedulerSystem
    + StackfulWorkerSystem<Worker: crate::resumable::stackful::worker::StackfulWorker<Self>>
    + PoolSystem<Desc: StackfulTaskDesc + TaskDescCore<Owned: HasExternalQueue<Self::Desc, Queue = Self::ExternalQueue>>>
{
}

impl<
    S: SchedulerSystem
        + StackfulWorkerSystem<Worker: crate::resumable::stackful::worker::StackfulWorker<S>>
        + PoolSystem<Desc: StackfulTaskDesc + TaskDescCore<Owned: HasExternalQueue<S::Desc, Queue = S::ExternalQueue>>>,
> StackfulSchedulerSystem for S
{
}

// ---------------------------------------------------------------------------
// Blanket ScopedStackfulTaskSystem/StackfulTaskSystem for every SpawnableStackfulTaskSystem
// ---------------------------------------------------------------------------

/// Pool-backed, fork-parent-first: `b` is built as a real task and pushed
/// to the run queue without ever switching into it
/// (`crate::resumable::stackful::thread::parallel_call`, backed by
/// `ContextPolicy::make_context`); `a` runs inline on the caller's own
/// stack. If `b` was never stolen, popping it back by identity affords
/// running it as a plain function call — no context switch, no join
/// protocol. See `docs/scoped-ult-promotion.md` §9.8.4 for the design
/// rationale and measurements.
impl<S: SpawnableStackfulTaskSystem + StackfulSchedulerSystem> crate::traits::scoped::ScopedStackfulTaskSystem for S
where
    S::Desc: StackfulTaskDesc,
    S::Worker: crate::resumable::stackful::worker::ContextSwitcher<S>
        + crate::resumable::common::worker::DescWorkerOps<S>,
{
    fn parallel_call<Fa, Fb, Ra, Rb>(a: Fa, b: Fb) -> (Ra, Rb)
    where
        Fa: FnOnce() -> Ra + Send + 'static,
        Fb: FnOnce() -> Rb + Send + 'static,
        Ra: Send + 'static,
        Rb: Send + 'static,
    {
        crate::resumable::stackful::thread::parallel_call::<S, Ra, Fa, Rb, Fb>(a, b)
    }
}

/// The bracketing/standalone-init entry point (`Builder::run`/`Builder::init`),
/// replacing what used to be `ScopedStackfulTaskSystem::run` — same blanket
/// condition as that trait's own impl just above, since both ultimately
/// need the same "real ULTs on a real stack" capability.
impl<S: SpawnableStackfulTaskSystem + StackfulSchedulerSystem> crate::traits::stackful::StackfulInitSystem for S
where
    S::Desc: StackfulTaskDesc,
    S: WorkerSystem<Worker = UltWorker<S>>,
{
    type Builder = crate::resumable::stackful::init::StackfulBuilderImpl<Self>;
    type Init = crate::resumable::stackful::init::StackfulInit<Self>;

    fn builder() -> Self::Builder {
        crate::resumable::stackful::init::StackfulBuilderImpl::new()
    }
}

/// Empty bundle: `SpawnableStackfulTaskSystem` is implemented directly (via `UltIdentity`'s
/// blanket impl or by hand); `ScopedStackfulTaskSystem` is blanket-derived
/// from it just above. This impl just ties the four bounds together as one.
impl<
    S: crate::traits::scoped::ScopedStackfulTaskSystem + SpawnableStackfulTaskSystem + StackfulSyncSystem + BlockOnSystem,
> crate::traits::stackful::StackfulTaskSystem for S
{
}

// ---------------------------------------------------------------------------
// UltIdentity
// ---------------------------------------------------------------------------

/// Assembles a complete stackful-only ULT system from a handful of
/// associated types — the config-trait replacement for what used to be the
/// `ult_system!` macro. Implement this for your own marker type and a
/// blanket `SchedulerSystem`/`StackfulSchedulerSystem`/`SpawnableStackfulTaskSystem` impl
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
/// `Lookup` must be resolved through `Self`, not a free type parameter of
/// some other type: `Lookup` exposing itself as a blanket-impl condition on
/// an unrelated generic parameter would create a self-referential
/// trait-resolution cycle through [`CurrentLookup`]'s own blanket impl
/// (`impl<S: WorkerSystem> CurrentLookup<S> for TlsCurrent`) — proving this
/// trait's own `where Self: WorkerSystem` gate (see [`Lookup`](Self::Lookup))
/// would require re-entering that very impl if `Lookup` weren't pinned to
/// `Self` directly.
///
/// The worker's TLS slot comes from the [`WorkerTlsAnchor`] supertrait,
/// implemented with [`worker_tls_anchor!`](crate::worker_tls_anchor):
/// every implementor needs a `static` of its own, which no default can
/// provide (a `static` in a generic function body is one instance shared
/// by every monomorphization).
///
/// ```
/// use cmpth::{SpawnableStackfulTaskSystem, StackfulBuilder, StackfulInitSystem, JoinHandleLike};
///
/// pub struct MySystem;
///
/// cmpth::worker_tls_anchor!(MySystem);
///
/// impl cmpth::UltIdentity for MySystem {
///     type Base = cmpth::OsSystem;
///     type Ctx = cmpth::NativeContext;
///     type Desc = cmpth::StackfulOnlyTaskDesc<Self>;
///     type RunQueue = cmpth::HybridRunQueue<cmpth::SuspendedTaskToken<cmpth::StackfulOnlyTaskDesc<Self>>>;
///     type Alloc = cmpth::HeapStack;
///     type Lookup = cmpth::TlsCurrent;
///     type Idle = cmpth::SpinIdle;
///     type Steal = cmpth::RandomSweepSteal;
///     type Dispatch = cmpth::StandardDispatch;
///     type Duties = cmpth::duty::GlobalDuties;
///     type Observe = cmpth::NoObserver;
///     type Yield = cmpth::LocalYield;
///     type ExternalQueue = cmpth::StealPathQueue<cmpth::StackfulOnlyTaskDesc<Self>>;
///     type PostChoice = cmpth::DefaultPostChoice;
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
pub trait UltIdentity: WorkerTlsAnchor + Sized + Send + Sync + 'static {
    /// The threading system this scheduler runs on.
    type Base: SpawnableStackfulTaskSystem + NestableSystem;

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
        + TaskDescCore<Owned: HasExternalQueue<<Self as UltIdentity>::Desc, Queue = <Self as PoolSystem>::ExternalQueue>>
    where
        Self: PoolSystem;

    /// Work-stealing run queue implementation.
    type RunQueue: WorkerRunQueue<SuspendedTaskToken<Self::Desc>> + Default;

    /// Idle/wakeup policy. `where Self: WorkerSystem`, same reasoning as
    /// [`Lookup`](Self::Lookup).
    type Idle: crate::resumable::common::idle::IdlePolicy;

    /// Steal policy; see [`crate::WorkerSystem::Steal`].
    type Steal: crate::resumable::common::steal::StealPolicy;

    /// Dispatch policy; see [`crate::WorkerSystem::Dispatch`].
    type Dispatch: crate::resumable::common::dispatch::DispatchPolicy;

    /// Duties the idle workers run; see [`crate::WorkerSystem::Duties`].
    type Duties: crate::duty::DutyDomain;

    /// Scheduler observer; see [`crate::WorkerSystem::Observe`].
    type Observe: crate::resumable::common::observe::SchedObserver;

    /// Where a yielding task waits, and the turn budget; see
    /// [`crate::WorkerSystem::Yield`].
    type Yield: crate::traits::component::yield_policy::YieldPolicy<<Self as WorkerSystem>::SuspendedToken>
    where
        Self: WorkerSystem;

    /// Where tasks woken from outside the pool wait; see
    /// [`PoolSystem::ExternalQueue`].
    type ExternalQueue: crate::resumable::common::external_queue::ExternalQueue<Self>
    where
        Self: PoolSystem;

    /// How `post` chooses a delegator; see
    /// [`PostingSystem::PostChoice`](crate::traits::PostingSystem::PostChoice).
    type PostChoice: crate::traits::PostChoice;

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
}

impl<M: UltIdentity> PoolSystem for M {
    type Desc  = M::Desc;
    type ExternalQueue = <M as UltIdentity>::ExternalQueue;
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
    type SuspendedToken  = SuspendedTaskToken<M::Desc>;
    type Worker = UltWorker<Self>;
    type RunQueue = M::RunQueue;
    type Lookup = <M as UltIdentity>::Lookup;
    type Idle = <M as UltIdentity>::Idle;
    type Steal = <M as UltIdentity>::Steal;
    type Dispatch = <M as UltIdentity>::Dispatch;
    type Duties = <M as UltIdentity>::Duties;
    type Observe = <M as UltIdentity>::Observe;
    type Yield = <M as UltIdentity>::Yield;

    fn worker_tls() -> &'static <M::Base as NestableSystem>::ThreadSpecific<UltWorker<Self>> {
        TlsSlot::from_anchor(<M as WorkerTlsAnchor>::anchor())
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

impl<M: UltIdentity + StackfulSchedulerSystem> SpawnableStackfulTaskSystem for M
where
    <M as PoolSystem>::Desc: StackfulTaskDesc,
    M::Worker: crate::resumable::stackful::worker::StackfulWorker<M>
        + crate::resumable::common::worker::DescWorkerOps<M>,
{
    fn yield_now() {
        use crate::resumable::common::worker::WorkerOps;
        use crate::resumable::stackful::worker::StackfulWorker;
        match M::Worker::current() {
            Some(wk) => { wk.yield_now(); }
            None => <<M as UltIdentity>::Base as SpawnableStackfulTaskSystem>::yield_now(),
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
    M::Worker: crate::resumable::common::worker::DescWorkerOps<M>,
{
    type Delegator<C: crate::traits::stackful::DelegatorConsumer<Self>> =
        crate::resumable::stackful::sync::McsDelegator<Self, C>;
}

impl<M: UltIdentity + StackfulSchedulerSystem> crate::traits::PostingSystem for M
where
    <M as PoolSystem>::Desc: StackfulTaskDesc,
{
    type PostChoice = <M as UltIdentity>::PostChoice;
}

impl<M: UltIdentity + StackfulSchedulerSystem> NestableSystem for M
where
    <M as PoolSystem>::Desc: StackfulTaskDesc,
    M::Worker: crate::resumable::common::worker::DescWorkerOps<M>,
{
    type ThreadSpecific<T: 'static> = crate::resumable::stackful::tls::UltTls<Self, T>;
}
