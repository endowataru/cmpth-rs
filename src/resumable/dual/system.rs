//! [`UltDualIdentity`] — the config trait that assembles a complete dual
//! system (stackful ULTs and stackless tasks on one scheduler) from a
//! handful of associated types, and [`UltDualSystem<M>`], the system it
//! builds.

use std::marker::PhantomData;

use crate::resumable::common::deque::WorkerRunQueue;
use crate::resumable::common::desc::SuspendedTaskToken;
use crate::resumable::common::lookup::CurrentLookup;
use crate::resumable::common::stack::StackAlloc;
use crate::resumable::common::system::{PoolSystem, WorkerSystem};
use crate::resumable::common::worker::UltWorker;
use crate::resumable::dual::desc::DualTaskDesc;
use crate::traits::stackful::{
    BlockOnSystem, ContextPolicy, DelegationSystem, NestableSystem, StackfulSyncSystem,
    SuspendableSystem, SpawnableStackfulTaskSystem,
};
use crate::traits::{TlsSlot, WorkerTlsAnchor};

/// Assembles a complete **dual** ULT system: stackful `spawn` and
/// stackless `spawn_async` tasks share one scheduler, and its `Mutex`/
/// `Barrier` are contended together from either calling convention
/// ([`DualResumable`](crate::DualResumable) waiters, unlike
/// [`UltIdentity`](crate::UltIdentity)'s stackful-only ones).  Implement
/// this for your own marker type and use [`UltDualSystem<M>`] as the
/// actual system — the [`UltAsyncIdentity`](crate::UltAsyncIdentity)/
/// [`UltAsyncSystem`](crate::UltAsyncSystem) shape, for the same reason: a
/// blanket impl on bare `M` would overlap with `UltIdentity`'s own under
/// Rust's coherence rules.
///
/// The task descriptor is not a choice here: a dual system's tasks are
/// [`DualTaskDesc`]s.  The worker's TLS slot comes from the
/// [`WorkerTlsAnchor`] supertrait ([`worker_tls_anchor!`](crate::worker_tls_anchor)).
///
/// ```
/// use cmpth::{JoinHandleLike, SpawnableStackfulTaskSystem, StackfulBuilder, StackfulInitSystem};
///
/// pub struct MyDualMarker;
///
/// cmpth::worker_tls_anchor!(MyDualMarker);
///
/// impl cmpth::UltDualIdentity for MyDualMarker {
///     type Base = cmpth::OsSystem;
///     type Ctx = cmpth::NativeContext;
///     type RunQueue = cmpth::HybridRunQueue<cmpth::SuspendedTaskToken<cmpth::DualTaskDesc<cmpth::UltDualSystem<Self>>>>;
///     type Alloc = cmpth::HeapStack;
///     type Lookup = cmpth::TlsCurrent;
///     type Idle = cmpth::SearchIdle;
/// }
///
/// type MyDualSystem = cmpth::UltDualSystem<MyDualMarker>;
///
/// MyDualSystem::builder().workers(2).run(|| {
///     let h = MyDualSystem::spawn(|| 42);
///     assert_eq!(JoinHandleLike::join(h), 42);
/// });
/// ```
///
/// `STACK_SIZE` defaults to 64 KiB and `ASYNC_POOL_SIZE` to 512; override
/// them like any other associated const.
pub trait UltDualIdentity: WorkerTlsAnchor + Sized + Send + Sync + 'static {
    /// The threading system this scheduler runs on.
    type Base: SpawnableStackfulTaskSystem + NestableSystem;

    /// Context-switch implementation.
    type Ctx: ContextPolicy;

    /// Work-stealing run queue implementation.
    type RunQueue: WorkerRunQueue<SuspendedTaskToken<DualTaskDesc<UltDualSystem<Self>>>> + Default;

    /// Stack allocation policy for spawned ULTs (and the scheduler's own
    /// stack).  `spawn_async` frames always come from
    /// [`MallocFrame`](crate::MallocFrame): they are written in bounds by
    /// construction, so a guard page would protect nothing.
    type Alloc: StackAlloc;

    /// Idle/wakeup policy; see [`WorkerSystem::Idle`].
    type Idle: crate::resumable::common::idle::IdlePolicy;

    /// Current-worker lookup policy.  `where UltDualSystem<Self>:
    /// WorkerSystem`, as for [`UltAsyncIdentity`](crate::UltAsyncIdentity)'s
    /// `Lookup`.
    type Lookup: CurrentLookup<UltDualSystem<Self>>
    where
        UltDualSystem<Self>: WorkerSystem;

    /// Stack size for each ULT (in bytes).
    const STACK_SIZE: usize = 64 * 1024;

    /// Fixed slot size for the `spawn_async` descriptor pool.
    const ASYNC_POOL_SIZE: usize = 512;
}

/// The actual dual system type: call `builder`/`spawn`/`spawn_async` on
/// `UltDualSystem<M>`, not on `M` itself.
pub struct UltDualSystem<M: UltDualIdentity> {
    _marker: PhantomData<fn() -> M>,
}

impl<M: UltDualIdentity> PoolSystem for UltDualSystem<M> {
    type Desc = DualTaskDesc<Self>;
    type ExternalQueue = crate::resumable::common::external_queue::StealPathQueue<DualTaskDesc<Self>>;
    type Pool = crate::resumable::common::pool::ReturnPool<DualTaskDesc<Self>, M::Alloc>;
    type AsyncPool = crate::resumable::common::pool::ReturnPool<DualTaskDesc<Self>, crate::resumable::common::stack::MallocFrame>;
    const ASYNC_POOL_SIZE: usize = <M as UltDualIdentity>::ASYNC_POOL_SIZE;
    type RecursionPool = crate::resumable::common::pool::ThresholdPool<crate::resumable::common::pool::BlockPool>;
}

impl<M: UltDualIdentity> WorkerSystem for UltDualSystem<M> {
    type Base = M::Base;
    type SuspendedToken = SuspendedTaskToken<DualTaskDesc<Self>>;
    type Worker = UltWorker<Self>;
    type RunQueue = M::RunQueue;
    type Lookup = <M as UltDualIdentity>::Lookup;
    type Idle = <M as UltDualIdentity>::Idle;

    fn worker_tls() -> &'static <M::Base as NestableSystem>::ThreadSpecific<UltWorker<Self>> {
        TlsSlot::from_anchor(<M as WorkerTlsAnchor>::anchor())
    }
}

// `SchedulerSystem for UltDualSystem<M>` is blanket-derived
// (`resumable::common::system`) from the `RunnableItem`/`ReclaimableDesc`
// impls for `DualTaskDesc<_>` (`resumable::dual::worker`).

impl<M: UltDualIdentity> crate::resumable::stackful::system::StackfulWorkerSystem for UltDualSystem<M> {
    type Ctx = M::Ctx;
    type StackAlloc = M::Alloc;
    const STACK_SIZE: usize = <M as UltDualIdentity>::STACK_SIZE;

    type SuspendedThread = crate::resumable::stackful::suspended::BasicStackfulOnlyResumable<Self>;

    fn pop_or_root(wk: &UltWorker<Self>) -> SuspendedTaskToken<<Self as PoolSystem>::Desc>
    where
        SuspendedTaskToken<<Self as PoolSystem>::Desc>: From<<Self as WorkerSystem>::SuspendedToken>,
    {
        crate::resumable::dual::worker::pop_or_root_dual(wk)
    }
}

impl<M: UltDualIdentity> SpawnableStackfulTaskSystem for UltDualSystem<M> {
    fn yield_now() {
        use crate::resumable::common::worker::WorkerOps;
        use crate::resumable::stackful::worker::StackfulWorker;
        match UltWorker::<Self>::current() {
            Some(wk) => { wk.yield_now(); }
            None => <M::Base as SpawnableStackfulTaskSystem>::yield_now(),
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

impl<M: UltDualIdentity> BlockOnSystem for UltDualSystem<M> {
    type Poller = crate::resumable::stackful::waker::UltPoller<Self>;
}

impl<M: UltDualIdentity> StackfulSyncSystem for UltDualSystem<M> {
    type Mutex<T: Send> = crate::resumable::common::sync::DualMutex<Self, T, crate::DualResumable<Self>>;
    type Barrier = crate::resumable::common::sync::DualBarrier<Self, crate::DualResumable<Self>>;
}

impl<M: UltDualIdentity> SuspendableSystem for UltDualSystem<M> {
    /// Dual, so that OS threads that are not workers can park on it too
    /// (a delegator or lock shared with them).
    type SuspendedThread = crate::DualResumable<Self>;
}

impl<M: UltDualIdentity> DelegationSystem for UltDualSystem<M> {
    type Delegator<C: crate::traits::stackful::DelegatorConsumer<Self>> =
        crate::resumable::stackful::sync::McsDelegator<Self, C>;
}

impl<M: UltDualIdentity> crate::traits::PostingSystem for UltDualSystem<M> {
    type PostChoice = crate::post::DefaultPostChoice;
}

impl<M: UltDualIdentity> NestableSystem for UltDualSystem<M> {
    type ThreadSpecific<T: 'static> = crate::resumable::stackful::tls::UltTls<Self, T>;
}
