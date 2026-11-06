#![doc = include_str!("../README.md")]
//!
//! # Architecture
//!
//! The design is trait-first: every component is defined as a trait, then
//! implemented by a concrete struct.  Three layers:
//!
//! * **`traits/`** — interface layer, no implementations:
//!   [`SpawnableStackfulTaskSystem`], [`traits::Resumable`], [`traits::StackfulMutex`], [`traits::StackfulBarrier`], …
//! * **`resumable/`** — implementation layer (parametric over [`SpawnableStackfulTaskSystem`])
//!   for schedulers whose defining property is that a spawned computation's
//!   *continuation* is reified into something independently resumable
//!   later (a real context-switch continuation for stackful ULTs, a
//!   pollable task for stackless ones): [`UltWorker<S>`],
//!   [`BasicStackfulOnlyResumable<S>`], sync primitives, scheduler. Sibling to
//!   [`scoped`], whose defining property is the opposite — a
//!   `parallel_call` call's own continuation is never reified/exposed,
//!   so its implementation needs none of this machinery.
//! * **`lib.rs`** — instantiations: [`DefaultDualTaskSystem`], [`DefaultNestedDualTaskSystem`],
//!   [`DefaultStackfulOnlyTaskSystem`], [`DefaultStacklessOnlyTaskSystem`].
//!
//! Every stackful system implements [`SpawnableStackfulTaskSystem`] directly, so schedulers
//! nest: set `type Base = DefaultDualTaskSystem` in a second `SpawnableStackfulTaskSystem`
//! implementation to run ULTs on top of ULTs.

#[macro_use]
mod macros;
pub mod traits;
mod os;
mod spin;
pub mod interchange;
pub mod resumable;
pub mod scoped;
pub mod duty;
pub mod post;

pub use traits::{BarrierWaitResult, StackfulParking, BlockOnSystem, Context, ContextPolicy, CallCx, CallSlot, Completion, DelegationSystem, DelegatorConsumer, Delegator, DelegatorRun, DualBarrier, FavorPosting, FavorServing, FnConsumer, PostChoice, PostConfig, PostConsumer, PostDelegator, PostDelegatorOf, PostFavor, PostingSystem, Ready, Ticket, DualMutex, HandoffTaskDesc, JoinHandleLike, NestableSystem, Poller, Resumable, ScopedStackfulTaskSystem, ScopedStacklessTaskSystem, StackfulBuilder, StackfulInitSystem, StackfulResumable, StackfulSyncSystem, StacklessBuilder, StacklessInitSystem, StacklessSyncSystem, SuspendableSystem, TaskDesc, TaskExitSink, TaskSystem, TlsAnchor, TlsSlot, SpawnableStackfulTaskSystem, Transfer, WakerTaskDesc, WorkerTlsAnchor};
pub use scoped::{DefaultScopedMarker, ScopedIdentity, ScopedSystem, ScopedTaskSystem};
pub use os::{available_parallelism, OsBarrier, OsCondvar, OsMutex, OsPoller, OsSystem, OsTls};
pub use resumable::stackful::context::{LeanFrameContext, NativeContext};
pub use resumable::stackful::waker::UltPoller;
pub use resumable::common::deque::{HybridRunQueue, RunQueueStealer, SpinRunQueue, Steal, WorkerRunQueue};
pub use resumable::common::desc::{SuspendedTaskToken, TaskDescAlloc};
pub use resumable::dual::desc::DualTaskDesc;
pub use resumable::stackful::desc::{StackfulOnlyTaskDesc, StackfulTaskDesc};
pub use resumable::stackless::desc::{AsyncTaskDesc, StacklessOnlyTaskDesc};
pub use resumable::common::external_queue::{ExternalQueue, PollerUltQueue, StealPathQueue};
pub use resumable::common::idle::{
    CondvarWait, FixedCap, FutexWait, GoCap, HalfCap, IdlePolicy, ParkIdle, ParkWait, SearchCap, SearchIdle, SpinIdle,
};
pub use resumable::common::lookup::{CurrentLookup, TlsCurrent};
pub use resumable::common::steal::{RandomSteal, RandomSweepSteal, StealPolicy, SweepSteal};
pub use resumable::common::dispatch::{DispatchPolicy, Site, StandardDispatch, TaskSources};
pub use resumable::common::observe::{NoObserver, SchedObserver, StealOutcome};
pub use resumable::common::idle::{IdleContext, IdleCx, IdleDuties};
pub use resumable::common::qos::{BackgroundQos, DemoteIdleQos, OsQos, QosClass, QosPolicy, SplitQos};
pub use resumable::stackless::lookup::InlineTlsCurrent;
pub use resumable::common::pool::{DescPool, ReturnPool, SimplePool};
pub use resumable::common::stack::{HeapStack, MallocFrame, StackAlloc};
pub use resumable::stackless::async_wait::SuspendedFuture;
pub use resumable::dual::dual_wait::DualResumable;
pub use resumable::dual::system::{UltDualIdentity, UltDualSystem};
pub use resumable::stackful::suspended::{BasicStackfulOnlyResumable, StackfulOnlyResumableCore};
pub use resumable::stackful::sync::{Barrier as UltBarrier, CombiningDelegator, McsDelegator, McsMutex, McsMutexGuard, McsCondvar, BarrierCore, MutexCore};
pub use resumable::common::sync::{DualBarrier as UltDualBarrier, DualMutex as UltDualMutex, DualMutexGuard as UltDualMutexGuard};
pub use resumable::common::system::{PoolSystem, SchedulerSystem, WorkerSystem};
pub use resumable::stackful::system::{StackfulSchedulerSystem, StackfulTaskSystem, StackfulWorkerSystem, UltIdentity};
pub use resumable::stackless::system::{StacklessSchedulerSystem, StacklessTaskSystem, UltAsyncIdentity, UltAsyncSystem};
pub use resumable::stackful::tls::UltTls;
pub use resumable::common::worker::{LocalQueue, TaskPool, UltWorker, WorkerOps, current_worker};
pub use resumable::stackful::worker::ContextSwitcher;
pub use post::{DefaultPostChoice, QueuePostDelegator, SlotPostDelegator};

// ---------------------------------------------------------------------------
// Default instantiations
// ---------------------------------------------------------------------------

// The dual systems host stackless `spawn_async` tasks as well as stackful
// ULTs, and their `Mutex`/`Barrier` are contended together from either
// calling convention; see `UltDualIdentity`.  Each is a marker type plus
// its config, named through `UltDualSystem<_>`; the markers stay hidden,
// as for `DefaultStacklessOnlyTaskSystem`.  Every system below states only
// what differs from `impl_with_defaults!`'s defaults.

#[doc(hidden)]
pub struct DefaultDualMarker;

worker_tls_anchor!(DefaultDualMarker);

impl_with_defaults! {
    impl UltDualIdentity for DefaultDualMarker {}
}

/// The default ULT system: runs on top of [`OsSystem`].
pub type DefaultDualTaskSystem = UltDualSystem<DefaultDualMarker>;

#[doc(hidden)]
pub struct ParkDualMarker;

worker_tls_anchor!(ParkDualMarker);

impl_with_defaults! {
    impl UltDualIdentity for ParkDualMarker {
        type Idle = SearchIdle;
    }
}

/// [`DefaultDualTaskSystem`] whose idle workers block their OS thread
/// instead of spinning: a few search, the others park ([`SearchIdle`]).
pub type ParkDualTaskSystem = UltDualSystem<ParkDualMarker>;

#[doc(hidden)]
pub struct ParkHalfDualMarker;

worker_tls_anchor!(ParkHalfDualMarker);

impl_with_defaults! {
    impl UltDualIdentity for ParkHalfDualMarker {
        type Idle = SearchIdle<HalfCap>;
    }
}

/// [`ParkDualTaskSystem`] letting half the workers search
/// ([`HalfCap`], Tokio's limit): faster to pick up work when
/// parallelism is low, at more energy.
pub type ParkHalfDualTaskSystem = UltDualSystem<ParkHalfDualMarker>;

#[doc(hidden)]
pub struct DefaultNestedDualMarker;

worker_tls_anchor!(DefaultNestedDualMarker);

impl_with_defaults! {
    impl UltDualIdentity for DefaultNestedDualMarker {
        type Base = DefaultDualTaskSystem;
    }
}

/// A second-level ULT system: runs on top of [`DefaultDualTaskSystem`]'s ULTs.
pub type DefaultNestedDualTaskSystem = UltDualSystem<DefaultNestedDualMarker>;

/// The default stackful-*only* ULT system: runs on top of [`OsSystem`],
/// genuinely no stackless capability exercised (`spawn_async`'s blanket
/// `StacklessTaskSystem` impl is technically still present, since
/// `DualTaskDesc: AsyncTaskDesc` unconditionally, but nothing on this
/// system ever calls it) -- unlike [`DefaultDualTaskSystem`], which pays
/// for dual-flavor dispatch (a `poll_fn`-tag check per popped
/// continuation, a tagged-word wait slot) on every task even when nothing
/// on it ever calls `spawn_async`.
///
/// Use this instead of `DefaultDualTaskSystem` whenever a system never
/// needs `spawn_async`/async-capable waiting -- the same "pick a system"
/// call site as any other. Implements [`UltIdentity`] directly: unlike
/// [`DefaultStacklessOnlyTaskSystem`], no wrapper type is needed (see
/// `UltIdentity`'s blanket `impl<M: UltIdentity> SchedulerSystem for M`).
pub struct DefaultStackfulOnlyTaskSystem;

worker_tls_anchor!(DefaultStackfulOnlyTaskSystem);

impl_with_defaults! {
    impl UltIdentity for DefaultStackfulOnlyTaskSystem {}
}

// `DefaultStacklessOnlyTaskSystem`'s marker: implements `UltAsyncIdentity`
// directly rather than being exposed itself -- callers only ever need to
// name the type alias below, the same way `UltAsyncSystem`'s own type
// parameter is never named at a `DefaultDualTaskSystem`-style call site.
#[doc(hidden)]
pub struct DefaultStacklessOnlyMarker;

worker_tls_anchor!(DefaultStacklessOnlyMarker);

impl_with_defaults! {
    impl UltAsyncIdentity for DefaultStacklessOnlyMarker {}
}

/// The default stackless-*only* ULT system: runs on top of [`OsSystem`],
/// genuinely no stackful capability at all (no context-switch policy, no
/// stack allocator) -- unlike [`DefaultDualTaskSystem`], which also hosts
/// `spawn_async` but pays for dual-flavor dispatch (a `poll_fn`-tag check
/// per popped continuation, a tagged-word wait slot) on every task even
/// when nothing on it ever calls the stackful `spawn`. Measured ~10-13%
/// faster than `DefaultDualTaskSystem` on a pure-async `spawn_async` workload for
/// exactly that reason.
///
/// Use this instead of `DefaultDualTaskSystem` whenever a system never needs
/// stackful `spawn`/blocking `Mutex`/`block_on` — the same "pick a system"
/// call site as any other, just via `UltAsyncIdentity` instead of
/// `UltIdentity`.
pub type DefaultStacklessOnlyTaskSystem = UltAsyncSystem<DefaultStacklessOnlyMarker>;

pub mod system {
    pub use crate::traits::{JoinHandleLike, StackfulBarrier, StackfulMutex, SpawnableStackfulTaskSystem};
    pub use crate::os::{OsBarrier, OsCondvar, OsMutex, OsSystem};

    pub type WssSystem = crate::DefaultDualTaskSystem;
}
