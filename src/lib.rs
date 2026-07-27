#![doc = include_str!("../README.md")]
//!
//! # Architecture
//!
//! The design is trait-first: every component is defined as a trait, then
//! implemented by a concrete struct.  Three layers:
//!
//! * **`traits/`** — interface layer, no implementations:
//!   [`ThreadSystem`], [`traits::Resumable`], [`traits::StackfulMutex`], [`traits::StackfulBarrier`], …
//! * **`resumable/`** — implementation layer (parametric over [`ThreadSystem`])
//!   for schedulers whose defining property is that a spawned computation's
//!   *continuation* is reified into something independently resumable
//!   later (a real context-switch continuation for stackful ULTs, a
//!   pollable task for stackless ones): [`UltWorker<S>`],
//!   [`BasicSuspendedThread<S>`], sync primitives, scheduler. Sibling to
//!   [`scoped`], whose defining property is the opposite — a
//!   `parallel_call` call's own continuation is never reified/exposed,
//!   so its implementation needs none of this machinery.
//! * **`lib.rs`** — instantiations: [`DefaultDualTaskSystem`], [`DefaultNestedDualTaskSystem`],
//!   [`DefaultStackfulOnlyTaskSystem`], [`DefaultStacklessOnlyTaskSystem`].
//!
//! Every stackful system implements [`ThreadSystem`] directly, so schedulers
//! nest: set `type Base = DefaultDualTaskSystem` in a second `ThreadSystem`
//! implementation to run ULTs on top of ULTs.

mod context;
pub mod traits;
mod os;
mod spin;
pub mod future;
pub mod resumable;
pub mod scoped;

pub use context::{CondTransfer, Context, ContextPolicy, NativeContext, Transfer};
pub use traits::{BarrierWaitResult, DelegatorConsumer, Delegator, DualBarrier, DualMutex, JoinHandleLike, Poller, Resumable, ScopedStackfulTaskSystem, ScopedStacklessTaskSystem, StackfulResumable, TaskSystem, TlsAnchor, TlsSlot, ThreadSystem};
pub use scoped::ScopedTaskSystem;
pub use os::{OsBarrier, OsCondvar, OsMutex, OsPoller, OsSystem, OsTls};
pub use resumable::stackful::waker::UltPoller;
pub use resumable::common::deque::{CrossbeamDeque, SpinDeque, WorkerDeque};
pub use resumable::common::desc::{BasicTaskDesc, SuspendedUlt, TaskDesc, TaskDescAlloc};
pub use resumable::stackful::desc::StackfulTaskDesc;
pub use resumable::stackless::desc::AsyncTaskDesc;
pub use resumable::common::external_queue::{ExternalQueue, PollerUltQueue, StealPathQueue};
pub use resumable::common::lookup::{CurrentLookup, TlsCurrent};
pub use resumable::stackless::lookup::InlineTlsCurrent;
pub use resumable::common::pool::{DescPool, ReturnPool, SimplePool};
pub use resumable::common::stack::{HeapStack, StackAlloc};
pub use resumable::stackless::async_wait::SuspendedFuture;
pub use resumable::dual::dual_wait::SuspendedTask;
pub use resumable::stackful::suspended::{BasicSuspendedThread, UltSuspendedThread};
pub use resumable::stackful::sync::{Barrier as UltBarrier, McsDelegator, McsMutex, McsMutexGuard, McsCondvar, BarrierCore, MutexCore};
pub use resumable::common::sync::{DualBarrier as UltDualBarrier, DualMutex as UltDualMutex, DualMutexGuard as UltDualMutexGuard};
pub use resumable::stackful::sync::{delegator, Producer as DelegatorProducer};
pub use resumable::common::system::SchedulerSystem;
pub use resumable::stackful::system::{StackfulSchedulerSystem, StackfulTaskSystem, UltIdentity};
pub use resumable::stackless::system::{StacklessTaskSystem, UltAsyncIdentity, UltAsyncSystem};
pub use resumable::stackful::tls::UltTls;
pub use resumable::common::worker::{LocalQueue, TaskPool, UltWorker, Worker, current_worker};
pub use resumable::stackful::worker::ContextSwitcher;

// ---------------------------------------------------------------------------
// Default instantiations
// ---------------------------------------------------------------------------

// Both default systems host stackless spawn_async-style tasks (the
// blanket `StacklessTaskSystem` impl, automatic for any async-capable
// `SchedulerSystem`) as well as stackful ULTs (`ThreadSystem`), and their
// `Mutex`/`Barrier` are meant to be contended-together from either
// calling convention -- so `UltIdentity` (whose blanket-derived `Mutex`/
// `Barrier` use the stackful-only `BasicSuspendedThread`, with no
// async-wait capability) isn't used here. Everything it would have
// generated is written out by hand instead, with `ThreadSystem::Mutex`/
// `Barrier` bound to a `SuspendedTask`-parameterized `DualMutex`/
// `DualBarrier`: since `SuspendedTask<S>` implements both
// `StackfulResumable` and `StacklessResumable` (unlike `UltIdentity`'s
// default `BasicSuspendedThread`, stackful-only), one type satisfies
// both `StackfulMutex` and `StacklessMutex`, so a single instance is
// genuinely shared and contended between stackful and stackless callers
// -- not two separate locks that happen to have the same name.

/// The default ULT system: runs on top of [`OsSystem`].
pub struct DefaultDualTaskSystem;

impl resumable::common::system::SchedulerSystem for DefaultDualTaskSystem {
    type Base  = OsSystem;
    type Desc  = resumable::common::desc::BasicTaskDesc;
    type Deque = CrossbeamDeque<resumable::common::desc::BasicTaskDesc>;
    type ExternalQueue   = resumable::common::external_queue::StealPathQueue<resumable::common::desc::BasicTaskDesc>;
    type Pool            = resumable::common::pool::ReturnPool<resumable::common::desc::BasicTaskDesc, resumable::common::stack::HeapStack>;
    type AsyncPool       = resumable::common::pool::ReturnPool<resumable::common::desc::BasicTaskDesc, resumable::stackless::stack::AsyncArenaStack>;
    const ASYNC_POOL_SIZE: usize = 512;
    type RecursionPool   = resumable::common::pool::ThresholdPool<resumable::common::pool::BlockPool>;
    type Lookup          = resumable::common::lookup::TlsCurrent;

    fn worker_tls() -> &'static <OsSystem as ThreadSystem>::ThreadSpecific<UltWorker<Self>> {
        static A: TlsAnchor = TlsAnchor::new();
        TlsSlot::from_anchor(&A)
    }

    // Dual system: a popped continuation may be either a real ULT or an
    // async task, so dispatch needs the poll_fn check (see execute_dual's
    // doc comment / StackfulSchedulerSystem::pop_or_root below for why this
    // can't just be the stackful-only default).
    fn execute(wk: &UltWorker<Self>, cont: SuspendedUlt<resumable::common::desc::BasicTaskDesc>) {
        resumable::dual::worker::execute_dual(wk, cont)
    }

    fn free_finished_desc(wk: &UltWorker<Self>, desc: *mut resumable::common::desc::BasicTaskDesc) {
        resumable::dual::worker::free_finished_desc_dual(wk, desc)
    }
}

impl resumable::stackful::system::StackfulSchedulerSystem for DefaultDualTaskSystem {
    type Ctx   = NativeContext;
    type StackAlloc = resumable::common::stack::HeapStack;
    const STACK_SIZE: usize = 64 * 1024;

    type SuspendedThread = resumable::stackful::suspended::BasicSuspendedThread<Self>;

    fn pop_or_root(wk: &UltWorker<Self>) -> SuspendedUlt<resumable::common::desc::BasicTaskDesc> {
        resumable::dual::worker::pop_or_root_dual(wk)
    }
}

impl ThreadSystem for DefaultDualTaskSystem {
    type Poller = resumable::stackful::waker::UltPoller<Self>;

    fn yield_now() {
        use resumable::stackful::worker::StackfulWorker;
        match UltWorker::<Self>::current() {
            Some(wk) => { wk.yield_now(); }
            None => <<Self as SchedulerSystem>::Base as ThreadSystem>::yield_now(),
        }
    }

    type JoinHandle<T: Send + 'static> = resumable::common::thread::JoinHandle<Self, T>;

    fn spawn<T, F>(f: F) -> resumable::common::thread::JoinHandle<Self, T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        resumable::stackful::thread::spawn::<Self, T, F>(f)
    }

    type Mutex<T: Send> = UltDualMutex<Self, T, SuspendedTask<Self>>;
    type Barrier         = UltDualBarrier<Self, SuspendedTask<Self>>;
    type SuspendedThread = resumable::stackful::suspended::BasicSuspendedThread<Self>;
    type Delegator<C: DelegatorConsumer<Self>> = McsDelegator<Self, C>;
    type ThreadSpecific<T: 'static> = resumable::stackful::tls::UltTls<Self, T>;
}

/// A second-level ULT system: runs on top of [`DefaultDualTaskSystem`]'s ULTs.
pub struct DefaultNestedDualTaskSystem;

impl resumable::common::system::SchedulerSystem for DefaultNestedDualTaskSystem {
    type Base  = DefaultDualTaskSystem;
    type Desc  = resumable::common::desc::BasicTaskDesc;
    type Deque = CrossbeamDeque<resumable::common::desc::BasicTaskDesc>;
    type ExternalQueue   = resumable::common::external_queue::StealPathQueue<resumable::common::desc::BasicTaskDesc>;
    type Pool            = resumable::common::pool::ReturnPool<resumable::common::desc::BasicTaskDesc, resumable::common::stack::HeapStack>;
    type AsyncPool       = resumable::common::pool::ReturnPool<resumable::common::desc::BasicTaskDesc, resumable::stackless::stack::AsyncArenaStack>;
    const ASYNC_POOL_SIZE: usize = 512;
    type RecursionPool   = resumable::common::pool::ThresholdPool<resumable::common::pool::BlockPool>;
    type Lookup          = resumable::common::lookup::TlsCurrent;

    fn worker_tls() -> &'static <DefaultDualTaskSystem as ThreadSystem>::ThreadSpecific<UltWorker<Self>> {
        static A: TlsAnchor = TlsAnchor::new();
        TlsSlot::from_anchor(&A)
    }

    fn execute(wk: &UltWorker<Self>, cont: SuspendedUlt<resumable::common::desc::BasicTaskDesc>) {
        resumable::dual::worker::execute_dual(wk, cont)
    }

    fn free_finished_desc(wk: &UltWorker<Self>, desc: *mut resumable::common::desc::BasicTaskDesc) {
        resumable::dual::worker::free_finished_desc_dual(wk, desc)
    }
}

impl resumable::stackful::system::StackfulSchedulerSystem for DefaultNestedDualTaskSystem {
    type Ctx   = NativeContext;
    type StackAlloc = resumable::common::stack::HeapStack;
    const STACK_SIZE: usize = 64 * 1024;

    type SuspendedThread = resumable::stackful::suspended::BasicSuspendedThread<Self>;

    fn pop_or_root(wk: &UltWorker<Self>) -> SuspendedUlt<resumable::common::desc::BasicTaskDesc> {
        resumable::dual::worker::pop_or_root_dual(wk)
    }
}

impl ThreadSystem for DefaultNestedDualTaskSystem {
    type Poller = resumable::stackful::waker::UltPoller<Self>;

    fn yield_now() {
        use resumable::stackful::worker::StackfulWorker;
        match UltWorker::<Self>::current() {
            Some(wk) => { wk.yield_now(); }
            None => <<Self as SchedulerSystem>::Base as ThreadSystem>::yield_now(),
        }
    }

    type JoinHandle<T: Send + 'static> = resumable::common::thread::JoinHandle<Self, T>;

    fn spawn<T, F>(f: F) -> resumable::common::thread::JoinHandle<Self, T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        resumable::stackful::thread::spawn::<Self, T, F>(f)
    }

    type Mutex<T: Send> = UltDualMutex<Self, T, SuspendedTask<Self>>;
    type Barrier         = UltDualBarrier<Self, SuspendedTask<Self>>;
    type SuspendedThread = resumable::stackful::suspended::BasicSuspendedThread<Self>;
    type Delegator<C: DelegatorConsumer<Self>> = McsDelegator<Self, C>;
    type ThreadSpecific<T: 'static> = resumable::stackful::tls::UltTls<Self, T>;
}

/// The default stackful-*only* ULT system: runs on top of [`OsSystem`],
/// genuinely no stackless capability exercised (`spawn_async`'s blanket
/// `StacklessTaskSystem` impl is technically still present, since
/// `BasicTaskDesc: AsyncTaskDesc` unconditionally, but nothing on this
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

impl UltIdentity for DefaultStackfulOnlyTaskSystem {
    type Base = OsSystem;
    type Ctx = NativeContext;
    type Deque = CrossbeamDeque<BasicTaskDesc>;
    type Alloc = HeapStack;
    type Lookup = TlsCurrent;

    fn worker_tls_anchor() -> &'static <OsSystem as ThreadSystem>::ThreadSpecific<UltWorker<Self>> {
        static A: TlsAnchor = TlsAnchor::new();
        TlsSlot::from_anchor(&A)
    }
}

// `DefaultStacklessOnlyTaskSystem`'s marker: implements `UltAsyncIdentity`
// directly rather than being exposed itself -- callers only ever need to
// name the type alias below, the same way `UltAsyncSystem`'s own type
// parameter is never named at a `DefaultDualTaskSystem`-style call site.
#[doc(hidden)]
pub struct DefaultStacklessOnlyMarker;

impl resumable::stackless::system::UltAsyncIdentity for DefaultStacklessOnlyMarker {
    type Base = OsSystem;
    type Deque = CrossbeamDeque<resumable::common::desc::BasicTaskDesc>;
    type Lookup = InlineTlsCurrent;

    fn worker_tls_anchor()
    -> &'static <OsSystem as ThreadSystem>::ThreadSpecific<UltWorker<UltAsyncSystem<Self>>>
    {
        static A: TlsAnchor = TlsAnchor::new();
        TlsSlot::from_anchor(&A)
    }
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

// ---------------------------------------------------------------------------
// Compatibility module for dependent crates (lite-rma, lite-dsm)
// ---------------------------------------------------------------------------

pub mod system {
    pub use crate::traits::{JoinHandleLike, StackfulBarrier, StackfulMutex, ThreadSystem};
    pub use crate::os::{OsBarrier, OsCondvar, OsMutex, OsSystem};

    pub type WssSystem = crate::DefaultDualTaskSystem;
}
