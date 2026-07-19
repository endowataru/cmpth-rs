#![doc = include_str!("../README.md")]
//!
//! # Architecture
//!
//! The design is trait-first: every component is defined as a trait, then
//! implemented by a concrete struct.  Three layers:
//!
//! * **`traits/`** — interface layer, no implementations:
//!   [`ThreadSystem`], [`traits::Resumable`], [`traits::Mutex`], [`traits::Barrier`], …
//! * **`ult/`** — ULT implementation layer (parametric over [`UltSystem`]):
//!   [`UltWorker<S>`], [`BasicSuspendedThread<S>`], sync primitives, scheduler.
//! * **`lib.rs`** — instantiations: [`DefaultUltSystem`], [`DefaultUltUltSystem`].
//!
//! Because every `UltSystem` is also a `ThreadSystem`, schedulers nest: set
//! `type Base = DefaultUltSystem` in a second `UltSystem` implementation to
//! run ULTs on top of ULTs.

mod context;
pub mod traits;
mod os;
mod spin;
pub mod future;
pub mod ult;

pub use context::{CondTransfer, Context, ContextPolicy, NativeContext, Transfer};
pub use traits::{BarrierWaitResult, DelegatorConsumer, Delegator, DualBarrier, DualMutex, JoinHandleLike, Poller, Resumable, StackfulResumable, TlsAnchor, TlsSlot, ThreadSystem};
pub use os::{OsBarrier, OsCondvar, OsMutex, OsPoller, OsSystem, OsTls};
pub use ult::waker::UltPoller;
pub use ult::deque::{CrossbeamDeque, SpinDeque, WorkerDeque};
pub use ult::external_queue::{ExternalQueue, PollerUltQueue, StealPathQueue};
pub use ult::lookup::{CurrentLookup, SpCurrent, TlsCurrent};
pub use ult::pool::{DescPool, ReturnPool, SimplePool};
pub use ult::stack::{ArenaStack, HeapStack, StackAlloc};
pub use ult::async_wait::SuspendedFuture;
pub use ult::dual_wait::SuspendedTask;
pub use ult::suspended::{BasicSuspendedThread, UltSuspendedThread};
pub use ult::sync::{Barrier as UltBarrier, McsDelegator, McsMutex, McsMutexGuard, McsCondvar, BarrierCore, MutexCore, DualBarrier as UltDualBarrier, DualMutex as UltDualMutex, DualMutexGuard as UltDualMutexGuard};
pub use ult::sync::{delegator, Producer as DelegatorProducer};
pub use ult::system::{AsyncWorkerSystem, UltContextSystem, UltSchedulerSystem, UltSystem};
pub use ult::tls::UltTls;
pub use ult::worker::{ContextSwitcher, LocalQueue, TaskPool, UltWorker, Worker, current_worker};

// ---------------------------------------------------------------------------
// Default instantiations
// ---------------------------------------------------------------------------

// Both default systems host stackless spawn_async-style tasks as well as
// stackful ULTs (`ult::system::AsyncWorkerSystem` + `UltSystem`), and their
// `Mutex`/`Barrier` are meant to be contended-together from either calling
// convention -- so `ult_system!` (which can't assume `AsyncWorkerSystem`
// is also implemented, since a stackful-only system must stay expressible)
// isn't used here. Everything it would have generated is written out by
// hand instead, with `UltSystem::Mutex`/`Barrier` bound to the same
// `SuspendedTask`-parameterized `DualMutex`/`DualBarrier` as
// `AsyncWorkerSystem::Mutex`/`Barrier`: since `SuspendedTask<S>`
// implements both `StackfulResumable` and `StacklessResumable` (unlike
// the macro's default `BasicSuspendedThread`, stackful-only), one type
// satisfies both `StackfulMutex` and `StacklessMutex`, so a single
// instance is genuinely shared and contended between stackful and
// stackless callers -- not two separate locks that happen to have the
// same name.

/// The default ULT system: runs on top of [`OsSystem`].
pub struct DefaultUltSystem;

impl ult::system::UltContextSystem for DefaultUltSystem {
    type StackAlloc = ult::stack::HeapStack;
}

impl ult::system::UltSchedulerSystem for DefaultUltSystem {
    type Base  = OsSystem;
    type Ctx   = NativeContext;
    type Deque = CrossbeamDeque;
    const STACK_SIZE: usize = 64 * 1024;

    type SuspendedThread = ult::suspended::BasicSuspendedThread<Self>;
    type ExternalQueue   = ult::external_queue::StealPathQueue;
    type Pool            = ult::pool::ReturnPool<ult::stack::HeapStack>;
    type Lookup          = ult::lookup::TlsCurrent;

    fn worker_tls() -> &'static <OsSystem as ThreadSystem>::ThreadSpecific<UltWorker<Self>> {
        static A: TlsAnchor = TlsAnchor::new();
        TlsSlot::from_anchor(&A)
    }
}

impl UltSystem for DefaultUltSystem {
    type Mutex<T: Send> = UltDualMutex<Self, T, SuspendedTask<Self>>;
    type Barrier         = UltDualBarrier<Self, SuspendedTask<Self>>;
    type Delegator<C: DelegatorConsumer<Self>> = McsDelegator<Self, C>;

    fn run<F>(num_workers: usize, root: F)
    where
        F: FnOnce() + Send + 'static,
    {
        ult::scheduler::run::<Self, F>(num_workers, root)
    }
}

impl ult::system::AsyncWorkerSystem for DefaultUltSystem {
    type Mutex<T: Send> = UltDualMutex<Self, T, SuspendedTask<Self>>;
    type Barrier = UltDualBarrier<Self, SuspendedTask<Self>>;
}

/// A second-level ULT system: runs on top of [`DefaultUltSystem`]'s ULTs.
pub struct DefaultUltUltSystem;

impl ult::system::UltContextSystem for DefaultUltUltSystem {
    type StackAlloc = ult::stack::HeapStack;
}

impl ult::system::UltSchedulerSystem for DefaultUltUltSystem {
    type Base  = DefaultUltSystem;
    type Ctx   = NativeContext;
    type Deque = CrossbeamDeque;
    const STACK_SIZE: usize = 64 * 1024;

    type SuspendedThread = ult::suspended::BasicSuspendedThread<Self>;
    type ExternalQueue   = ult::external_queue::StealPathQueue;
    type Pool            = ult::pool::ReturnPool<ult::stack::HeapStack>;
    type Lookup          = ult::lookup::TlsCurrent;

    fn worker_tls() -> &'static <DefaultUltSystem as ThreadSystem>::ThreadSpecific<UltWorker<Self>> {
        static A: TlsAnchor = TlsAnchor::new();
        TlsSlot::from_anchor(&A)
    }
}

impl UltSystem for DefaultUltUltSystem {
    type Mutex<T: Send> = UltDualMutex<Self, T, SuspendedTask<Self>>;
    type Barrier         = UltDualBarrier<Self, SuspendedTask<Self>>;
    type Delegator<C: DelegatorConsumer<Self>> = McsDelegator<Self, C>;

    fn run<F>(num_workers: usize, root: F)
    where
        F: FnOnce() + Send + 'static,
    {
        ult::scheduler::run::<Self, F>(num_workers, root)
    }
}

impl ult::system::AsyncWorkerSystem for DefaultUltUltSystem {
    type Mutex<T: Send> = UltDualMutex<Self, T, SuspendedTask<Self>>;
    type Barrier = UltDualBarrier<Self, SuspendedTask<Self>>;
}

// ---------------------------------------------------------------------------
// Convenience API for the default system
//
// Import with `use cmpth::default::*` to make the choice of default
// implementation visible at the call site.
// ---------------------------------------------------------------------------

pub mod default {
    //! Convenience aliases and entry points bound to [`DefaultUltSystem`](crate::DefaultUltSystem).
    //!
    //! Import with `use cmpth::default::*` to bring the default-system API
    //! into scope.  The explicit module path signals that you are using the
    //! *default* ULT scheduler; code that needs a different system should
    //! call `MySystem::run(...)`, etc. directly on its own marker struct.

    use crate::traits::ThreadSystem as _;
    use crate::traits::UltSystem as _;

    /// Start the default scheduler with `num_workers` OS threads and run
    /// `root` as the first task.  Returns when `root` and every task it
    /// spawned have completed.
    ///
    /// ```
    /// cmpth::default::run(2, || {
    ///     println!("running on a ULT");
    /// });
    /// ```
    pub fn run<F>(num_workers: usize, root: F)
    where
        F: FnOnce() + Send + 'static,
    {
        crate::DefaultUltSystem::run(num_workers, root);
    }

    /// Spawn a ULT (child-first: the child starts immediately and the
    /// caller's continuation becomes stealable).  Must be called on a ULT,
    /// i.e. inside [`run`].
    ///
    /// ```
    /// cmpth::default::run(2, || {
    ///     let h = cmpth::default::spawn(|| 6 * 7);
    ///     assert_eq!(h.join().unwrap(), 42);
    /// });
    /// ```
    pub fn spawn<T, F>(f: F) -> JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        crate::ult::thread::spawn::<crate::DefaultUltSystem, T, F>(f)
    }

    /// Spawn a `Future` as a stackless task: the executor polls it in place,
    /// with no stack allocation and no context switch per poll.
    ///
    /// ```
    /// cmpth::default::run(2, || {
    ///     let h = cmpth::default::spawn_async(async { 6 * 7 });
    ///     assert_eq!(h.join().unwrap(), 42);
    /// });
    /// ```
    pub fn spawn_async<T, F>(f: F) -> JoinHandle<T>
    where
        F: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        crate::ult::thread::spawn_async::<crate::DefaultUltSystem, T, F>(f)
    }

    pub fn yield_now() {
        crate::DefaultUltSystem::yield_now();
    }

    pub type JoinHandle<T> = crate::ult::thread::JoinHandle<crate::DefaultUltSystem, T>;
    pub type Mutex<T> = crate::ult::sync::McsMutex<crate::DefaultUltSystem, T>;
    pub type MutexGuard<'a, T> = crate::ult::sync::McsMutexGuard<'a, crate::DefaultUltSystem, T>;
    pub type Condvar = crate::ult::sync::McsCondvar<crate::DefaultUltSystem>;
    pub type Barrier = crate::ult::sync::Barrier<crate::DefaultUltSystem>;
}

// ---------------------------------------------------------------------------
// Compatibility module for dependent crates (lite-rma, lite-dsm)
// ---------------------------------------------------------------------------

pub mod system {
    pub use crate::traits::{Barrier, Condvar, JoinHandleLike, Mutex, ThreadSystem};
    pub use crate::os::{OsBarrier, OsCondvar, OsMutex, OsSystem};

    pub type WssSystem = crate::DefaultUltSystem;
}
