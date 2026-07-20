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

crate::ult_system! {
    /// The default ULT system: runs on top of [`OsSystem`].
    pub struct DefaultUltSystem {
        base:       crate::OsSystem,
        context:    crate::NativeContext,
        deque:      crate::CrossbeamDeque,
        stack_size: 64 * 1024,
    }
}

crate::ult_system! {
    /// A second-level ULT system: runs on top of [`DefaultUltSystem`]'s ULTs.
    pub struct DefaultUltUltSystem {
        base:       crate::DefaultUltSystem,
        context:    crate::NativeContext,
        deque:      crate::CrossbeamDeque,
        stack_size: 64 * 1024,
    }
}

// Both default systems can also host stackless spawn_async-style tasks.
// See `ult::system::AsyncWorkerSystem` — kept separate from the
// `ult_system!` macro deliberately (see docs/sync-async-unification.md).
// `Mutex`/`Barrier` here are bound to `DualMutex`/`DualBarrier` over
// `SuspendedTask` (not the `BasicSuspendedThread`-based ones `UltSystem`
// gets from the macro) so stackless callers get real, contended-together-
// with-ULTs capability, not just a marker.
impl ult::system::AsyncWorkerSystem for DefaultUltSystem {
    type Mutex<T: Send> = UltDualMutex<Self, T, SuspendedTask<Self>>;
    type Barrier = UltDualBarrier<Self, SuspendedTask<Self>>;
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
