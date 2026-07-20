//! [`UltSystem`] trait and the [`ult_system!`](crate::ult_system) macro.
//!
//! # Design
//!
//! [`UltSystem`] is the configuration trait for a ULT scheduler.  It bundles
//! the base system, context implementation, deque, and constants into a marker
//! struct via associated types.  The blanket
//! `impl<S: UltSystem> ThreadSystem for S` then automatically derives the full
//! [`ThreadSystem`] implementation from those choices, so there is no need to write
//! `impl ThreadSystem for …` by hand.
//!
//! This replaces the old `UltPolicy` + `Ult<P>` pair: instead of a generic
//! struct parameterised by a policy, we have a *trait* with associated types
//! implemented by a concrete marker struct (`DefaultUltSystem`, etc.).
//!
//! # Nesting
//!
//! Because the blanket gives every `UltSystem` a full `ThreadSystem`, setting
//! `type Base = DefaultUltSystem` in a second `UltSystem` stacks one ULT
//! scheduler on top of another without any extra boilerplate:
//!
//! ```ignore
//! cmpth::ult_system! {
//!     pub struct DefaultUltSystem {
//!         base:       cmpth::OsSystem,
//!         context:    cmpth::NativeContext,
//!         deque:      cmpth::CrossbeamDeque,
//!         stack_size: 64 * 1024,
//!     }
//! }
//!
//! cmpth::ult_system! {
//!     pub struct DefaultUltUltSystem {
//!         base:       DefaultUltSystem,   // runs on ULTs, not OS threads
//!         context:    cmpth::NativeContext,
//!         deque:      cmpth::CrossbeamDeque,
//!         stack_size: 64 * 1024,
//!     }
//! }
//! ```

use crate::context::ContextPolicy;
use crate::traits::DelegatorConsumer;
use crate::traits::thread_system::ThreadSystem;
use crate::ult::deque::WorkerDeque;
use crate::ult::external_queue::ExternalQueue;
use crate::ult::pool::DescPool;
use crate::ult::suspended::UltSuspendedThread;
use crate::ult::worker::{LocalQueue, UltWorker, Worker};

// ---------------------------------------------------------------------------
// Trait hierarchy
//
//   UltContextSystem          stack memory type (what UltDesc needs)
//       ↑
//   UltSchedulerSystem        worker / scheduler infrastructure
//       ↑
//   UltSystem                 user-facing high-level primitives
// ---------------------------------------------------------------------------

/// Minimal system interface required by [`BasicTaskDesc`](crate::ult::desc::BasicTaskDesc)
/// and the context-switch shims.
///
/// Carries only the stack-allocation policy; everything else lives at a higher
/// level.  Deliberately kept narrow so future generic `UltDesc<C>` can avoid
/// pulling in mutex/delegator types.
pub trait UltContextSystem: Sized + Send + Sync + 'static {
    /// Stack allocation policy for this system.
    type StackAlloc: crate::ult::stack::StackAlloc;
}

/// System interface required by [`UltWorker`] and
/// [`Scheduler`](crate::ult::scheduler::Scheduler).
///
/// Extends [`UltContextSystem`] with the context-switch, deque, pool, lookup,
/// and external-queue machinery.  Sync primitives (mutex, barrier, delegator)
/// are NOT included here; they live in [`UltSystem`].
pub trait UltSchedulerSystem: UltContextSystem {
    /// The threading system this scheduler runs on.
    type Base: ThreadSystem;

    /// Context-switch implementation.
    type Ctx: ContextPolicy;

    /// Work-stealing deque implementation.
    type Deque: WorkerDeque;

    /// Stack size for each ULT (in bytes).
    const STACK_SIZE: usize;

    /// Descriptor pool implementation for this system.
    type Pool: DescPool;

    /// Current-worker lookup policy.
    type Lookup: crate::ult::lookup::CurrentLookup<Self>;

    /// Parked-continuation type for this system.
    type SuspendedThread: UltSuspendedThread<UltSystem = Self>;

    /// Queue for continuations pushed by external (non-worker) OS threads.
    type ExternalQueue: ExternalQueue<Self>;

    /// The one TLS slot that stores the worker pointer for this scheduler
    /// level.  Each concrete system gets its own `static`, anchored by the
    /// function body of this implementation.
    fn worker_tls() -> &'static <Self::Base as ThreadSystem>::ThreadSpecific<UltWorker<Self>>;
}

// `UltSystem`/`AsyncWorkerSystem` now live in `crate::traits::ult_system` —
// re-exported below for callers that still spell out `ult::system::UltSystem`.
pub use crate::traits::ult_system::{AsyncWorkerSystem, UltSystem};

// ---------------------------------------------------------------------------
// Blanket ThreadSystem implementation for every UltSystem
// ---------------------------------------------------------------------------

impl<S: UltSystem> ThreadSystem for S {
    type Poller = crate::ult::waker::UltPoller<S>;

    fn yield_now() {
        match UltWorker::<S>::current() {
            Some(wk) => { wk.yield_now(); }
            None => S::Base::yield_now(),
        }
    }

    type JoinHandle<T: Send + 'static> = crate::ult::thread::JoinHandle<S, T>;

    fn spawn<T, F>(f: F) -> crate::ult::thread::JoinHandle<S, T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        crate::ult::thread::spawn::<S, T, F>(f)
    }

    type Mutex<T: Send> = <S as UltSystem>::Mutex<T>;
    type Barrier = <S as UltSystem>::Barrier;
    type SuspendedThread = S::SuspendedThread;
    type Delegator<C: DelegatorConsumer<Self>> = <S as UltSystem>::Delegator<C>;
    type ThreadSpecific<T: 'static> = crate::ult::tls::UltTls<S, T>;

    fn num_workers() -> usize {
        match UltWorker::<S>::current() {
            Some(wk) => wk.num_workers(),
            None => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// ult_system! macro
// ---------------------------------------------------------------------------

/// Define a complete ULT system in one declaration.
///
/// Generates a marker struct and a `UltSystem` implementation that includes
/// the one `static` TLS slot for the per-worker pointer.  The `ThreadSystem`
/// implementation is provided automatically by the blanket.
///
/// ```
/// use cmpth::{ThreadSystem, UltSystem, JoinHandleLike};
///
/// cmpth::ult_system! {
///     struct MySystem {
///         base:       cmpth::OsSystem,
///         context:    cmpth::NativeContext,
///         deque:      cmpth::CrossbeamDeque,
///         stack_size: 64 * 1024,
///     }
/// }
///
/// MySystem::run(2, || {
///     let h = MySystem::spawn(|| 42);
///     assert_eq!(JoinHandleLike::join(h), 42);
/// });
/// ```
///
/// The full form adds the stack-allocation and worker-lookup policies
/// (defaults: `HeapStack`, `TlsCurrent`):
///
/// ```
/// cmpth::ult_system! {
///     struct GuardedSystem {
///         base:        cmpth::OsSystem,
///         context:     cmpth::NativeContext,
///         deque:       cmpth::CrossbeamDeque,
///         stack_size:  64 * 1024,
///         stack_alloc: cmpth::ArenaStack,  // guard pages, sp lookup support
///         lookup:      cmpth::SpCurrent,   // worker from the stack pointer
///     }
/// }
/// # use cmpth::UltSystem;
/// # GuardedSystem::run(1, || {});
/// ```
#[macro_export]
macro_rules! ult_system {
    // Short form: heap stacks + TLS lookup (the classic configuration).
    ($(#[$meta:meta])* $vis:vis struct $name:ident {
        base:       $base:ty,
        context:    $ctx:ty,
        deque:      $deque:ty,
        stack_size: $stack:expr $(,)?
    }) => {
        $crate::ult_system! {
            $(#[$meta])* $vis struct $name {
                base:        $base,
                context:     $ctx,
                deque:       $deque,
                stack_size:  $stack,
                stack_alloc: $crate::ult::stack::HeapStack,
                lookup:      $crate::ult::lookup::TlsCurrent,
            }
        }
    };
    // Full form: explicit stack-allocation and worker-lookup policies.
    ($(#[$meta:meta])* $vis:vis struct $name:ident {
        base:        $base:ty,
        context:     $ctx:ty,
        deque:       $deque:ty,
        stack_size:  $stack:expr,
        stack_alloc: $alloc:ty,
        lookup:      $lookup:ty $(,)?
    }) => {
        $(#[$meta])*
        $vis struct $name;

        impl $crate::ult::system::UltContextSystem for $name {
            type StackAlloc = $alloc;
        }

        impl $crate::ult::system::UltSchedulerSystem for $name {
            type Base  = $base;
            type Ctx   = $ctx;
            type Deque = $deque;
            const STACK_SIZE: usize = $stack;

            type SuspendedThread = $crate::ult::suspended::BasicSuspendedThread<Self>;
            type ExternalQueue   = $crate::ult::external_queue::StealPathQueue;
            type Pool            = $crate::ult::pool::ReturnPool<$alloc>;
            type Lookup          = $lookup;

            fn worker_tls()
            -> &'static <$base as $crate::ThreadSystem>::ThreadSpecific<$crate::UltWorker<$name>>
            {
                static A: $crate::TlsAnchor = $crate::TlsAnchor::new();
                $crate::TlsSlot::from_anchor(&A)
            }
        }

        impl $crate::UltSystem for $name {
            type Mutex<T: Send>  = $crate::ult::sync::DualMutex<Self, T, $crate::ult::suspended::BasicSuspendedThread<Self>>;
            type Barrier         = $crate::ult::sync::DualBarrier<Self, $crate::ult::suspended::BasicSuspendedThread<Self>>;
            type Delegator<C: $crate::DelegatorConsumer<Self>> =
                $crate::ult::sync::McsDelegator<Self, C>;
        }
    };
}
