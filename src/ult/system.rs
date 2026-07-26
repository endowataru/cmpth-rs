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
//!         deque:      cmpth::CrossbeamDeque<cmpth::BasicTaskDesc>,
//!         stack_size: 64 * 1024,
//!     }
//! }
//!
//! cmpth::ult_system! {
//!     pub struct DefaultUltUltSystem {
//!         base:       DefaultUltSystem,   // runs on ULTs, not OS threads
//!         context:    cmpth::NativeContext,
//!         deque:      cmpth::CrossbeamDeque<cmpth::BasicTaskDesc>,
//!         stack_size: 64 * 1024,
//!     }
//! }
//! ```

use std::future::Future;

use crate::context::ContextPolicy;
use crate::traits::DelegatorConsumer;
use crate::traits::thread_system::ThreadSystem;
use crate::ult::deque::WorkerDeque;
use crate::ult::desc::{AsyncTaskDesc, SuspendedUlt, StackfulTaskDesc, TaskDescAlloc};
use crate::ult::external_queue::ExternalQueue;
use crate::ult::pool::{DescPool, DynamicPool};
use crate::ult::suspended::UltSuspendedThread;
use crate::ult::worker::{LocalQueue, UltWorker, Worker};

// ---------------------------------------------------------------------------
// Trait hierarchy
//
//   SchedulerSystem           worker / scheduler infrastructure, any task
//   flavor (stackful, stackless, or dual) — never names Ctx/StackAlloc
//       ↑
//   UltSchedulerSystem        adds real-stack-switch capability (Ctx,
//   stack allocation, SuspendedThread) — only implementable when
//   Self::Desc: StackfulTaskDesc
//       ↑
//   UltSystem                 user-facing high-level primitives
// ---------------------------------------------------------------------------

/// Base system interface required by [`UltWorker`] and
/// [`Scheduler`](crate::ult::scheduler::Scheduler), independent of whether
/// tasks are stackful ULTs, stackless `spawn_async` futures, or both.
///
/// Deliberately does **not** name a context-switch policy or stack
/// allocator: a stackless-only system has no real stack to switch into, so
/// requiring one here would force it to name machinery it never uses. See
/// [`UltSchedulerSystem`] for the stackful extension.
pub trait SchedulerSystem: Sized + Send + Sync + 'static {
    /// The threading system this scheduler runs on.
    type Base: ThreadSystem;

    /// Task descriptor type for this system. Every concrete system today
    /// sets this to `BasicTaskDesc`; the associated type exists so
    /// `SuspendedUlt`/`WorkerDeque`/`DescPool`/the worker traits never
    /// hardcode a concrete descriptor, in preparation for narrower
    /// stackful-only/stackless-only descriptor types later.
    type Desc: TaskDescAlloc;

    /// Work-stealing deque implementation.
    type Deque: WorkerDeque<Self::Desc>;

    /// Descriptor pool implementation for this system, used by the stackful
    /// `spawn` path (fixed-size ULT stacks, `STACK_SIZE` on
    /// [`UltSchedulerSystem`]).
    type Pool: DescPool<Self::Desc>;

    /// Descriptor pool used by `spawn_async` (variable-size Future storage,
    /// capped at `ASYNC_POOL_SIZE`) — deliberately a *separate* pool/type
    /// from [`Pool`](Self::Pool) rather than sharing it: a dual system needs
    /// both a large fixed-size ULT-stack pool and a small fixed-size
    /// async-task pool live at once, and the two have nothing in common
    /// beyond both implementing [`DescPool`]. Required on every
    /// `SchedulerSystem` (even stackful-only ones that never call
    /// `spawn_async`) purely so the type is nameable uniformly; an unused
    /// `AsyncPool` costs nothing beyond declaring it, matching the existing
    /// precedent of stackless-only systems declaring an unused [`Pool`].
    type AsyncPool: DescPool<Self::Desc>;

    /// Fixed slot size for [`AsyncPool`](Self::AsyncPool). Futures that fit
    /// are served from its free list; larger ones fall back to a one-off
    /// allocation (see [`DescPool::alloc`]).
    const ASYNC_POOL_SIZE: usize;

    /// Frame-only pool backing [`crate::ult::thread::recurse`] — the same
    /// fixed-slot free-list mechanism [`Pool`](Self::Pool)/[`AsyncPool`](Self::AsyncPool)
    /// use ([`crate::ult::pool::DynamicPool`]'s doc comment has the full
    /// layering), just without any `TaskDescAlloc`-specific construction:
    /// no descriptor, no join-protocol state, nothing schedulable — a
    /// recursion frame is never pushed to a deque, stolen, or joined by
    /// anyone but its immediate caller.
    type RecursionPool: DynamicPool;

    /// Current-worker lookup policy.
    type Lookup: crate::ult::lookup::CurrentLookup<Self>;

    /// Queue for continuations pushed by external (non-worker) OS threads.
    type ExternalQueue: ExternalQueue<Self>;

    /// The one TLS slot that stores the worker pointer for this scheduler
    /// level.  Each concrete system gets its own `static`, anchored by the
    /// function body of this implementation.
    fn worker_tls() -> &'static <Self::Base as ThreadSystem>::ThreadSpecific<UltWorker<Self>>;

    /// Run one continuation popped off a deque/root/external-queue.
    ///
    /// Required, with **no default**: the correct body depends entirely on
    /// which task flavors this system supports, and a base `SchedulerSystem`
    /// can't know that. Every concrete system supplies this directly by
    /// calling exactly one of the free functions in `worker.rs`:
    ///
    /// - stackful-only: [`crate::ult::worker::execute_stackful`] (always a
    ///   real context switch — `Self::Desc` need not even implement
    ///   `AsyncTaskDesc`, so there is no tag to check).
    /// - dual: [`crate::ult::worker::execute_dual`] (today's poll_fn check).
    /// - stackless-only (added when that flavor lands): always polls.
    ///
    /// This is ordinary trait-method overriding, not specialization: each
    /// concrete marker struct gets exactly one `impl SchedulerSystem for
    /// Self` block, so the compiler picks the right body statically.
    fn execute(wk: &UltWorker<Self>, cont: SuspendedUlt<Self::Desc>);

    /// Free a finished task's descriptor once its `JoinHandle` is done with
    /// it (`take_result`/`Drop`, both in `thread.rs`).
    ///
    /// Required, with **no default**, for the same reason as [`execute`]:
    /// stackful-only frees always go through the pool
    /// ([`crate::ult::worker::free_finished_desc_stackful`]); stackless-only
    /// descriptors always bypass the pool (variable-size `spawn_async`
    /// allocations — [`crate::ult::worker::free_finished_desc_async`]); dual
    /// systems check `poll_fn` first
    /// ([`crate::ult::worker::free_finished_desc_dual`]).
    ///
    /// [`execute`]: Self::execute
    fn free_finished_desc(wk: &UltWorker<Self>, desc: *mut Self::Desc);
}

/// Extends [`SchedulerSystem`] with real-stack context-switch machinery:
/// context-switch policy, stack allocator, stack size, and the
/// stackful parked-continuation type.
///
/// Only implementable when `Self::Desc: StackfulTaskDesc` — a stackless-only
/// descriptor type (no saved context to switch into) cannot satisfy this
/// trait at all, which is exactly the point: it makes "this system can run
/// real ULTs" a checkable, compile-time fact instead of a convention.
pub trait UltSchedulerSystem: SchedulerSystem
where
    Self::Desc: StackfulTaskDesc,
{
    /// Context-switch implementation.
    type Ctx: ContextPolicy;

    /// Stack allocation policy for this system.
    type StackAlloc: crate::ult::stack::StackAlloc;

    /// Stack size for each ULT (in bytes).
    const STACK_SIZE: usize;

    /// Parked-continuation type for this system.
    type SuspendedThread: UltSuspendedThread<UltSchedulerSystem = Self>;

    /// Resolve what a suspending/exiting ULT switches into when its local
    /// deque is empty: the worker's own root (scheduler-loop) continuation.
    ///
    /// Default: [`crate::ult::worker::pop_or_root_stackful`] — correct
    /// whenever `Self::Desc` isn't also `AsyncTaskDesc` (stackful-only),
    /// since every popped item is then guaranteed to be a real, switchable
    /// continuation. Dual configs override with
    /// [`crate::ult::worker::pop_or_root_dual`], which requeues an async
    /// task popped off the top instead of trying to switch into it.
    fn pop_or_root(wk: &UltWorker<Self>) -> SuspendedUlt<Self::Desc> {
        crate::ult::worker::pop_or_root_stackful(wk)
    }
}

// `UltSystem`/`AsyncWorkerSystem` now live in `crate::traits::ult_system` —
// re-exported below for callers that still spell out `ult::system::UltSystem`.
pub use crate::traits::ult_system::{AsyncWorkerSystem, UltSystem};

/// `S::spawn(...)`/`S::recurse(...)` associated-function form of
/// [`crate::ult::thread::spawn_async`]/[`crate::ult::thread::recurse`].
///
/// Blanket-implemented for every [`SchedulerSystem`] whose `Desc` supports
/// async tasks, exactly like [`ThreadSystem`]'s blanket derivation from
/// [`UltSystem`] above — no concrete system ever writes `impl
/// AsyncTaskSystem for ...` by hand.
pub trait AsyncTaskSystem: SchedulerSystem
where
    Self::Desc: AsyncTaskDesc,
{
    /// See [`crate::ult::thread::spawn_async`].
    fn spawn<T, F, Mk>(mk: Mk) -> crate::ult::thread::SpawnAction<Self, T>
    where
        F: Future<Output = T> + Send + 'static,
        Mk: FnOnce() -> F + Send + 'static,
        T: Send + 'static,
    {
        crate::ult::thread::spawn_async::<Self, T, F, Mk>(mk)
    }

    /// See [`crate::ult::thread::recurse`].
    fn recurse<F, Mk>(mk: Mk) -> crate::ult::thread::RecursionFrame<Self, F>
    where
        F: Future,
        Mk: FnOnce() -> F,
    {
        crate::ult::thread::recurse::<Self, F, Mk>(mk)
    }

    /// See [`crate::ult::scheduler::run_async`]. Named `run_async`, not
    /// `run`, so it never collides with [`UltSystem::run`](crate::UltSystem::run)
    /// on a dual system that implements both.
    fn run_async<F>(num_workers: usize, root: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        crate::ult::scheduler::run_async::<Self, F>(num_workers, root)
    }

    /// Yield once to the executor from inside an async task on this system
    /// — see [`crate::future::yield_now`], which this just forwards to.
    /// Not generic over `Self` at all (unlike `spawn`/`recurse`/`run_async`):
    /// provided here purely so generic code bounded by `S: AsyncTaskSystem`
    /// can write `S::yield_now().await` instead of a separate
    /// `cmpth::future` import, matching this trait's other methods.
    ///
    /// Deliberately shares its name with
    /// [`ThreadSystem::yield_now`](crate::ThreadSystem::yield_now) (the
    /// stackful, synchronous, whole-ULT-suspending version) rather than being
    /// renamed to dodge the collision — on a dual system implementing both
    /// traits, calling `Concrete::yield_now()` is ambiguous by design (same
    /// resolution as `spawn` above) and must be disambiguated with
    /// `<Concrete as AsyncTaskSystem>::yield_now()` /
    /// `<Concrete as ThreadSystem>::yield_now()`; a generic caller bounded by
    /// only one of the two traits never sees the ambiguity.
    fn yield_now() -> impl Future<Output = ()> {
        crate::future::yield_now()
    }
}

impl<S: SchedulerSystem> AsyncTaskSystem for S where S::Desc: AsyncTaskDesc {}

// ---------------------------------------------------------------------------
// Blanket ThreadSystem implementation for every UltSystem
// ---------------------------------------------------------------------------

impl<S: UltSystem + UltSchedulerSystem> ThreadSystem for S
where
    S::Desc: StackfulTaskDesc + crate::ult::desc::WakerTaskDesc,
{
    type Poller = crate::ult::waker::UltPoller<S>;

    fn yield_now() {
        use crate::ult::worker::StackfulWorker;
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
///         deque:      cmpth::CrossbeamDeque<cmpth::BasicTaskDesc>,
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
///         deque:       cmpth::CrossbeamDeque<cmpth::BasicTaskDesc>,
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
    //
    // `deque` must already name the fully-parametrized type (e.g.
    // `cmpth::CrossbeamDeque<cmpth::BasicTaskDesc>`) — a `ty` fragment can't
    // have `<...>` appended to it after the fact inside the macro body, so
    // the descriptor type argument has to be spelled out at the call site.
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

        impl $crate::ult::system::SchedulerSystem for $name {
            type Base  = $base;
            type Desc  = $crate::ult::desc::BasicTaskDesc;
            type Deque = $deque;
            type ExternalQueue   = $crate::ult::external_queue::StealPathQueue<$crate::ult::desc::BasicTaskDesc>;
            type Pool            = $crate::ult::pool::ReturnPool<$crate::ult::desc::BasicTaskDesc, $alloc>;
            // Never actually allocated through: this system never calls
            // spawn_async (no AsyncWorkerSystem impl below). Mirrors
            // ult_async_system!'s unused `Pool` in the other direction.
            type AsyncPool       = $crate::ult::pool::SimplePool<$crate::ult::desc::BasicTaskDesc>;
            const ASYNC_POOL_SIZE: usize = 0;
            // Never actually taken from: this system never calls `recurse`
            // (no AsyncWorkerSystem impl below). Mirrors `AsyncPool` above.
            type RecursionPool   = $crate::ult::pool::ThresholdPool<$crate::ult::pool::BlockPool>;
            type Lookup          = $lookup;

            fn worker_tls()
            -> &'static <$base as $crate::ThreadSystem>::ThreadSpecific<$crate::UltWorker<$name>>
            {
                static A: $crate::TlsAnchor = $crate::TlsAnchor::new();
                $crate::TlsSlot::from_anchor(&A)
            }

            // Stackful-only: always a real context switch, no poll_fn tag
            // check — `execute_stackful`'s whole point is that this bound
            // never needs `AsyncTaskDesc` at all.
            fn execute(
                wk: &$crate::UltWorker<Self>,
                cont: $crate::SuspendedUlt<$crate::ult::desc::BasicTaskDesc>,
            ) {
                $crate::ult::worker::execute_stackful(wk, cont)
            }

            fn free_finished_desc(wk: &$crate::UltWorker<Self>, desc: *mut $crate::ult::desc::BasicTaskDesc) {
                $crate::ult::worker::free_finished_desc_stackful(wk, desc)
            }
        }

        impl $crate::ult::system::UltSchedulerSystem for $name {
            type Ctx   = $ctx;
            type StackAlloc = $alloc;
            const STACK_SIZE: usize = $stack;

            type SuspendedThread = $crate::ult::suspended::BasicSuspendedThread<Self>;
        }

        impl $crate::UltSystem for $name {
            type Mutex<T: Send>  = $crate::ult::sync::DualMutex<Self, T, $crate::ult::suspended::BasicSuspendedThread<Self>>;
            type Barrier         = $crate::ult::sync::DualBarrier<Self, $crate::ult::suspended::BasicSuspendedThread<Self>>;
            type Delegator<C: $crate::DelegatorConsumer<Self>> =
                $crate::ult::sync::McsDelegator<Self, C>;

            fn run<F>(num_workers: usize, root: F)
            where
                F: FnOnce() + Send + 'static,
            {
                $crate::ult::scheduler::run::<Self, F>(num_workers, root)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// ult_async_system! macro
// ---------------------------------------------------------------------------

/// Define a complete **stackless-only** ULT system in one declaration.
///
/// Unlike [`ult_system!`], the generated marker struct implements only
/// [`SchedulerSystem`] — never [`UltSchedulerSystem`], so it never names a
/// context-switch policy or stack allocator, because it has none. Its only
/// entry points are [`crate::ult::scheduler::run_async`] (run) and
/// [`crate::ult::thread::spawn_async`] (spawn); there is no `spawn`, no
/// `block_on`, no `ThreadSystem` impl for it via the `UltSystem` blanket
/// (that blanket requires stackful capability).
///
/// `Worker::execute`'s dispatch is [`crate::ult::worker::execute_stackful`]
/// -shaped in spirit but for polling instead of switching: it always polls,
/// with no `poll_fn`-tag branch, because every task on this system is one.
///
/// ```
/// use cmpth::SuspendedUlt;
/// use cmpth::ult::system::AsyncTaskSystem;
///
/// cmpth::ult_async_system! {
///     struct MyAsyncSystem {
///         base:  cmpth::OsSystem,
///         deque: cmpth::CrossbeamDeque<cmpth::BasicTaskDesc>,
///     }
/// }
///
/// MyAsyncSystem::run_async(2, async {
///     let h = MyAsyncSystem::spawn(|| async { 6 * 7 }).await;
///     assert_eq!(h.await, 42);
/// });
/// ```
#[macro_export]
macro_rules! ult_async_system {
    ($(#[$meta:meta])* $vis:vis struct $name:ident {
        base:  $base:ty,
        deque: $deque:ty $(,)?
    }) => {
        $crate::ult_async_system! {
            $(#[$meta])* $vis struct $name {
                base:            $base,
                deque:           $deque,
                // Sound as the default here specifically: this macro's
                // output never implements UltSchedulerSystem, so it never
                // does a real context switch (see InlineTlsCurrent's doc
                // comment for the hazard that would otherwise apply).
                lookup:          $crate::ult::lookup::InlineTlsCurrent,
                async_pool_size: 512,
            }
        }
    };
    ($(#[$meta:meta])* $vis:vis struct $name:ident {
        base:    $base:ty,
        deque:   $deque:ty,
        lookup:  $lookup:ty $(,)?
    }) => {
        $crate::ult_async_system! {
            $(#[$meta])* $vis struct $name {
                base:            $base,
                deque:           $deque,
                lookup:          $lookup,
                async_pool_size: 512,
            }
        }
    };
    ($(#[$meta:meta])* $vis:vis struct $name:ident {
        base:            $base:ty,
        deque:           $deque:ty,
        lookup:          $lookup:ty,
        async_pool_size: $async_pool_size:expr $(,)?
    }) => {
        $(#[$meta])*
        $vis struct $name;

        impl $crate::ult::system::SchedulerSystem for $name {
            type Base  = $base;
            type Desc  = $crate::ult::desc::BasicTaskDesc;
            type Deque = $deque;
            type ExternalQueue = $crate::ult::external_queue::StealPathQueue<$crate::ult::desc::BasicTaskDesc>;
            // Never actually allocated through: this flavor has no `spawn`,
            // only `spawn_async` (which goes through AsyncPool below).
            // SimplePool is the cheapest DescPool to instantiate for a type
            // that's never used.
            type Pool = $crate::ult::pool::SimplePool<$crate::ult::desc::BasicTaskDesc>;
            const ASYNC_POOL_SIZE: usize = $async_pool_size;
            type AsyncPool = $crate::ult::pool::ReturnPool<$crate::ult::desc::BasicTaskDesc, $crate::ult::stack::AsyncArenaStack>;
            type RecursionPool = $crate::ult::pool::ThresholdPool<$crate::ult::pool::BlockPool>;
            type Lookup = $lookup;

            fn worker_tls()
            -> &'static <$base as $crate::ThreadSystem>::ThreadSpecific<$crate::UltWorker<$name>>
            {
                static A: $crate::TlsAnchor = $crate::TlsAnchor::new();
                $crate::TlsSlot::from_anchor(&A)
            }

            // Stackless-only: always poll, never switch — no poll_fn tag
            // check, because every task on this system is a poll_fn task.
            fn execute(
                wk: &$crate::UltWorker<Self>,
                cont: $crate::SuspendedUlt<$crate::ult::desc::BasicTaskDesc>,
            ) {
                $crate::ult::worker::execute_async(wk, cont)
            }

            fn free_finished_desc(wk: &$crate::UltWorker<Self>, desc: *mut $crate::ult::desc::BasicTaskDesc) {
                $crate::ult::worker::free_finished_desc_async(wk, desc)
            }
        }
    };
}
