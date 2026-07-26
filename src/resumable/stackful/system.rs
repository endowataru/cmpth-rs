//! [`StackfulSchedulerSystem`] — extends
//! [`SchedulerSystem`](crate::resumable::common::system::SchedulerSystem)
//! with real-stack context-switch capability, and the
//! [`ult_system!`](crate::ult_system) macro that generates a stackful-only
//! system (a full [`ThreadSystem`] impl plus this trait).
//!
//! # Nesting
//!
//! Every `ult_system!`-generated system is a full `ThreadSystem`, so setting
//! `type Base = DualTaskSystem` in a second one stacks one ULT
//! scheduler on top of another without any extra boilerplate:
//!
//! ```ignore
//! cmpth::ult_system! {
//!     pub struct DualTaskSystem {
//!         base:       cmpth::OsSystem,
//!         context:    cmpth::NativeContext,
//!         deque:      cmpth::CrossbeamDeque<cmpth::BasicTaskDesc>,
//!         stack_size: 64 * 1024,
//!     }
//! }
//!
//! cmpth::ult_system! {
//!     pub struct NestedDualTaskSystem {
//!         base:       DualTaskSystem,   // runs on ULTs, not OS threads
//!         context:    cmpth::NativeContext,
//!         deque:      cmpth::CrossbeamDeque<cmpth::BasicTaskDesc>,
//!         stack_size: 64 * 1024,
//!     }
//! }
//! ```

use crate::context::ContextPolicy;
use crate::traits::thread_system::ThreadSystem;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::desc::SuspendedUlt;
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::stackful::suspended::UltSuspendedThread;
use crate::resumable::common::worker::UltWorker;

// `StackfulTaskSystem` now lives in `crate::traits::system` — re-exported
// below for callers that still spell out
// `resumable::stackful::system::StackfulTaskSystem`.
pub use crate::traits::system::StackfulTaskSystem;

/// Extends [`SchedulerSystem`] with real-stack context-switch machinery:
/// context-switch policy, stack allocator, stack size, and the
/// stackful parked-continuation type.
///
/// Only implementable when `Self::Desc: StackfulTaskDesc` — a stackless-only
/// descriptor type (no saved context to switch into) cannot satisfy this
/// trait at all, which is exactly the point: it makes "this system can run
/// real ULTs" a checkable, compile-time fact instead of a convention.
pub trait StackfulSchedulerSystem: SchedulerSystem
where
    Self::Desc: StackfulTaskDesc,
{
    /// Context-switch implementation.
    type Ctx: ContextPolicy;

    /// Stack allocation policy for this system.
    type StackAlloc: crate::resumable::common::stack::StackAlloc;

    /// Stack size for each ULT (in bytes).
    const STACK_SIZE: usize;

    /// Parked-continuation type for this system.
    type SuspendedThread: UltSuspendedThread<StackfulSchedulerSystem = Self>;

    /// Resolve what a suspending/exiting ULT switches into when its local
    /// deque is empty: the worker's own root (scheduler-loop) continuation.
    ///
    /// Default: [`crate::resumable::stackful::worker::pop_or_root_stackful`] — correct
    /// whenever `Self::Desc` isn't also `AsyncTaskDesc` (stackful-only),
    /// since every popped item is then guaranteed to be a real, switchable
    /// continuation. Dual configs override with
    /// [`crate::resumable::dual::worker::pop_or_root_dual`], which requeues an async
    /// task popped off the top instead of trying to switch into it.
    fn pop_or_root(wk: &UltWorker<Self>) -> SuspendedUlt<Self::Desc> {
        crate::resumable::stackful::worker::pop_or_root_stackful(wk)
    }
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
    S::Desc: StackfulTaskDesc + crate::resumable::common::desc::WakerTaskDesc,
{
    fn run<F, R>(num_workers: usize, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        crate::resumable::stackful::scheduler::run_with_result::<S, F, R>(num_workers, f)
    }

    fn parallel_call<Fa, Fb, Ra, Rb>(a: Fa, b: Fb) -> (Ra, Rb)
    where
        Fa: FnOnce() -> Ra + Send + 'static,
        Fb: FnOnce() -> Rb + Send + 'static,
        Ra: Send + 'static,
        Rb: Send + 'static,
    {
        let h = <S as ThreadSystem>::spawn(a);
        let rb = b();
        (crate::traits::thread_system::JoinHandleLike::join(h), rb)
    }
}

/// Empty bundle: `ThreadSystem` is implemented directly (by the
/// `ult_system!` macro or by hand); `ScopedStackfulTaskSystem` is
/// blanket-derived from it just above. This impl just ties the two
/// together as one bound.
impl<S: crate::traits::scoped::ScopedStackfulTaskSystem + ThreadSystem> crate::traits::system::StackfulTaskSystem
    for S
{
}

// ---------------------------------------------------------------------------
// ult_system! macro
// ---------------------------------------------------------------------------

/// Define a complete ULT system in one declaration.
///
/// Generates a marker struct, a [`SchedulerSystem`]/[`StackfulSchedulerSystem`]
/// implementation, and a full [`ThreadSystem`] implementation (with the one
/// `static` TLS slot for the per-worker pointer) — everything needed to
/// `spawn`/`join`/`parallel_call`/`run` (the latter two via the blanket
/// `ScopedStackfulTaskSystem` impl above).
///
/// ```
/// use cmpth::{ThreadSystem, ScopedStackfulTaskSystem, JoinHandleLike};
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
/// # use cmpth::ScopedStackfulTaskSystem;
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
                stack_alloc: $crate::resumable::common::stack::HeapStack,
                lookup:      $crate::resumable::common::lookup::TlsCurrent,
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

        impl $crate::resumable::common::system::SchedulerSystem for $name {
            type Base  = $base;
            type Desc  = $crate::resumable::common::desc::BasicTaskDesc;
            type Deque = $deque;
            type ExternalQueue   = $crate::resumable::common::external_queue::StealPathQueue<$crate::resumable::common::desc::BasicTaskDesc>;
            type Pool            = $crate::resumable::common::pool::ReturnPool<$crate::resumable::common::desc::BasicTaskDesc, $alloc>;
            // Never actually allocated through: nothing calls spawn_async on
            // a system built by this macro (StacklessTaskSystem's blanket
            // impl still applies, since BasicTaskDesc: AsyncTaskDesc
            // unconditionally, but the capability just goes unused here).
            // Mirrors ult_async_system!'s unused `Pool` in the other
            // direction.
            type AsyncPool       = $crate::resumable::common::pool::SimplePool<$crate::resumable::common::desc::BasicTaskDesc>;
            const ASYNC_POOL_SIZE: usize = 0;
            // Never actually taken from: nothing calls `recurse` on a
            // system built by this macro either. Mirrors `AsyncPool` above.
            type RecursionPool   = $crate::resumable::common::pool::ThresholdPool<$crate::resumable::common::pool::BlockPool>;
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
                cont: $crate::SuspendedUlt<$crate::resumable::common::desc::BasicTaskDesc>,
            ) {
                $crate::resumable::stackful::worker::execute_stackful(wk, cont)
            }

            fn free_finished_desc(wk: &$crate::UltWorker<Self>, desc: *mut $crate::resumable::common::desc::BasicTaskDesc) {
                $crate::resumable::stackful::worker::free_finished_desc_stackful(wk, desc)
            }
        }

        impl $crate::resumable::stackful::system::StackfulSchedulerSystem for $name {
            type Ctx   = $ctx;
            type StackAlloc = $alloc;
            const STACK_SIZE: usize = $stack;

            type SuspendedThread = $crate::resumable::stackful::suspended::BasicSuspendedThread<Self>;
        }

        impl $crate::ThreadSystem for $name {
            type Poller = $crate::resumable::stackful::waker::UltPoller<$name>;

            fn yield_now() {
                use $crate::resumable::common::worker::Worker;
                use $crate::resumable::stackful::worker::StackfulWorker;
                match $crate::UltWorker::<$name>::current() {
                    Some(wk) => { wk.yield_now(); }
                    None => <$base as $crate::ThreadSystem>::yield_now(),
                }
            }

            type JoinHandle<T: Send + 'static> = $crate::resumable::common::thread::JoinHandle<$name, T>;

            fn spawn<T, F>(f: F) -> $crate::resumable::common::thread::JoinHandle<$name, T>
            where
                F: FnOnce() -> T + Send + 'static,
                T: Send + 'static,
            {
                $crate::resumable::stackful::thread::spawn::<$name, T, F>(f)
            }

            type Mutex<T: Send>  = $crate::resumable::common::sync::DualMutex<Self, T, $crate::resumable::stackful::suspended::BasicSuspendedThread<Self>>;
            type Barrier         = $crate::resumable::common::sync::DualBarrier<Self, $crate::resumable::stackful::suspended::BasicSuspendedThread<Self>>;
            type SuspendedThread = $crate::resumable::stackful::suspended::BasicSuspendedThread<Self>;
            type Delegator<C: $crate::DelegatorConsumer<Self>> =
                $crate::resumable::stackful::sync::McsDelegator<Self, C>;
            type ThreadSpecific<T: 'static> = $crate::resumable::stackful::tls::UltTls<$name, T>;
        }
    };
}
