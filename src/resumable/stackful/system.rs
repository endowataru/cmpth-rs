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

use crate::traits::stackful::{ContextPolicy, ThreadSystem};
use crate::resumable::common::deque::WorkerDeque;
use crate::resumable::common::lookup::CurrentLookup;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::desc::SuspendedTaskToken;
use crate::resumable::common::stack::StackAlloc;
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::stackful::suspended::StackfulOnlyResumableCore;
use crate::resumable::common::worker::UltWorker;

// `StackfulTaskSystem` now lives in `crate::traits::stackful` — re-exported
// below for callers that still spell out
// `resumable::stackful::system::StackfulTaskSystem`.
pub use crate::traits::stackful::StackfulTaskSystem;

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
    type SuspendedThread: StackfulOnlyResumableCore<StackfulSchedulerSystem = Self>;

    /// Resolve what a suspending/exiting ULT switches into when its local
    /// deque is empty: the worker's own root (scheduler-loop) continuation.
    ///
    /// Default: [`crate::resumable::stackful::worker::pop_or_root_stackful`] — correct
    /// whenever `Self::Desc` isn't also `AsyncTaskDesc` (stackful-only),
    /// since every popped item is then guaranteed to be a real, switchable
    /// continuation. Dual configs override with
    /// [`crate::resumable::dual::worker::pop_or_root_dual`], which requeues an async
    /// task popped off the top instead of trying to switch into it.
    fn pop_or_root(wk: &UltWorker<Self>) -> SuspendedTaskToken<Self::Desc> {
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
    S::Desc: StackfulTaskDesc,
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
        (crate::traits::stackful::JoinHandleLike::join(h), rb)
    }
}

/// Empty bundle: `ThreadSystem` is implemented directly (via `UltIdentity`'s
/// blanket impl or by hand); `ScopedStackfulTaskSystem` is blanket-derived
/// from it just above. This impl just ties the two together as one bound.
impl<S: crate::traits::scoped::ScopedStackfulTaskSystem + ThreadSystem> crate::traits::stackful::StackfulTaskSystem
    for S
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
/// (`impl<S: SchedulerSystem> CurrentLookup<S> for TlsCurrent`) — proving
/// this trait's `SchedulerSystem` blanket impl would require re-entering
/// that very impl (`error[E0275]: overflow evaluating the requirement`,
/// reproduced directly). `worker_tls_anchor`'s `static` has the same
/// requirement for an unrelated reason: a `static` declared inside a
/// generic function body is one shared instance across every
/// monomorphization, not one per instantiation (also verified directly) —
/// every implementor needs its own `static`, anchored by its own function
/// body.
///
/// ```
/// use cmpth::{ThreadSystem, ScopedStackfulTaskSystem, JoinHandleLike};
///
/// pub struct MySystem;
///
/// impl cmpth::UltIdentity for MySystem {
///     type Base = cmpth::OsSystem;
///     type Ctx = cmpth::NativeContext;
///     type Desc = cmpth::StackfulOnlyTaskDesc;
///     type Deque = cmpth::CrossbeamDeque<cmpth::StackfulOnlyTaskDesc>;
///     type Alloc = cmpth::HeapStack;
///     type Lookup = cmpth::TlsCurrent;
///
///     fn worker_tls_anchor() -> &'static <cmpth::OsSystem as ThreadSystem>::ThreadSpecific<cmpth::UltWorker<Self>> {
///         static A: cmpth::TlsAnchor = cmpth::TlsAnchor::new();
///         cmpth::TlsSlot::from_anchor(&A)
///     }
/// }
///
/// MySystem::run(2, || {
///     let h = MySystem::spawn(|| 42);
///     assert_eq!(JoinHandleLike::join(h), 42);
/// });
/// ```
///
/// `STACK_SIZE` defaults to 64 KiB; override it like any other associated
/// const.
pub trait UltIdentity: Sized + Send + Sync + 'static {
    /// The threading system this scheduler runs on.
    type Base: ThreadSystem;

    /// Context-switch implementation.
    type Ctx: ContextPolicy;

    /// Task descriptor type. Most implementors want
    /// [`StackfulOnlyTaskDesc`](crate::resumable::stackful::desc::StackfulOnlyTaskDesc)
    /// (no unused `poll_fn` slot); a system that also needs `spawn_async`/
    /// dual capability on the same tasks wants
    /// [`DualTaskDesc`](crate::resumable::dual::desc::DualTaskDesc)
    /// instead.
    type Desc: crate::resumable::common::desc::TaskDescAlloc + StackfulTaskDesc;

    /// Work-stealing deque implementation.
    type Deque: WorkerDeque<Self::Desc>;

    /// Stack allocation policy.
    type Alloc: StackAlloc;

    /// Stack size for each ULT (in bytes).
    const STACK_SIZE: usize = 64 * 1024;

    /// Current-worker lookup policy.
    type Lookup: CurrentLookup<Self>
    where
        Self: SchedulerSystem;

    /// The per-system TLS anchor backing [`SchedulerSystem::worker_tls`].
    fn worker_tls_anchor() -> &'static <<Self as UltIdentity>::Base as ThreadSystem>::ThreadSpecific<UltWorker<Self>>
    where
        Self: SchedulerSystem;
}

impl<M: UltIdentity> SchedulerSystem for M {
    type Base  = M::Base;
    type Desc  = M::Desc;
    type Deque = M::Deque;
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
    type Lookup = <M as UltIdentity>::Lookup;

    fn worker_tls() -> &'static <M::Base as ThreadSystem>::ThreadSpecific<UltWorker<Self>> {
        <M as UltIdentity>::worker_tls_anchor()
    }

    // Stackful-only: always a real context switch, no poll_fn tag check —
    // `execute_stackful`'s whole point is that this bound never needs
    // `AsyncTaskDesc` at all.
    fn execute(wk: &UltWorker<Self>, cont: SuspendedTaskToken<M::Desc>) {
        crate::resumable::stackful::worker::execute_stackful(wk, cont)
    }

    fn free_finished_desc(wk: &UltWorker<Self>, desc: *mut M::Desc) {
        crate::resumable::stackful::worker::free_finished_desc_stackful(wk, desc)
    }
}

impl<M: UltIdentity> StackfulSchedulerSystem for M
where
    <M as SchedulerSystem>::Desc: StackfulTaskDesc,
{
    type Ctx = M::Ctx;
    type StackAlloc = M::Alloc;
    const STACK_SIZE: usize = <M as UltIdentity>::STACK_SIZE;

    type SuspendedThread = crate::resumable::stackful::suspended::BasicStackfulOnlyResumable<Self>;
}

impl<M: UltIdentity + StackfulSchedulerSystem> ThreadSystem for M
where
    <M as SchedulerSystem>::Desc: StackfulTaskDesc,
{
    type Poller = crate::resumable::stackful::waker::ResumablePoller<Self>;

    fn yield_now() {
        use crate::resumable::common::worker::Worker;
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

    type Mutex<T: Send>  = crate::resumable::common::sync::DualMutex<Self, T, crate::resumable::stackful::suspended::BasicStackfulOnlyResumable<Self>>;
    type Barrier         = crate::resumable::common::sync::DualBarrier<Self, crate::resumable::stackful::suspended::BasicStackfulOnlyResumable<Self>>;
    type SuspendedThread = crate::resumable::stackful::suspended::BasicStackfulOnlyResumable<Self>;
    type Delegator<C: crate::traits::stackful::DelegatorConsumer<Self>> =
        crate::resumable::stackful::sync::McsDelegator<Self, C>;
    type ThreadSpecific<T: 'static> = crate::resumable::stackful::tls::UltTls<Self, T>;
}
