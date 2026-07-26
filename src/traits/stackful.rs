//! Stackful (real-ULT, blocking-call) interface: [`ThreadSystem`],
//! [`Delegator`], [`StackfulMutex`]/[`StackfulBarrier`],
//! [`StackfulResumable`], [`Poller`], [`StackfulTaskSystem`].
//!
//! `use cmpth::traits::stackful::*;` also brings in the shared
//! [`TaskSystem`]/[`Resumable`] (re-exported from [`crate::traits::common`])
//! and [`ScopedStackfulTaskSystem`] (re-exported from
//! [`crate::traits::scoped`]) — everything a caller working purely in the
//! stackful flavor needs in one `use`.

use std::future::Future;
use std::ops::DerefMut;
use std::pin::pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

pub use crate::traits::common::{Resumable, TaskSystem};
pub use crate::traits::scoped::ScopedStackfulTaskSystem;

use crate::traits::common::{BarrierWaitResult, TlsSlot};

/// Threading system interface bundle — swap the entire backend by changing
/// one type parameter.
pub trait ThreadSystem: TaskSystem {
    /// Drives a single `block_on` call; the customisation point for async
    /// integration.  See [`Poller`].
    type Poller: Poller;

    /// Block the current thread/ULT until `future` completes.
    ///
    /// On a ULT system this suspends only the calling ULT; the OS thread
    /// underneath keeps running other tasks.
    ///
    /// ```
    /// use cmpth::ThreadSystem;
    ///
    /// cmpth::default::run(2, || {
    ///     let x = cmpth::DualTaskSystem::block_on(async { 6 * 7 });
    ///     assert_eq!(x, 42);
    /// });
    /// ```
    ///
    /// The default implementation drives the future through [`Self::Poller`].
    fn block_on<F, T>(f: F) -> T
    where
        F: Future<Output = T> + Send,
        T: Send,
    {
        let pol = Self::Poller::new();
        let mut f = pin!(f);
        loop {
            match f.as_mut().poll(&mut pol.context()) {
                Poll::Ready(v) => return v,
                Poll::Pending => pol.wait(),
            }
        }
    }

    /// Yield the current thread/ULT so other tasks can run.
    fn yield_now();

    /// Spawn a new thread or ULT; returns a handle that can be joined.
    type JoinHandle<T: Send + 'static>: JoinHandleLike<T>;
    fn spawn<T, F>(f: F) -> Self::JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static;

    /// Mutex type for this system.
    type Mutex<T: Send>: StackfulMutex<T> + Send + Sync;

    /// Barrier type for this system.
    type Barrier: StackfulBarrier + Send + Sync;

    /// Parked-continuation handle for this system.
    type SuspendedThread: Send + Default;

    /// Delegator type for this system.
    type Delegator<C: DelegatorConsumer<Self>>: Delegator<Self, C>;

    /// Thread-specific storage slot: one `*mut T` per thread (or per ULT) of
    /// this system.  A nested scheduler stores its per-worker pointer here,
    /// which is why a single slot per level is enough — everything else is
    /// reached through the worker pointer.
    type ThreadSpecific<T: 'static>: TlsSlot<T>;
}

/// Common interface for join handles returned by [`ThreadSystem::spawn`].
pub trait JoinHandleLike<T: Send + 'static>: Send {
    fn join(self) -> T;
}

/// Drives a single `block_on` invocation.
///
/// `Poller` is to `block_on` what a wait-slot (`StackfulResumable`) is to `wait_with`: a thin
/// type that encapsulates the system-specific park/wake mechanism, leaving the
/// poll loop itself as a generic default on [`ThreadSystem`].
///
/// Implementations are always stack-local inside `block_on`.  They are `!Send`
/// by convention — bound to the same ULT, not to a specific OS thread.  In
/// cmpth, `!Send` means "bound to the same ULT", not "bound to the same OS
/// thread": work-stealing moves the entire ULT stack atomically, so a `!Send`
/// value is safe across `yield_now` even when the ULT migrates.
///
/// [`Drop`] performs cleanup (e.g. resetting `waker_refs` to `IDLE`).
pub trait Poller {
    /// Initialise for the current thread/ULT.
    fn new() -> Self;

    /// Return a [`Context`] whose [`Waker`] resumes the current thread/ULT.
    fn context<'a>(&'a self) -> Context<'a>;

    /// Suspend until the waker fires.
    ///
    /// For ULT systems this uses `cond_suspend_to_sched` and handles the race
    /// where `wake()` fires between `poll()` returning `Pending` and the actual
    /// suspension (NOTIFIED → re-poll without parking).
    fn wait(&self);
}

/// A waker whose `wake()` is a no-op.  Used for busy-polling fallbacks where
/// the poll loop drives re-polling itself (via `yield_now`). Shared by every
/// [`Poller`] implementation that busy-polls (`OsPoller`, `UltPoller`'s
/// fallback).
pub(crate) fn noop_waker() -> Waker {
    static VTABLE: RawWakerVTable =
        RawWakerVTable::new(|p| RawWaker::new(p, &VTABLE), |_| {}, |_| {}, |_| {});
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

/// User-supplied consumer that the delegator executes on behalf of callers.
///
/// The consumer has exclusive access to a hardware resource (RDMA QP, epoll
/// fd, …). Callers that cannot acquire the delegator lock write their work
/// into a queue node; the consumer ULT drains the queue and calls `progress`
/// to poll for completions.
pub trait DelegatorConsumer<S: ThreadSystem>: Send + 'static {
    /// Per-call work descriptor written into the queue by a delegating caller.
    type Work: Send + Default;

    /// Execute one work item (called by whoever holds the delegator lock).
    /// Returns `(is_done, thread_to_wake)`.
    fn execute(&mut self, work: &mut Self::Work) -> (bool, Option<S::SuspendedThread>);

    /// Poll for completions (e.g. ibv_poll_cq).  Called while `is_active`.
    /// Returns a thread to wake if a completion was found.
    fn progress(&mut self) -> Option<S::SuspendedThread>;

    /// True while there are posted-but-not-completed operations.
    /// When false the consumer ULT suspends instead of spinning.
    fn is_active(&self) -> bool;
}

/// Delegator: a queue-based lock that serialises hardware access and batches
/// work items on behalf of callers that miss the lock.
pub trait Delegator<S: ThreadSystem, C: DelegatorConsumer<S>>:
    Sized + Send + Sync + 'static
{
    /// Start accepting delegations. Implementations may spawn the consumer
    /// ULT eagerly here or lazily on first use — see the implementation for
    /// which, and why (a `Self`-address-stability concern rules eager
    /// spawning out for `Delegator<S, C, Q>`).
    fn start(consumer: C) -> Self;

    /// Stop the consumer ULT (blocks until it exits).
    fn stop(self);

    /// Either execute `imm` inline (if the lock is free) or write work via
    /// `del` into the queue and suspend until the consumer executes it.
    ///
    /// `imm` — called with `&mut Consumer` when the caller wins the lock.
    ///   Returns `(is_done, Option<suspended_thread_to_wake_on_unlock>)`.
    /// `del` — called with `&mut C::Work` when delegating; fills in the work
    ///   and returns a reference to the `SuspendedThread` to park on.
    fn execute_or_delegate<Imm, Del>(&self, imm: Imm, del: Del)
    where
        Imm: FnOnce(&mut C) -> (bool, Option<S::SuspendedThread>),
        Del: FnOnce(&mut C::Work) -> &S::SuspendedThread;
}

/// [`StackfulMutex`]/[`StacklessMutex`](crate::traits::stackless::StacklessMutex) —
/// same-named stackful/stackless mutex traits.
///
/// Same disambiguation pattern as [`StackfulResumable`]/
/// [`StacklessResumable`](crate::traits::stackless::StacklessResumable): both
/// traits define a method literally named `lock`; which one resolves at a
/// call site depends on which trait is `use`d there, not on a `_async`
/// suffix.
///
/// Each carries its own `new`. There is no generic `Condvar` trait: it was
/// never used generically through `S::Mutex`, only via concrete types like
/// `McsCondvar`, so pairing types (`McsMutex`/`McsCondvar`,
/// `OsMutex`/`OsCondvar`, …) expose their condvar as an inherent type with
/// inherent methods instead.
///
/// The interface owns the name here, not the implementation:
/// [`crate::traits::common::DualMutex`] is the trait; the concrete
/// generic-over-N type (`resumable::common::sync::DualMutex`) is
/// re-exported under an alias (`UltDualMutex`) at the crate root to make
/// room, the same pattern already used for `Barrier`/`UltBarrier`.
pub trait StackfulMutex<T: Send>: Sized + Send + Sync {
    type Guard<'a>: DerefMut<Target = T> + 'a
    where
        Self: 'a,
        T: 'a;

    fn new(val: T) -> Self;

    fn lock(&self) -> Self::Guard<'_>;
}

/// Stackful/stackless-flavored barrier `wait`, same disambiguation pattern
/// as [`StackfulMutex`]/[`StacklessMutex`](crate::traits::stackless::StacklessMutex):
/// both traits define a method literally named `wait`, resolved by which
/// trait is `use`d at the call site. Each carries its own `new`, same
/// reasoning as `StackfulMutex`/`StacklessMutex`.
pub trait StackfulBarrier: Sized + Send + Sync {
    fn new(count: usize) -> Self;
    fn wait(&self) -> BarrierWaitResult;
}

/// Stackful (real-context-switch) flavor of parking. These do a real
/// context switch and must only be called from a genuine ULT stack —
/// checked dynamically via `cur_task.is_root` (see
/// `docs/sync-async-unification.md`), not via an explicit capability token.
pub trait StackfulResumable<S>: Resumable<S> {
    /// Suspend the current ULT into this slot. `f` runs after the context
    /// is fully saved (release any spinlock protecting this slot inside
    /// it).
    fn wait_with<F: FnOnce()>(&self, f: F);

    /// Like [`wait_with`](Self::wait_with), but `f` may cancel the
    /// suspension by returning `false`.
    fn wait_with_cond<F: FnOnce() -> bool>(&self, f: F);

    /// Switch directly to the parked continuation, pushing the caller's own
    /// continuation to the local deque. If the slot didn't hold a real
    /// continuation — only possible when `Self` also admits async waiters
    /// (e.g. `SuspendedTask`) — falls back to waking it the
    /// [`Resumable::notify`] way internally, so callers never need to
    /// branch on whether a real switch happened.
    fn enter(&self);

    /// Symmetric handoff: park the current ULT here and switch to `next`.
    /// Same async-target fallback as [`enter`](Self::enter).
    fn swap(&self, next: &Self);
}

/// Everything a "complete" stackful system offers: `spawn`/`join` (via
/// `ThreadSystem`) *and* `run`/`parallel_call` (via
/// `ScopedStackfulTaskSystem`). An empty bundle — no methods of its own —
/// blanket-derived for any `S: ScopedStackfulTaskSystem + ThreadSystem`
/// (see [`resumable::stackful::system`](crate::resumable::stackful::system)
/// for both blankets), never implemented by hand. Kept as its own trait
/// (rather than just writing `S: ScopedStackfulTaskSystem + ThreadSystem`
/// at every call site) since it may grow members of its own later.
///
/// There is no `DualTaskSystem` trait: a concrete system implementing both
/// this and [`StacklessTaskSystem`](crate::traits::stackless::StacklessTaskSystem)
/// simply *is* dual, no separate marker needed.
pub trait StackfulTaskSystem: ScopedStackfulTaskSystem + ThreadSystem {}
