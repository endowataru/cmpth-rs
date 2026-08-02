//! Stackless (`spawn_async`, `.await`-based) interface: [`StacklessMutex`]/
//! [`StacklessBarrier`], [`StacklessResumable`], [`StacklessTaskSystem`].
//!
//! `use cmpth::traits::stackless::*;` also brings in the shared
//! [`TaskSystem`]/[`Resumable`] (re-exported from [`crate::traits::common`])
//! and [`ScopedStacklessTaskSystem`] (re-exported from
//! [`crate::traits::scoped`]) — everything a caller working purely in the
//! stackless flavor needs in one `use`.

use std::future::Future;
use std::ops::DerefMut;
use std::task::{Context, Poll, Waker};

pub use crate::traits::common::{Resumable, TaskSystem};
pub use crate::traits::scoped::ScopedStacklessTaskSystem;

use crate::traits::common::{BarrierWaitResult, TaskDesc, WakeOutcome};

// ---------------------------------------------------------------------------
// WakerTaskDesc
// ---------------------------------------------------------------------------

/// Descriptor operations needed by a task driven via a real
/// [`std::task::Waker`] whose wake state must live on the descriptor itself
/// — currently only `spawn_async` (no stack to anchor the state anywhere
/// else; a real ULT's `block_on` uses the same state machine but keeps it
/// in a call-scoped `Poller` instead, since it has a stack to anchor on).
///
/// Bodyless — pure behavior, same spirit as [`TaskDesc`]. Implement
/// directly for a custom representation, or implement
/// [`WakerTaskDescCore`](crate::resumable::stackless::desc::WakerTaskDescCore)
/// (together with [`TaskDescCore`](crate::resumable::common::desc::TaskDescCore))
/// instead to get this crate's own word-based algorithm for free via a
/// blanket impl.
///
/// The async join-registration methods (`try_register_async_joiner`/
/// `try_register_waker`) live here rather than on the base `TaskDesc`
/// specifically so a descriptor with no async capability at all never gets
/// them — they're only ever called from `JoinHandle::poll`, which itself
/// requires `S::Desc: AsyncTaskDesc: WakerTaskDesc`.
pub trait WakerTaskDesc: TaskDesc {
    fn mark_polling(&self);
    fn mark_idle(&self);
    fn decide_park(&self) -> bool;
    fn park_after_poll(&self) -> bool;

    /// Core wake outcome, shared by the stackful and async wake paths.
    fn try_wake_state(&self) -> WakeOutcome;

    /// True once this waker has been cloned at least once. Sticky — never
    /// clears back to false.
    fn is_ever_shared(&self) -> bool;

    /// First-clone transition: mark this waker as shared, preserving
    /// whatever poll state is currently set.
    fn transition_to_shared(&self);

    /// `JoinHandle::poll`'s fast-path registration: install `joiner` (the
    /// currently-polling task's own descriptor) directly, with no
    /// allocation. Returns `false` if the task turned out to already be
    /// finished (caller should proceed to take the result instead) —
    /// otherwise commits the registration.
    ///
    /// # Safety
    /// `joiner` must be a stable pointer to the currently-polling task's own
    /// descriptor for as long as it might be woken through this slot.
    unsafe fn try_register_async_joiner(&self, joiner: *mut Self) -> bool;

    /// `JoinHandle::poll`'s waker registration: try to install `waker` as
    /// this task's async waiter. Returns `false` if the task turned out to
    /// already be finished (caller should proceed to take the result
    /// instead) — otherwise commits the registration.
    fn try_register_waker(&self, waker: Waker) -> bool;
}

/// [`StackfulMutex`](crate::traits::stackful::StackfulMutex)/`StacklessMutex` —
/// same-named stackful/stackless mutex traits.
///
/// Same disambiguation pattern as
/// [`StackfulResumable`](crate::traits::stackful::StackfulResumable)/
/// [`StacklessResumable`]: both traits define a method literally named
/// `lock`; which one resolves at a call site depends on which trait is
/// `use`d there, not on a `_async` suffix.
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
pub trait StacklessMutex<T: Send>: Sized + Send + Sync {
    type Guard<'a>: DerefMut<Target = T> + 'a
    where
        Self: 'a,
        T: 'a;

    fn new(val: T) -> Self;

    fn lock<'a>(&'a self) -> impl Future<Output = Self::Guard<'a>> + Send
    where
        T: 'a;
}

/// Stackful/stackless-flavored barrier `wait`, same disambiguation pattern
/// as [`StacklessMutex`]/[`StackfulMutex`](crate::traits::stackful::StackfulMutex):
/// both traits define a method literally named `wait`, resolved by which
/// trait is `use`d at the call site. Each carries its own `new`, same
/// reasoning as `StackfulMutex`/`StacklessMutex`.
pub trait StacklessBarrier: Sized + Send + Sync {
    fn new(count: usize) -> Self;
    fn wait<'a>(&'a self) -> impl Future<Output = BarrierWaitResult> + Send + 'a;
}

/// Stackless (poll-based) flavor of parking.
///
/// `enter`/`swap` are deliberately not part of this trait yet: a correct
/// stackless implementation needs the caller to defer itself to the
/// FIFO/steal end of the local deque (`push_local_bottom`) so the target it
/// hands off to isn't overtaken by the caller's own re-queued continuation,
/// and [`StacklessTaskSystem::yield_now`]'s self-wake path doesn't do that
/// today (see its doc comment). Adding them before that's fixed would
/// silently invert the intended priority.
pub trait StacklessResumable<S>: Resumable<S> {
    /// Register `cx`'s waker.
    fn register(&self, cx: &mut Context<'_>);

    /// `.await`-able equivalent of
    /// [`StackfulResumable::wait_with`](crate::traits::stackful::StackfulResumable::wait_with):
    /// registers this task's waker, then runs `f`, then suspends by
    /// returning `Poll::Pending` once, completing once notified.
    ///
    /// `register` must run *before* `f`, mirroring the ordering
    /// `StackfulResumable::wait_with`'s implementations use (store the
    /// parked continuation, *then* publish the link). `f` is what makes
    /// this slot reachable by a concurrent notifier (e.g. publishing this
    /// node into an MCS chain's `next` pointer) — if it ran first, a
    /// notifier could observe the link and call `notify()` while the waker
    /// slot is still empty, losing the wakeup permanently. This was caught
    /// by an hour-long hang in `async_only_flavor` under `cargo test
    /// --all`'s parallel scheduling — the tight, low-worker-count runs used
    /// while developing this never hit the race window.
    ///
    /// Desugared (rather than a native `async fn`) specifically to pin down
    /// `Send` on the returned `Future` — this needs to be usable inside a
    /// `spawn_async`'d task, which requires `F: Future + Send`.
    fn wait_with<F: FnOnce() + Send>(
        &self,
        f: F,
    ) -> impl Future<Output = ()> + Send
    where
        Self: Sync,
    {
        let mut f = Some(f);
        std::future::poll_fn(move |cx| {
            if let Some(f) = f.take() {
                self.register(cx);
                f();
                Poll::Pending
            } else {
                Poll::Ready(())
            }
        })
    }
}

/// `S::spawn(...)`/`S::recurse(...)`/`S::run_async(...)` — the stackless
/// counterpart of [`ThreadSystem`](crate::traits::stackful::ThreadSystem): a
/// capability every [`SchedulerSystem`](crate::resumable::common::system::SchedulerSystem)
/// with an async-capable descriptor gets automatically (see the blanket
/// impl in [`resumable::stackless::system`](crate::resumable::stackless::system)),
/// not something any concrete system implements by hand.
///
/// `: ScopedStacklessTaskSystem` because `parallel_call`'s "nothing
/// outlives this call" constraint is strictly stricter than `spawn`'s (a
/// spawned task may outlive the caller) — anything with `spawn` capability
/// trivially satisfies the more restricted one too (spawn one branch,
/// await the other inline). `run_async` therefore lives on
/// `ScopedStacklessTaskSystem` only, not redeclared here — redeclaring an
/// identical signature in a subtrait would make `S::run_async(...)`
/// ambiguous between the two traits for any `S: StacklessTaskSystem`.
pub trait StacklessTaskSystem: ScopedStacklessTaskSystem {
    /// Handle returned once a spawned task has started: `.await` it again
    /// to get the task's result. (Two-step — `S::spawn(mk).await.await` —
    /// because the first `.await` is what makes the spawn actually happen;
    /// see [`crate::resumable::stackless::thread::spawn_async`].)
    type SpawnHandle<T: Send + 'static>: Future<Output = T> + Send;

    /// Spawn `mk()`'s future as a stackless task — see
    /// [`crate::resumable::stackless::thread::spawn_async`].
    fn spawn<T, F, Mk>(mk: Mk) -> impl Future<Output = Self::SpawnHandle<T>> + Send
    where
        F: Future<Output = T> + Send + 'static,
        Mk: FnOnce() -> F + Send + 'static,
        T: Send + 'static;

    /// Await `mk()`'s future in place through a pooled, non-schedulable
    /// frame instead of `Box::pin` — see
    /// [`crate::resumable::stackless::thread::recurse`].
    ///
    /// Requires `F: Send` here even though the underlying
    /// [`recurse`](crate::resumable::stackless::thread::recurse) free
    /// function doesn't: a recursion frame is never pushed to a deque,
    /// stolen, or awaited by anyone but its immediate caller, so *it*
    /// never needs to cross a thread boundary — but this trait method's
    /// return type is an opaque `impl Future`, and unlike a concretely
    /// named type (which the free function returns, letting Rust's
    /// auto-trait inference see straight through to whether `F` happens to
    /// be `Send`), an opaque return type only gets to claim `Send` if the
    /// trait signature says so unconditionally. Every real caller already
    /// has a `Send` `F` (the recursive call sits inside a `Send`-bounded
    /// `async fn`, same as `spawn`'s callers), so this costs nothing in
    /// practice; callers who genuinely need a non-`Send` `F` can still
    /// call the free function directly on a concrete system.
    fn recurse<F, Mk>(mk: Mk) -> impl Future<Output = F::Output> + Send
    where
        F: Future + Send,
        Mk: FnOnce() -> F;

    /// Returns `Pending` on the first poll (waking itself immediately),
    /// then `Ready` on the next — a single suspend/resume round-trip.
    ///
    /// **Not a fair yield**: the self-wake goes through the same generic
    /// waker path any other wakeup does, which re-queues this task at the
    /// *LIFO* end of its worker's local deque (`push_local_top`) — the
    /// same end `pop_local` pops from next — not the FIFO end
    /// (`push_local_bottom`) that would actually let already-queued
    /// sibling tasks run first. A correct fair yield needs to reach the
    /// scheduler directly (bypassing the waker) to request the FIFO end
    /// specifically; nothing here does that yet. Useful today for "come
    /// back to me after one poll round-trip" (e.g. a busy-poll retry
    /// loop watching a flag another worker sets), not for cooperative
    /// scheduling between tasks that share a worker.
    ///
    /// Deliberately shares its name with
    /// [`ThreadSystem::yield_now`](crate::traits::stackful::ThreadSystem::yield_now)
    /// (the stackful, synchronous, whole-ULT-suspending version, which
    /// *is* fair — it goes through the real scheduler loop) rather than
    /// being renamed to dodge the collision — on a dual system
    /// implementing both traits, calling `Concrete::yield_now()` is
    /// ambiguous by design (same resolution as `spawn` above) and must be
    /// disambiguated with `<Concrete as StacklessTaskSystem>::yield_now()`
    /// / `<Concrete as ThreadSystem>::yield_now()`; a generic caller
    /// bounded by only one of the two traits never sees the ambiguity.
    fn yield_now() -> impl Future<Output = ()> {
        let mut yielded = false;
        std::future::poll_fn(move |cx| {
            if yielded {
                Poll::Ready(())
            } else {
                yielded = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        })
    }
}
