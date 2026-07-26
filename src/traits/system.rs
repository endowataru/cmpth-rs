//! [`StackfulTaskSystem`] and [`StacklessTaskSystem`] — the capability
//! traits a concrete system composes (`docs/sync-async-unification.md`).
//!
//! There used to be a separate `StackfulSystem` trait here (`Mutex`/
//! `Barrier`/`Delegator` only), with `ThreadSystem` blanket-derived from
//! it. It was dropped: `ThreadSystem` already declares those same
//! associated types itself, so the split bought nothing but an extra name
//! and an extra indirection to trace — every concrete system implements
//! `ThreadSystem` directly now (the [`ult_system!`](crate::ult_system)
//! macro generates the full impl; hand-rolled systems like
//! `DualTaskSystem` write it directly too).
//!
//! There used to also be a `StacklessSystem` trait (`Mutex`/`Barrier` in
//! the `StacklessMutex`/`StacklessBarrier` flavor, for a system that
//! wants an `.await`-lockable mutex independent of `ThreadSystem`). It was
//! dropped too, for a different reason than `StackfulSystem`: not because
//! it duplicated another trait's members (its bounds are genuinely
//! distinct from `ThreadSystem`'s), but because nothing implemented it
//! standalone — every implementor (`DualTaskSystem`, `NestedDualTaskSystem`)
//! also implements `ThreadSystem`, and its only consumers
//! ([`SuspendedTask`](crate::resumable::dual::dual_wait::SuspendedTask),
//! [`SuspendedFuture`](crate::resumable::stackless::async_wait::SuspendedFuture))
//! never actually read its associated types, just carried it as an unused
//! bound. When a genuinely stack-free system needs an async mutex/barrier,
//! the natural home is [`StacklessTaskSystem`] (add `Mutex`/`Barrier`
//! there directly) rather than reviving a standalone marker trait ahead of
//! that actual need.
//!
//! [`StacklessTaskSystem`] follows the exact same "no implementation-layer
//! type in the trait's own signature" discipline `ThreadSystem` does:
//! every method that needs `resumable`-layer machinery
//! (spawn/recurse/run_async) is declared with **no default body** —
//! bodies live entirely in the blanket `impl<S: SchedulerSystem>
//! StacklessTaskSystem for S` in
//! [`resumable::stackless::system`](crate::resumable::stackless::system),
//! where naming `SchedulerSystem` and concrete resumable types is fine.
//! Only `yield_now`, which never touches `SchedulerSystem` at all, keeps a
//! default body here.

use std::future::Future;

use crate::traits::scoped::{ScopedStackfulTaskSystem, ScopedStacklessTaskSystem};
use crate::traits::thread_system::ThreadSystem;

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
/// this and [`StacklessTaskSystem`] simply *is* dual, no separate marker
/// needed.
pub trait StackfulTaskSystem: ScopedStackfulTaskSystem + ThreadSystem {}

/// `S::spawn(...)`/`S::recurse(...)`/`S::run_async(...)` — the stackless
/// counterpart of [`ThreadSystem`]: a capability every [`SchedulerSystem`](crate::resumable::common::system::SchedulerSystem)
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

    /// Yield once to the executor from inside an async task on this
    /// system — see [`crate::future::yield_now`], which this just
    /// forwards to.  Not generic over `Self` at all (unlike
    /// `spawn`/`recurse`/`run_async`): provided here purely so generic
    /// code bounded by `S: StacklessTaskSystem` can write
    /// `S::yield_now().await` instead of a separate `cmpth::future`
    /// import, matching this trait's other methods.
    ///
    /// Deliberately shares its name with [`ThreadSystem::yield_now`] (the
    /// stackful, synchronous, whole-ULT-suspending version) rather than
    /// being renamed to dodge the collision — on a dual system
    /// implementing both traits, calling `Concrete::yield_now()` is
    /// ambiguous by design (same resolution as `spawn` above) and must be
    /// disambiguated with `<Concrete as StacklessTaskSystem>::yield_now()`
    /// / `<Concrete as ThreadSystem>::yield_now()`; a generic caller
    /// bounded by only one of the two traits never sees the ambiguity.
    fn yield_now() -> impl Future<Output = ()> {
        crate::future::yield_now()
    }
}
