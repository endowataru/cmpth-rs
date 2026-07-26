//! [`StackfulSystem`], [`StacklessSystem`], and [`StacklessTaskSystem`] —
//! the capability traits a concrete system composes
//! (`docs/sync-async-unification.md`).
//!
//! Kept separate from [`SchedulerSystem`](crate::resumable::common::system::SchedulerSystem)/
//! [`StackfulSchedulerSystem`](crate::resumable::stackful::system::StackfulSchedulerSystem), which stay in
//! `resumable::system`: those reference implementation-layer types
//! (`UltWorker`) directly in their signatures, so they belong with the
//! scheduler implementation, not the interface layer. `StackfulSystem` is a
//! sibling of `StackfulSchedulerSystem`, not built on top of it (no supertrait
//! relationship) — a concrete marker struct implements both independently
//! (via [`ult_system!`](crate::ult_system)), but generic code that only
//! needs scheduler internals (almost all of `resumable/`) should bound on
//! `StackfulSchedulerSystem` alone rather than pulling in `StackfulSystem`'s
//! user-facing `Mutex`/`Barrier`/`Delegator` capabilities it never uses.
//! `run` *is* a trait method here, despite needing `Self: StackfulSchedulerSystem`
//! to implement (every implementor is a concrete marker struct that also
//! implements `StackfulSchedulerSystem` directly, so the bound is satisfied at
//! each `impl` site without this trait's own signature needing to name it) —
//! `bench/examples/quick.rs` calls it generically over an abstract
//! `S: StackfulSystem` to run the same benchmark body against several system
//! configurations, so it can't be inherent-only.
//!
//! [`StacklessTaskSystem`] follows the exact same "no implementation-layer
//! type in the trait's own signature" discipline, using the same technique
//! [`ThreadSystem`] does: every method that needs `resumable`-layer
//! machinery (spawn/recurse/run_async) is declared with **no default
//! body** — bodies live entirely in the blanket
//! `impl<S: SchedulerSystem> StacklessTaskSystem for S` in
//! [`resumable::stackless::system`](crate::resumable::stackless::system),
//! where naming `SchedulerSystem` and concrete resumable types is fine.
//! Only `yield_now`, which never touches `SchedulerSystem` at all, keeps a
//! default body here. (An earlier version of this trait lived directly in
//! `resumable::stackless::system` with default bodies for every method,
//! which forced `SchedulerSystem` onto the trait's own supertrait list —
//! the one asymmetry in this file's otherwise-consistent split, fixed by
//! this restructuring.)

use std::future::Future;

use crate::traits::{Delegator, DelegatorConsumer, StackfulBarrier, StackfulMutex, StacklessBarrier, StacklessMutex};
use crate::traits::scoped::{ScopedStackfulTaskSystem, ScopedStacklessTaskSystem};
use crate::traits::thread_system::ThreadSystem;

/// Full ULT system configuration trait.
///
/// Implement this on a concrete marker struct to define a complete ULT system.
/// The blanket `impl<S: StackfulSystem + StackfulSchedulerSystem> ThreadSystem for S`
/// automatically provides all threading-system methods; there is no need to
/// write `impl ThreadSystem for …` separately.
///
/// Use [`ult_system!`](crate::ult_system) to implement this trait.
///
/// No `run` method here — that lives solely on
/// [`ScopedStackfulTaskSystem`](crate::traits::scoped::ScopedStackfulTaskSystem)
/// (also blanket-derived for any `S: StackfulSystem + StackfulSchedulerSystem`),
/// matching how [`StacklessSystem`] has no `run_async` either. Declaring
/// "start the scheduler" on both this trait and `ScopedStackfulTaskSystem`
/// would make `S::run(...)` ambiguous for any bundled system implementing
/// both (and it very often will — this bit a real `use cmpth::*;` call
/// site during development). Code that needs to start a system should
/// bound on `ScopedStackfulTaskSystem` (or
/// [`StackfulTaskSystem`](crate::traits::system::StackfulTaskSystem) for
/// the full bundle) rather than `StackfulSystem` alone.
pub trait StackfulSystem: Sized + Send + Sync + 'static {
    /// Mutex type for this system.
    type Mutex<T: Send>: StackfulMutex<T> + Send + Sync;

    /// Barrier type for this system.
    type Barrier: StackfulBarrier + Send + Sync;

    /// Delegator type for this system.
    type Delegator<C: DelegatorConsumer<Self>>: Delegator<Self, C>
    where
        Self: ThreadSystem;
}

/// Everything a "complete" stackful system offers: `spawn`/`join` (via
/// `ThreadSystem`) *and* `run`/`parallel_call` (via
/// `ScopedStackfulTaskSystem`). An empty bundle — no methods of its own —
/// blanket-derived for any `S: ScopedStackfulTaskSystem + ThreadSystem`
/// (see [`resumable::stackful::system`](crate::resumable::stackful::system)
/// for both blankets), never implemented by hand.
///
/// There is no `DualTaskSystem` trait: a concrete system implementing both
/// this and [`StacklessTaskSystem`] simply *is* dual, no separate marker
/// needed.
pub trait StackfulTaskSystem: ScopedStackfulTaskSystem + ThreadSystem {}

/// Marker for systems that can run stackless `spawn_async`-style tasks.
///
/// Independent of [`StackfulSystem`] (see `docs/sync-async-unification.md`): a
/// system can implement `StackfulSystem` only (real ULTs, no async),
/// `StacklessSystem` only (no stack machinery at all — not yet expressible
/// here since `spawn_async` is still defined in terms of `StackfulSystem`, see
/// the doc's "open / not yet decided"), or both (today's `DualTaskSystem`).
///
/// Deliberately not blanket-implemented for every `StackfulSystem`: that would
/// make a genuinely async-free configuration inexpressible.
pub trait StacklessSystem: Sized + Send + Sync + 'static {
    /// Mutex type for this system's stackless (`.await`-based) callers.
    type Mutex<T: Send>: StacklessMutex<T> + Send + Sync;

    /// Barrier type for this system's stackless callers.
    type Barrier: StacklessBarrier + Send + Sync;
}

/// `S::spawn(...)`/`S::recurse(...)`/`S::run_async(...)` — the stackless
/// counterpart of [`ThreadSystem`]: a capability every [`SchedulerSystem`](crate::resumable::common::system::SchedulerSystem)
/// with an async-capable descriptor gets automatically (see the blanket
/// impl in [`resumable::stackless::system`](crate::resumable::stackless::system)),
/// not something any concrete system implements by hand — matching how
/// `ThreadSystem` is blanket-derived from [`StackfulSystem`] rather than
/// hand-written.
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
