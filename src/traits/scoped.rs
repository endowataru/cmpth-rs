//! [`ScopedStackfulTaskSystem`]/[`ScopedStacklessTaskSystem`] — a rayon-
//! `join`-like binary divide-and-conquer primitive: run two function
//! objects (or futures), potentially in parallel via work stealing, and
//! either synchronously (stackful) or via `.await` (stackless) wait for
//! both results.
//!
//! Lives in [`crate::scoped`] (module name, not the method name — see that
//! module's docs) because the defining property here isn't really "fork" or
//! "join" at all: unlike `spawn`/`spawn_async`, which split off *one*
//! function and reify the caller's own continuation as the other,
//! separately schedulable half, `parallel_call` takes two already-closed
//! function objects and never exposes the caller's continuation as anything
//! stealable — control returns to the caller's own next line exactly like
//! an ordinary (if parallel) function call would. That's the same
//! "nothing spawned here outlives this call" property `std::thread::scope`
//! names after itself, just restricted to exactly two branches. This is a
//! *stricter* constraint than [`ThreadSystem`](crate::ThreadSystem)'s
//! spawn/join (whose spawned task may outlive the caller) — so anything
//! with `ThreadSystem`'s looser capability can trivially satisfy this one
//! too (spawn one branch, run the other inline, join). See
//! [`StackfulTaskSystem`](crate::traits::system::StackfulTaskSystem) for
//! that blanket derivation.
//!
//! The method was originally named after Intel TBB's `tbb::parallel_invoke`
//! (renamed `parallel_call` here to fit this trait family's naming, not
//! because the TBB precedent stopped applying) for this specific shape of
//! primitive — not "join" (collides in spirit with
//! [`JoinHandleLike::join`](crate::JoinHandleLike::join)), not "fork_join"
//! (rayon itself reserves "fork-join" language for its N-ary, heap-allocated
//! `scope()`/`Scope::spawn()` API, not for the binary, stack-only `join()`
//! this mirrors — confirmed against rayon's own docs: `join`'s description
//! never uses "fork-join"; `scope`'s does).
//!
//! Implemented directly (no `ThreadSystem`/`SchedulerSystem` involved) by
//! [`crate::scoped`]'s standalone engine for systems that want *only* this
//! capability, and blanket-derived for anything that already has
//! `ThreadSystem`/[`StacklessTaskSystem`](crate::StacklessTaskSystem) — see
//! [`crate::scoped`]'s docs for why the standalone engine stays independent
//! of `resumable`'s `SchedulerSystem`/`UltWorker` machinery.

use std::future::Future;

use crate::traits::thread_system::TaskSystem;

/// Stackful (blocking) flavor: `a`/`b` are plain closures, run to
/// completion synchronously before [`parallel_call`](Self::parallel_call)
/// returns.
pub trait ScopedStackfulTaskSystem: TaskSystem {
    /// Start `num_workers` worker threads and run `f` as the root job.
    /// Blocks until `f` (and everything it transitively
    /// [`parallel_call`](Self::parallel_call)s) completes.
    ///
    /// `F`/`R` need `'static` here (the standalone engine's internal `run`
    /// free function doesn't — `f`/its result never actually outlive this
    /// blocking call — but a system blanket-derived from
    /// [`ThreadSystem`](crate::ThreadSystem) satisfies this by spawning and
    /// joining internally, which does need it: spawned/stolen tasks there
    /// can genuinely run for an unbounded time after this call starts).
    /// Same "opaque trait bound can't conditionally relax" shape as
    /// [`ScopedStacklessTaskSystem::parallel_call`](crate::traits::scoped::ScopedStacklessTaskSystem::parallel_call)'s
    /// `MkA`/`MkB`. The non-`'static` capability is only exercised inside
    /// the crate today (`scoped::sync_engine`'s own tests) — its free
    /// functions aren't `pub`, so there's currently no way to reach the
    /// relaxed version from outside `cmpth` itself.
    fn run<F, R>(num_workers: usize, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static;

    /// Run `a` and `b`, potentially in parallel, and return both results.
    /// Must be called from within [`run`](Self::run) (on one of its worker
    /// threads, possibly nested inside another `parallel_call`'s `a`/`b`).
    ///
    /// `Fa`/`Fb` need `'static` here for the same reason [`run`](Self::run)
    /// does: the standalone engine's own internal `parallel_call` free
    /// function doesn't need it (`a`/`b` are provably both finished before
    /// this returns, so borrowing the caller's own stack data is sound —
    /// see `scoped::sync_engine`'s `borrows_non_static_data` test), but a
    /// system blanket-derived from
    /// [`ThreadSystem`](crate::ThreadSystem) satisfies this via
    /// `spawn`+`join`, and a plain (non-scoped) `spawn` can never promise a
    /// task won't outlive its caller — the type system has no way to know
    /// this particular spawned task always gets joined before returning.
    fn parallel_call<Fa, Fb, Ra, Rb>(a: Fa, b: Fb) -> (Ra, Rb)
    where
        Fa: FnOnce() -> Ra + Send + 'static,
        Fb: FnOnce() -> Rb + Send + 'static,
        Ra: Send + 'static,
        Rb: Send + 'static;
}

/// Stackless (`.await`) flavor: `a`/`b` are futures, driven by polling
/// (potentially on different worker threads) rather than run to completion
/// synchronously — composes with async code the way `futures::join!`/
/// `tokio::join!` do, just with work-stealing instead of same-task polling.
///
/// Unlike [`ScopedStackfulTaskSystem::parallel_call`], `b` here needs
/// `'static`: the returned future can be dropped (cancelled) before `b`
/// finishes even after it's been genuinely stolen onto another worker
/// thread, so `b`'s storage is kept alive by an `Arc` shared with the thief
/// rather than borrowed from the caller's own stack frame — sound only if
/// nothing it closes over can go out of scope first. This is the same
/// trade rayon itself makes for its heap-backed `scope()`/`spawn()` API
/// (vs. the stack-only, non-`'static` `join()`), just applied to keep this
/// binary primitive's stackless flavor safely cancellable.
///
/// Takes **thunks** (`mk_a`/`mk_b`), not already-constructed futures —
/// same reason [`crate::resumable::stackless::thread::spawn_async`]/
/// [`crate::resumable::stackless::thread::recurse`] do: a directly self-recursive `async fn`
/// (`fib(n) = ...parallel_call(fib(n-1), fib(n-2))...`) can't pass its own
/// opaque return type as a bare generic argument to anything without
/// hitting E0733, regardless of what the callee does with it internally —
/// only a distinct, non-opaque *closure* type sidesteps that. `mk_a`/`mk_b`
/// are called eagerly, synchronously, inside `parallel_call` itself (not
/// deferred to a `poll`), exactly like `recurse`.
pub trait ScopedStacklessTaskSystem: TaskSystem {
    /// Start `num_workers` worker threads and run `root` as the first async
    /// job. Returns once `root` (and everything it transitively
    /// [`parallel_call`](Self::parallel_call)s) completes.
    fn run_async<F>(num_workers: usize, root: F)
    where
        F: Future<Output = ()> + Send + 'static;

    /// Run `mk_a()`/`mk_b()`'s futures, potentially in parallel, and
    /// resolve to both results once both complete. Must be polled from
    /// within [`run_async`](Self::run_async).
    ///
    /// `MkA`/`MkB` need `Send + 'static` here (the standalone engine calls
    /// both eagerly and never actually needs it, but a system blanket-
    /// derived from [`StacklessTaskSystem`](crate::StacklessTaskSystem)
    /// implements this via `spawn_async`, which does — see that trait's
    /// `recurse` doc comment for the general shape of this "opaque return
    /// type can't conditionally relax a bound" limitation).
    fn parallel_call<Fa, Fb, Ra, Rb, MkA, MkB>(mk_a: MkA, mk_b: MkB) -> impl Future<Output = (Ra, Rb)> + Send
    where
        MkA: FnOnce() -> Fa + Send + 'static,
        MkB: FnOnce() -> Fb + Send + 'static,
        Fa: Future<Output = Ra> + Send + 'static,
        Fb: Future<Output = Rb> + Send + 'static,
        Ra: Send + 'static,
        Rb: Send + 'static;
}
