//! [`StackfulParallelInvoke`]/[`StacklessParallelInvoke`] — a rayon-`join`-
//! like binary divide-and-conquer primitive: run two function objects (or
//! futures), potentially in parallel via work stealing, and either
//! synchronously (stackful) or via `.await` (stackless) wait for both
//! results.
//!
//! Lives in [`crate::scoped`] (module name, not the method name — see that
//! module's docs) because the defining property here isn't really "fork" or
//! "join" at all: unlike `spawn`/`spawn_async`, which split off *one*
//! function and reify the caller's own continuation as the other,
//! separately schedulable half, `parallel_invoke` takes two already-closed
//! function objects and never exposes the caller's continuation as anything
//! stealable — control returns to the caller's own next line exactly like
//! an ordinary (if parallel) function call would. That's the same
//! "nothing spawned here outlives this call" property `std::thread::scope`
//! names after itself, just restricted to exactly two branches.
//!
//! The method itself keeps Intel TBB's `tbb::parallel_invoke` naming for
//! this specific shape of primitive — not "join" (collides in spirit with
//! [`JoinHandleLike::join`](crate::JoinHandleLike::join)), not "fork_join"
//! (rayon itself reserves "fork-join" language for its N-ary, heap-allocated
//! `scope()`/`Scope::spawn()` API, not for the binary, stack-only `join()`
//! this mirrors — confirmed against rayon's own docs: `join`'s description
//! never uses "fork-join"; `scope`'s does).
//!
//! Implemented by [`crate::scoped`], a scheduler family deliberately
//! independent of [`crate::ult`]'s `SchedulerSystem`/`UltWorker` machinery
//! — see that module's docs for why.

use std::future::Future;

/// Stackful (blocking) flavor: `a`/`b` are plain closures, run to
/// completion synchronously before [`parallel_invoke`](Self::parallel_invoke)
/// returns.
pub trait StackfulParallelInvoke: Sized + Send + Sync + 'static {
    /// Start `num_workers` worker threads and run `f` as the root job.
    /// Blocks until `f` (and everything it transitively
    /// [`parallel_invoke`](Self::parallel_invoke)s) completes.
    fn run<F, R>(num_workers: usize, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send;

    /// Run `a` and `b`, potentially in parallel, and return both results.
    /// Must be called from within [`run`](Self::run) (on one of its worker
    /// threads, possibly nested inside another `parallel_invoke`'s `a`/`b`).
    fn parallel_invoke<Fa, Fb, Ra, Rb>(a: Fa, b: Fb) -> (Ra, Rb)
    where
        Fa: FnOnce() -> Ra + Send,
        Fb: FnOnce() -> Rb + Send,
        Ra: Send,
        Rb: Send;
}

/// Stackless (`.await`) flavor: `a`/`b` are futures, driven by polling
/// (potentially on different worker threads) rather than run to completion
/// synchronously — composes with async code the way `futures::join!`/
/// `tokio::join!` do, just with work-stealing instead of same-task polling.
///
/// Unlike [`StackfulParallelInvoke::parallel_invoke`], `b` here needs
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
/// same reason [`crate::ult::thread::spawn_async`]/
/// [`crate::ult::thread::recurse`] do: a directly self-recursive `async fn`
/// (`fib(n) = ...parallel_invoke(fib(n-1), fib(n-2))...`) can't pass its own
/// opaque return type as a bare generic argument to anything without
/// hitting E0733, regardless of what the callee does with it internally —
/// only a distinct, non-opaque *closure* type sidesteps that. `mk_a`/`mk_b`
/// are called eagerly, synchronously, inside `parallel_invoke` itself (not
/// deferred to a `poll`), exactly like `recurse`.
pub trait StacklessParallelInvoke: Sized + Send + Sync + 'static {
    /// Start `num_workers` worker threads and run `root` as the first async
    /// job. Returns once `root` (and everything it transitively
    /// [`parallel_invoke`](Self::parallel_invoke)s) completes.
    fn run_async<F>(num_workers: usize, root: F)
    where
        F: Future<Output = ()> + Send + 'static;

    /// Run `mk_a()`/`mk_b()`'s futures, potentially in parallel, and
    /// resolve to both results once both complete. Must be polled from
    /// within [`run_async`](Self::run_async).
    fn parallel_invoke<Fa, Fb, Ra, Rb, MkA, MkB>(mk_a: MkA, mk_b: MkB) -> impl Future<Output = (Ra, Rb)> + Send
    where
        MkA: FnOnce() -> Fa,
        MkB: FnOnce() -> Fb,
        Fa: Future<Output = Ra> + Send + 'static,
        Fb: Future<Output = Rb> + Send + 'static,
        Ra: Send + 'static,
        Rb: Send + 'static;
}
