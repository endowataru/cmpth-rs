use std::future::Future;

use crate::traits::system::TaskSystem;
use crate::traits::system::scoped::ScopedStacklessTaskSystem;

/// `S::spawn(...)`/`S::recurse(...)` — the stackless counterpart of
/// [`ThreadSystem`](crate::traits::system::stackful::ThreadSystem): a
/// capability every [`SchedulerSystem`](crate::resumable::common::system::SchedulerSystem)
/// with an async-capable descriptor gets automatically (see the blanket
/// impl in [`resumable::stackless::system`](crate::resumable::stackless::system)),
/// not something any concrete system implements by hand.
///
/// `: ScopedStacklessTaskSystem` because `parallel_call`'s "nothing
/// outlives this call" constraint is strictly stricter than `spawn`'s (a
/// spawned task may outlive the caller) — anything with `spawn` capability
/// trivially satisfies the more restricted one too (spawn one branch,
/// await the other inline). `run_async` itself lives on
/// [`StacklessBuilder`] (reached via [`StacklessInitSystem::builder`]),
/// not on either trait here — see that trait's doc comment.
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

    /// Returns `Pending` on the first poll, then `Ready` on the next — a
    /// single suspend/resume round-trip that is a **fair** yield: on the
    /// worker actually driving this task (`run_async_poll`), the
    /// implementation routes the requeue through `WorkerRunQueue::defer`
    /// rather than the ordinary self-wake path's `push`, so already-queued
    /// sibling tasks on that worker run first — see the blanket impl in
    /// [`resumable::stackless::system`](crate::resumable::stackless::system)
    /// for the mechanism.
    ///
    /// No default body: a correct implementation needs to reach the
    /// current worker's run queue directly to make that `defer` vs. `push`
    /// choice, and this interface-layer trait must not name any
    /// `resumable`-layer type to do so — the same `traits/` vs.
    /// `resumable/` layering rule every other method on this trait already
    /// follows (a default body here would leak worker/queue types into the
    /// interface layer). Each `StacklessTaskSystem` implementor supplies
    /// its own.
    ///
    /// Deliberately shares its name with
    /// [`ThreadSystem::yield_now`](crate::traits::system::stackful::ThreadSystem::yield_now)
    /// (the stackful, synchronous, whole-ULT-suspending version) rather than
    /// being renamed to dodge the collision — on a dual system
    /// implementing both traits, calling `Concrete::yield_now()` is
    /// ambiguous by design (same resolution as `spawn` above) and must be
    /// disambiguated with `<Concrete as StacklessTaskSystem>::yield_now()`
    /// / `<Concrete as ThreadSystem>::yield_now()`; a generic caller
    /// bounded by only one of the two traits never sees the ambiguity.
    fn yield_now() -> impl Future<Output = ()>;
}

// ---------------------------------------------------------------------------
// StacklessInitSystem / StacklessBuilder — replaces the old
// `ScopedStacklessTaskSystem::run_async(num_workers, root)`.
// ---------------------------------------------------------------------------

/// A stackless system that can bring up its worker pool through a
/// [`StacklessBuilder`].
///
/// **No standalone `init()`** here, deliberately — unlike
/// [`StackfulInitSystem`](crate::traits::system::stackful::StackfulInitSystem)'s
/// `init`, there is no caller continuation to save: a stackless task is a
/// polled `Future`, not a real stack, so there is nothing for "everything
/// after this call keeps running as a task" to mean. [`StacklessBuilder`]
/// only ever offers the bracketing [`run_async`](StacklessBuilder::run_async).
pub trait StacklessInitSystem: TaskSystem {
    /// Builder type — accumulates configuration (`workers`, ...) before
    /// `run_async`.
    type Builder: StacklessBuilder<Self>;

    /// Start building a configuration for this system.
    fn builder() -> Self::Builder;
}

/// Accumulates configuration for a [`StacklessInitSystem`] before bringing
/// up its worker pool.
pub trait StacklessBuilder<S: StacklessInitSystem>: Sized {
    /// Set the worker count. Defaults to
    /// [`available_parallelism`](crate::available_parallelism) if never
    /// called.
    fn workers(self, n: usize) -> Self;

    /// Start the worker pool, run `root` as the first async job, and block
    /// until it (and everything it transitively
    /// [`parallel_call`](crate::traits::scoped::ScopedStacklessTaskSystem::parallel_call)s)
    /// completes.
    fn run_async<F>(self, root: F)
    where
        F: Future<Output = ()> + Send + 'static;
}
