use crate::traits::system::TaskSystem;

/// Threading system interface — spawn/join only.  Split off from what used
/// to be a six-capability bundle (see `block_on.rs`, `sync.rs`,
/// `suspend.rs`, `delegation.rs`, `nesting.rs` in this module for the rest)
/// so that code which only wants `spawn` isn't forced to also supply a
/// `Poller`, a `Mutex`, a `Barrier`, a `Delegator`, a TLS slot type, and a
/// parked-continuation type.
pub trait ThreadSystem: TaskSystem {
    /// Spawn a new thread or ULT; returns a handle that can be joined.
    type JoinHandle<T: Send + 'static>: JoinHandleLike<T>;
    fn spawn<T, F>(f: F) -> Self::JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static;

    /// Yield the current thread/ULT so other tasks can run.
    fn yield_now();
}

/// Common interface for join handles returned by [`ThreadSystem::spawn`].
pub trait JoinHandleLike<T: Send + 'static>: Send {
    fn join(self) -> T;
}

// ---------------------------------------------------------------------------
// StackfulInitSystem / StackfulBuilder — standalone initializer + bracketing
// `run`, replacing the old `ScopedStackfulTaskSystem::run(num_workers, f)`.
// ---------------------------------------------------------------------------

/// A stackful system that can bring up its worker pool through a
/// [`StackfulBuilder`] — either the bracketing [`StackfulBuilder::run`]
/// (start, run `f`, block until done, tear down) or the standalone
/// [`StackfulBuilder::init`] (start, then return: everything the caller does
/// *after* that call keeps running, as an ordinary — stealable — task on
/// the pool, until the returned guard is dropped).
///
/// Mirrors ComposableThreads' `basic_scheduler<P>::initializer`: see
/// [`crate::resumable::stackful::init`] for the mechanism (child-first fork
/// of the scheduler loop, so the *caller's* continuation — not a
/// separately-forked root task — is what keeps running).
///
/// Blanket-derived for any `S: ThreadSystem + StackfulSchedulerSystem`
/// (`resumable::stackful::system`) — never implemented by hand for a
/// `resumable`-backed system. [`crate::ScopedTaskSystem`] (the independent,
/// non-`resumable` `parallel_call`-only engine) also implements it, using
/// its own OS-thread pool directly: it has no ULT/context-switch concept at
/// all, so `init`'s "keeps running as a schedulable task" property there
/// just means "the calling OS thread stays this pool's worker 0" — nothing
/// about it ever migrates, same as `parallel_call`'s existing capability.
pub trait StackfulInitSystem: TaskSystem {
    /// Builder type — accumulates configuration (`workers`, ...) before
    /// `init`/`run`.
    type Builder: StackfulBuilder<Self>;

    /// RAII guard returned by [`StackfulBuilder::init`]. Dropping it tears
    /// the pool down. See [`StackfulBuilder::init`]'s doc comment for the
    /// full contract, including the panic-across-drop hazard.
    type Init;

    /// Start building a configuration for this system.
    fn builder() -> Self::Builder;
}

/// Accumulates configuration for a [`StackfulInitSystem`] before bringing up
/// its worker pool.
pub trait StackfulBuilder<S: StackfulInitSystem>: Sized {
    /// Set the worker count. Defaults to
    /// [`available_parallelism`](crate::available_parallelism) if never
    /// called.
    fn workers(self, n: usize) -> Self;

    /// Standalone init: start the worker pool, then return. Everything the
    /// caller does *after* this call — including the call's own return —
    /// keeps running as an ordinary, stealable task on the pool: `spawn`,
    /// `join`, and (unlike inside the old bracketing `run`, where it always
    /// panicked) [`ScopedStackfulTaskSystem::parallel_call`](crate::traits::scoped::ScopedStackfulTaskSystem::parallel_call)
    /// all work immediately, with no enclosing `run` needed.
    ///
    /// # Panics across the returned guard's `Drop`
    ///
    /// A Rust panic unwinding across a context switch is unsound (the
    /// unwinder's in-flight state is OS-thread-local, but a suspended ULT
    /// may resume on a different OS thread, or on the same one after other
    /// tasks — which may themselves unwind — have run in between). The
    /// guard's `Drop` does exactly one context switch (back to the
    /// scheduler) internally, so it checks
    /// [`std::thread::panicking`] and
    /// `std::process::abort`s, with a clear message on stderr, rather than
    /// attempt that switch while unwinding. Use [`run`](Self::run) instead
    /// of standalone `init` whenever panic propagation out of the pool is
    /// needed — it never lets an unwind reach the guard in the first place.
    ///
    /// # `!Send` values across a suspension point
    ///
    /// Since ordinary code after `init()` can now suspend (via `join`,
    /// blocking `Mutex`, `block_on`, ...) the same hazard
    /// [`Poller`](crate::traits::component::stackful::Poller)'s doc comment
    /// describes for tasks applies here too, right from `main`'s own stack:
    /// do not hold a [`std::sync::MutexGuard`] (or anything else `!Send`
    /// because its invariant is tied to *OS-thread* identity, not just
    /// data-race safety) across a suspension point — use this crate's own
    /// ULT `Mutex` instead.
    fn init(self) -> S::Init;

    /// Bracketing convenience: start the worker pool, run `f` to completion,
    /// tear the pool down, and return `f`'s result — defined in terms of
    /// [`init`](Self::init), so every [`StackfulInitSystem`] gets it for
    /// free. Unlike bare `init`, a panic inside `f` is caught here (never
    /// reaching the guard's `Drop` while unwinding) and re-raised on the
    /// caller's own thread once teardown finishes normally — the same
    /// catch/re-raise shape `spawn`'s task boundary already uses.
    fn run<F, R>(self, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let guard = self.init();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        drop(guard);
        match result {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        }
    }
}
