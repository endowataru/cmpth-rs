//! Dispatch policy: in what order a worker looks at its task sources when
//! it picks the next task.
//!
//! A worker picks a task at three [`Site`]s: its scheduler loop, a task
//! suspending or exiting on it (which switches to the next task directly,
//! without going back to the loop), and, on `scoped`, a task waiting for a
//! join on its own stack.  One policy orders the sources for all three, so
//! the order a system documents is the order it follows whichever way it
//! got there; the sites differ only in what the caller does with the
//! result (see [`Site`]).
//!
//! A system picks one via `WorkerSystem::Dispatch`; implementations:
//! `resumable::common::dispatch`.

use super::run_queue::Steal;

/// Where a worker picked the task it runs next (see
/// [`DispatchPolicy`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    /// The worker's scheduler loop: anything runnable.
    Loop,
    /// A task suspending or exiting on this worker: only tasks the worker
    /// can switch to directly, else the worker returns to its loop.
    Switch,
    /// A task waiting for a join on its own stack (`scoped`): anything
    /// runnable, run on top of the waiting task.
    Wait,
}

/// What a worker can draw a task from, in the calling worker's context.
pub trait TaskSources {
    type Item;

    /// The worker's own run queue.
    fn local(&self) -> Option<Self::Item>;

    /// One steal round through the system's
    /// [`StealPolicy`](crate::StealPolicy).
    fn steal(&self) -> Steal<Self::Item>;

    /// Tasks that yielded and wait outside the worker's own queue (the
    /// pool's [`YieldPolicy`](super::yield_policy::YieldPolicy)); always
    /// `None` with `LocalYield`.
    fn yielded(&self) -> Option<Self::Item>;

    /// Tasks woken from outside the pool (`ExternalQueue`); always `None`
    /// on a system without one.
    fn external(&self) -> Option<Self::Item>;
}

/// The order a worker looks at its task sources.
///
/// # Contract
///
/// * `next` is called by the worker that owns `state` only.
/// * It returns [`Steal::Retry`] when it found nothing but saw work it could
///   not take (a steal round's `Retry`), so that the idle policy does not
///   back off on its account; else [`Steal::Empty`].
/// * At [`Site::Switch`] it runs on a suspending task's stack, on every
///   suspension: keep it short.  An item that cannot be switched to is put
///   back by the caller, which then returns to the loop.
pub trait DispatchPolicy: Send + Sync + 'static {
    /// Per-worker state (a round counter, say).
    type State: Default;

    fn next<Q: TaskSources>(state: &Self::State, q: &Q, site: Site) -> Steal<Q::Item>;
}
