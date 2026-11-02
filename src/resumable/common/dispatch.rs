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
//! Swap the implementation via [`crate::WorkerSystem::Dispatch`].

use crate::resumable::common::deque::Steal;

pub use crate::resumable::common::observe::Site;

/// What a worker can draw a task from, in the calling worker's context.
pub trait TaskSources {
    type Item;

    /// The worker's own run queue.
    fn local(&self) -> Option<Self::Item>;

    /// One steal round through the system's
    /// [`StealPolicy`](crate::StealPolicy).
    fn steal(&self) -> Steal<Self::Item>;

    /// Tasks that yielded and wait outside the worker's own queue (the
    /// pool's [`YieldPolicy`](crate::YieldPolicy)); always `None` with
    /// [`LocalYield`](crate::LocalYield).
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

/// The standard order:
///
/// * [`Site::Loop`] and [`Site::Wait`]: own queue, then the pool's yielded
///   tasks, then a steal round, then the external queue;
/// * [`Site::Switch`]: the external queue, then the own queue, then the
///   yielded tasks, never a steal (the worker steals from its loop).  The
///   external queue comes first because tasks that only yield to each
///   other never let the worker reach its loop.
#[derive(Default)]
pub struct StandardDispatch;

impl DispatchPolicy for StandardDispatch {
    type State = ();

    #[inline(always)]
    fn next<Q: TaskSources>(_state: &(), q: &Q, site: Site) -> Steal<Q::Item> {
        match site {
            Site::Loop | Site::Wait => {
                if let Some(c) = q.local() {
                    return Steal::Success(c);
                }
                if let Some(c) = q.yielded() {
                    return Steal::Success(c);
                }
                let s = q.steal();
                if let Steal::Success(_) = s {
                    return s;
                }
                if let Some(c) = q.external() {
                    return Steal::Success(c);
                }
                s
            }
            Site::Switch => {
                if let Some(c) = q.external() {
                    return Steal::Success(c);
                }
                if let Some(c) = q.local() {
                    return Steal::Success(c);
                }
                match q.yielded() {
                    Some(c) => Steal::Success(c),
                    None => Steal::Empty,
                }
            }
        }
    }
}
