//! Stackless-only descriptor operations: the `spawn_async` poll entry point.

use std::cell::Cell;
use std::task::Context;

use crate::resumable::common::desc::WakerTaskDesc;

/// Result of driving one `spawn_async` task's poll to completion or a
/// suspend point (named `TaskPollResult`, not `PollResult`, to keep it out
/// of the way of `std::task::Poll` and `std::future::poll_fn` at a glance).
pub enum TaskPollResult<D> {
    /// The future finished; nothing left to do for this task.
    Ready,
    /// The future finished, and its completion claimed exclusive ownership
    /// of a waiting [`JoinState::AsyncJoiner`](crate::resumable::common::desc::JoinState::AsyncJoiner)
    /// — the caller's poll loop should continue directly into that
    /// descriptor next (symmetric transfer), instead of pushing it to a
    /// deque and waiting for some worker to pop it back out. Safe because
    /// `try_wake_state`'s `ClaimedParked` outcome (the only case this is
    /// constructed for) proves nobody else can be concurrently polling that
    /// descriptor.
    ReadyAndContinue(*mut D),
    /// The future returned `Poll::Pending`; the caller should park (or
    /// requeue immediately if a wake raced in during the poll).
    Pending,
}

/// Type-erased poll function stored on an async task's descriptor. Not
/// `PollFn` — that reads too much like `std::future::poll_fn` for a type
/// that has nothing to do with it.
pub type TaskPollFn<D> = for<'cx> unsafe fn(*mut D, &mut Context<'cx>) -> TaskPollResult<D>;

/// Descriptor operations needed only by tasks that represent a `spawn_async`
/// Future — the type-erased poll entry point. Builds on [`WakerTaskDesc`]
/// (a `spawn_async` task's own poll loop, `run_async_poll`, uses
/// `mark_polling`/`park_after_poll` on itself just like `block_on` does).
pub trait AsyncTaskDesc: WakerTaskDesc {
    /// Non-null for async tasks spawned via `spawn_async`; null for sync
    /// ULTs.
    ///
    /// When set, `Worker::execute` calls this instead of doing a context
    /// switch.  The function polls the Future stored in the task's "stack"
    /// buffer; see [`TaskPollResult`] for what it reports back (`Ready`:
    /// don't touch `desc` again; `Pending`: park it; `ReadyAndContinue`:
    /// poll the named descriptor next instead).
    fn poll_fn(&self) -> &Cell<Option<TaskPollFn<Self>>>;
}
