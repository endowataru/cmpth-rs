use std::ops::DerefMut;

// ---------------------------------------------------------------------------
// TaskDesc
// ---------------------------------------------------------------------------

/// A task descriptor: the per-task join-protocol state (has it finished?
/// who's waiting?) that every scheduling flavor needs, regardless of how it
/// represents a task's stack, poll function, or anything else about *how*
/// the task actually runs.
///
/// Bodyless — pure behavior, no storage representation implied. Implement
/// this directly for a completely custom descriptor, or implement
/// [`TaskDescCore`](crate::resumable::common::desc::TaskDescCore) instead
/// to get this crate's own word-based join-protocol algorithm for free via
/// a blanket impl.
///
/// `Owned` is this descriptor's owner-exclusive data (whatever a concrete
/// implementation wants to store there — e.g. this crate's own worker/
/// slot/result/tls bookkeeping) reachable *only* through a live
/// [`Suspended`](Self::Suspended)/[`Running`](Self::Running) token: holding
/// one of these tokens is itself the proof of exclusive access, the same
/// "the token proves the precondition" pattern `MutexGuard`/`RefMut` use.
/// `TaskDesc` says nothing about *how* that exclusivity is implemented
/// (`UnsafeCell`, a lock, whatever) — only that a token gives `&mut Owned`
/// via `DerefMut`.
///
/// Deliberately says nothing about *who* (if anyone) is waiting on a task —
/// that's this descriptor's own private business, encoded however it likes
/// internally (this crate's own implementation uses a single tagged-pointer
/// word — see [`resumable::common::desc`](crate::resumable::common::desc)'s
/// module doc comment). The only two things the outside world is ever
/// allowed to do about a waiter are captured by [`TaskExitSink`]:
/// resume it, or notice nobody's there to collect the result.
pub trait TaskDesc: Send + Sync + Sized + 'static {
    /// Owner-exclusive data, reached only through a live `Suspended`/
    /// `Running` token.
    type Owned;

    /// Owning handle to a suspended (parked) task — proof that nothing
    /// else can be concurrently accessing its `Owned` data.
    type Suspended: DerefMut<Target = Self::Owned> + Send;

    /// Owning handle to the task currently running — same exclusivity
    /// proof as `Suspended`, for the task actively executing rather than
    /// parked.
    type Running: DerefMut<Target = Self::Owned> + Send;

    /// Fast check for the hot join path: is the task already finished?
    fn is_finished(&self) -> bool;

    /// Direct-handoff exit: the exiting task already switched straight
    /// into the parked sync joiner's continuation — just publish
    /// `Finished` so the joiner (now running) observes its result is
    /// ready.
    fn commit_finished(&self);

    /// General-case exit: publish FINISHED and settle whoever was waiting,
    /// delegating the only two possible actions to `sink`. WHO was waiting
    /// is this descriptor's private business and must not appear in this
    /// signature — `sink.resume` for a waiter that can be resumed,
    /// `sink.reclaim` for a dropped handle with nobody left to collect the
    /// result. A registered async waker (a foreign `Waker`, not a
    /// same-system task) is woken directly, inline, with no sink call at
    /// all — see the concrete implementation for why.
    fn finish_and_settle<K: TaskExitSink<Self>>(&self, sink: &K);

    /// Try to mark this task abandoned (no handle left to collect the
    /// result). Returns `true` if the task was already finished (caller
    /// now owns the result and the descriptor) — otherwise commits
    /// abandoned.
    fn try_abandon(&self) -> bool;

    /// Pre-exit check: was this task's handle dropped early (abandoned)?
    /// Stable once true — an abandoned task never comes back — so this is
    /// safe to read before a context switch and act on afterward, same as
    /// [`HandoffTaskDesc::try_take_handoff_target`](crate::traits::component::stackful::HandoffTaskDesc::try_take_handoff_target).
    fn is_abandoned(&self) -> bool;
}

/// The two things anything settling a finished task's join protocol is ever
/// allowed to do with whoever (if anyone) was waiting — see
/// [`TaskDesc::finish_and_settle`]. Implemented at each exit call site,
/// capturing whatever that call site needs (the worker, the descriptor
/// pointer, the result slot) to actually perform the action.
pub trait TaskExitSink<D: TaskDesc> {
    /// Resume a waiter that can be resumed (a parked sync joiner, or a
    /// same-system async joiner reclaimed directly) — typically by pushing
    /// its continuation back onto a worker's deque.
    fn resume(&self, cont: D::Suspended);

    /// Nobody is left to collect the result (the `JoinHandle` was dropped
    /// early) — reclaim whatever the exiting task itself can't clean up
    /// until after the context switch (the result slot, the descriptor).
    fn reclaim(&self);
}
