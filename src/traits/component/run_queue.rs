//! The worker run-queue axis: what a worker pushes runnable items to and
//! steals them from.
//!
//! **Not a deque**: no lock-free implementation can offer an owner-side push
//! at the *steal* end (Chase-Lev's steal index is advanced only by thieves,
//! via CAS), so a "deque" contract with a real `push_bottom` is a promise no
//! implementation can actually keep — see [`WorkerRunQueue`]'s doc comment.
//! The interface is defined by *scheduling intent* (`push` / `defer`)
//! rather than by position (`top` / `bottom`).  Implementations:
//! `resumable::common::deque`.

// ---------------------------------------------------------------------------
// Steal
// ---------------------------------------------------------------------------

/// Same shape as [`crossbeam_deque::Steal`], defined here so this axis's
/// contract does not expose crossbeam. `Retry` means "work exists but could
/// not be taken right now" (lost a race with another thief, or the owner) —
/// it must never be collapsed into `Empty`: a caller that treats a `Retry`
/// scan as "genuinely nothing to do" may back off (e.g. yield to the OS)
/// when work was actually available.
#[derive(Debug)]
pub enum Steal<T> {
    Empty,
    Success(T),
    Retry,
}

// ---------------------------------------------------------------------------
// WorkerRunQueue / RunQueueStealer
// ---------------------------------------------------------------------------

/// A queue of runnable items that other workers may steal from. Only the
/// owning worker holds this handle — `push`/`defer`/`try_pop` are only
/// called by the worker that owns the queue; stealing goes exclusively
/// through [`stealer`](Self::stealer)'s [`RunQueueStealer::try_steal`],
/// callable from any thread.
///
/// **Not a deque**: no single total order over items is promised. Each
/// operation is defined by *when the caller wants the item to run*, not by
/// where it is placed. This is a deliberate retreat from a `WorkerDeque`
/// contract this crate used to expose: that contract promised operations at
/// both ends of a deque, but no lock-free implementation can provide an
/// owner-side push at the *steal* end (Chase-Lev's steal index is advanced
/// by thieves via CAS only), so the old `push_bottom` silently degraded to
/// `push_top` — fair yielding was only approximated, and only by accident.
/// [`HybridRunQueue`](crate::resumable::common::deque::HybridRunQueue) keeps a second, explicitly lock-based queue for
/// `defer`, so the ordering this trait promises is real, not aspirational.
///
/// Generic over the element type `T` moved through the queue — neither
/// provided implementation ever inspects `T`, only stores/returns it. Every
/// concrete system today sets `T = SuspendedTaskToken<Self::Desc>` via
/// [`crate::WorkerSystem::SuspendedToken`].
pub trait WorkerRunQueue<T: Send>: Send + Sync + 'static {
    type Stealer: RunQueueStealer<T>;

    /// Run next on this worker (a forked child, a woken task). Keeps
    /// locality; least likely to be stolen.
    fn push(&self, v: T);

    /// Run after work already queued here (`yield`). Most likely to be
    /// stolen.
    fn defer(&self, v: T);

    /// Take what this worker should run next.
    fn try_pop(&self) -> Option<T>;

    /// A cloneable handle other workers steal through.
    fn stealer(&self) -> Self::Stealer;
}

pub trait RunQueueStealer<T: Send>: Clone + Send + Sync + 'static {
    fn try_steal(&self) -> Steal<T>;
}

