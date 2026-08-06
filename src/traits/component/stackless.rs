use std::future::Future;
use std::task::{Context, Poll, Waker};

use crate::traits::component::task::TaskDesc;
use crate::traits::component::wait::Resumable;

// ---------------------------------------------------------------------------
// WakerTaskDesc
// ---------------------------------------------------------------------------

/// Descriptor operations needed by a task driven via a real
/// [`std::task::Waker`] whose wake state must live on the descriptor itself
/// — currently only `spawn_async` (no stack to anchor the state anywhere
/// else; a real ULT's `block_on` uses the same state machine but keeps it
/// in a call-scoped `Poller` instead, since it has a stack to anchor on).
///
/// Bodyless — pure behavior, same spirit as [`TaskDesc`]. Implement
/// directly for a custom representation, or implement
/// [`WakerTaskDescCore`](crate::resumable::stackless::desc::WakerTaskDescCore)
/// (together with [`TaskDescCore`](crate::resumable::common::desc::TaskDescCore))
/// instead to get this crate's own word-based algorithm for free via a
/// blanket impl.
///
/// The async join-registration methods (`try_register_async_joiner`/
/// `try_register_waker`) live here rather than on the base `TaskDesc`
/// specifically so a descriptor with no async capability at all never gets
/// them — they're only ever called from `JoinHandle::poll`, which itself
/// requires `S::Desc: AsyncTaskDesc: WakerTaskDesc`.
pub trait WakerTaskDesc: TaskDesc {
    fn mark_polling(&self);
    fn mark_idle(&self);
    fn decide_park(&self) -> bool;
    fn park_after_poll(&self) -> bool;

    /// Try to claim a parked waiter directly, shared by the stackful and
    /// async wake paths — `Some` means this call won the PARKED -> POLLING
    /// transition and now owns delivering the returned continuation (push
    /// to a worker deque or the external queue); `None` means there was
    /// nothing to claim (was POLLING — the task will notice on its own and
    /// re-poll — or was already NOTIFIED/IDLE).
    fn try_claim_parked(&self) -> Option<Self::Suspended>;

    /// True once this waker has been cloned at least once. Sticky — never
    /// clears back to false.
    fn is_waker_shared(&self) -> bool;

    /// First-clone transition: mark this waker as shared, preserving
    /// whatever poll state is currently set.
    fn note_waker_shared(&self);

    /// `JoinHandle::poll`'s fast-path registration: install `joiner` (the
    /// currently-polling task's own continuation) directly, with no
    /// allocation. `Err` means the task turned out to already be finished
    /// — hands `joiner` straight back so it cannot leak; the caller should
    /// proceed to take the result instead. `Ok` commits the registration.
    fn try_register_async_joiner(&self, joiner: Self::Suspended) -> Result<(), Self::Suspended>;

    /// `JoinHandle::poll`'s waker registration: try to install `waker` as
    /// this task's async waiter. Returns `false` if the task turned out to
    /// already be finished (caller should proceed to take the result
    /// instead) — otherwise commits the registration.
    fn try_register_waker(&self, waker: Waker) -> bool;
}

/// Stackless (poll-based) flavor of parking.
///
/// `enter`/`swap` are deliberately not part of this trait yet: a correct
/// stackless implementation needs the caller to defer itself to the
/// FIFO/steal end of the local deque (`push_local_bottom`) so the target it
/// hands off to isn't overtaken by the caller's own re-queued continuation,
/// and [`StacklessTaskSystem::yield_now`](crate::traits::system::stackless::StacklessTaskSystem::yield_now)'s self-wake path doesn't do that
/// today (see its doc comment). Adding them before that's fixed would
/// silently invert the intended priority.
pub trait StacklessResumable<S>: Resumable<S> {
    /// Register `cx`'s waker.
    fn register(&self, cx: &mut Context<'_>);

    /// `.await`-able equivalent of
    /// [`StackfulResumable::wait_with`](crate::traits::component::stackful::StackfulResumable::wait_with):
    /// registers this task's waker, then runs `f`, then suspends by
    /// returning `Poll::Pending` once, completing once notified.
    ///
    /// `register` must run *before* `f`, mirroring the ordering
    /// `StackfulResumable::wait_with`'s implementations use (store the
    /// parked continuation, *then* publish the link). `f` is what makes
    /// this slot reachable by a concurrent notifier (e.g. publishing this
    /// node into an MCS chain's `next` pointer) — if it ran first, a
    /// notifier could observe the link and call `notify()` while the waker
    /// slot is still empty, losing the wakeup permanently. This was caught
    /// by an hour-long hang in `async_only_flavor` under `cargo test
    /// --all`'s parallel scheduling — the tight, low-worker-count runs used
    /// while developing this never hit the race window.
    ///
    /// Desugared (rather than a native `async fn`) specifically to pin down
    /// `Send` on the returned `Future` — this needs to be usable inside a
    /// `spawn_async`'d task, which requires `F: Future + Send`.
    fn wait_with<F: FnOnce() + Send>(
        &self,
        f: F,
    ) -> impl Future<Output = ()> + Send
    where
        Self: Sync,
    {
        let mut f = Some(f);
        std::future::poll_fn(move |cx| {
            if let Some(f) = f.take() {
                self.register(cx);
                f();
                Poll::Pending
            } else {
                Poll::Ready(())
            }
        })
    }
}
