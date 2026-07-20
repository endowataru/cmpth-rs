//! Sync/async wait-slot traits: [`Resumable`], [`StackfulResumable`],
//! [`StacklessResumable`].
//!
//! These generalize the older, now-retired `SuspendedThread` trait to also
//! cover stackless `spawn_async`-style waiters. The stackful and stackless
//! flavors use the *same* method names (`wait_with`, `enter`, `swap`);
//! callers pick the flavor by which trait they `use`, mirroring how
//! [`crate::future::yield_now`] is disambiguated from
//! [`ThreadSystem::yield_now`](crate::traits::ThreadSystem::yield_now) by
//! module path rather than by a `_async` suffix. See
//! `docs/sync-async-unification.md` for the full design.

use std::task::{Context, Poll};

/// The durable capability every wait-slot has, regardless of what kind of
/// waiter (if any) is currently parked: a real ULT continuation, a
/// registered async [`Waker`](std::task::Waker), or nothing. Unlike
/// `is_set`'s answer, which changes per instance over time, this trait
/// itself is a fixed property of the type — same spirit as `Send`/`Sync`.
pub trait Resumable<S>: Default {
    /// True if a waiter is currently parked here.
    fn is_set(&self) -> bool;

    /// Wake whatever is parked here, if anything. Cheap and direct for a
    /// real ULT continuation; goes through `Waker::wake` only when the slot
    /// actually holds a registered async waiter.
    fn notify(&self);
}

/// Stackful (real-context-switch) flavor of parking. These do a real
/// context switch and must only be called from a genuine ULT stack —
/// checked dynamically via `cur_task.is_root` (see
/// `docs/sync-async-unification.md`), not via an explicit capability token.
pub trait StackfulResumable<S>: Resumable<S> {
    /// Suspend the current ULT into this slot. `f` runs after the context
    /// is fully saved (release any spinlock protecting this slot inside
    /// it).
    fn wait_with<F: FnOnce()>(&self, f: F);

    /// Like [`wait_with`](Self::wait_with), but `f` may cancel the
    /// suspension by returning `false`.
    fn wait_with_cond<F: FnOnce() -> bool>(&self, f: F);

    /// Switch directly to the parked continuation, pushing the caller's own
    /// continuation to the local deque. If the slot didn't hold a real
    /// continuation — only possible when `Self` also admits async waiters
    /// (e.g. `SuspendedTask`) — falls back to waking it the
    /// [`Resumable::notify`] way internally, so callers never need to
    /// branch on whether a real switch happened.
    fn enter(&self);

    /// Symmetric handoff: park the current ULT here and switch to `next`.
    /// Same async-target fallback as [`enter`](Self::enter).
    fn swap(&self, next: &Self);
}

/// Stackless (poll-based) flavor of parking.
///
/// `enter`/`swap` are deliberately not part of this trait yet: a correct
/// stackless implementation needs the caller to defer itself to the
/// FIFO/steal end of the local deque (`push_local_bottom`) so the target it
/// hands off to isn't overtaken by the caller's own re-queued continuation,
/// and `cmpth::future::yield_now()`'s self-wake path doesn't do that today
/// (see `ISSUES.md`). Adding them before that's fixed would silently invert
/// the intended priority.
pub trait StacklessResumable<S>: Resumable<S> {
    /// Register `cx`'s waker.
    fn register(&self, cx: &mut Context<'_>);

    /// `.await`-able equivalent of [`StackfulResumable::wait_with`]:
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
    ) -> impl std::future::Future<Output = ()> + Send
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
