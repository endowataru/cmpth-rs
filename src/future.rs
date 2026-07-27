//! Executor-agnostic helpers for `Future`-based code.
//!
//! Unlike [`ThreadSystem`](crate::ThreadSystem)'s methods, nothing here is
//! generic over a thread-system type — these compose correctly whether the
//! surrounding `Future` is driven by [`ThreadSystem::block_on`] on a real
//! ULT stack, polled in place by `spawn_async`'s stackless executor, or run
//! under an unrelated executor entirely.
//!
//! [`ThreadSystem::block_on`]: crate::ThreadSystem::block_on

use std::future::poll_fn;
use std::task::Poll;

/// Yield once to the executor: returns `Pending` on the first poll (waking
/// itself immediately so the task is re-queued), then `Ready` on the next.
///
/// Use this inside a busy-poll retry loop (e.g. testing a completion flag)
/// to let other tasks make progress between attempts.
/// [`ThreadSystem::yield_now`](crate::ThreadSystem::yield_now) is not a
/// substitute here: it suspends the whole calling ULT stack, which does not
/// exist for a `spawn_async` task polled in place.
///
/// ```
/// use cmpth::{DefaultStackfulOnlyTaskSystem, ScopedStackfulTaskSystem, ThreadSystem};
///
/// DefaultStackfulOnlyTaskSystem::run(2, || {
///     DefaultStackfulOnlyTaskSystem::block_on(async {
///         cmpth::future::yield_now().await;
///     });
/// });
/// ```
pub async fn yield_now() {
    let mut yielded = false;
    poll_fn(|cx| {
        if yielded {
            Poll::Ready(())
        } else {
            yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    })
    .await
}
