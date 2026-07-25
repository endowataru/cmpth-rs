//! An independent, rayon-`join`-like scheduler family:
//! [`StackfulParallelInvoke`](crate::traits::StackfulParallelInvoke)/
//! [`StacklessParallelInvoke`](crate::traits::StacklessParallelInvoke),
//! implemented by [`ParallelInvokeSystem`].
//!
//! Named `scoped`, not `parallel_invoke` or `fork_join` — see
//! [`crate::traits::scoped`]'s doc comment for why: the defining property
//! here is that the caller's own continuation is never reified/exposed as
//! stealable work (unlike `spawn`/`spawn_async`), the same "nothing
//! spawned here outlives this call" property `std::thread::scope` names
//! itself after, just restricted to exactly two branches.
//!
//! Deliberately *not* built on [`crate::ult`]'s `SchedulerSystem`/
//! `UltWorker` machinery: a `parallel_invoke` branch is represented as a
//! plain value on the caller's own native stack frame ([`job::JobRef`]),
//! with a single-purpose completion latch, not a separately allocated,
//! pooled task descriptor with a general join-protocol. That's what makes
//! the common (unstolen) path cheap — see
//! `docs/stackless-perf-investigation.md`'s measurements of the original
//! `fork_join` prototype this replaces (~6-7x faster than
//! `spawn`/`spawn_async` on `fib`, because only the handful of calls that
//! actually get stolen ever pay for deque/latch/help-first machinery at
//! all).
//!
//! Two independent engines share [`job::JobRef`]'s stack-resident,
//! type-erased job representation:
//! - [`sync_engine`] — OS threads, blocking `parallel_invoke`, mirrors the
//!   original `fork_join.rs` almost exactly.
//! - [`async_engine`] — OS threads that poll [`Future`](std::future::Future)
//!   bodies instead of calling plain closures; `parallel_invoke` itself
//!   returns a future that only blocks its *own* worker thread while
//!   driving the un-stolen fast path (same "pay only when it's real"
//!   property as the sync engine); waiting on a genuinely stolen branch
//!   registers a waker instead of busy-spinning.

mod async_engine;
mod job;
mod sync_engine;
mod system;

pub use system::ParallelInvokeSystem;
