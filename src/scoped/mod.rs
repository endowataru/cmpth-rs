//! An independent, rayon-`join`-like scheduler family:
//! [`ScopedStackfulTaskSystem`](crate::traits::ScopedStackfulTaskSystem)/
//! [`ScopedStacklessTaskSystem`](crate::traits::ScopedStacklessTaskSystem),
//! implemented by [`ScopedTaskSystem`].
//!
//! Named `scoped`, not `parallel_call` or `fork_join` — see
//! [`crate::traits::scoped`]'s doc comment for why: the defining property
//! here is that the caller's own continuation is never reified/exposed as
//! stealable work (unlike `spawn`/`spawn_async`), the same "nothing
//! spawned here outlives this call" property `std::thread::scope` names
//! itself after, just restricted to exactly two branches.
//!
//! Shares [`crate::resumable::common::system::WorkerSystem`]'s worker-pool
//! machinery (`WorkerRunQueue`/`CurrentLookup`/`WorkerOps`/`LocalQueue` —
//! see `worker.rs`) with the `resumable` engine, but deliberately *not*
//! [`PoolSystem`](crate::resumable::common::system::PoolSystem)/`UltWorker`:
//! a `parallel_call` branch is represented as a plain value on the caller's
//! own native stack frame (`task::TaskRef`), with a single-purpose
//! completion latch, not a separately allocated, pooled task descriptor
//! with a general join-protocol. That's what makes the common (unstolen)
//! path cheap — see `docs/stackless-perf-investigation.md`'s measurements
//! of the original `fork_join` prototype this replaces (~6-7x faster than
//! `spawn`/`spawn_async` on `fib`, because only the handful of calls that
//! actually get stolen ever pay for deque/latch/help-first machinery at
//! all). Integrating the worker pool without integrating task
//! representation/pooling is exactly what `docs/traits-redesign.md`§11
//! item 9 and `docs/design-vision.md`§4.1 call for.
//!
//! Two independent engines share `task::TaskRef`'s stack-resident,
//! type-erased task representation:
//! - `sync_engine` — OS threads, blocking `parallel_call`, mirrors the
//!   original `fork_join.rs` almost exactly.
//! - `async_engine` — OS threads that poll [`Future`]
//!   bodies instead of calling plain closures; `parallel_call` itself
//!   returns a future that only blocks its *own* worker thread while
//!   driving the un-stolen fast path (same "pay only when it's real"
//!   property as the sync engine); waiting on a genuinely stolen branch
//!   registers a waker instead of busy-spinning.

mod async_engine;
mod sync_engine;
mod system;
mod task;
mod worker;

pub use system::ScopedTaskSystem;
// `SyncInit` is `ScopedTaskSystem`'s `StackfulInitSystem::Init` — a public
// associated type needs an at-least-as-public backing type, so it needs a
// fully public path even though the `sync_engine` module itself (and
// everything else in it) stays crate-private.
pub use sync_engine::SyncInit;
