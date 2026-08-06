use crate::traits::system::scoped::ScopedStackfulTaskSystem;
use crate::traits::system::stackful::ThreadSystem;

/// Everything a "complete" stackful system offers: `spawn`/`join` (via
/// `ThreadSystem`) *and* `run`/`parallel_call` (via
/// `ScopedStackfulTaskSystem`). An empty bundle — no methods of its own —
/// blanket-derived for any `S: ScopedStackfulTaskSystem + ThreadSystem`
/// (see [`resumable::stackful::system`](crate::resumable::stackful::system)
/// for both blankets), never implemented by hand. Kept as its own trait
/// (rather than just writing `S: ScopedStackfulTaskSystem + ThreadSystem`
/// at every call site) since it may grow members of its own later.
///
/// There is no `DefaultDualTaskSystem` trait: a concrete system implementing both
/// this and [`StacklessTaskSystem`](crate::traits::system::stackless::StacklessTaskSystem)
/// simply *is* dual, no separate marker needed.
pub trait StackfulTaskSystem: ScopedStackfulTaskSystem + ThreadSystem {}
