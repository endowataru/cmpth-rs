use crate::traits::system::block_on::BlockOnSystem;
use crate::traits::system::scoped::ScopedStackfulTaskSystem;
use crate::traits::system::stackful::SpawnableStackfulTaskSystem;
use crate::traits::system::sync::StackfulSyncSystem;

/// Everything a "complete" stackful system offers: `spawn`/`join` (via
/// `SpawnableStackfulTaskSystem`), `parallel_call` (via `ScopedStackfulTaskSystem`; see
/// also `StackfulInitSystem`/`StackfulBuilder` for `run`/standalone
/// `init`), a `Mutex`/`Barrier` (via `StackfulSyncSystem`), and `block_on`
/// (via `BlockOnSystem`). An empty bundle — no methods of its own —
/// blanket-derived for any `S: ScopedStackfulTaskSystem + SpawnableStackfulTaskSystem +
/// StackfulSyncSystem + BlockOnSystem` (see
/// [`resumable::stackful::system`](crate::resumable::stackful::system) for
/// the blankets), never implemented by hand. Kept as its own trait (rather
/// than just writing out all four bounds at every call site) since it may
/// grow members of its own later.
///
/// There is no `DefaultDualTaskSystem` trait: a concrete system implementing both
/// this and [`StacklessTaskSystem`](crate::traits::system::stackless::StacklessTaskSystem)
/// simply *is* dual, no separate marker needed.
pub trait StackfulTaskSystem:
    SpawnableStackfulTaskSystem + ScopedStackfulTaskSystem + StackfulSyncSystem + BlockOnSystem
{
}
