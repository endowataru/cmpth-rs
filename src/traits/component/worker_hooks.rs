//! Worker hooks: what a worker does on its own thread when it starts and
//! when its pool stops — pin itself to a CPU, name or register its thread,
//! set up per-thread state a library needs.
//!
//! `start` runs before the worker first looks for work and `stop` after it
//! last did, both on the worker's thread.  Worker 0's thread is the
//! caller's (`run`), so `stop` must put back what `start` changed.
//!
//! A system picks them via `WorkerSystem::Hooks`; implementations:
//! `resumable::common::worker_hooks` (the default, `NoHooks`, compiles to
//! nothing).

use std::cell::Cell;

/// Per-worker start/stop hooks.  One value per pool (it lives in
/// `PoolCore`).
pub trait WorkerHooks: Default + Send + Sync + 'static {
    /// Worker `index` of `num` starts; `st` is its own state.
    #[inline(always)]
    fn start(&self, _index: usize, _num: usize, _st: &Cell<u64>) {}

    /// The worker that `start`ed with `st` stops.
    #[inline(always)]
    fn stop(&self, _st: &Cell<u64>) {}
}
