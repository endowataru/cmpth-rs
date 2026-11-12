//! Worker hooks: what a worker does on its own thread when it starts and
//! when its pool stops — pin itself to a CPU, lower its QoS class, name or
//! register its thread, set up per-thread state a library needs.
//!
//! `start` runs before the worker first looks for work and `stop` after it
//! last did, both on the worker's thread.  Worker 0's thread is the
//! caller's (`run`), so `stop` must put back what `start` changed.
//!
//! A system picks them via `WorkerSystem::Hooks`; implementations:
//! `resumable::common::worker_hooks` (the default, `NoHooks`, compiles to
//! nothing).

/// Per-worker start/stop hooks.  One value per pool (it lives in
/// `PoolCore`).
pub trait WorkerHooks: Default + Send + Sync + 'static {
    /// What `start` hands to `stop`: whatever it needs to undo.  It never
    /// leaves the worker's thread.
    type State;

    /// Worker `index` of `num` starts.
    fn start(&self, index: usize, num: usize) -> Self::State;

    /// The worker whose `start` returned `state` stops.
    #[inline(always)]
    fn stop(&self, state: Self::State) {
        drop(state);
    }
}
