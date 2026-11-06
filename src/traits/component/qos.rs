//! QoS policy: what a worker tells the OS about how urgent the work on its
//! thread is, so that the OS can choose the core type and the clock.
//!
//! On macOS a thread's QoS class steers placement on performance or
//! efficiency cores and the clock the core runs at; `background` keeps the
//! thread on efficiency cores at a low clock.  The runtime knows what the
//! OS cannot see: whether a worker is running tasks or only looking for
//! them.  Other platforms have no such hint yet; the policies leave the
//! thread unchanged there.
//!
//! The hooks run in the shared idle loop only (never between two tasks a
//! worker runs back to back), so a policy that changes nothing costs
//! nothing.
//!
//! A system picks one via `WorkerSystem::Qos`; implementations:
//! `resumable::common::qos`.

use std::cell::Cell;

/// Per-worker QoS rule.  One value per pool (it lives in `PoolCore`); the
/// per-worker state is the `Cell` each hook receives, owned by the worker.
///
/// # Contract
///
/// * `start` runs on the worker's own thread before it looks for work, and
///   `stop` on the same thread when the pool shuts down; `stop` must put
///   back what `start` changed, since worker 0's thread is the caller's.
/// * `idle` runs when the worker has found nothing for
///   [`IDLE_ROUNDS`](Self::IDLE_ROUNDS) consecutive rounds, and `busy` when
///   it then finds work.  `IDLE_ROUNDS = 0` means `idle`/`busy` are never
///   called.
pub trait QosPolicy: Default + Send + Sync + 'static {
    const IDLE_ROUNDS: u32 = 0;

    fn start(&self, _index: usize, _num: usize, _st: &Cell<u32>) {}

    fn stop(&self, _st: &Cell<u32>) {}

    fn idle(&self, _st: &Cell<u32>) {}

    fn busy(&self, _st: &Cell<u32>) {}
}
