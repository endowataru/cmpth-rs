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
//! nothing.  Swap the implementation via [`crate::WorkerSystem::Qos`].

use std::cell::Cell;

/// One worker's QoS state: the policy's value and whether `start` ran.
#[derive(Default)]
pub struct QosSlot {
    pub(crate) state: Cell<u32>,
    pub(crate) started: Cell<bool>,
}

/// A thread's QoS class.  The values are macOS's `qos_class_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum QosClass {
    UserInteractive = 0x21,
    UserInitiated = 0x19,
    Default = 0x15,
    Utility = 0x11,
    Background = 0x09,
}

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

/// Leaves the threads' QoS to the OS (the default).
#[derive(Default)]
pub struct OsQos;

impl QosPolicy for OsQos {}

/// Workers `0..FAST` keep their QoS; the others run at `background`.
#[derive(Default)]
pub struct SplitQos<const FAST: usize>;

impl<const FAST: usize> QosPolicy for SplitQos<FAST> {
    fn start(&self, index: usize, _num: usize, st: &Cell<u32>) {
        if index >= FAST {
            st.set(sys::get());
            sys::set(QosClass::Background as u32);
        }
    }

    fn stop(&self, st: &Cell<u32>) {
        restore(st);
    }
}

/// Every worker runs at `background`.
#[derive(Default)]
pub struct BackgroundQos;

impl QosPolicy for BackgroundQos {
    fn start(&self, _index: usize, _num: usize, st: &Cell<u32>) {
        st.set(sys::get());
        sys::set(QosClass::Background as u32);
    }

    fn stop(&self, st: &Cell<u32>) {
        restore(st);
    }
}

/// A worker that has found nothing for `ROUNDS` rounds drops to
/// `background` until it finds work: a spinning worker then spins on an
/// efficiency core at a low clock, and the cores that run tasks stay fast.
#[derive(Default)]
pub struct DemoteIdleQos<const ROUNDS: u32 = 256>;

impl<const ROUNDS: u32> QosPolicy for DemoteIdleQos<ROUNDS> {
    const IDLE_ROUNDS: u32 = ROUNDS;

    fn idle(&self, st: &Cell<u32>) {
        st.set(sys::get());
        sys::set(QosClass::Background as u32);
    }

    fn busy(&self, st: &Cell<u32>) {
        restore(st);
    }

    fn stop(&self, st: &Cell<u32>) {
        restore(st);
    }
}

fn restore(st: &Cell<u32>) {
    let q = st.replace(0);
    if q != 0 {
        sys::set(q);
    }
}

#[cfg(target_os = "macos")]
mod sys {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos: u32, rel: i32) -> i32;
        fn pthread_get_qos_class_np(t: libc::pthread_t, qos: *mut u32, rel: *mut i32) -> i32;
    }

    /// The calling thread's class; `Default` when it has none.
    pub(super) fn get() -> u32 {
        let (mut q, mut r) = (0u32, 0i32);
        unsafe { pthread_get_qos_class_np(libc::pthread_self(), &mut q, &mut r) };
        if q == 0 { super::QosClass::Default as u32 } else { q }
    }

    pub(super) fn set(q: u32) {
        unsafe { pthread_set_qos_class_self_np(q, 0) };
    }
}

#[cfg(not(target_os = "macos"))]
mod sys {
    pub(super) fn get() -> u32 {
        0
    }

    pub(super) fn set(_q: u32) {}
}
