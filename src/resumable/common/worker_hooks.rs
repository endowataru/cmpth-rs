//! Worker hooks ([`WorkerHooks`]): [`NoHooks`] (the default),
//! [`PinWorkers`] and [`BackgroundWorkers`].

use std::cell::Cell;

pub use crate::traits::component::worker_hooks::WorkerHooks;

/// One worker's hook state: `Some` from `start` until `stop`, since a
/// worker can enter its loop more than once.
pub struct HooksSlot<H: WorkerHooks> {
    state: Cell<Option<H::State>>,
}

impl<H: WorkerHooks> Default for HooksSlot<H> {
    fn default() -> Self {
        HooksSlot { state: Cell::new(None) }
    }
}

impl<H: WorkerHooks> HooksSlot<H> {
    /// Run `hooks.start` unless this worker already did.
    #[inline(always)]
    pub(crate) fn start(&self, hooks: &H, index: usize, num: usize) {
        let st = match self.state.take() {
            Some(st) => st,
            None => hooks.start(index, num),
        };
        self.state.set(Some(st));
    }

    /// Run `hooks.stop` if this worker started.
    #[inline(always)]
    pub(crate) fn stop(&self, hooks: &H) {
        if let Some(st) = self.state.take() {
            hooks.stop(st);
        }
    }
}

/// Does nothing.  The default.
#[derive(Default)]
pub struct NoHooks;

impl WorkerHooks for NoHooks {
    type State = ();

    #[inline(always)]
    fn start(&self, _index: usize, _num: usize) {}
}

/// Pins worker `i` to CPU `i` modulo the CPUs this process may use
/// (Linux), and restores the thread's mask when it stops.  Elsewhere it
/// does nothing: macOS has no CPU pinning (its affinity tags are hints
/// for grouping threads, not placement).
#[derive(Default)]
pub struct PinWorkers;

impl WorkerHooks for PinWorkers {
    type State = sys::Saved;

    fn start(&self, index: usize, _num: usize) -> sys::Saved {
        sys::pin(index)
    }

    fn stop(&self, saved: sys::Saved) {
        sys::unpin(saved);
    }
}

/// Runs every worker at macOS's background QoS class and restores the
/// thread's class when it stops: the workers move to efficiency cores at
/// a low clock (fib on 12 workers: about a ninth of the energy for 5.5
/// times the time).  For a pool whose work is not urgent.  Elsewhere it
/// does nothing.
#[derive(Default)]
pub struct BackgroundWorkers;

impl WorkerHooks for BackgroundWorkers {
    type State = qos::Saved;

    fn start(&self, _index: usize, _num: usize) -> qos::Saved {
        qos::lower()
    }

    fn stop(&self, saved: qos::Saved) {
        qos::restore(saved);
    }
}

#[cfg(target_os = "linux")]
mod sys {
    use std::mem::{size_of, zeroed};

    /// The thread's mask before pinning; `None` if it was not pinned.
    pub type Saved = Option<libc::cpu_set_t>;

    pub(super) fn pin(index: usize) -> Saved {
        unsafe {
            let mut old: libc::cpu_set_t = zeroed();
            if libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut old) != 0 {
                return None;
            }
            let allowed: Vec<usize> = (0..libc::CPU_SETSIZE as usize).filter(|&c| libc::CPU_ISSET(c, &old)).collect();
            if allowed.is_empty() {
                return None;
            }
            let mut set: libc::cpu_set_t = zeroed();
            libc::CPU_SET(allowed[index % allowed.len()], &mut set);
            if libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set) != 0 {
                return None;
            }
            Some(old)
        }
    }

    pub(super) fn unpin(saved: Saved) {
        if let Some(old) = saved {
            unsafe { libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &old) };
        }
    }

    #[cfg(test)]
    pub(super) fn current_cpus() -> usize {
        unsafe {
            let mut s: libc::cpu_set_t = zeroed();
            libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut s);
            libc::CPU_COUNT(&s) as usize
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod sys {
    /// Nothing to restore.
    pub type Saved = ();

    pub(super) fn pin(_index: usize) {}

    pub(super) fn unpin(_saved: ()) {}
}

#[cfg(target_os = "macos")]
mod qos {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos: u32, rel: i32) -> i32;
        fn pthread_get_qos_class_np(t: libc::pthread_t, qos: *mut u32, rel: *mut i32) -> i32;
    }

    /// `QOS_CLASS_BACKGROUND` and `QOS_CLASS_DEFAULT` (`qos_class_t`).
    const BACKGROUND: u32 = 0x09;
    const DEFAULT: u32 = 0x15;

    /// The thread's class before lowering it.
    pub type Saved = u32;

    pub(super) fn lower() -> Saved {
        let (mut q, mut r) = (0u32, 0i32);
        unsafe { pthread_get_qos_class_np(libc::pthread_self(), &mut q, &mut r) };
        unsafe { pthread_set_qos_class_self_np(BACKGROUND, 0) };
        // A thread with no class runs as the default one.
        if q == 0 { DEFAULT } else { q }
    }

    pub(super) fn restore(saved: Saved) {
        unsafe { pthread_set_qos_class_self_np(saved, 0) };
    }
}

#[cfg(not(target_os = "macos"))]
mod qos {
    /// Nothing to restore.
    pub type Saved = ();

    pub(super) fn lower() {}

    pub(super) fn restore(_saved: ()) {}
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn pin_and_restore() {
        let before = sys::current_cpus();
        let saved = PinWorkers.start(0, 1);
        if saved.is_some() {
            assert_eq!(sys::current_cpus(), 1);
        }
        PinWorkers.stop(saved);
        assert_eq!(sys::current_cpus(), before);
    }
}
