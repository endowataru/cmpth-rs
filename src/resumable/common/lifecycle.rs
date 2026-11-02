//! Lifecycle policy: what a worker does on its own thread when it starts
//! and when its pool stops — pin itself to a CPU, name or register its
//! thread, set up per-thread state a library needs.
//!
//! `start` runs before the worker first looks for work and `stop` after it
//! last did, both on the worker's thread.  Worker 0's thread is the
//! caller's (`run`), so `stop` must put back what `start` changed.  Swap
//! the implementation via [`crate::WorkerSystem::Lifecycle`]; the default
//! [`NoLifecycle`] compiles to nothing.

use std::cell::Cell;

/// One worker's lifecycle state (whatever `start` needs to undo).
#[derive(Default)]
pub struct LifecycleSlot {
    pub(crate) state: Cell<u64>,
}

/// Per-worker start/stop hooks.  One value per pool (it lives in
/// `PoolCore`).
pub trait LifecyclePolicy: Default + Send + Sync + 'static {
    /// Worker `index` of `num` starts; `st` is its own state.
    #[inline(always)]
    fn start(&self, _index: usize, _num: usize, _st: &Cell<u64>) {}

    /// The worker that `start`ed with `st` stops.
    #[inline(always)]
    fn stop(&self, _st: &Cell<u64>) {}
}

/// Does nothing.  The default.
#[derive(Default)]
pub struct NoLifecycle;
impl LifecyclePolicy for NoLifecycle {}

/// Pins worker `i` to CPU `i` modulo the CPUs this process may use
/// (Linux), and restores the thread's mask when it stops.  Elsewhere it
/// does nothing: macOS has no CPU pinning (its affinity tags are hints
/// for grouping threads, not placement).
#[derive(Default)]
pub struct PinWorkers;

impl LifecyclePolicy for PinWorkers {
    fn start(&self, index: usize, _num: usize, st: &Cell<u64>) {
        st.set(sys::pin(index));
    }

    fn stop(&self, st: &Cell<u64>) {
        sys::unpin(st.replace(0));
    }
}

#[cfg(target_os = "linux")]
mod sys {
    use std::mem::{size_of, zeroed};

    /// The previous mask, leaked as a raw `Box` pointer (0: none).
    pub(super) fn pin(index: usize) -> u64 {
        unsafe {
            let mut old: libc::cpu_set_t = zeroed();
            if libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut old) != 0 {
                return 0;
            }
            let allowed: Vec<usize> = (0..libc::CPU_SETSIZE as usize).filter(|&c| libc::CPU_ISSET(c, &old)).collect();
            if allowed.is_empty() {
                return 0;
            }
            let mut set: libc::cpu_set_t = zeroed();
            libc::CPU_SET(allowed[index % allowed.len()], &mut set);
            if libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set) != 0 {
                return 0;
            }
            Box::into_raw(Box::new(old)) as u64
        }
    }

    pub(super) fn unpin(old: u64) {
        if old == 0 {
            return;
        }
        let old = unsafe { Box::from_raw(old as *mut libc::cpu_set_t) };
        unsafe { libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &*old) };
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
    pub(super) fn pin(_index: usize) -> u64 {
        0
    }

    pub(super) fn unpin(_old: u64) {}
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn pin_and_restore() {
        let before = sys::current_cpus();
        let st = Cell::new(0);
        PinWorkers.start(0, 1, &st);
        if st.get() != 0 {
            assert_eq!(sys::current_cpus(), 1);
        }
        PinWorkers.stop(&st);
        assert_eq!(sys::current_cpus(), before);
    }
}
