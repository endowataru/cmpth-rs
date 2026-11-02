//! Worker hooks ([`WorkerHooks`]): [`NoHooks`] (the default) and
//! [`PinWorkers`].

use std::cell::Cell;

pub use crate::traits::component::worker_hooks::WorkerHooks;

/// One worker's hook state (whatever `start` needs to undo).
#[derive(Default)]
pub struct HooksSlot {
    pub(crate) state: Cell<u64>,
}


/// Does nothing.  The default.
#[derive(Default)]
pub struct NoHooks;
impl WorkerHooks for NoHooks {}

/// Pins worker `i` to CPU `i` modulo the CPUs this process may use
/// (Linux), and restores the thread's mask when it stops.  Elsewhere it
/// does nothing: macOS has no CPU pinning (its affinity tags are hints
/// for grouping threads, not placement).
#[derive(Default)]
pub struct PinWorkers;

impl WorkerHooks for PinWorkers {
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
