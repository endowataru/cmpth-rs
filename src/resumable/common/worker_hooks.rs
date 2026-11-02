//! Worker hooks ([`WorkerHooks`]): [`NoHooks`] (the default) and
//! [`PinWorkers`].

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
