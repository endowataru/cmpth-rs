//! Stackful-only descriptor operations: a real, switchable saved context.

use std::sync::atomic::AtomicPtr;

use crate::resumable::common::desc::TaskDesc;

/// Descriptor operations needed only by tasks with a real, switchable
/// execution stack (stackful ULTs). A pure-stackless descriptor type would
/// not implement this — there is no saved context to hand off, since
/// `run_async_poll` never does a context switch.
pub trait StackfulTaskDesc: TaskDesc {
    /// Saved context pointer; null while the task is running.
    ///
    /// Written with `Release` by the context-switch shim; claimed with
    /// `Acquire` or `AcqRel` by resumer or waker.
    fn ctx(&self) -> &AtomicPtr<u8>;

    /// Claim this task's saved context before switching into it (`Acquire`
    /// swap-to-null). The caller is expected to `debug_assert` the returned
    /// pointer is non-null (a null result means a double-resume — the exact
    /// diagnostic message differs per call site, so that check stays there).
    fn claim_saved_context(&self) -> *mut u8 {
        self.ctx().swap(std::ptr::null_mut(), std::sync::atomic::Ordering::Acquire)
    }

    /// Look at this task's saved context without consuming it (`Acquire`
    /// load) — used when the caller might not actually commit to switching
    /// (`cond_suspend_to_cont`).
    fn peek_saved_context(&self) -> *mut u8 {
        self.ctx().load(std::sync::atomic::Ordering::Acquire)
    }

    /// Publish a just-saved context (`Release` swap), making this task
    /// resumable. Returns the previous value so the caller can
    /// `debug_assert` it was null (overwriting a live context is a bug).
    fn publish_saved_context(&self, ptr: *mut u8) -> *mut u8 {
        self.ctx().swap(ptr, std::sync::atomic::Ordering::Release)
    }

    /// Initialize the context of a freshly allocated task that has never
    /// been suspended (`Release` store — cheaper than `publish_saved_context`
    /// since there is provably nothing to overwrite, so no swap-and-check
    /// is needed).
    fn init_saved_context(&self, ptr: *mut u8) {
        self.ctx().store(ptr, std::sync::atomic::Ordering::Release);
    }

    /// Clear this task's saved context (`Relaxed` store) when synchronization
    /// is already established by other means — used by `cond_suspend_shim`'s
    /// commit/cancel cleanup, after the ordering-relevant handoff already
    /// happened via the context switch itself.
    fn clear_saved_context(&self) {
        self.ctx().store(std::ptr::null_mut(), std::sync::atomic::Ordering::Relaxed);
    }
}
