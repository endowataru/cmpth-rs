//! [`SuspendedTask`] — the dual (ULT-or-async) wait-slot from
//! `docs/sync-async-unification.md`.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Waker};

use crate::traits::{Resumable, StackfulResumable, StacklessResumable};
use crate::ult::desc::{BasicTaskDesc, SuspendedUlt, TaskDesc};
use crate::ult::system::{AsyncWorkerSystem, UltSystem};
use crate::ult::worker::{ContextSwitcher, LocalQueue, UltWorker, Worker};

const EMPTY: usize = 0;
const ASYNC_TAG: usize = 1;

/// Dual wait slot: holds zero or one waiter, which may be a real ULT
/// continuation *or* a registered async [`Waker`] — chosen
/// per wait attempt by whichever entry point (sync or async) the caller
/// used. Internally a single tagged word (bit 0 = async), matching
/// `cmpth-rs`'s own existing "task" vocabulary (`TaskDesc::TaskResult`,
/// `UltWorker::cur_task` already mean "whichever kind is running").
///
/// `enter`/`swap` (via [`StackfulResumable`]) return `false` without acting when
/// the slot turns out to hold an async waiter — a real context jump is only
/// possible into a genuine continuation. `wait_with`/`register` never need
/// this: they only ever *write* a fresh registration into what must already
/// be an empty slot, so there's no ambiguity about prior content.
pub struct SuspendedTask<S: UltSystem + AsyncWorkerSystem> {
    state: AtomicUsize,
    _marker: PhantomData<S>,
}

unsafe impl<S: UltSystem + AsyncWorkerSystem> Send for SuspendedTask<S> {}
unsafe impl<S: UltSystem + AsyncWorkerSystem> Sync for SuspendedTask<S> {}

impl<S: UltSystem + AsyncWorkerSystem> Default for SuspendedTask<S> {
    fn default() -> Self {
        SuspendedTask { state: AtomicUsize::new(EMPTY), _marker: PhantomData }
    }
}

/// Panics if the caller is not currently running as a real, dedicated ULT
/// stack — i.e. if called (incorrectly) from inside `run_async_poll`, which
/// runs as a plain call on the worker's own shared dispatch-loop stack and
/// therefore never has a `cur_task` other than the worker's `root_desc`.
/// See `docs/sync-async-unification.md` for why this replaces an explicit
/// capability-token parameter: `cur_task` already carries exactly this
/// information, correctly maintained by the context-switch shims.
fn assert_on_real_ult<S: UltSystem>(wk: &UltWorker<S>) {
    let is_root = unsafe { (*wk.cur_task.get()).is_root() };
    assert!(
        !is_root,
        "cmpth: StackfulResumable operation called outside a real ULT \
         (e.g. from inside spawn_async's poll — use StacklessResumable instead)"
    );
}

impl<S: UltSystem + AsyncWorkerSystem> Resumable<S> for SuspendedTask<S> {
    fn is_set(&self) -> bool {
        self.state.load(Ordering::Acquire) != EMPTY
    }

    fn notify(&self) {
        let v = self.state.swap(EMPTY, Ordering::AcqRel);
        if v == EMPTY {
            return;
        }
        if v & ASYNC_TAG != 0 {
            let waker_ptr = (v & !ASYNC_TAG) as *mut Waker;
            let w = unsafe { Box::from_raw(waker_ptr) };
            w.wake();
        } else {
            let wk = UltWorker::<S>::current()
                .expect("cmpth: SuspendedTask::notify called outside a worker");
            wk.push_local_top(SuspendedUlt(v as *mut BasicTaskDesc));
        }
    }
}

impl<S: UltSystem + AsyncWorkerSystem> StackfulResumable<S> for SuspendedTask<S> {
    fn wait_with<F: FnOnce()>(&self, f: F) {
        let wk = UltWorker::<S>::current()
            .expect("cmpth: SuspendedTask::wait_with called outside a worker");
        assert_on_real_ult(wk);
        let slot = &self.state as *const AtomicUsize;
        wk.suspend_to_sched(move |_wk, prev| {
            // Release: publishes the context saved just before this callback.
            unsafe { (*slot).store(prev.into_raw() as usize, Ordering::Release) };
            f();
        });
    }

    fn wait_with_cond<F: FnOnce() -> bool>(&self, f: F) {
        let wk = UltWorker::<S>::current()
            .expect("cmpth: SuspendedTask::wait_with_cond called outside a worker");
        assert_on_real_ult(wk);
        let slot = &self.state as *const AtomicUsize;
        wk.cond_suspend_to_sched(move |_wk, prev| {
            unsafe {
                (*slot).store(prev.take().unwrap().into_raw() as usize, Ordering::Release)
            };
            if !f() {
                let v = unsafe { (*slot).swap(EMPTY, Ordering::Acquire) };
                debug_assert_ne!(v, EMPTY);
                *prev = Some(SuspendedUlt(v as *mut BasicTaskDesc));
            }
        });
    }

    fn enter(&self) -> bool {
        let wk = UltWorker::<S>::current()
            .expect("cmpth: SuspendedTask::enter called outside a worker");
        assert_on_real_ult(wk);
        let v = self.state.swap(EMPTY, Ordering::AcqRel);
        if v == EMPTY {
            return false;
        }
        if v & ASYNC_TAG != 0 {
            // Not a real continuation — put the registration back untouched
            // and let the caller fall back to `notify()`.
            self.state.store(v, Ordering::Release);
            return false;
        }
        let c = SuspendedUlt(v as *mut BasicTaskDesc);
        wk.suspend_to_cont(c, |wk, prev| wk.push_local_top(prev));
        true
    }

    fn swap(&self, next: &Self) -> bool {
        debug_assert!(!self.is_set(), "SuspendedTask::swap: self must be empty");
        let wk = UltWorker::<S>::current()
            .expect("cmpth: SuspendedTask::swap called outside a worker");
        assert_on_real_ult(wk);
        let v = next.state.swap(EMPTY, Ordering::AcqRel);
        if v == EMPTY {
            return false;
        }
        if v & ASYNC_TAG != 0 {
            next.state.store(v, Ordering::Release);
            return false;
        }
        let c = SuspendedUlt(v as *mut BasicTaskDesc);
        let slot = &self.state as *const AtomicUsize;
        wk.suspend_to_cont(c, move |_wk, prev| {
            unsafe { (*slot).store(prev.into_raw() as usize, Ordering::Release) };
        });
        true
    }
}

impl<S: UltSystem + AsyncWorkerSystem> StacklessResumable<S> for SuspendedTask<S> {
    fn register(&self, cx: &mut Context<'_>) -> bool {
        let boxed = Box::new(cx.waker().clone());
        let ptr = Box::into_raw(boxed) as usize | ASYNC_TAG;
        let old = self.state.swap(ptr, Ordering::AcqRel);
        debug_assert_eq!(old, EMPTY, "SuspendedTask::register called on an already-set slot");
        true
    }
}
