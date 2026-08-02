//! [`DualResumable`] — the dual (ULT-or-async) wait-slot from
//! `docs/sync-async-unification.md`.

use crate::resumable::stackful::worker::StackfulWorker;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Waker};

use crate::traits::{Resumable, StackfulResumable, StacklessResumable};
use crate::resumable::common::desc::{SuspendedTaskToken, TaskDescCore};
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::stackless::desc::AsyncTaskDesc;
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::common::worker::{LocalQueue, UltWorker, Worker};
use crate::resumable::stackful::worker::ContextSwitcher;

const EMPTY: usize = 0;
const ASYNC_TAG: usize = 1;

/// Dual wait slot: holds zero or one waiter, which may be a real ULT
/// continuation *or* a registered async [`Waker`] — chosen
/// per wait attempt by whichever entry point (sync or async) the caller
/// used. Internally a single tagged word (bit 0 = async), matching
/// `cmpth-rs`'s own existing "task" vocabulary (`TaskDesc::TaskResult`,
/// `UltWorker::cur_task` already mean "whichever kind is running").
///
/// `enter`/`swap` (via [`StackfulResumable`]) fall back to a plain wake when
/// the slot turns out to hold an async waiter — a real context jump is only
/// possible into a genuine continuation. `wait_with`/`register` never need
/// this: they only ever *write* a fresh registration into what must already
/// be an empty slot, so there's no ambiguity about prior content.
pub struct DualResumable<S: StackfulSchedulerSystem> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {
    state: AtomicUsize,
    _marker: PhantomData<S>,
}

unsafe impl<S: StackfulSchedulerSystem> Send for DualResumable<S> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {}
unsafe impl<S: StackfulSchedulerSystem> Sync for DualResumable<S> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {}

impl<S: StackfulSchedulerSystem> Default for DualResumable<S> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {
    fn default() -> Self {
        DualResumable { state: AtomicUsize::new(EMPTY), _marker: PhantomData }
    }
}

/// Panics if the caller is not currently running as a real, dedicated ULT
/// stack — i.e. if called (incorrectly) from inside `run_async_poll`, which
/// runs as a plain call on the worker's own shared dispatch-loop stack and
/// therefore never has a `cur_task` other than the worker's `root_desc`.
/// See `docs/sync-async-unification.md` for why this replaces an explicit
/// capability-token parameter: `cur_task` already carries exactly this
/// information, correctly maintained by the context-switch shims.
fn assert_on_real_ult<S: StackfulSchedulerSystem>(wk: &UltWorker<S>)
where
    S::Desc: StackfulTaskDesc,
{
    let is_root = wk.cur_task_ref().is_root();
    assert!(
        !is_root,
        "cmpth: StackfulResumable operation called outside a real ULT \
         (e.g. from inside spawn_async's poll — use StacklessResumable instead)"
    );
}

impl<S: StackfulSchedulerSystem> DualResumable<S> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {
    /// Wake whatever `v` (a raw slot value already taken via
    /// `state.swap(EMPTY, ..)`) represents: push a real ULT continuation to
    /// the local deque, or wake a boxed [`Waker`]. `v == EMPTY` is a no-op.
    /// Shared by `notify()` and by `enter`/`swap`'s fallback when the slot
    /// didn't hold a real continuation to switch into.
    fn wake_raw(v: usize) {
        if v == EMPTY {
            return;
        }
        if v & ASYNC_TAG != 0 {
            let waker_ptr = (v & !ASYNC_TAG) as *mut Waker;
            let w = unsafe { Box::from_raw(waker_ptr) };
            w.wake();
        } else {
            let wk = UltWorker::<S>::current()
                .expect("cmpth: DualResumable wake called outside a worker");
            wk.push_local_top(SuspendedTaskToken(v as *mut S::Desc));
        }
    }
}

impl<S: StackfulSchedulerSystem> Resumable<S> for DualResumable<S> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {
    fn is_set(&self) -> bool {
        self.state.load(Ordering::Acquire) != EMPTY
    }

    fn notify(&self) {
        let v = self.state.swap(EMPTY, Ordering::AcqRel);
        Self::wake_raw(v);
    }
}

impl<S: StackfulSchedulerSystem> StackfulResumable<S> for DualResumable<S> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {
    fn wait_with<F: FnOnce()>(&self, f: F) {
        let wk = UltWorker::<S>::current()
            .expect("cmpth: DualResumable::wait_with called outside a worker");
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
            .expect("cmpth: DualResumable::wait_with_cond called outside a worker");
        assert_on_real_ult(wk);
        let slot = &self.state as *const AtomicUsize;
        wk.cond_suspend_to_sched(move |_wk, prev| {
            unsafe {
                (*slot).store(prev.take().unwrap().into_raw() as usize, Ordering::Release)
            };
            if !f() {
                let v = unsafe { (*slot).swap(EMPTY, Ordering::Acquire) };
                debug_assert_ne!(v, EMPTY);
                *prev = Some(SuspendedTaskToken(v as *mut S::Desc));
            }
        });
    }

    fn enter(&self) {
        let wk = UltWorker::<S>::current()
            .expect("cmpth: DualResumable::enter called outside a worker");
        assert_on_real_ult(wk);
        let v = self.state.swap(EMPTY, Ordering::AcqRel);
        if v != EMPTY && v & ASYNC_TAG == 0 {
            let c = SuspendedTaskToken(v as *mut S::Desc);
            wk.suspend_to_cont(c, |wk, prev| wk.push_local_top(prev));
        } else {
            // Not a real continuation — no context jump is possible here,
            // so fall back to a plain wake instead.
            Self::wake_raw(v);
        }
    }

    fn swap(&self, next: &Self) {
        debug_assert!(!self.is_set(), "DualResumable::swap: self must be empty");
        let wk = UltWorker::<S>::current()
            .expect("cmpth: DualResumable::swap called outside a worker");
        assert_on_real_ult(wk);
        let v = next.state.swap(EMPTY, Ordering::AcqRel);
        if v != EMPTY && v & ASYNC_TAG == 0 {
            let c = SuspendedTaskToken(v as *mut S::Desc);
            let slot = &self.state as *const AtomicUsize;
            wk.suspend_to_cont(c, move |_wk, prev| {
                unsafe { (*slot).store(prev.into_raw() as usize, Ordering::Release) };
            });
        } else {
            // Not a real continuation — fall back to a plain wake; `self`
            // never becomes parked since no switch happens.
            Self::wake_raw(v);
        }
    }
}

impl<S: StackfulSchedulerSystem> StacklessResumable<S> for DualResumable<S> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {
    fn register(&self, cx: &mut Context<'_>) {
        let boxed = Box::new(cx.waker().clone());
        let ptr = Box::into_raw(boxed) as usize | ASYNC_TAG;
        let old = self.state.swap(ptr, Ordering::AcqRel);
        debug_assert_eq!(old, EMPTY, "DualResumable::register called on an already-set slot");
    }
}
