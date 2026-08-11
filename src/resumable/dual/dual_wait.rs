//! [`DualResumable`] — the dual (ULT-or-async) wait-slot from
//! `docs/sync-async-unification.md`.

use crate::resumable::stackful::worker::StackfulWorker;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::task::{Context, Waker};

use crate::traits::{Resumable, StackfulResumable, StacklessResumable};
use crate::resumable::common::desc::{SuspendedTaskToken, TaskDescCore};
use crate::interchange::{AtomicTaggedSlot, TaggedPtr};
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::stackless::desc::AsyncTaskDesc;
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::common::worker::{DescWorkerOps, LocalQueue, WorkerOps};
use crate::resumable::stackful::worker::ContextSwitcher;

const ASYNC_TAG: usize = 1;

/// Dual wait slot: holds zero or one waiter, which may be a real ULT
/// continuation *or* a registered async [`Waker`] — chosen
/// per wait attempt by whichever entry point (sync or async) the caller
/// used. Internally a single tagged word (bit 0 = async), matching
/// `cmpth-rs`'s own existing "task" vocabulary (`WorkerOps::cur_task`
/// already means "whichever kind is running").
///
/// `enter`/`swap` (via [`StackfulResumable`]) fall back to a plain wake when
/// the slot turns out to hold an async waiter — a real context jump is only
/// possible into a genuine continuation. `wait_with`/`register` never need
/// this: they only ever *write* a fresh registration into what must already
/// be an empty slot, so there's no ambiguity about prior content.
pub struct DualResumable<S: StackfulSchedulerSystem> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {
    state: AtomicTaggedSlot<1>,
    _marker: PhantomData<S>,
}

unsafe impl<S: StackfulSchedulerSystem> Send for DualResumable<S> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {}
unsafe impl<S: StackfulSchedulerSystem> Sync for DualResumable<S> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {}

impl<S: StackfulSchedulerSystem> Default for DualResumable<S> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {
    fn default() -> Self {
        DualResumable { state: AtomicTaggedSlot::empty(), _marker: PhantomData }
    }
}

/// Panics if the caller is not currently running as a real, dedicated ULT
/// stack — i.e. if called (incorrectly) from inside `run_async_poll`, which
/// runs as a plain call on the worker's own shared dispatch-loop stack and
/// therefore never has a `cur_task` other than the worker's `root_desc`.
/// See `docs/sync-async-unification.md` for why this replaces an explicit
/// capability-token parameter: `cur_task` already carries exactly this
/// information, correctly maintained by the context-switch shims.
fn assert_on_real_ult<S: StackfulSchedulerSystem>(wk: &S::Worker)
where
    S::Desc: StackfulTaskDesc,
    S::Worker: DescWorkerOps<S>,
{
    let is_root = wk.cur_task_ref().is_root();
    assert!(
        !is_root,
        "cmpth: StackfulResumable operation called outside a real ULT \
         (e.g. from inside spawn_async's poll — use StacklessResumable instead)"
    );
}

impl<S: StackfulSchedulerSystem> DualResumable<S> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {
    /// Wake whatever `v` (already taken via `state.take(..)`) represents:
    /// push a real ULT continuation to the local deque, or wake a boxed
    /// [`Waker`]. `None` is a no-op. Shared by `notify()` and by
    /// `enter`/`swap`'s fallback when the slot didn't hold a real
    /// continuation to switch into.
    fn wake_raw(v: Option<(usize, TaggedPtr<1>)>) {
        let Some((tag, raw)) = v else { return };
        if tag == ASYNC_TAG {
            // SAFETY: `tag == ASYNC_TAG` means this slot's `publish` was
            // called from `register` with a `Box<Waker>`.
            let w: Box<Waker> = unsafe { raw.into_typed() };
            w.wake();
        } else {
            let wk = S::Worker::current()
                .expect("cmpth: DualResumable wake called outside a worker");
            // SAFETY: `tag == 0` means this slot's `publish` was called
            // from `wait_with`/`wait_with_cond`/`swap` with a real
            // `SuspendedTaskToken`.
            let c: SuspendedTaskToken<S::Desc> = unsafe { raw.into_typed() };
            wk.push(c.into());
        }
    }
}

impl<S: StackfulSchedulerSystem> Resumable<S> for DualResumable<S> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {
    fn is_set(&self) -> bool {
        self.state.is_set(Ordering::Acquire)
    }

    fn notify(&self) {
        let v = self.state.take(Ordering::AcqRel);
        Self::wake_raw(v);
    }
}

impl<S: StackfulSchedulerSystem> StackfulResumable<S> for DualResumable<S>
where
    S::Desc: StackfulTaskDesc + AsyncTaskDesc,
    S::Worker: StackfulWorker<S> + DescWorkerOps<S>,
{
    fn wait_with<F: FnOnce()>(&self, f: F) {
        let wk = S::Worker::current()
            .expect("cmpth: DualResumable::wait_with called outside a worker");
        assert_on_real_ult::<S>(wk);
        let slot = &self.state as *const AtomicTaggedSlot<1>;
        wk.suspend_to_sched(move |_wk, prev| {
            // Release: publishes the context saved just before this
            // callback. SAFETY: `slot` outlives this callback (it's
            // `&self`'s own field, borrowed for as long as `wait_with` is
            // suspended).
            unsafe { (*slot).publish(prev, 0, Ordering::Release) };
            f();
        });
    }

    fn wait_with_cond<F: FnOnce() -> bool>(&self, f: F) {
        let wk = S::Worker::current()
            .expect("cmpth: DualResumable::wait_with_cond called outside a worker");
        assert_on_real_ult::<S>(wk);
        let slot = &self.state as *const AtomicTaggedSlot<1>;
        wk.cond_suspend_to_sched(move |_wk, prev| {
            // SAFETY: `slot` outlives this callback.
            unsafe { (*slot).publish(prev.take().unwrap(), 0, Ordering::Release) };
            if !f() {
                // SAFETY: `slot` outlives this callback; `take` pairs with
                // this same closure's `publish` a few lines up.
                let v = unsafe { (*slot).take(Ordering::Acquire) };
                let (tag, raw) = v.expect("DualResumable: wait_with_cond cancel raced");
                debug_assert_eq!(tag, 0, "cmpth: cond-suspend slot held an async waiter");
                // SAFETY: `tag == 0` means a `SuspendedTaskToken` was
                // published a few lines up by this same closure.
                *prev = Some(unsafe { raw.into_typed() });
            }
        });
    }

    fn enter(&self) {
        let wk = S::Worker::current()
            .expect("cmpth: DualResumable::enter called outside a worker");
        assert_on_real_ult::<S>(wk);
        match self.state.take(Ordering::AcqRel) {
            Some((tag, raw)) if tag != ASYNC_TAG => {
                // SAFETY: `tag != ASYNC_TAG` means a `SuspendedTaskToken`
                // was published by `wait_with`/`wait_with_cond`.
                let c: SuspendedTaskToken<S::Desc> = unsafe { raw.into_typed() };
                wk.suspend_to_cont(c, |wk, prev| wk.push(prev.into()));
            }
            // Not a real continuation (or empty) — no context jump is
            // possible here, so fall back to a plain wake instead.
            v => Self::wake_raw(v),
        }
    }

    fn swap(&self, next: &Self) {
        debug_assert!(!self.is_set(), "DualResumable::swap: self must be empty");
        let wk = S::Worker::current()
            .expect("cmpth: DualResumable::swap called outside a worker");
        assert_on_real_ult::<S>(wk);
        match next.state.take(Ordering::AcqRel) {
            Some((tag, raw)) if tag != ASYNC_TAG => {
                // SAFETY: same provenance as `enter` above.
                let c: SuspendedTaskToken<S::Desc> = unsafe { raw.into_typed() };
                let slot = &self.state as *const AtomicTaggedSlot<1>;
                wk.suspend_to_cont(c, move |_wk, prev| {
                    // SAFETY: `slot` outlives this callback (it's `self`'s
                    // own field, and `self` outlives the suspend/resume it
                    // spans).
                    unsafe { (*slot).publish(prev, 0, Ordering::Release) };
                });
            }
            // Not a real continuation — fall back to a plain wake; `self`
            // never becomes parked since no switch happens.
            v => Self::wake_raw(v),
        }
    }
}

impl<S: StackfulSchedulerSystem> StacklessResumable<S> for DualResumable<S> where S::Desc: StackfulTaskDesc + AsyncTaskDesc {
    fn register(&self, cx: &mut Context<'_>) {
        debug_assert!(!self.state.is_set(Ordering::Relaxed), "DualResumable::register called on an already-set slot");
        let boxed: Box<Waker> = Box::new(cx.waker().clone());
        self.state.publish(boxed, ASYNC_TAG, Ordering::Release);
    }
}
