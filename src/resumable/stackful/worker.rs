//! Stackful worker extension traits ([`ContextSwitcher`]/
//! [`StackfulLocalQueue`]/[`StackfulWorker`]),
//! `StackfulSchedulerSystem::pop_or_root`'s stackful-only body, the
//! [`RunnableItem`]/[`ReclaimableDesc`] impls for [`StackfulOnlyTaskDesc`],
//! and the `extern "C"` context-switch shims. See
//! [`common::worker`](crate::resumable::common::worker) for the base
//! traits and [`UltWorker<S>`](crate::resumable::common::worker::UltWorker) itself.

use std::mem::ManuallyDrop;
use std::ptr;

use crate::traits::stackful::{CondTransfer, Context, ContextPolicy, Transfer};
use crate::resumable::common::deque::WorkerRunQueue;
use crate::resumable::common::worker::{LocalQueue, TaskPool, UltWorker, WorkerOps};
use crate::resumable::common::system::{ReclaimableDesc, RunnableItem, WorkerSystem};
use crate::resumable::stackful::system::StackfulWorkerSystem;
use crate::resumable::common::desc::{RunningTaskToken, SuspendedTaskToken, TaskDescCore};
use crate::interchange::Transferred;
use crate::resumable::stackful::desc::{StackfulOnlyTaskDesc, StackfulTaskDesc};

// ---------------------------------------------------------------------------
// ContextSwitcher (stackful-only)
// ---------------------------------------------------------------------------

/// Raw context-switch operations at the worker level.
///
/// Every method executes a callback **on the destination stack**, after the
/// current context is fully saved.  Publishing the suspended continuation from
/// inside the callback is therefore inherently race-free; no "saving in
/// progress" flags or spin-wait handshakes are needed anywhere.
///
/// Only implementable when `S: StackfulWorkerSystem` (needs `S::Ctx`) — a
/// stackless-only system has no context-switch policy to name. Worker-layer
/// (`S: StackfulWorkerSystem`), not gated on dispatch (`SchedulerSystem`):
/// switching contexts never touches `RunnableItem`/`ReclaimableDesc`. The
/// concrete impl below (for [`UltWorker<S>`]) needs no `Worker` pin at all —
/// every method is `Self`-typed (`Self` is already the concrete `UltWorker<S>`
/// because that's the impl's own target), so `S::Worker` is never named.
pub trait ContextSwitcher<S: StackfulWorkerSystem>: Sized
where
    S::Desc: StackfulTaskDesc,
{
    /// Save the current task's context, switch to `next`, run `f(wk, prev)`
    /// on that stack where `prev` is the just-saved continuation, and return
    /// when the current task is later resumed.
    fn suspend_to_cont<F>(&self, next: SuspendedTaskToken<S::Desc>, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedTaskToken<S::Desc>);

    /// Like [`suspend_to_cont`](Self::suspend_to_cont), but `f` may cancel
    /// the switch.  `f` receives `&mut Option<SuspendedTaskToken<S::Desc>>` holding
    /// the current task's continuation; consuming it (`Option::take`) commits
    /// the switch, leaving it in place cancels it and resumes the caller.
    fn cond_suspend_to_cont<F>(&self, next: &mut Option<SuspendedTaskToken<S::Desc>>, f: F) -> &Self
    where
        F: FnOnce(&Self, &mut Option<SuspendedTaskToken<S::Desc>>);

    /// Save the current context, switch to a **fresh** stack at `stack_top`,
    /// run `f(wk, prev)` there.  Used for child-first fork; `f` must never
    /// return.
    ///
    /// # Safety
    /// `next` must be a freshly allocated descriptor that has never been
    /// wrapped in a token before, exclusively owned by the caller.
    unsafe fn suspend_to_new<F>(&self, stack_top: *mut u8, next: *mut S::Desc, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedTaskToken<S::Desc>);

    /// Abandon (do not save) the current context and switch to `next`.
    fn exit_to_cont<F>(&self, next: SuspendedTaskToken<S::Desc>, f: F) -> !
    where
        F: FnOnce(&Self);
}

// ---------------------------------------------------------------------------
// StackfulLocalQueue (stackful-only)
// ---------------------------------------------------------------------------

/// Root-continuation management: only meaningful when there is a real
/// scheduler-loop stack a suspending ULT can fall back into. Worker-layer
/// (`S: StackfulWorkerSystem`), same reasoning as [`ContextSwitcher`].
pub trait StackfulLocalQueue<S: StackfulWorkerSystem>: LocalQueue<S>
where
    S::Desc: StackfulTaskDesc,
{
    /// Pop the next runnable continuation: local deque first, then the root
    /// (scheduler-loop) continuation. Forwards to
    /// [`StackfulWorkerSystem::pop_or_root`] — see that method for why the
    /// dispatch body lives on the system trait, not here.
    fn pop_or_root(&self) -> SuspendedTaskToken<S::Desc>;

    /// Store the scheduler-loop context as the root continuation.
    fn set_root_cont(&self, c: SuspendedTaskToken<S::Desc>);
}

// ---------------------------------------------------------------------------
// StackfulWorker (stackful-only)
// ---------------------------------------------------------------------------

/// Scheduler-level operations that only make sense with a real, switchable
/// stack: suspending the calling ULT and resuming whatever's next.
pub trait StackfulWorker<S: StackfulWorkerSystem>:
    WorkerOps<S> + ContextSwitcher<S> + StackfulLocalQueue<S>
where
    S::Desc: StackfulTaskDesc,
{
    /// Suspend to the next continuation from the local deque / root.
    fn suspend_to_sched<F>(&self, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedTaskToken<S::Desc>),
    {
        let next = self.pop_or_root();
        self.suspend_to_cont(next, f)
    }

    /// Conditionally suspend to the scheduler.  On cancellation the popped
    /// continuation is returned to its source (deque top or root slot).
    fn cond_suspend_to_sched<F>(&self, f: F) -> &Self
    where
        F: FnOnce(&Self, &mut Option<SuspendedTaskToken<S::Desc>>),
    {
        let mut next = Some(self.pop_or_root());
        let wk = self.cond_suspend_to_cont(&mut next, f);
        if let Some(c) = next.take() {
            if c.is_root() {
                wk.set_root_cont(c);
            } else {
                wk.push(c.into());
            }
        }
        wk
    }

    /// Terminate the current task and switch to the scheduler.
    fn exit_to_sched<F>(&self, f: F) -> !
    where
        F: FnOnce(&Self),
    {
        let next = self.pop_or_root();
        self.exit_to_cont(next, f)
    }

    /// Cooperative yield: defer so other tasks already queued run first.
    fn yield_now(&self) -> &Self {
        self.suspend_to_sched(|wk, prev| wk.defer(prev.into()))
    }
}

// ---------------------------------------------------------------------------
// StackfulWorkerSystem::pop_or_root's stackful-only body
// ---------------------------------------------------------------------------

/// `pop_or_root` body for stackful-only systems: every popped item is a
/// real, switchable continuation, so no requeue check is needed.
///
/// Lowest rung: plain [`WorkerSystem`] — `wk.deque.try_pop()` returns
/// `S::SuspendedToken`, converted to this function's
/// `SuspendedTaskToken<S::Desc>` return type via the `Into` bound on
/// [`WorkerSystem::SuspendedToken`] rather than an equality pin;
/// `wk.take_root_cont()` is a plain `WorkerSystem`-level accessor, already
/// implied. Does *not* need `StackfulWorkerSystem` (no context switch
/// happens here, so `S::Ctx` is never named) or `S::Desc: StackfulTaskDesc`.
pub fn pop_or_root_stackful<S>(wk: &UltWorker<S>) -> SuspendedTaskToken<S::Desc>
where
    S: WorkerSystem,
{
    if let Some(c) = wk.deque.try_pop() {
        return c.into();
    }
    wk.take_root_cont()
}

// ---------------------------------------------------------------------------
// RunnableItem / ReclaimableDesc for StackfulOnlyTaskDesc
// ---------------------------------------------------------------------------

/// `cont` is always a real ULT continuation (no `poll_fn` tag ever gets
/// set, since `spawn_async` isn't reachable when `S::Desc` isn't
/// `AsyncTaskDesc`), so this always performs a real context switch — no
/// runtime check.
///
/// Bound: `StackfulWorkerSystem + WorkerSystem<Desc = StackfulOnlyTaskDesc<S>>`
/// plus `S::Worker: ContextSwitcher<S> + StackfulLocalQueue<S>` — strictly
/// below `SchedulerSystem`, and no identity pin at all: `wk.suspend_to_cont`/
/// `wk.set_root_cont` only need those two *capabilities* on `S::Worker`, not
/// `S::Worker` to literally be `UltWorker<S>` — the only implementer today
/// happens to be `UltWorker<S>`, but the bound doesn't have to say so, and
/// `S::SuspendedToken` is never named in this impl's body either. Neither
/// `SchedulerSystem` nor any of its folds is needed: `SchedulerSystem` itself
/// becomes derivable for `S` only *after* this impl (plus the matching
/// `ReclaimableDesc` impl below) exist, not before.
///
/// `Desc` is pinned via `WorkerSystem<Desc = ...>` directly, not restated on
/// `StackfulWorkerSystem` — `StackfulWorkerSystem` no longer carries any
/// bound on `Desc` at all (see that trait's doc comment: nesting one there
/// broke unrelated obligations on the same concrete descriptor, such as
/// `DualTaskDesc`'s `HasPollFn`, once pinned), so there is nothing left to
/// pin it *against*.
impl<S: StackfulWorkerSystem + WorkerSystem<Desc = StackfulOnlyTaskDesc<S>>>
    RunnableItem<S> for SuspendedTaskToken<StackfulOnlyTaskDesc<S>>
where
    S::Worker: ContextSwitcher<S> + StackfulLocalQueue<S>,
{
    fn run_on(self, wk: &S::Worker) {
        let wk2 = wk.suspend_to_cont(self, |wk, prev| wk.set_root_cont(prev));
        debug_assert!(std::ptr::eq(wk2 as *const S::Worker, wk as *const S::Worker));
    }
}

/// Every descriptor came from the pool (there is no `spawn_async`
/// allocation path to bypass it), so always return it there.
///
/// Bound: plain [`WorkerSystem`], `Desc` pinned directly (not via
/// `DescScheduler`) — `wk.free_task` only needs `TaskPool<S>`, reachable
/// through `S::Worker: WorkerOps<S>` alone, so this needs neither `Worker =
/// UltWorker<S>` nor any fold of `SchedulerSystem`.
impl<S: WorkerSystem<Desc = StackfulOnlyTaskDesc<S>>> ReclaimableDesc<S> for StackfulOnlyTaskDesc<S> {
    unsafe fn reclaim(wk: &S::Worker, desc: *mut Self) {
        unsafe { wk.free_task(desc) };
    }
}

// --- StackfulLocalQueue ---

// `S: StackfulSchedulerSystem` was the old bound here; relaxed further to
// plain `StackfulWorkerSystem`: `pop_or_root` below calls `S::pop_or_root`,
// whose default body needs no `Worker`/`SuspendedToken` identity pin (see
// that method's doc comment — it converts via `Into` now), and
// `set_root_cont` only ever touches the concrete `root_cont` field, never
// `S::SuspendedToken`.
impl<S: StackfulWorkerSystem> StackfulLocalQueue<S> for UltWorker<S>
where
    S::Desc: StackfulTaskDesc,
{
    fn pop_or_root(&self) -> SuspendedTaskToken<S::Desc> {
        S::pop_or_root(self)
    }

    fn set_root_cont(&self, cont: SuspendedTaskToken<S::Desc>) {
        debug_assert!(cont.is_root());
        let old = self.root_cont.replace(Some(cont));
        debug_assert!(old.is_none(), "cmpth: overwriting a live root_cont");
    }
}

// --- ContextSwitcher ---

impl<S: StackfulWorkerSystem> ContextSwitcher<S> for UltWorker<S>
where
    S::Desc: StackfulTaskDesc,
{
    fn suspend_to_cont<F>(&self, mut next: SuspendedTaskToken<S::Desc>, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedTaskToken<S::Desc>),
    {
        let next_ctx = Context(next.claim_saved_context());
        debug_assert!(!next_ctx.is_null(), "double-resume in suspend_to_cont (is_root={})", next.is_root());
        let mut payload = SuspendPayload::<S, F> {
            wk: self,
            next: Transferred::new(next),
            f: ManuallyDrop::new(f),
        };
        let tr = unsafe {
            S::Ctx::swap_context(
                next_ctx,
                suspend_shim::<S, F>,
                &mut payload as *mut _ as *mut (),
                ptr::null_mut(),
            )
        };
        unsafe { &*(tr.0 as *const UltWorker<S>) }
    }

    fn cond_suspend_to_cont<F>(&self, next: &mut Option<SuspendedTaskToken<S::Desc>>, f: F) -> &Self
    where
        F: FnOnce(&Self, &mut Option<SuspendedTaskToken<S::Desc>>),
    {
        let next_ctx = Context(
            next.as_ref().expect("cond_suspend without target").peek_saved_context()
        );
        debug_assert!(!next_ctx.is_null());
        let mut payload = CondSuspendPayload::<S, F> {
            wk: self,
            next: next as *mut Option<SuspendedTaskToken<S::Desc>>,
            f: ManuallyDrop::new(f),
        };
        let tr = unsafe {
            S::Ctx::cond_swap_context(
                next_ctx,
                cond_suspend_shim::<S, F>,
                &mut payload as *mut _ as *mut (),
                ptr::null_mut(),
            )
        };
        unsafe { &*(tr.0 as *const UltWorker<S>) }
    }

    unsafe fn suspend_to_new<F>(&self, stack_top: *mut u8, next: *mut S::Desc, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedTaskToken<S::Desc>),
    {
        // SAFETY: `next` is a freshly allocated descriptor (child-first
        // fork) that has never been wrapped in a token before — trivially
        // exclusive. Unlike `suspend_to_cont`/`exit_to_cont`, there's no
        // predecessor token to consume via `Transferred::new`.
        let next = unsafe { Transferred::<SuspendedTaskToken<S::Desc>>::from_raw(next) };
        let mut payload = SuspendPayload::<S, F> { wk: self, next, f: ManuallyDrop::new(f) };
        let tr = unsafe {
            S::Ctx::save_context(
                stack_top,
                suspend_shim::<S, F>,
                &mut payload as *mut _ as *mut (),
                ptr::null_mut(),
            )
        };
        unsafe { &*(tr.0 as *const UltWorker<S>) }
    }

    fn exit_to_cont<F>(&self, mut next: SuspendedTaskToken<S::Desc>, f: F) -> !
    where
        F: FnOnce(&Self),
    {
        let next_ctx = Context(next.claim_saved_context());
        debug_assert!(!next_ctx.is_null(), "double-resume in exit_to_cont (is_root={})", next.is_root());
        let mut payload = ExitPayload::<S, F> {
            wk: self,
            next: Transferred::new(next),
            f: ManuallyDrop::new(f),
        };
        unsafe {
            S::Ctx::restore_context(
                next_ctx,
                exit_shim::<S, F>,
                &mut payload as *mut _ as *mut (),
                ptr::null_mut(),
            )
        }
    }
}

// --- StackfulWorker ---

impl<S: StackfulWorkerSystem, W: WorkerOps<S> + ContextSwitcher<S> + StackfulLocalQueue<S>> StackfulWorker<S> for W
where
    S::Desc: StackfulTaskDesc,
{
}

// ---------------------------------------------------------------------------
// Shims: extern "C" callbacks handed to the context-switch layer.
//
// Each shim runs on the destination stack.  Read everything out of the
// payload (which lives on the now-frozen previous stack) *before* doing
// anything that could allow the previous context to resume.
// ---------------------------------------------------------------------------

struct SuspendPayload<S: StackfulWorkerSystem, F>
where
    S::Desc: StackfulTaskDesc,
{
    wk: *const UltWorker<S>,
    next: Transferred<SuspendedTaskToken<S::Desc>>,
    f: ManuallyDrop<F>,
}

unsafe extern "C" fn suspend_shim<S, F>(prev: Context, a1: *mut (), _a2: *mut ()) -> Transfer
where
    S: StackfulWorkerSystem,
    S::Desc: StackfulTaskDesc,
    F: FnOnce(&UltWorker<S>, SuspendedTaskToken<S::Desc>),
{
    let (wk, next, f) = unsafe {
        let payload = &mut *(a1 as *mut SuspendPayload<S, F>);
        (&*payload.wk, ptr::read(&payload.next), ManuallyDrop::take(&mut payload.f))
    };
    let mut prev_task = wk.take_cur_task();
    let old = prev_task.publish_saved_context(prev.0);
    debug_assert!(old.is_null(), "suspend over live ctx in suspend_shim (is_root={})", prev_task.as_desc().is_root());
    let next_running = next.into_inner::<RunningTaskToken<S::Desc>>();
    wk.set_cur_task(next_running);
    f(wk, prev_task.into_suspended());
    Transfer(wk as *const UltWorker<S> as *mut ())
}

struct CondSuspendPayload<S: StackfulWorkerSystem, F>
where
    S::Desc: StackfulTaskDesc,
{
    wk: *const UltWorker<S>,
    next: *mut Option<SuspendedTaskToken<S::Desc>>,
    f: ManuallyDrop<F>,
}

unsafe extern "C" fn cond_suspend_shim<S, F>(prev: Context, a1: *mut (), _a2: *mut ()) -> CondTransfer
where
    S: StackfulWorkerSystem,
    S::Desc: StackfulTaskDesc,
    F: FnOnce(&UltWorker<S>, &mut Option<SuspendedTaskToken<S::Desc>>),
{
    let (wk, next_slot, next_cont, f) = unsafe {
        let payload = &mut *(a1 as *mut CondSuspendPayload<S, F>);
        let next_cont = (*payload.next).take().unwrap();
        (&*payload.wk, payload.next, next_cont, ManuallyDrop::take(&mut payload.f))
    };
    let mut prev_task = wk.take_cur_task();
    let prev_desc = prev_task.desc();
    let old = prev_task.publish_saved_context(prev.0);
    debug_assert!(old.is_null(), "suspend over live ctx in cond_suspend_shim (is_root={})", prev_task.as_desc().is_root());

    // Promote + commit immediately: `wk.cur_task()` correctly reflects
    // physical reality (this *is* what's running) for the entire duration
    // of `f` below, and nothing else holds a second, independent handle to
    // the same descriptor at the same time -- unlike the old
    // `Cell<*mut S::Desc>` design, there is no window where `cur_task` and
    // a live `SuspendedTaskToken`/local variable alias the same task while
    // owner-exclusive fields (`ctx`) are mutated through one of them. See
    // `RunningTaskToken`'s doc comment.
    let next_running = next_cont.into_running();
    wk.set_cur_task(next_running);

    let mut prev_cont = Some(prev_task.into_suspended());
    f(wk, &mut prev_cont);

    match prev_cont {
        None => {
            // Committed: `next_running` is already `cur_task` -- just
            // finish publishing it (peek, not take; nothing else needs to
            // claim it right now).
            wk.cur_task_token_mut().clear_saved_context();
            CondTransfer { value: wk as *const UltWorker<S> as *mut (), flag: 1 }
        }
        Some(c) => {
            // Cancelled: take the provisional commit back out, restore
            // `prev` as the running task, hand `next` back to the caller
            // as suspended again.
            debug_assert!(std::ptr::eq(c.desc(), prev_desc));
            let mut c_running = c.into_running();
            c_running.clear_saved_context();
            let next_running = wk.take_cur_task();
            wk.set_cur_task(c_running);
            unsafe { *next_slot = Some(next_running.into_suspended()) };
            CondTransfer { value: wk as *const UltWorker<S> as *mut (), flag: 0 }
        }
    }
}

struct ExitPayload<S: StackfulWorkerSystem, F>
where
    S::Desc: StackfulTaskDesc,
{
    wk: *const UltWorker<S>,
    next: Transferred<SuspendedTaskToken<S::Desc>>,
    f: ManuallyDrop<F>,
}

unsafe extern "C" fn exit_shim<S, F>(a1: *mut (), _a2: *mut ()) -> Transfer
where
    S: StackfulWorkerSystem,
    S::Desc: StackfulTaskDesc,
    F: FnOnce(&UltWorker<S>),
{
    let (wk, next, f) = unsafe {
        let payload = &mut *(a1 as *mut ExitPayload<S, F>);
        (&*payload.wk, ptr::read(&payload.next), ManuallyDrop::take(&mut payload.f))
    };
    // The exiting task's own descriptor isn't being saved anywhere -- `f`
    // is responsible for its cleanup/freeing via the join protocol -- so
    // just take it out of `cur_task` and drop the (zero-cost, no `Drop`
    // impl) `RunningTaskToken` wrapper without doing anything else with it.
    let _ = wk.take_cur_task();
    let next_running = next.into_inner::<RunningTaskToken<S::Desc>>();
    wk.set_cur_task(next_running);
    f(wk);
    Transfer(wk as *const UltWorker<S> as *mut ())
}
