//! Stackful worker extension traits ([`ContextSwitcher`]/
//! [`StackfulLocalQueue`]/[`StackfulWorker`]), the stackful-only/dual
//! dispatch bodies for `SchedulerSystem::execute`/
//! `StackfulSchedulerSystem::pop_or_root`/`SchedulerSystem::free_finished_desc`,
//! and the `extern "C"` context-switch shims. See
//! [`common::worker`](crate::resumable::common::worker) for the base
//! traits and [`UltWorker<S>`](crate::resumable::common::worker::UltWorker) itself.

use std::mem::ManuallyDrop;
use std::ptr;

use crate::traits::stackful::{CondTransfer, Context, ContextPolicy, Transfer};
use crate::resumable::common::deque::WorkerDeque;
use crate::resumable::common::worker::{LocalQueue, TaskPool, UltWorker, Worker};
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::common::desc::{RunningTaskToken, SuspendedTaskToken, TaskDescCore};
use crate::resumable::stackful::desc::StackfulTaskDesc;

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
/// Only implementable when `S: StackfulSchedulerSystem` (needs `S::Ctx`) — a
/// stackless-only system has no context-switch policy to name.
pub trait ContextSwitcher<S: StackfulSchedulerSystem>: Sized
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
    fn suspend_to_new<F>(&self, stack_top: *mut u8, next: *mut S::Desc, f: F) -> &Self
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
/// scheduler-loop stack a suspending ULT can fall back into.
pub trait StackfulLocalQueue<S: StackfulSchedulerSystem>: LocalQueue<S>
where
    S::Desc: StackfulTaskDesc,
{
    /// Pop the next runnable continuation: local deque first, then the root
    /// (scheduler-loop) continuation. Forwards to
    /// [`StackfulSchedulerSystem::pop_or_root`] — see that method for why the
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
pub trait StackfulWorker<S: StackfulSchedulerSystem>:
    Worker<S> + ContextSwitcher<S> + StackfulLocalQueue<S>
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
                wk.push_local_top(c);
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

    /// Cooperative yield: requeue at the FIFO end so other tasks run first.
    fn yield_now(&self) -> &Self {
        self.suspend_to_sched(|wk, prev| wk.push_local_bottom(prev))
    }
}

// ---------------------------------------------------------------------------
// Dispatch bodies for SchedulerSystem::execute / StackfulSchedulerSystem::pop_or_root
//
// Plain functions, not trait defaults directly: each concrete system's
// `impl SchedulerSystem`/`impl StackfulSchedulerSystem` block calls exactly one
// of these from its own `execute`/`pop_or_root` method. No specialization is
// involved — every concrete marker struct (DefaultDualTaskSystem, a
// stackful-only `UltIdentity` implementor, ...) gets exactly one such
// `impl` block, so this is ordinary static dispatch, monomorphized per
// system.
// ---------------------------------------------------------------------------

/// `execute` body for stackful-only systems: `cont` is always a real ULT
/// continuation (no `poll_fn` tag ever gets set, since `spawn_async` isn't
/// reachable when `S::Desc` isn't `AsyncTaskDesc`), so this always performs
/// a real context switch — no runtime check.
pub fn execute_stackful<S>(wk: &UltWorker<S>, cont: SuspendedTaskToken<S::Desc>)
where
    S: StackfulSchedulerSystem,
    S::Desc: StackfulTaskDesc,
{
    let wk2 = wk.suspend_to_cont(cont, |wk, prev| wk.set_root_cont(prev));
    debug_assert!(std::ptr::eq(wk2 as *const UltWorker<S>, wk as *const UltWorker<S>));
}

/// `pop_or_root` body for stackful-only systems: every popped item is a
/// real, switchable continuation, so no requeue check is needed.
pub fn pop_or_root_stackful<S>(wk: &UltWorker<S>) -> SuspendedTaskToken<S::Desc>
where
    S: StackfulSchedulerSystem,
    S::Desc: StackfulTaskDesc,
{
    if let Some(c) = wk.deque.try_pop_top() {
        return c;
    }
    wk.take_root_cont()
}

/// `free_finished_desc` body for stackful-only systems: every descriptor
/// came from the pool (there is no `spawn_async` allocation path to bypass
/// it), so always return it there.
///
/// # Safety
/// No other references to `desc` may exist after this call (same contract
/// as [`TaskPool::free_task`]).
pub unsafe fn free_finished_desc_stackful<S>(wk: &UltWorker<S>, desc: *mut S::Desc)
where
    S: SchedulerSystem,
{
    unsafe { wk.free_task(desc) };
}

// --- StackfulLocalQueue ---

impl<S: StackfulSchedulerSystem> StackfulLocalQueue<S> for UltWorker<S>
where
    S::Desc: StackfulTaskDesc,
{
    fn pop_or_root(&self) -> SuspendedTaskToken<S::Desc> {
        S::pop_or_root(self)
    }

    fn set_root_cont(&self, cont: SuspendedTaskToken<S::Desc>) {
        debug_assert!(self.root_cont.get().is_null());
        debug_assert!(cont.is_root());
        self.root_cont.set(cont.into_raw());
    }
}

// --- ContextSwitcher ---

impl<S: StackfulSchedulerSystem> ContextSwitcher<S> for UltWorker<S>
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
            next: next.into_raw(),
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

    fn suspend_to_new<F>(&self, stack_top: *mut u8, next: *mut S::Desc, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedTaskToken<S::Desc>),
    {
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
            next: next.into_raw(),
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

impl<S: StackfulSchedulerSystem> StackfulWorker<S> for UltWorker<S> where S::Desc: StackfulTaskDesc {}

// ---------------------------------------------------------------------------
// Shims: extern "C" callbacks handed to the context-switch layer.
//
// Each shim runs on the destination stack.  Read everything out of the
// payload (which lives on the now-frozen previous stack) *before* doing
// anything that could allow the previous context to resume.
// ---------------------------------------------------------------------------

struct SuspendPayload<S: StackfulSchedulerSystem, F>
where
    S::Desc: StackfulTaskDesc,
{
    wk: *const UltWorker<S>,
    next: *mut S::Desc,
    f: ManuallyDrop<F>,
}

unsafe extern "C" fn suspend_shim<S, F>(prev: Context, a1: *mut (), _a2: *mut ()) -> Transfer
where
    S: StackfulSchedulerSystem,
    S::Desc: StackfulTaskDesc,
    F: FnOnce(&UltWorker<S>, SuspendedTaskToken<S::Desc>),
{
    let (wk, next, f) = unsafe {
        let payload = &mut *(a1 as *mut SuspendPayload<S, F>);
        (&*payload.wk, payload.next, ManuallyDrop::take(&mut payload.f))
    };
    let mut prev_task = wk.take_cur_task();
    let old = prev_task.publish_saved_context(prev.0);
    debug_assert!(old.is_null(), "suspend over live ctx in suspend_shim (is_root={})", prev_task.as_desc().is_root());
    // SAFETY: `next` (carried across the FFI context switch as a raw
    // pointer in `SuspendPayload`, since a move-only Rust token can't cross
    // an `extern "C"` boundary) was already-linear at the call site before
    // the switch — consumed from a real `SuspendedTaskToken`'s `into_raw()`
    // (or is a freshly allocated descriptor, `suspend_to_new`) — so it's
    // exclusively ours here.
    let mut next_running = unsafe { RunningTaskToken::from_raw(next) };
    let wkp = wk as *const UltWorker<S> as *const ();
    next_running.mark_resumed_on(wkp);
    wk.set_cur_task(next_running);
    f(wk, prev_task.into_suspended());
    Transfer(wk as *const UltWorker<S> as *mut ())
}

struct CondSuspendPayload<S: StackfulSchedulerSystem, F>
where
    S::Desc: StackfulTaskDesc,
{
    wk: *const UltWorker<S>,
    next: *mut Option<SuspendedTaskToken<S::Desc>>,
    f: ManuallyDrop<F>,
}

unsafe extern "C" fn cond_suspend_shim<S, F>(prev: Context, a1: *mut (), _a2: *mut ()) -> CondTransfer
where
    S: StackfulSchedulerSystem,
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
    // owner-exclusive fields (`worker`/`slot`/`ctx`) are mutated through
    // one of them. See `RunningTaskToken`'s doc comment.
    let mut next_running = next_cont.into_running();
    let wkp = wk as *const UltWorker<S> as *const ();
    next_running.mark_resumed_on(wkp);
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

struct ExitPayload<S: StackfulSchedulerSystem, F>
where
    S::Desc: StackfulTaskDesc,
{
    wk: *const UltWorker<S>,
    next: *mut S::Desc,
    f: ManuallyDrop<F>,
}

unsafe extern "C" fn exit_shim<S, F>(a1: *mut (), _a2: *mut ()) -> Transfer
where
    S: StackfulSchedulerSystem,
    S::Desc: StackfulTaskDesc,
    F: FnOnce(&UltWorker<S>),
{
    let (wk, next, f) = unsafe {
        let payload = &mut *(a1 as *mut ExitPayload<S, F>);
        (&*payload.wk, payload.next, ManuallyDrop::take(&mut payload.f))
    };
    // The exiting task's own descriptor isn't being saved anywhere -- `f`
    // is responsible for its cleanup/freeing via the join protocol -- so
    // just take it out of `cur_task` and drop the (zero-cost, no `Drop`
    // impl) `RunningTaskToken` wrapper without doing anything else with it.
    let _ = wk.take_cur_task();
    // SAFETY: same reasoning as `suspend_shim` — `next` was already-linear
    // at the call site before the switch and is carried across the FFI
    // boundary as a raw pointer in `ExitPayload`.
    let mut next_running = unsafe { RunningTaskToken::from_raw(next) };
    let wkp = wk as *const UltWorker<S> as *const ();
    next_running.mark_resumed_on(wkp);
    wk.set_cur_task(next_running);
    f(wk);
    Transfer(wk as *const UltWorker<S> as *mut ())
}
