//! Worker traits and the concrete ULT worker implementation.
//!
//! # Trait hierarchy
//!
//! * [`ContextSwitcher`] — raw context-switch primitives (save / swap /
//!   cond-swap / restore), each taking a callback that runs *after* the stack
//!   switch.
//! * [`TaskPool`] — task-descriptor allocation backed by a per-worker free
//!   list.
//! * [`LocalQueue`] — the per-worker deque and root-continuation slot.
//! * [`Worker`] — composes the above; provides scheduler-level operations
//!   (`suspend_to_sched`, `yield_now`, `execute`) as default implementations
//!   built from the sub-trait primitives alone.
//!
//! The concrete type [`UltWorker<S>`] satisfies all four traits for any
//! [`UltSchedulerSystem`] `S`.  Everything that depends on workers — sync primitives,
//! [`super::suspended`] — is generic over `W: Worker`.  The `UltSchedulerSystem`
//! type appears only at the construction boundary (`scheduler`, `thread`).

use std::cell::Cell;
use std::mem::ManuallyDrop;
use std::ptr;

use std::task::{RawWaker, Waker};

use crate::context::{CondTransfer, Context, ContextPolicy, Transfer};
use crate::ult::deque::WorkerDeque;
use crate::ult::pool::DescPool;
use crate::ult::scheduler::Scheduler;
use crate::ult::system::UltSchedulerSystem;
use crate::ult::desc::{
    AsyncTaskDesc, BasicTaskDesc, StackfulTaskDesc, SuspendedUlt, TaskDesc,
};
use crate::ult::waker::async_task_private_vtable;

// ---------------------------------------------------------------------------
// ContextSwitcher
// ---------------------------------------------------------------------------

/// Raw context-switch operations at the worker level.
///
/// Every method executes a callback **on the destination stack**, after the
/// current context is fully saved.  Publishing the suspended continuation from
/// inside the callback is therefore inherently race-free; no "saving in
/// progress" flags or spin-wait handshakes are needed anywhere.
pub trait ContextSwitcher: Sized {
    /// Save the current task's context, switch to `next`, run `f(wk, prev)`
    /// on that stack where `prev` is the just-saved continuation, and return
    /// when the current task is later resumed.
    fn suspend_to_cont<F>(&self, next: SuspendedUlt, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedUlt);

    /// Like [`suspend_to_cont`](Self::suspend_to_cont), but `f` may cancel
    /// the switch.  `f` receives `&mut Option<SuspendedUlt>` holding the
    /// current task's continuation; consuming it (`Option::take`) commits the
    /// switch, leaving it in place cancels it and resumes the caller.
    fn cond_suspend_to_cont<F>(&self, next: &mut Option<SuspendedUlt>, f: F) -> &Self
    where
        F: FnOnce(&Self, &mut Option<SuspendedUlt>);

    /// Save the current context, switch to a **fresh** stack at `stack_top`,
    /// run `f(wk, prev)` there.  Used for child-first fork; `f` must never
    /// return.
    fn suspend_to_new<F>(&self, stack_top: *mut u8, next: *mut BasicTaskDesc, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedUlt);

    /// Abandon (do not save) the current context and switch to `next`.
    fn exit_to_cont<F>(&self, next: SuspendedUlt, f: F) -> !
    where
        F: FnOnce(&Self);
}

// ---------------------------------------------------------------------------
// TaskPool
// ---------------------------------------------------------------------------

/// Task-descriptor allocation with a per-worker free list.
pub trait TaskPool {
    fn alloc_task(&self, has_handle: bool) -> *mut BasicTaskDesc;

    /// Return a dead descriptor to the pool.
    ///
    /// # Safety
    /// No other references to `desc` may exist after this call.
    unsafe fn free_task(&self, desc: *mut BasicTaskDesc);
}

// ---------------------------------------------------------------------------
// LocalQueue
// ---------------------------------------------------------------------------

/// Per-worker work queue and root-continuation management.
pub trait LocalQueue {
    /// Push `c` to the **LIFO** end (will run before anything already queued).
    fn push_local_top(&self, c: SuspendedUlt);

    /// Push `c` to the **FIFO** end (yield: let other tasks run first).
    fn push_local_bottom(&self, c: SuspendedUlt);

    /// Pop from the LIFO end of this worker's local deque.
    fn pop_local(&self) -> Option<SuspendedUlt>;

    /// Try to steal one task from another worker's FIFO end.
    fn try_steal(&self) -> Option<SuspendedUlt>;

    /// Pop the next runnable continuation: local deque first, then the root
    /// (scheduler-loop) continuation.
    fn pop_or_root(&self) -> SuspendedUlt;

    /// Store the scheduler-loop context as the root continuation.
    fn set_root_cont(&self, c: SuspendedUlt);

    /// This worker's index within its scheduler.
    fn num(&self) -> usize;

    /// Total number of workers in this scheduler instance.
    fn num_workers(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Combined worker interface.
///
/// Blanket default implementations of the scheduler-level operations
/// (`suspend_to_sched`, `cond_suspend_to_sched`, `exit_to_sched`, `execute`,
/// `yield_now`) are provided here, composed from the three sub-traits.
/// Concrete types implement only the sub-trait primitives.
pub trait Worker: ContextSwitcher + TaskPool + LocalQueue + Send + Sync + 'static {
    /// The worker currently running on this base thread, if any.
    fn current() -> Option<&'static Self>
    where
        Self: Sized;

    /// Suspend to the next continuation from the local deque / root.
    fn suspend_to_sched<F>(&self, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedUlt),
    {
        let next = self.pop_or_root();
        self.suspend_to_cont(next, f)
    }

    /// Conditionally suspend to the scheduler.  On cancellation the popped
    /// continuation is returned to its source (deque top or root slot).
    fn cond_suspend_to_sched<F>(&self, f: F) -> &Self
    where
        F: FnOnce(&Self, &mut Option<SuspendedUlt>),
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

    /// Run one task to its next suspension point (scheduler-loop side).
    fn execute(&self, cont: SuspendedUlt) {
        let wk = self.suspend_to_cont(cont, |wk, prev| wk.set_root_cont(prev));
        debug_assert!(std::ptr::eq(wk as *const Self, self as *const Self));
    }

    /// Cooperative yield: requeue at the FIFO end so other tasks run first.
    fn yield_now(&self) -> &Self {
        self.suspend_to_sched(|wk, prev| wk.push_local_bottom(prev))
    }
}

// ---------------------------------------------------------------------------
// Concrete implementation: UltWorker<S>
// ---------------------------------------------------------------------------

pub struct UltWorker<S: UltSchedulerSystem> {
    num: usize,
    deque: S::Deque,
    pub(crate) cur_task: Cell<*mut BasicTaskDesc>,
    root_desc: BasicTaskDesc,
    root_cont: Cell<*mut BasicTaskDesc>,
    steal_seed: Cell<usize>,
    pub(crate) shared: Cell<*const Scheduler<S>>,
}

// `Cell` fields are only accessed by the owning base thread; `deque` is
// internally synchronized; `shared` is read-only after init.
unsafe impl<S: UltSchedulerSystem> Send for UltWorker<S> {}
unsafe impl<S: UltSchedulerSystem> Sync for UltWorker<S> {}

impl<S: UltSchedulerSystem> UltWorker<S> {
    pub(crate) fn new(num: usize) -> Self {
        UltWorker {
            num,
            deque: S::Deque::default(),
            cur_task: Cell::new(ptr::null_mut()),
            root_desc: BasicTaskDesc::new_root(),
            root_cont: Cell::new(ptr::null_mut()),
            steal_seed: Cell::new(num.wrapping_mul(0x9E37_79B9).wrapping_add(1)),
            shared: Cell::new(ptr::null()),
        }
    }

    pub(crate) fn root_desc(&self) -> &BasicTaskDesc {
        &self.root_desc
    }

    pub(crate) fn shared(&self) -> &Scheduler<S> {
        unsafe { &*self.shared.get() }
    }
}

// --- ContextSwitcher ---

impl<S: UltSchedulerSystem> ContextSwitcher for UltWorker<S> {
    fn suspend_to_cont<F>(&self, next: SuspendedUlt, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedUlt),
    {
        let next_ctx = Context(unsafe { (*next.desc()).claim_saved_context() });
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

    fn cond_suspend_to_cont<F>(&self, next: &mut Option<SuspendedUlt>, f: F) -> &Self
    where
        F: FnOnce(&Self, &mut Option<SuspendedUlt>),
    {
        let next_ctx = Context(unsafe {
            (*next.as_ref().expect("cond_suspend without target").desc()).peek_saved_context()
        });
        debug_assert!(!next_ctx.is_null());
        let mut payload = CondSuspendPayload::<S, F> {
            wk: self,
            next: next as *mut Option<SuspendedUlt>,
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

    fn suspend_to_new<F>(&self, stack_top: *mut u8, next: *mut BasicTaskDesc, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedUlt),
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

    fn exit_to_cont<F>(&self, next: SuspendedUlt, f: F) -> !
    where
        F: FnOnce(&Self),
    {
        let next_ctx = Context(unsafe { (*next.desc()).claim_saved_context() });
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

// --- TaskPool ---

impl<S: UltSchedulerSystem> TaskPool for UltWorker<S> {
    fn alloc_task(&self, has_handle: bool) -> *mut BasicTaskDesc {
        self.shared().task_pool.alloc(self.num, has_handle)
    }

    unsafe fn free_task(&self, desc: *mut BasicTaskDesc) {
        unsafe { self.shared().task_pool.dealloc(self.num, desc) };
    }
}

// --- LocalQueue ---

impl<S: UltSchedulerSystem> LocalQueue for UltWorker<S> {
    fn push_local_top(&self, c: SuspendedUlt) {
        self.deque.push_top(c);
    }

    fn push_local_bottom(&self, c: SuspendedUlt) {
        self.deque.push_bottom(c);
    }

    fn pop_local(&self) -> Option<SuspendedUlt> {
        self.deque.try_pop_top()
    }

    fn try_steal(&self) -> Option<SuspendedUlt> {
        let shared = self.shared();
        let n = shared.workers.len();
        if n <= 1 {
            return None;
        }
        let seed = self.steal_seed.get();
        self.steal_seed.set(seed.wrapping_add(1));
        for i in 0..n {
            let victim = (seed + i) % n;
            if victim == self.num {
                continue;
            }
            if let Some(c) = shared.workers[victim].deque.try_steal_bottom() {
                return Some(c);
            }
        }
        None
    }

    fn pop_or_root(&self) -> SuspendedUlt {
        if let Some(c) = self.deque.try_pop_top() {
            if unsafe { (*c.desc()).poll_fn().get().is_some() } {
                // Async tasks have no saved context; they can only be executed
                // by the scheduler loop via execute().  Push the async task back
                // to the LIFO end and return root so the scheduler loop handles it.
                self.deque.push_top(c);
            } else {
                return c;
            }
        }
        let root = self.root_cont.replace(ptr::null_mut());
        assert!(!root.is_null(), "no runnable continuation on worker {}", self.num);
        SuspendedUlt(root)
    }

    fn set_root_cont(&self, cont: SuspendedUlt) {
        debug_assert!(self.root_cont.get().is_null());
        debug_assert!(cont.is_root());
        self.root_cont.set(cont.into_raw());
    }

    fn num(&self) -> usize {
        self.num
    }

    fn num_workers(&self) -> usize {
        self.shared().workers.len()
    }
}

// --- Worker ---

impl<S: UltSchedulerSystem> Worker for UltWorker<S> {
    fn current() -> Option<&'static Self> {
        <S::Lookup as crate::ult::lookup::CurrentLookup<S>>::current()
    }

    fn execute(&self, cont: SuspendedUlt) {
        let desc = cont.desc();
        if let Some(poll_fn) = unsafe { (*desc).poll_fn().get() } {
            let _ = cont.into_raw(); // consumed; no context switch
            self.run_async_poll(desc, poll_fn);
        } else {
            // Sync ULT: context switch as usual.
            let wk = self.suspend_to_cont(cont, |wk, prev| wk.set_root_cont(prev));
            debug_assert!(std::ptr::eq(wk as *const Self, self as *const Self));
        }
    }
}

impl<S: UltSchedulerSystem> UltWorker<S> {
    /// Execute one poll of an async task.  Called from `execute` when
    /// `desc.poll_fn` is `Some`.
    fn run_async_poll(
        &self,
        desc: *mut BasicTaskDesc,
        poll_fn: for<'cx> unsafe fn(*mut BasicTaskDesc, &mut std::task::Context<'cx>) -> bool,
    ) {
        // Mark as POLLING so the waker's state machine works correctly.
        unsafe { (*desc).mark_polling() };

        let raw = RawWaker::new(desc as *const (), async_task_private_vtable::<S>());
        let waker = unsafe { Waker::from_raw(raw) };
        let mut cx = std::task::Context::from_waker(&waker);

        // Returns true = Ready (desc must not be touched after this).
        let done = unsafe { poll_fn(desc, &mut cx) };

        // waker is dropped here; drop_async_private is a no-op for PRIVATE mode.
        drop(waker);

        if !done {
            // Pending: park, unless a wake raced in during poll() -- then
            // re-queue immediately instead.
            if !unsafe { (*desc).park_after_poll() } {
                self.push_local_top(SuspendedUlt(desc));
            }
        }
        // done=true: async_poll_fn handled everything; desc may have been freed.
    }
}

// ---------------------------------------------------------------------------
// Free function kept for call-site compatibility
// ---------------------------------------------------------------------------

pub fn current_worker<S: UltSchedulerSystem>() -> Option<&'static UltWorker<S>> {
    UltWorker::<S>::current()
}

// ---------------------------------------------------------------------------
// Shims: extern "C" callbacks handed to the context-switch layer.
//
// Each shim runs on the destination stack.  Read everything out of the
// payload (which lives on the now-frozen previous stack) *before* doing
// anything that could allow the previous context to resume.
// ---------------------------------------------------------------------------

struct SuspendPayload<S: UltSchedulerSystem, F> {
    wk: *const UltWorker<S>,
    next: *mut BasicTaskDesc,
    f: ManuallyDrop<F>,
}

unsafe extern "C" fn suspend_shim<S, F>(prev: Context, a1: *mut (), _a2: *mut ()) -> Transfer
where
    S: UltSchedulerSystem,
    F: FnOnce(&UltWorker<S>, SuspendedUlt),
{
    let (wk, next, f) = unsafe {
        let payload = &mut *(a1 as *mut SuspendPayload<S, F>);
        (&*payload.wk, payload.next, ManuallyDrop::take(&mut payload.f))
    };
    let prev_desc = wk.cur_task.get();
    let old = unsafe { (*prev_desc).publish_saved_context(prev.0) };
    debug_assert!(old.is_null(), "suspend over live ctx in suspend_shim (is_root={})", unsafe { (*prev_desc).is_root() });
    wk.cur_task.set(next);
    let wkp = wk as *const UltWorker<S> as *const ();
    unsafe { (*next).mark_resumed_on(wkp) };
    f(wk, SuspendedUlt(prev_desc));
    Transfer(wk as *const UltWorker<S> as *mut ())
}

struct CondSuspendPayload<S: UltSchedulerSystem, F> {
    wk: *const UltWorker<S>,
    next: *mut Option<SuspendedUlt>,
    f: ManuallyDrop<F>,
}

unsafe extern "C" fn cond_suspend_shim<S, F>(prev: Context, a1: *mut (), _a2: *mut ()) -> CondTransfer
where
    S: UltSchedulerSystem,
    F: FnOnce(&UltWorker<S>, &mut Option<SuspendedUlt>),
{
    let (wk, next_slot, next_cont, f) = unsafe {
        let payload = &mut *(a1 as *mut CondSuspendPayload<S, F>);
        let next_cont = (*payload.next).take().unwrap();
        (&*payload.wk, payload.next, next_cont, ManuallyDrop::take(&mut payload.f))
    };
    let prev_desc = wk.cur_task.get();
    let old = unsafe { (*prev_desc).publish_saved_context(prev.0) };
    debug_assert!(old.is_null(), "suspend over live ctx in cond_suspend_shim (is_root={})", unsafe { (*prev_desc).is_root() });
    wk.cur_task.set(next_cont.desc());
    let wkp = wk as *const UltWorker<S> as *const ();
    unsafe { (*next_cont.desc()).mark_resumed_on(wkp) };

    let mut prev_cont = Some(SuspendedUlt(prev_desc));
    f(wk, &mut prev_cont);

    match prev_cont {
        None => {
            unsafe { (*next_cont.desc()).clear_saved_context() };
            let _ = next_cont.into_raw();
            CondTransfer { value: wk as *const UltWorker<S> as *mut (), flag: 1 }
        }
        Some(c) => {
            debug_assert!(std::ptr::eq(c.desc(), prev_desc));
            unsafe { (*prev_desc).clear_saved_context() };
            let _ = c.into_raw();
            wk.cur_task.set(prev_desc);
            unsafe { *next_slot = Some(next_cont) };
            CondTransfer { value: wk as *const UltWorker<S> as *mut (), flag: 0 }
        }
    }
}

struct ExitPayload<S: UltSchedulerSystem, F> {
    wk: *const UltWorker<S>,
    next: *mut BasicTaskDesc,
    f: ManuallyDrop<F>,
}

unsafe extern "C" fn exit_shim<S, F>(a1: *mut (), _a2: *mut ()) -> Transfer
where
    S: UltSchedulerSystem,
    F: FnOnce(&UltWorker<S>),
{
    let (wk, next, f) = unsafe {
        let payload = &mut *(a1 as *mut ExitPayload<S, F>);
        (&*payload.wk, payload.next, ManuallyDrop::take(&mut payload.f))
    };
    wk.cur_task.set(next);
    let wkp = wk as *const UltWorker<S> as *const ();
    unsafe { (*next).mark_resumed_on(wkp) };
    f(wk);
    Transfer(wk as *const UltWorker<S> as *mut ())
}
