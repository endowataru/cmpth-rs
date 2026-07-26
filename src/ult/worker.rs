//! Worker traits and the concrete ULT worker implementation.
//!
//! # Trait hierarchy
//!
//! * [`LocalQueue`]/[`TaskPool`]/[`Worker`] — base-level, parameterized by
//!   `S: SchedulerSystem`. No context-switch machinery: usable by a
//!   stackful-only, dual, *or* (eventually) stackless-only system alike.
//! * [`ContextSwitcher`]/[`StackfulLocalQueue`]/[`StackfulWorker`] — the
//!   real-stack extension, parameterized by `S: UltSchedulerSystem`. A
//!   stackless-only system never implements these at all.
//!
//! [`Worker::execute`] and [`UltSchedulerSystem::pop_or_root`] are the two
//! places dispatch used to be a single hardcoded (dual, poll_fn-checking)
//! body for every system. Both are now *required* hooks on the system trait
//! itself (`SchedulerSystem::execute`, `UltSchedulerSystem::pop_or_root`),
//! with a stackful-only default and a `_dual` variant dual configs override
//! to — an ordinary trait-default override, monomorphized per concrete
//! system, not runtime dispatch: see `execute_stackful`/`execute_dual` and
//! `pop_or_root_stackful`/`pop_or_root_dual` below.
//!
//! The concrete type [`UltWorker<S>`] satisfies the base traits for any
//! `S: SchedulerSystem`, and the stackful extension traits whenever
//! `S: UltSchedulerSystem`.

use std::cell::Cell;
use std::mem::ManuallyDrop;
use std::ptr;

use std::task::{RawWaker, Waker};

use crate::context::{CondTransfer, Context, ContextPolicy, Transfer};
use crate::ult::deque::WorkerDeque;
use crate::ult::pool::DescPool;
use crate::ult::scheduler::Scheduler;
use crate::ult::system::{SchedulerSystem, UltSchedulerSystem};
use crate::ult::desc::{
    AsyncTaskDesc, StackfulTaskDesc, SuspendedUlt, TaskDesc, TaskDescAlloc, TaskPollFn,
    TaskPollResult, WakerTaskDesc,
};

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
/// Only implementable when `S: UltSchedulerSystem` (needs `S::Ctx`) — a
/// stackless-only system has no context-switch policy to name.
pub trait ContextSwitcher<S: UltSchedulerSystem>: Sized
where
    S::Desc: StackfulTaskDesc,
{
    /// Save the current task's context, switch to `next`, run `f(wk, prev)`
    /// on that stack where `prev` is the just-saved continuation, and return
    /// when the current task is later resumed.
    fn suspend_to_cont<F>(&self, next: SuspendedUlt<S::Desc>, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedUlt<S::Desc>);

    /// Like [`suspend_to_cont`](Self::suspend_to_cont), but `f` may cancel
    /// the switch.  `f` receives `&mut Option<SuspendedUlt<S::Desc>>` holding
    /// the current task's continuation; consuming it (`Option::take`) commits
    /// the switch, leaving it in place cancels it and resumes the caller.
    fn cond_suspend_to_cont<F>(&self, next: &mut Option<SuspendedUlt<S::Desc>>, f: F) -> &Self
    where
        F: FnOnce(&Self, &mut Option<SuspendedUlt<S::Desc>>);

    /// Save the current context, switch to a **fresh** stack at `stack_top`,
    /// run `f(wk, prev)` there.  Used for child-first fork; `f` must never
    /// return.
    fn suspend_to_new<F>(&self, stack_top: *mut u8, next: *mut S::Desc, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedUlt<S::Desc>);

    /// Abandon (do not save) the current context and switch to `next`.
    fn exit_to_cont<F>(&self, next: SuspendedUlt<S::Desc>, f: F) -> !
    where
        F: FnOnce(&Self);
}

// ---------------------------------------------------------------------------
// TaskPool (base)
// ---------------------------------------------------------------------------

/// Task-descriptor allocation with a per-worker free list.
pub trait TaskPool<S: SchedulerSystem> {
    /// Allocate a descriptor with storage for at least `size` bytes (see
    /// [`DescPool::alloc`](crate::ult::pool::DescPool::alloc) — `spawn`
    /// always requests the same fixed `S::STACK_SIZE`, but the size
    /// parameter is here so a future per-task custom stack size needs no
    /// further interface change).
    fn alloc_task(&self, has_handle: bool, size: usize) -> *mut S::Desc;

    /// Return a dead descriptor to the pool.
    ///
    /// # Safety
    /// No other references to `desc` may exist after this call.
    unsafe fn free_task(&self, desc: *mut S::Desc);
}

// ---------------------------------------------------------------------------
// LocalQueue (base)
// ---------------------------------------------------------------------------

/// Per-worker work-stealing deque, independent of task flavor.
pub trait LocalQueue<S: SchedulerSystem> {
    /// Push `c` to the **LIFO** end (will run before anything already queued).
    fn push_local_top(&self, c: SuspendedUlt<S::Desc>);

    /// Push `c` to the **FIFO** end (yield: let other tasks run first).
    fn push_local_bottom(&self, c: SuspendedUlt<S::Desc>);

    /// Pop from the LIFO end of this worker's local deque.
    fn pop_local(&self) -> Option<SuspendedUlt<S::Desc>>;

    /// Try to steal one task from another worker's FIFO end.
    fn try_steal(&self) -> Option<SuspendedUlt<S::Desc>>;

    /// This worker's index within its scheduler.
    fn num(&self) -> usize;

    /// Total number of workers in this scheduler instance.
    fn num_workers(&self) -> usize;
}

// ---------------------------------------------------------------------------
// StackfulLocalQueue (stackful-only)
// ---------------------------------------------------------------------------

/// Root-continuation management: only meaningful when there is a real
/// scheduler-loop stack a suspending ULT can fall back into.
pub trait StackfulLocalQueue<S: UltSchedulerSystem>: LocalQueue<S>
where
    S::Desc: StackfulTaskDesc,
{
    /// Pop the next runnable continuation: local deque first, then the root
    /// (scheduler-loop) continuation. Forwards to
    /// [`UltSchedulerSystem::pop_or_root`] — see that method for why the
    /// dispatch body lives on the system trait, not here.
    fn pop_or_root(&self) -> SuspendedUlt<S::Desc>;

    /// Store the scheduler-loop context as the root continuation.
    fn set_root_cont(&self, c: SuspendedUlt<S::Desc>);
}

// ---------------------------------------------------------------------------
// Worker (base)
// ---------------------------------------------------------------------------

/// Base worker interface: locating the current worker, and running one
/// popped continuation.
pub trait Worker<S: SchedulerSystem>: TaskPool<S> + LocalQueue<S> + Send + Sync + 'static {
    /// The worker currently running on this base thread, if any.
    fn current() -> Option<&'static Self>
    where
        Self: Sized;

    /// Run one task to its next suspension point (scheduler-loop side).
    /// Forwards to [`SchedulerSystem::execute`] — see that method for why
    /// the dispatch body lives on the system trait, not here.
    fn execute(&self, cont: SuspendedUlt<S::Desc>);
}

// ---------------------------------------------------------------------------
// StackfulWorker (stackful-only)
// ---------------------------------------------------------------------------

/// Scheduler-level operations that only make sense with a real, switchable
/// stack: suspending the calling ULT and resuming whatever's next.
pub trait StackfulWorker<S: UltSchedulerSystem>:
    Worker<S> + ContextSwitcher<S> + StackfulLocalQueue<S>
where
    S::Desc: StackfulTaskDesc,
{
    /// Suspend to the next continuation from the local deque / root.
    fn suspend_to_sched<F>(&self, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedUlt<S::Desc>),
    {
        let next = self.pop_or_root();
        self.suspend_to_cont(next, f)
    }

    /// Conditionally suspend to the scheduler.  On cancellation the popped
    /// continuation is returned to its source (deque top or root slot).
    fn cond_suspend_to_sched<F>(&self, f: F) -> &Self
    where
        F: FnOnce(&Self, &mut Option<SuspendedUlt<S::Desc>>),
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
// Dispatch bodies for SchedulerSystem::execute / UltSchedulerSystem::pop_or_root
//
// Plain functions, not trait defaults directly: each concrete system's
// `impl SchedulerSystem`/`impl UltSchedulerSystem` block calls exactly one
// of these from its own `execute`/`pop_or_root` method. No specialization is
// involved — every concrete marker struct (DefaultUltSystem, a
// stackful-only `ult_system!` struct, ...) gets exactly one such `impl`
// block, so this is ordinary static dispatch, monomorphized per system.
// ---------------------------------------------------------------------------

/// `execute` body for stackful-only systems: `cont` is always a real ULT
/// continuation (no `poll_fn` tag ever gets set, since `spawn_async` isn't
/// reachable when `S::Desc` isn't `AsyncTaskDesc`), so this always performs
/// a real context switch — no runtime check.
pub fn execute_stackful<S>(wk: &UltWorker<S>, cont: SuspendedUlt<S::Desc>)
where
    S: UltSchedulerSystem,
    S::Desc: StackfulTaskDesc,
{
    let wk2 = wk.suspend_to_cont(cont, |wk, prev| wk.set_root_cont(prev));
    debug_assert!(std::ptr::eq(wk2 as *const UltWorker<S>, wk as *const UltWorker<S>));
}

/// `execute` body for dual systems: today's original logic — check
/// `poll_fn` first, and either poll inline or perform a real context switch.
pub fn execute_dual<S>(wk: &UltWorker<S>, cont: SuspendedUlt<S::Desc>)
where
    S: UltSchedulerSystem,
    S::Desc: StackfulTaskDesc + AsyncTaskDesc,
{
    let desc = cont.desc();
    if let Some(poll_fn) = unsafe { (*desc).poll_fn().get() } {
        let _ = cont.into_raw(); // consumed; no context switch
        run_async_poll(wk, desc, poll_fn);
    } else {
        // Sync ULT: context switch as usual.
        let wk2 = wk.suspend_to_cont(cont, |wk, prev| wk.set_root_cont(prev));
        debug_assert!(std::ptr::eq(wk2 as *const UltWorker<S>, wk as *const UltWorker<S>));
    }
}

/// `pop_or_root` body for stackful-only systems: every popped item is a
/// real, switchable continuation, so no requeue check is needed.
pub fn pop_or_root_stackful<S>(wk: &UltWorker<S>) -> SuspendedUlt<S::Desc>
where
    S: UltSchedulerSystem,
    S::Desc: StackfulTaskDesc,
{
    if let Some(c) = wk.deque.try_pop_top() {
        return c;
    }
    wk.take_root_cont()
}

/// `pop_or_root` body for dual systems: today's original logic — an async
/// task popped off the top has no saved context to switch into, so requeue
/// it and fall back to the root (scheduler-loop) continuation instead.
pub fn pop_or_root_dual<S>(wk: &UltWorker<S>) -> SuspendedUlt<S::Desc>
where
    S: UltSchedulerSystem,
    S::Desc: StackfulTaskDesc + AsyncTaskDesc,
{
    if let Some(c) = wk.deque.try_pop_top() {
        if unsafe { (*c.desc()).poll_fn().get().is_some() } {
            // Async tasks have no saved context; they can only be executed
            // by the scheduler loop via execute().  Push the async task back
            // to the LIFO end and return root so the scheduler loop handles it.
            wk.deque.push_top(c);
        } else {
            return c;
        }
    }
    wk.take_root_cont()
}

/// `free_finished_desc` body for dual systems: async tasks go through
/// `S::AsyncPool` (a separate pool from the ULT-stack `S::Pool`, see
/// [`SchedulerSystem::AsyncPool`]); everything else goes through the
/// ULT-stack pool as usual.
pub fn free_finished_desc_dual<S>(wk: &UltWorker<S>, desc: *mut S::Desc)
where
    S: UltSchedulerSystem,
    S::Desc: StackfulTaskDesc + AsyncTaskDesc,
{
    if unsafe { (*desc).poll_fn().get().is_some() } {
        unsafe { wk.shared().async_task_pool.dealloc(wk.num(), desc) };
    } else {
        unsafe { wk.free_task(desc) };
    }
}

/// `free_finished_desc` body for stackful-only systems: every descriptor
/// came from the pool (there is no `spawn_async` allocation path to bypass
/// it), so always return it there.
pub fn free_finished_desc_stackful<S>(wk: &UltWorker<S>, desc: *mut S::Desc)
where
    S: SchedulerSystem,
{
    unsafe { wk.free_task(desc) };
}

/// `free_finished_desc` body for stackless-only systems: every descriptor
/// is a `spawn_async` allocation, so always route it through `S::AsyncPool`
/// (which itself decides pool-return vs. raw-free via
/// [`TaskDesc::oversized`]).
pub fn free_finished_desc_async<S>(wk: &UltWorker<S>, desc: *mut S::Desc)
where
    S: SchedulerSystem,
{
    unsafe { wk.shared().async_task_pool.dealloc(wk.num(), desc) };
}

// ---------------------------------------------------------------------------
// Concrete implementation: UltWorker<S>
// ---------------------------------------------------------------------------

pub struct UltWorker<S: SchedulerSystem> {
    num: usize,
    deque: S::Deque,
    pub(crate) cur_task: Cell<*mut S::Desc>,
    root_desc: S::Desc,
    root_cont: Cell<*mut S::Desc>,
    steal_seed: Cell<usize>,
    pub(crate) shared: Cell<*const Scheduler<S>>,
    /// The descriptor currently being driven by `run_async_poll` on this
    /// worker, or null. Distinct from `cur_task` (which tracks real
    /// context-switch state and is meaningless for async polling): this is
    /// how `JoinHandle::poll` recognizes "the ambient waker is verifiably
    /// this task's own" without inspecting the waker itself, avoiding a
    /// `Box<Waker>` allocation on the common `spawn_async`/`.await` path.
    pub(crate) polling_async: Cell<*mut S::Desc>,
}

// `Cell` fields are only accessed by the owning base thread; `deque` is
// internally synchronized; `shared` is read-only after init.
unsafe impl<S: SchedulerSystem> Send for UltWorker<S> {}
unsafe impl<S: SchedulerSystem> Sync for UltWorker<S> {}

impl<S: SchedulerSystem> UltWorker<S> {
    pub(crate) fn new(num: usize) -> Self {
        UltWorker {
            num,
            deque: S::Deque::default(),
            cur_task: Cell::new(ptr::null_mut()),
            root_desc: S::Desc::new_root(),
            root_cont: Cell::new(ptr::null_mut()),
            steal_seed: Cell::new(num.wrapping_mul(0x9E37_79B9).wrapping_add(1)),
            shared: Cell::new(ptr::null()),
            polling_async: Cell::new(ptr::null_mut()),
        }
    }

    pub(crate) fn root_desc(&self) -> &S::Desc {
        &self.root_desc
    }

    pub(crate) fn shared(&self) -> &Scheduler<S> {
        unsafe { &*self.shared.get() }
    }

    /// Take the stored root (scheduler-loop) continuation. Shared by
    /// `pop_or_root_stackful`/`pop_or_root_dual`.
    fn take_root_cont(&self) -> SuspendedUlt<S::Desc> {
        let root = self.root_cont.replace(ptr::null_mut());
        assert!(!root.is_null(), "no runnable continuation on worker {}", self.num);
        SuspendedUlt(root)
    }
}

// --- TaskPool ---

impl<S: SchedulerSystem> TaskPool<S> for UltWorker<S> {
    fn alloc_task(&self, has_handle: bool, size: usize) -> *mut S::Desc {
        self.shared().task_pool.alloc(self.num, has_handle, size)
    }

    unsafe fn free_task(&self, desc: *mut S::Desc) {
        unsafe { self.shared().task_pool.dealloc(self.num, desc) };
    }
}

// --- LocalQueue ---

impl<S: SchedulerSystem> LocalQueue<S> for UltWorker<S> {
    fn push_local_top(&self, c: SuspendedUlt<S::Desc>) {
        self.deque.push_top(c);
    }

    fn push_local_bottom(&self, c: SuspendedUlt<S::Desc>) {
        self.deque.push_bottom(c);
    }

    fn pop_local(&self) -> Option<SuspendedUlt<S::Desc>> {
        self.deque.try_pop_top()
    }

    fn try_steal(&self) -> Option<SuspendedUlt<S::Desc>> {
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

    fn num(&self) -> usize {
        self.num
    }

    fn num_workers(&self) -> usize {
        self.shared().workers.len()
    }
}

// --- StackfulLocalQueue ---

impl<S: UltSchedulerSystem> StackfulLocalQueue<S> for UltWorker<S>
where
    S::Desc: StackfulTaskDesc,
{
    fn pop_or_root(&self) -> SuspendedUlt<S::Desc> {
        S::pop_or_root(self)
    }

    fn set_root_cont(&self, cont: SuspendedUlt<S::Desc>) {
        debug_assert!(self.root_cont.get().is_null());
        debug_assert!(cont.is_root());
        self.root_cont.set(cont.into_raw());
    }
}

// --- ContextSwitcher ---

impl<S: UltSchedulerSystem> ContextSwitcher<S> for UltWorker<S>
where
    S::Desc: StackfulTaskDesc,
{
    fn suspend_to_cont<F>(&self, next: SuspendedUlt<S::Desc>, f: F) -> &Self
    where
        F: FnOnce(&Self, SuspendedUlt<S::Desc>),
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

    fn cond_suspend_to_cont<F>(&self, next: &mut Option<SuspendedUlt<S::Desc>>, f: F) -> &Self
    where
        F: FnOnce(&Self, &mut Option<SuspendedUlt<S::Desc>>),
    {
        let next_ctx = Context(unsafe {
            (*next.as_ref().expect("cond_suspend without target").desc()).peek_saved_context()
        });
        debug_assert!(!next_ctx.is_null());
        let mut payload = CondSuspendPayload::<S, F> {
            wk: self,
            next: next as *mut Option<SuspendedUlt<S::Desc>>,
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
        F: FnOnce(&Self, SuspendedUlt<S::Desc>),
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

    fn exit_to_cont<F>(&self, next: SuspendedUlt<S::Desc>, f: F) -> !
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

// --- Worker ---

impl<S: SchedulerSystem> Worker<S> for UltWorker<S> {
    fn current() -> Option<&'static Self> {
        <S::Lookup as crate::ult::lookup::CurrentLookup<S>>::current()
    }

    fn execute(&self, cont: SuspendedUlt<S::Desc>) {
        S::execute(self, cont);
    }
}

// --- StackfulWorker ---

impl<S: UltSchedulerSystem> StackfulWorker<S> for UltWorker<S> where S::Desc: StackfulTaskDesc {}

/// Drive one async task's poll to completion or a suspend point. Called
/// from [`execute_dual`] (when `desc.poll_fn` is `Some`) and from
/// [`execute_async`] (always). Base-level (`S: SchedulerSystem`): polling a
/// `spawn_async` task never touches context-switch machinery, so a
/// stackless-only system needs this exactly as much as a dual one does.
///
/// A `loop`, not a single poll: when a completion reports
/// [`TaskPollResult::ReadyAndContinue`] (its completion directly claimed a
/// waiting `AsyncJoiner`), this continues straight into that descriptor's
/// own poll on the next iteration instead of returning control to the
/// outer dispatch loop — symmetric transfer, skipping a deque push/pop
/// round trip for the common case where a parent was waiting on exactly
/// the task that just finished. A `loop` rather than a recursive call, so
/// an arbitrarily long completion chain (however deep the fork-join
/// recursion) costs no native call-stack depth.
pub(crate) fn run_async_poll<S>(
    wk: &UltWorker<S>,
    mut desc: *mut S::Desc,
    mut poll_fn: TaskPollFn<S::Desc>,
) where
    S: SchedulerSystem,
    S::Desc: AsyncTaskDesc,
{
    // Whatever this worker was polling (if anything) before this call —
    // restored once, after the whole chain below is done, not per
    // iteration (see the loop body for why).
    let prev_polling = wk.polling_async.get();

    loop {
        // Mark as POLLING so the waker's state machine works correctly.
        unsafe { (*desc).mark_polling() };

        // Same bookkeeping the stackful switch shims do on every real
        // context switch: publish `wk` on the descriptor itself (and, for
        // an arena-backed AsyncPool, on the cell slot too). Lets anything
        // holding a pointer into this task's own arena cell — e.g.
        // `JoinHandle::poll`'s `self` address, see
        // `worker_from_async_arena_addr` — find `wk` via address masking
        // instead of a TLS lookup.
        unsafe { (*desc).mark_resumed_on(wk as *const UltWorker<S> as *const ()) };

        let raw = RawWaker::new(desc as *const (), crate::ult::waker::async_task_private_vtable::<S>());
        let waker = unsafe { Waker::from_raw(raw) };

        // Record that `desc` is the task this worker is polling right now,
        // so `JoinHandle::poll` (reachable synchronously from `poll_fn`
        // below via any `.await` on a child) can recognize its ambient
        // waker as this task's own instead of boxing a fresh one.
        wk.polling_async.set(desc);

        let mut cx = std::task::Context::from_waker(&waker);
        let result = unsafe { poll_fn(desc, &mut cx) };

        // waker is dropped here; drop_async_private is a no-op for PRIVATE mode.
        drop(waker);

        match result {
            TaskPollResult::Ready => break,
            TaskPollResult::Pending => {
                // Park, unless a wake raced in during poll() -- then
                // re-queue immediately instead.
                if !unsafe { (*desc).park_after_poll() } {
                    wk.push_local_top(SuspendedUlt(desc));
                }
                break;
            }
            TaskPollResult::ReadyAndContinue(next) => {
                poll_fn = unsafe { (*next).poll_fn().get() }.expect(
                    "cmpth: symmetric-transfer target has no poll_fn (not a spawn_async task)",
                );
                desc = next;
                // loop: poll `next` directly, no deque round trip.
            }
        }
    }

    wk.polling_async.set(prev_polling);
}

/// `execute` body for stackless-only systems: every popped continuation is
/// a `spawn_async` task, so always poll — no `poll_fn` tag check, because
/// there is nothing else it could be.
pub fn execute_async<S>(wk: &UltWorker<S>, cont: SuspendedUlt<S::Desc>)
where
    S: SchedulerSystem,
    S::Desc: AsyncTaskDesc,
{
    let desc = cont.desc();
    let poll_fn = unsafe { (*desc).poll_fn().get() }
        .expect("cmpth: execute_async called on a continuation with no poll_fn (not a spawn_async task)");
    let _ = cont.into_raw(); // consumed; no context switch
    run_async_poll(wk, desc, poll_fn);
}

// ---------------------------------------------------------------------------
// Free function kept for call-site compatibility
// ---------------------------------------------------------------------------

pub fn current_worker<S: SchedulerSystem>() -> Option<&'static UltWorker<S>> {
    UltWorker::<S>::current()
}

// ---------------------------------------------------------------------------
// Shims: extern "C" callbacks handed to the context-switch layer.
//
// Each shim runs on the destination stack.  Read everything out of the
// payload (which lives on the now-frozen previous stack) *before* doing
// anything that could allow the previous context to resume.
// ---------------------------------------------------------------------------

struct SuspendPayload<S: UltSchedulerSystem, F>
where
    S::Desc: StackfulTaskDesc,
{
    wk: *const UltWorker<S>,
    next: *mut S::Desc,
    f: ManuallyDrop<F>,
}

unsafe extern "C" fn suspend_shim<S, F>(prev: Context, a1: *mut (), _a2: *mut ()) -> Transfer
where
    S: UltSchedulerSystem,
    S::Desc: StackfulTaskDesc,
    F: FnOnce(&UltWorker<S>, SuspendedUlt<S::Desc>),
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

struct CondSuspendPayload<S: UltSchedulerSystem, F>
where
    S::Desc: StackfulTaskDesc,
{
    wk: *const UltWorker<S>,
    next: *mut Option<SuspendedUlt<S::Desc>>,
    f: ManuallyDrop<F>,
}

unsafe extern "C" fn cond_suspend_shim<S, F>(prev: Context, a1: *mut (), _a2: *mut ()) -> CondTransfer
where
    S: UltSchedulerSystem,
    S::Desc: StackfulTaskDesc,
    F: FnOnce(&UltWorker<S>, &mut Option<SuspendedUlt<S::Desc>>),
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

struct ExitPayload<S: UltSchedulerSystem, F>
where
    S::Desc: StackfulTaskDesc,
{
    wk: *const UltWorker<S>,
    next: *mut S::Desc,
    f: ManuallyDrop<F>,
}

unsafe extern "C" fn exit_shim<S, F>(a1: *mut (), _a2: *mut ()) -> Transfer
where
    S: UltSchedulerSystem,
    S::Desc: StackfulTaskDesc,
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
