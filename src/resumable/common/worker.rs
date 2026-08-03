//! Base worker traits ([`TaskPool`]/[`LocalQueue`]/[`Worker`]) and the
//! concrete [`UltWorker<S>`] implementation — usable by a stackful-only,
//! stackless-only, or dual system alike, no context-switch machinery named
//! anywhere here.
//!
//! [`Worker::execute`] forwards to [`SchedulerSystem::execute`] — see that
//! method's doc comment for why the dispatch body lives on the system
//! trait rather than here (a required hook, monomorphized per concrete
//! system, not runtime dispatch). The stackful extension traits
//! ([`ContextSwitcher`](crate::resumable::stackful::worker::ContextSwitcher)/
//! [`StackfulLocalQueue`](crate::resumable::stackful::worker::StackfulLocalQueue)/
//! [`StackfulWorker`](crate::resumable::stackful::worker::StackfulWorker))
//! live in `stackful::worker`.

use std::cell::Cell;
use std::ptr;

use crate::resumable::common::deque::WorkerDeque;
use crate::resumable::common::pool::DescPool;
use crate::resumable::common::scheduler::Scheduler;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::desc::{RunningTaskToken, SuspendedTaskToken, TaskDescAlloc};

// ---------------------------------------------------------------------------
// TaskPool (base)
// ---------------------------------------------------------------------------

/// Task-descriptor allocation with a per-worker free list.
pub trait TaskPool<S: SchedulerSystem> {
    /// Allocate a descriptor with storage for at least `size` bytes (see
    /// [`DescPool::alloc`] — `spawn`
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
    fn push_local_top(&self, c: SuspendedTaskToken<S::Desc>);

    /// Push `c` to the **FIFO** end (yield: let other tasks run first).
    fn push_local_bottom(&self, c: SuspendedTaskToken<S::Desc>);

    /// Pop from the LIFO end of this worker's local deque.
    fn pop_local(&self) -> Option<SuspendedTaskToken<S::Desc>>;

    /// Try to steal one task from another worker's FIFO end.
    fn try_steal(&self) -> Option<SuspendedTaskToken<S::Desc>>;

    /// This worker's index within its scheduler.
    fn num(&self) -> usize;

    /// Total number of workers in this scheduler instance.
    fn num_workers(&self) -> usize;
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
    fn execute(&self, cont: SuspendedTaskToken<S::Desc>);
}

// ---------------------------------------------------------------------------
// Concrete implementation: UltWorker<S>
// ---------------------------------------------------------------------------

pub struct UltWorker<S: SchedulerSystem> {
    num: usize,
    pub(crate) deque: S::Deque,
    /// The task currently running on this worker, if any. `None` means
    /// nothing is running (mirrors the old `Cell<*mut S::Desc>`'s null
    /// convention). Deliberately `Option<RunningTaskToken<S::Desc>>`, not a bare
    /// pointer: `RunningTaskToken` is move-only, so `.take()`-ing it out of this
    /// cell is the *only* way to get a live handle, and the cell is
    /// provably empty for as long as that handle is in use — see
    /// `cur_task`/`take_cur_task`/`set_cur_task` below, and
    /// `RunningTaskToken`'s own doc comment (`resumable::common::desc`) for why
    /// this exists (it closes a real, load-bearing aliasing window that
    /// used to exist in `cond_suspend_shim`, verified by an Explore-agent
    /// audit of every place a "current task" pointer flowed through this
    /// scheduler, 2026-07-30).
    cur_task_cell: Cell<Option<RunningTaskToken<S::Desc>>>,
    root_desc: S::Desc,
    pub(crate) root_cont: Cell<*mut S::Desc>,
    steal_seed: Cell<usize>,
    pub(crate) shared: Cell<*const Scheduler<S>>,
    /// The descriptor currently being driven by `run_async_poll` on this
    /// worker, or null. Distinct from `cur_task` (which tracks real
    /// context-switch state and is meaningless for async polling): this is
    /// how `JoinHandle::poll` recognizes "the ambient waker is verifiably
    /// this task's own" without inspecting the waker itself, avoiding a
    /// `Box<Waker>` allocation on the common `spawn_async`/`.await` path.
    /// A bare pointer, not `RunningTaskToken`-wrapped: unlike `cur_task`, this
    /// field is a marker read by a *different* call chain
    /// (`JoinHandle::poll`) than the one that owns the descriptor
    /// (`run_async_poll`'s own `desc` local) — it never itself grants
    /// exclusive rights, so move discipline doesn't apply to it. The
    /// aliasing risk here (a stale marker outliving the moment the
    /// descriptor becomes reachable by another worker) is closed instead by
    /// `run_async_poll` clearing it *before* publishing the descriptor to
    /// the deque, not after — see that function.
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
            cur_task_cell: Cell::new(None),
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

    /// Peek at the raw pointer to the task currently running on this
    /// worker, without taking ownership — for callers that only need to
    /// read "what am I running right now" (sanity checks, `UltTls`,
    /// `UltPoller`, `DualResumable::assert_on_real_ult`), never to move or
    /// replace it. Null if nothing is running.
    ///
    /// # Safety of the shared read
    /// Constructs a `&Option<RunningTaskToken<S::Desc>>` via `Cell::as_ptr`
    /// instead of `Cell::get` (which would require `T: Copy`) — sound
    /// under the same "only the owning base thread ever touches this
    /// worker's `Cell` fields" protocol `UltWorker`'s `unsafe impl Sync`
    /// already rests on (see that impl), same as every other `Cell` field
    /// here.
    pub(crate) fn cur_task(&self) -> *mut S::Desc {
        let opt: &Option<RunningTaskToken<S::Desc>> = unsafe { &*self.cur_task_cell.as_ptr() };
        opt.as_ref().map_or(ptr::null_mut(), RunningTaskToken::desc)
    }

    /// Safe counterpart to [`cur_task`](Self::cur_task) for callers that
    /// know (structurally, not just by luck) that a task is actually
    /// running right now — i.e. every caller except the internal
    /// switch-shim window between `take_cur_task`/`set_cur_task`, where the
    /// cell is legitimately empty. Panics rather than risking a null deref
    /// if that assumption is ever wrong, same as
    /// [`cur_task_token_mut`](Self::cur_task_token_mut)'s existing
    /// `.expect()`.
    pub(crate) fn cur_task_ref(&self) -> &S::Desc {
        let opt: &Option<RunningTaskToken<S::Desc>> = unsafe { &*self.cur_task_cell.as_ptr() };
        opt.as_ref().expect("cmpth: no current task on worker").as_desc()
    }

    /// Mutable peek at the currently-running task's token, for callers with
    /// no explicit `RunningTaskToken` in scope (e.g. `UltTls::get`/`set`,
    /// reached through the generic `TlsSlot` trait) that still need
    /// `Owned`-field access. `D::Owned` is reached only through a token
    /// (never a separate direct path — see `TaskDesc::owned_cell`'s doc
    /// comment), so this is the one place besides an explicit token value
    /// that can produce one.
    ///
    /// Sound for the same reason as [`cur_task`](Self::cur_task)'s peek:
    /// only the OS thread currently running as this worker's current task
    /// ever calls this, so there is no concurrent access to guard against —
    /// checked by the `debug_assert` below rather than merely assumed. That
    /// invariant is exactly what clippy's `mut_from_ref` can't see from the
    /// `&self` signature alone (it would need proof "no second live
    /// `&mut`/`&` from this same `&self` exists," which the single-caller
    /// discipline above provides but the type system doesn't express).
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn cur_task_token_mut(&self) -> &mut RunningTaskToken<S::Desc> {
        debug_assert!(
            Self::current().is_some_and(|cur| std::ptr::eq(cur, self)),
            "cmpth: cur_task_token_mut called from a thread not currently running as this worker"
        );
        let opt: &mut Option<RunningTaskToken<S::Desc>> = unsafe { &mut *self.cur_task_cell.as_ptr() };
        opt.as_mut().expect("cmpth: no current task on worker")
    }

    /// Take exclusive ownership of the currently-running task out of this
    /// worker's slot, leaving it empty. Panics if nothing is running —
    /// every real call site only calls this while a task is known to be
    /// running (same implicit invariant the old `Cell<*mut S::Desc>`
    /// carried, just now checked instead of silently dereferencing null).
    pub(crate) fn take_cur_task(&self) -> RunningTaskToken<S::Desc> {
        self.cur_task_cell.take().expect("cmpth: no current task on worker")
    }

    /// Commit `task` as the task now running on this worker. Panics if the
    /// slot wasn't already empty — every real call site is expected to
    /// have `take_cur_task`d (or never populated) the slot first; silently
    /// overwriting a live `RunningTaskToken` would drop it without anyone
    /// noticing the ownership it represented just vanished.
    pub(crate) fn set_cur_task(&self, task: RunningTaskToken<S::Desc>) {
        let old = self.cur_task_cell.replace(Some(task));
        debug_assert!(old.is_none(), "cmpth: overwriting a live cur_task");
    }

    pub(crate) fn shared(&self) -> &Scheduler<S> {
        unsafe { &*self.shared.get() }
    }

    /// Take the stored root (scheduler-loop) continuation. Shared by
    /// `pop_or_root_stackful`/`pop_or_root_dual`.
    pub(crate) fn take_root_cont(&self) -> SuspendedTaskToken<S::Desc> {
        let root = self.root_cont.replace(ptr::null_mut());
        assert!(!root.is_null(), "no runnable continuation on worker {}", self.num);
        // SAFETY: `root_cont` only ever holds a pointer published by
        // `set_root_cont`'s `cont.into_raw()`, and `root_cont` (a plain
        // `Cell`, not shared across threads) is only ever touched by this
        // worker's own OS thread — no ordering is needed beyond that
        // single-thread discipline, and this `replace` is the sole
        // consumer of whatever was there.
        unsafe { SuspendedTaskToken::from_raw(root) }
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
    fn push_local_top(&self, c: SuspendedTaskToken<S::Desc>) {
        self.deque.push_top(c);
    }

    fn push_local_bottom(&self, c: SuspendedTaskToken<S::Desc>) {
        self.deque.push_bottom(c);
    }

    fn pop_local(&self) -> Option<SuspendedTaskToken<S::Desc>> {
        self.deque.try_pop_top()
    }

    fn try_steal(&self) -> Option<SuspendedTaskToken<S::Desc>> {
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

// --- Worker ---

impl<S: SchedulerSystem> Worker<S> for UltWorker<S> {
    fn current() -> Option<&'static Self> {
        <S::Lookup as crate::resumable::common::lookup::CurrentLookup<S>>::current()
    }

    fn execute(&self, cont: SuspendedTaskToken<S::Desc>) {
        S::execute(self, cont);
    }
}

// ---------------------------------------------------------------------------
// Free function kept for call-site compatibility
// ---------------------------------------------------------------------------

pub fn current_worker<S: SchedulerSystem>() -> Option<&'static UltWorker<S>> {
    UltWorker::<S>::current()
}
