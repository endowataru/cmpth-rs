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
use crate::resumable::common::desc::{SuspendedUlt, TaskDescAlloc};

// ---------------------------------------------------------------------------
// TaskPool (base)
// ---------------------------------------------------------------------------

/// Task-descriptor allocation with a per-worker free list.
pub trait TaskPool<S: SchedulerSystem> {
    /// Allocate a descriptor with storage for at least `size` bytes (see
    /// [`DescPool::alloc`](crate::resumable::common::pool::DescPool::alloc) — `spawn`
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
// Concrete implementation: UltWorker<S>
// ---------------------------------------------------------------------------

pub struct UltWorker<S: SchedulerSystem> {
    num: usize,
    pub(crate) deque: S::Deque,
    pub(crate) cur_task: Cell<*mut S::Desc>,
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
    pub(crate) fn take_root_cont(&self) -> SuspendedUlt<S::Desc> {
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

// --- Worker ---

impl<S: SchedulerSystem> Worker<S> for UltWorker<S> {
    fn current() -> Option<&'static Self> {
        <S::Lookup as crate::resumable::common::lookup::CurrentLookup<S>>::current()
    }

    fn execute(&self, cont: SuspendedUlt<S::Desc>) {
        S::execute(self, cont);
    }
}

// ---------------------------------------------------------------------------
// Free function kept for call-site compatibility
// ---------------------------------------------------------------------------

pub fn current_worker<S: SchedulerSystem>() -> Option<&'static UltWorker<S>> {
    UltWorker::<S>::current()
}
