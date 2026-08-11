//! [`ScopedWorker`]/[`ScopedRegistry`] — [`ScopedTaskSystem`](super::ScopedTaskSystem)'s
//! [`WorkerSystem`] implementation, reusing `resumable/common`'s
//! [`WorkerRunQueue`]/[`CurrentLookup`] machinery instead of a bespoke
//! deque/thread-local pair.
//!
//! Deliberately *not* [`UltWorker`](crate::resumable::common::worker::UltWorker):
//! that struct is the `resumable`-engine's own concrete worker, tied to
//! [`PoolSystem`](crate::resumable::common::system::PoolSystem) (pooled
//! descriptors, root continuations, context-switch state) that `scoped` has
//! none of. `ScopedWorker` is the second, genuinely independent
//! implementation of [`WorkerOps`]/[`LocalQueue`] this crate has — proof
//! that the `Worker`/`SuspendedToken` axes are actually swappable, not just
//! declared as such (`docs/traits-redesign.md`§11 item 9).

use std::cell::Cell;
use std::sync::atomic::AtomicBool;

use crate::os::OsSystem;
use crate::resumable::common::deque::{HybridRunQueue, RunQueueStealer, Steal, WorkerRunQueue};
use crate::resumable::common::lookup::CurrentLookup;
use crate::resumable::common::system::{RunnableItem, WorkerSystem};
use crate::resumable::common::worker::{LocalQueue, WorkerOps};
use crate::traits::component::tls::TlsAnchor;
use crate::traits::common::TlsSlot;
use crate::traits::stackful::NestableSystem;

use super::system::ScopedTaskSystem;
use super::task::TaskRef;

// ---------------------------------------------------------------------------
// ScopedLookup — a fast current-worker lookup, deliberately not TlsCurrent
// ---------------------------------------------------------------------------

// A plain, ordinarily-inlinable native thread-local — not `OsTls` (reached
// via `TlsCurrent`/`WorkerSystem::worker_tls()`), whose `get()` is
// deliberately `#[inline(never)]` to survive a *ULT* context switch (an
// opaque `extern "C"` call the compiler must not see through, or it may CSE
// the thread-local base address across it — see `OsTls::get`'s own doc
// comment). `scoped` has no context switch at all: a `ScopedWorker` never
// migrates to a different OS thread mid-flight, so that hazard does not
// apply here, but reusing `TlsCurrent` anyway paid its cost regardless —
// measured 20-30% slower on `parallel_call`'s hot unstolen path on `fib`.
// `Lookup` is its own pluggable axis on `WorkerSystem` precisely so a
// system that doesn't need `OsTls`'s safety property doesn't have to pay
// for it; this is that axis actually being exercised, not a workaround.
thread_local! {
    static CURRENT: Cell<*const ScopedWorker> = const { Cell::new(std::ptr::null()) };
}

pub struct ScopedLookup;

impl CurrentLookup<ScopedTaskSystem> for ScopedLookup {
    fn current() -> Option<&'static ScopedWorker> {
        let p = CURRENT.with(|c| c.get());
        if p.is_null() { None } else { Some(unsafe { &*p }) }
    }
}

/// Set (or clear, with a null pointer) the current OS thread's worker.
/// `scoped`'s own construction sites use this directly instead of
/// `WorkerSystem::worker_tls().set(...)`, matching how `ScopedLookup::current`
/// reads from this same thread-local rather than `worker_tls()`'s `OsTls`
/// slot — `worker_tls()` still exists to satisfy the trait, but nothing
/// here actually routes through it.
pub(super) fn set_current(wk: *const ScopedWorker) {
    CURRENT.with(|c| c.set(wk));
}

// ---------------------------------------------------------------------------
// ScopedRegistry — shared worker-pool state (the scoped-flavor Scheduler<S>)
// ---------------------------------------------------------------------------

/// Worker-pool state shared by one `run`/`run_async`/`init` call's workers.
/// The `scoped`-flavor counterpart of
/// [`Scheduler<S>`](crate::resumable::common::scheduler::Scheduler) — leaner,
/// since there is no task-pool axis at all here (`ScopedTaskSystem`
/// implements `WorkerSystem` alone, never `PoolSystem`).
pub(super) struct ScopedRegistry {
    pub(super) workers: Box<[ScopedWorker]>,
    /// Cloneable stealer handles, one per worker, indexed the same as
    /// `workers` — same "thieves never reach `workers[victim]` directly"
    /// structure as `Scheduler::stealers`, for the same reason.
    pub(super) stealers: Box<[<HybridRunQueue<TaskRef> as WorkerRunQueue<TaskRef>>::Stealer]>,
    pub(super) finished: AtomicBool,
}

// ---------------------------------------------------------------------------
// ScopedWorker
// ---------------------------------------------------------------------------

pub struct ScopedWorker {
    num: usize,
    deque: HybridRunQueue<TaskRef>,
    steal_seed: Cell<usize>,
    shared: Cell<*const ScopedRegistry>,
}

// SAFETY: same justification as `UltWorker`'s `unsafe impl Send/Sync` —
// `Cell` fields are only ever touched by the owning OS thread (this engine
// pins one worker per OS thread for its whole lifetime, never migrates a
// worker struct itself), `deque` is internally synchronized.
unsafe impl Send for ScopedWorker {}
unsafe impl Sync for ScopedWorker {}

impl ScopedWorker {
    pub(super) fn new(num: usize) -> Self {
        ScopedWorker {
            num,
            deque: HybridRunQueue::default(),
            steal_seed: Cell::new(num.wrapping_mul(0x9E37_79B9).wrapping_add(1)),
            shared: Cell::new(std::ptr::null()),
        }
    }

    pub(super) fn deque_stealer(&self) -> <HybridRunQueue<TaskRef> as WorkerRunQueue<TaskRef>>::Stealer {
        self.deque.stealer()
    }

    pub(super) fn bind_registry(&self, registry: *const ScopedRegistry) {
        self.shared.set(registry);
    }

    fn shared(&self) -> &ScopedRegistry {
        // SAFETY: set once via `bind_registry`, before this worker is ever
        // driven, by the same construction sequence `UltWorker::bind_scheduler`
        // uses; outlives every call through it (the registry owns the OS
        // threads that could still be running).
        unsafe { &*self.shared.get() }
    }
}

impl LocalQueue<ScopedTaskSystem> for ScopedWorker {
    fn push(&self, c: TaskRef) {
        self.deque.push(c);
    }

    fn defer(&self, c: TaskRef) {
        self.deque.defer(c);
    }

    fn try_pop(&self) -> Option<TaskRef> {
        self.deque.try_pop()
    }

    fn try_steal(&self) -> Steal<TaskRef> {
        let shared = self.shared();
        let n = shared.workers.len();
        if n <= 1 {
            return Steal::Empty;
        }
        let seed = self.steal_seed.get();
        self.steal_seed.set(seed.wrapping_add(1));
        let mut saw_retry = false;
        for i in 0..n {
            let victim = (seed + i) % n;
            if victim == self.num {
                continue;
            }
            match shared.stealers[victim].try_steal() {
                Steal::Success(c) => return Steal::Success(c),
                Steal::Retry => saw_retry = true,
                Steal::Empty => {}
            }
        }
        if saw_retry { Steal::Retry } else { Steal::Empty }
    }

    fn num(&self) -> usize {
        self.num
    }

    fn num_workers(&self) -> usize {
        self.shared().workers.len()
    }
}

impl WorkerOps<ScopedTaskSystem> for ScopedWorker {
    fn current() -> Option<&'static Self> {
        <ScopedLookup as CurrentLookup<ScopedTaskSystem>>::current()
    }
}

// ---------------------------------------------------------------------------
// WorkerSystem for ScopedTaskSystem
// ---------------------------------------------------------------------------

impl WorkerSystem for ScopedTaskSystem {
    type Base = OsSystem;
    type SuspendedToken = TaskRef;
    type RunQueue = HybridRunQueue<TaskRef>;
    type Lookup = ScopedLookup;
    type Worker = ScopedWorker;

    fn worker_tls() -> &'static <OsSystem as NestableSystem>::ThreadSpecific<ScopedWorker> {
        static ANCHOR: TlsAnchor = TlsAnchor::new();
        TlsSlot::from_anchor(&ANCHOR)
    }
}

// ---------------------------------------------------------------------------
// RunnableItem for TaskRef — dispatch is just the existing trampoline
// ---------------------------------------------------------------------------

impl RunnableItem<ScopedTaskSystem> for TaskRef {
    fn run_on(self, _wk: &ScopedWorker) {
        // SAFETY: every `TaskRef` in circulation was built by
        // `StackTask::as_task_ref`/`AsyncTask::as_task_ref`, whose
        // `execute_fn` is a valid trampoline for `data` by construction.
        unsafe { self.execute() }
    }
}
