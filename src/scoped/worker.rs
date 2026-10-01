//! [`ScopedWorker`]/[`ScopedRegistry`] — [`ScopedSystem`](super::ScopedSystem)'s
//! [`WorkerSystem`] implementation. Everything here is `resumable/common`'s
//! shared substrate: the worker is a bare [`WorkerCore`] (run queue + steal
//! loop, no descriptor/context state), the pool state is a [`PoolCore`], and
//! worker threads run the same [`idle_loop`] the descriptor-backed flavors
//! run. What `scoped` does *not* take from the descriptor-backed flavors is
//! the task representation: no `PoolSystem`/`UltWorker`, since a branch is a
//! stack-resident [`TaskRef`], not a pooled descriptor.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::os::OsSystem;
use crate::resumable::common::deque::HybridRunQueue;
use crate::resumable::common::lookup::CurrentLookup;
use crate::resumable::common::system::{RunnableItem, WorkerSystem};
use crate::resumable::common::worker_core::{idle_loop, try_run_one, PoolCore, WorkerCore};
use crate::traits::common::TlsSlot;
use crate::traits::component::tls::TlsAnchor;
use crate::traits::stackful::{NestableSystem, SpawnableStackfulTaskSystem};
use crate::traits::stackful::JoinHandleLike;

use super::system::{ScopedIdentity, ScopedSystem};
use super::task::TaskRef;

/// `scoped`'s whole worker: just the shared queue layer.
pub type ScopedWorker<M> = WorkerCore<ScopedSystem<M>>;

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
//
// One slot, shared by every `M`: a `static` inside a generic fn/impl can't
// depend on the generic parameter (`E0401`), so this stores a type-erased
// `*const ()` rather than one genuine thread-local per `M`. Sound because a
// worker OS thread is always dedicated to exactly one concrete
// `ScopedSystem<M>` pool for its whole lifetime (spawned in
// `start_workers`, joined in `shutdown`, never reused for another pool) —
// nesting a *different* `M` inside a `parallel_call` running on the same
// thread is as unsupported as nesting the *same* `M` already was (see the
// assert in `sync_engine::init`); both clobber this shared slot the same
// way.
thread_local! {
    static CURRENT: Cell<*const ()> = const { Cell::new(std::ptr::null()) };
}

pub struct ScopedLookup<M>(PhantomData<M>);

impl<M: ScopedIdentity> CurrentLookup<ScopedSystem<M>> for ScopedLookup<M> {
    fn current() -> Option<&'static ScopedWorker<M>> {
        let p = CURRENT.with(Cell::get) as *const ScopedWorker<M>;
        if p.is_null() { None } else { Some(unsafe { &*p }) }
    }
}

/// Set (or clear, with a null pointer) the current OS thread's worker.
/// `scoped`'s own construction sites use this directly instead of
/// `WorkerSystem::worker_tls().set(...)`, matching how `ScopedLookup::current`
/// reads from this same thread-local rather than `worker_tls()`'s `OsTls`
/// slot — `worker_tls()` still exists to satisfy the trait, but nothing
/// here actually routes through it.
pub(super) fn set_current<M: ScopedIdentity>(wk: *const ScopedWorker<M>) {
    CURRENT.with(|c| c.set(wk as *const ()));
}

// ---------------------------------------------------------------------------
// ScopedRegistry — one pool's workers plus their shared core
// ---------------------------------------------------------------------------

/// Worker-pool state shared by one `run`/`run_async`/`init` call's workers.
/// The `scoped` counterpart of
/// [`Scheduler<S>`](crate::resumable::common::scheduler::Scheduler): the same
/// [`PoolCore`], minus the descriptor pools and external queue.
pub(super) struct ScopedRegistry<M: ScopedIdentity> {
    pub(super) workers: Box<[ScopedWorker<M>]>,
    pub(super) pool: PoolCore<ScopedSystem<M>>,
}

impl<M: ScopedIdentity> ScopedRegistry<M> {
    pub(super) fn new(num_workers: usize) -> Arc<Self> {
        assert!(num_workers >= 1, "need at least one worker");
        let workers: Box<[ScopedWorker<M>]> = (0..num_workers).map(ScopedWorker::<M>::new).collect();
        let pool = PoolCore::new(workers.iter());
        let registry = Arc::new(ScopedRegistry { workers, pool });
        for w in registry.workers.iter() {
            w.bind(&registry.pool);
        }
        registry
    }

    /// Start worker threads `1..n` running the idle loop; worker 0 is the
    /// calling thread's to drive.
    pub(super) fn start_workers(self: &Arc<Self>) -> Vec<<OsSystem as SpawnableStackfulTaskSystem>::JoinHandle<()>> {
        (1..self.workers.len())
            .map(|idx| {
                let registry = Arc::clone(self);
                OsSystem::spawn(move || {
                    let wk = &registry.workers[idx];
                    set_current(wk as *const ScopedWorker<M>);
                    idle_loop::<ScopedSystem<M>, _>(wk, &registry.pool, || None);
                    // Nested calls always leave the deque as they found it,
                    // so this only catches a straggler pushed just before
                    // shutdown was observed.
                    while try_run_one::<ScopedSystem<M>>(wk) {}
                    set_current::<M>(std::ptr::null());
                })
            })
            .collect()
    }

    pub(super) fn shutdown(&self, handles: Vec<<OsSystem as SpawnableStackfulTaskSystem>::JoinHandle<()>>) {
        self.pool.finish();
        for h in handles {
            JoinHandleLike::join(h);
        }
    }
}

// ---------------------------------------------------------------------------
// WorkerSystem for ScopedSystem
// ---------------------------------------------------------------------------

impl<M: ScopedIdentity> WorkerSystem for ScopedSystem<M> {
    type Base = OsSystem;
    type SuspendedToken = TaskRef;
    type RunQueue = HybridRunQueue<TaskRef>;
    type Lookup = ScopedLookup<M>;
    type Idle = M::Idle;
    type Worker = ScopedWorker<M>;

    // Dead code for every `M`: nothing in `scoped` ever calls
    // `Self::worker_tls()` (`ScopedLookup`/`CURRENT` above is what
    // `ScopedSystem`'s own `Lookup` actually reads), and the only generic
    // callers in `resumable::common`/`stackful::init` require `Worker =
    // UltWorker<S>`, which `ScopedWorker<M>` never is — so this never gets
    // monomorphized into a real call. That matters here because `static
    // ANCHOR: TlsAnchor` is declared once in this single generic `impl<M>`
    // block: a `static`'s *type* can't mention an outer generic parameter
    // (`E0401`), so unlike `worker_tls_anchor`'s per-concrete-type contract
    // elsewhere (each `UltIdentity` leaf writes its own `static`), one
    // `ANCHOR`/index here would be shared by every `M` if this were ever
    // actually exercised. Only satisfies `WorkerSystem`'s required
    // associated fn.
    fn worker_tls() -> &'static <OsSystem as NestableSystem>::ThreadSpecific<ScopedWorker<M>> {
        static ANCHOR: TlsAnchor = TlsAnchor::new();
        TlsSlot::from_anchor(&ANCHOR)
    }
}

// ---------------------------------------------------------------------------
// RunnableItem for TaskRef — dispatch is just the existing trampoline
// ---------------------------------------------------------------------------

impl<M: ScopedIdentity> RunnableItem<ScopedSystem<M>> for TaskRef {
    fn run_on(self, _wk: &ScopedWorker<M>) {
        // SAFETY: every `TaskRef` in circulation was built by
        // `StackTask::as_task_ref`/`AsyncTask::as_task_ref`, whose
        // `execute_fn` is a valid trampoline for `data` by construction.
        unsafe { self.execute() }
    }
}
