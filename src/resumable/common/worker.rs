//! Base worker traits ([`TaskPool`]/[`LocalQueue`]/[`WorkerOps`]) and the
//! concrete [`UltWorker<S>`] implementation — usable by a stackful-only,
//! stackless-only, or dual system alike, no context-switch machinery named
//! anywhere here.
//!
//! Running a popped item is now [`RunnableItem::run_on`](crate::resumable::common::system::RunnableItem::run_on),
//! implemented directly on the item type per descriptor flavor — see that
//! trait's doc comment. `WorkerOps` here only locates the current worker;
//! it no longer has a dispatch method of its own. The stackful extension
//! traits ([`ContextSwitcher`](crate::resumable::stackful::worker::ContextSwitcher)/
//! [`StackfulLocalQueue`](crate::resumable::stackful::worker::StackfulLocalQueue)/
//! [`StackfulWorker`](crate::resumable::stackful::worker::StackfulWorker))
//! live in `stackful::worker`.

use std::alloc::Layout;
use std::cell::Cell;
use std::ptr;

use crate::resumable::common::deque::{RunQueueStealer, Steal, WorkerRunQueue};
use crate::resumable::common::pool::{DescPool, DynamicPool};
use crate::resumable::common::scheduler::Scheduler;
use crate::resumable::common::system::{PoolSystem, WorkerSystem};
use crate::resumable::common::desc::{RunningTaskToken, SuspendedTaskToken, TaskDescAlloc};

// ---------------------------------------------------------------------------
// TaskPool (base)
// ---------------------------------------------------------------------------

/// Task-descriptor allocation with a per-worker free list. Descriptor-flavor
/// only — `S: PoolSystem` too, unlike [`WorkerOps`]/[`LocalQueue`], which a
/// bare `WorkerSystem` with no pooled descriptor concept at all (e.g.
/// `scoped`'s stack-resident items) never needs.
pub trait TaskPool<S: WorkerSystem + PoolSystem> {
    /// Allocate a descriptor for a ULT stack. The size comes from the
    /// pool's own configuration (`Scheduler::stack_size`, set by
    /// [`StackfulBuilder::stack_size`](crate::traits::system::stackful::StackfulBuilder::stack_size)),
    /// not from the caller — callers neither know nor pass it. Returns it
    /// already wrapped as an owned token — see [`DescPool::alloc`], which
    /// this delegates to.
    fn alloc_task(&self, has_handle: bool) -> SuspendedTaskToken<S::Desc>;

    /// Return a dead descriptor to the pool.
    ///
    /// # Safety
    /// No other references to `desc` may exist after this call.
    unsafe fn free_task(&self, desc: *mut S::Desc);
}

// ---------------------------------------------------------------------------
// AsyncTaskPool (base)
// ---------------------------------------------------------------------------

/// `spawn_async`-descriptor allocation with a per-worker free list. Separate
/// from [`TaskPool`]: a `spawn_async` slot comes from
/// [`PoolSystem::AsyncPool`],
/// a different pool from the ULT-stack
/// `S::Pool` `TaskPool` allocates from (a dual system needs both live at
/// once — see [`PoolSystem::AsyncPool`]'s doc comment) — and this trait
/// is stackless-only, unlike `TaskPool`, which every system needs.
pub trait AsyncTaskPool<S: WorkerSystem + PoolSystem> {
    /// Allocate a descriptor with storage for at least `size` bytes.
    /// Returns it already wrapped as an owned token — see
    /// [`DescPool::alloc`], which this delegates to.
    fn alloc_async_task(&self, has_handle: bool, size: usize) -> SuspendedTaskToken<S::Desc>;

    /// Return a dead descriptor to the pool.
    ///
    /// # Safety
    /// No other references to `desc` may exist after this call.
    unsafe fn free_async_task(&self, desc: *mut S::Desc);
}

// ---------------------------------------------------------------------------
// RecursionAlloc (base)
// ---------------------------------------------------------------------------

/// Per-worker free-list allocator backing
/// [`stackless::thread::recurse`](crate::resumable::stackless::thread::recurse)'s
/// per-frame storage. Unlike [`TaskPool`]/[`AsyncTaskPool`], this names no
/// descriptor type in its signature — it is a plain sized allocator (see
/// [`PoolSystem::RecursionPool`]),
/// unrelated to descriptors, so it needs neither `S` nor `D` to spell out
/// what it hands back.
pub trait RecursionAlloc {
    /// Allocate (or reuse a freed block for) `layout`.
    fn alloc_recursion_frame(&self, layout: Layout) -> *mut u8;

    /// Return a block to the pool.
    ///
    /// # Safety
    /// `ptr` must have come from [`alloc_recursion_frame`](Self::alloc_recursion_frame)
    /// on this same worker with this same `layout`, and no other references
    /// to it may exist after this call.
    unsafe fn free_recursion_frame(&self, ptr: *mut u8, layout: Layout);
}

// ---------------------------------------------------------------------------
// LocalQueue (base)
// ---------------------------------------------------------------------------

/// Per-worker work-stealing run queue, independent of task flavor.
///
/// Speaks [`WorkerSystem::SuspendedToken`], never `SuspendedTaskToken<S::Desc>`:
/// this layer moves work around without looking inside it, so naming the
/// descriptor here would be a claim it does not need to make. The item stops
/// being opaque exactly one step later, when a caller with a concrete token
/// in hand calls
/// [`RunnableItem::run_on`](crate::resumable::common::system::RunnableItem::run_on)
/// on it directly.
pub trait LocalQueue<S: WorkerSystem> {
    /// Run `c` next on this worker (will run before anything already
    /// queued).
    fn push(&self, c: S::SuspendedToken);

    /// Run `c` after work already queued here (yield: let other tasks run
    /// first).
    fn defer(&self, c: S::SuspendedToken);

    /// Take what this worker should run next.
    fn try_pop(&self) -> Option<S::SuspendedToken>;

    /// Try to steal one task from another worker. `Steal::Retry` means some
    /// victim had work but it could not be taken right now — distinct from
    /// `Steal::Empty` (every victim scanned was genuinely empty) so callers
    /// don't mistake contention for idleness.
    fn try_steal(&self) -> Steal<S::SuspendedToken>;

    /// This worker's index within its scheduler.
    fn num(&self) -> usize;

    /// Total number of workers in this scheduler instance.
    fn num_workers(&self) -> usize;
}

// ---------------------------------------------------------------------------
// WorkerOps (base)
// ---------------------------------------------------------------------------

/// Base worker interface: locating the current worker. Named `WorkerOps`
/// (not `Worker`) to keep the name free for [`WorkerSystem::Worker`] — the
/// associated type naming which concrete struct implements this trait for a
/// given system.
///
/// Deliberately minimal — just `current()` — so a bare `WorkerSystem` with
/// no pooled descriptor concept at all (`scoped`'s stack-resident items,
/// which have no `Desc`/`PoolSystem` at all) can satisfy
/// `WorkerSystem::Worker: WorkerOps<Self>` without needing a `Desc` to name.
/// The descriptor-flavor accessors that used to live here
/// (`cur_task`/`external_queue`/`polling_async`/...) moved to
/// [`DescWorkerOps`], which additionally requires `S: PoolSystem`.
pub trait WorkerOps<S: WorkerSystem>: LocalQueue<S> + Send + Sync + 'static {
    /// The worker currently running on this base thread, if any.
    fn current() -> Option<&'static Self>
    where
        Self: Sized;
}

// ---------------------------------------------------------------------------
// DescWorkerOps — the descriptor-flavor extension of WorkerOps
// ---------------------------------------------------------------------------

/// Descriptor-flavor worker accessors: everything a `resumable`-engine
/// worker (stackful/stackless/dual) needs beyond plain [`WorkerOps`], all
/// typed by `S::Desc`/`S::ExternalQueue` (so `S: PoolSystem` too). A bare
/// `WorkerSystem` with no pooled descriptor concept (`scoped`) never
/// implements this — its worker only needs [`WorkerOps`]/[`LocalQueue`].
pub trait DescWorkerOps<S: WorkerSystem + PoolSystem>: WorkerOps<S> + TaskPool<S> {
    /// Raw pointer to the task currently running on this worker, without
    /// taking ownership; null if nothing is running. Crate-internal —
    /// lets call sites generic over `S::Worker` reach
    /// [`UltWorker::cur_task`], which they cannot name directly. `#[doc(hidden)]`
    /// because a trait method can't be narrowed below its trait's own
    /// visibility (same pattern as
    /// [`StackAlloc::alloc_stack`](crate::resumable::common::stack::StackAlloc::alloc_stack)) —
    /// this is `pub` only because `DescWorkerOps` itself is, not because
    /// it's meant to be called outside this crate.
    #[doc(hidden)]
    fn cur_task(&self) -> *mut S::Desc;

    /// Safe counterpart to [`cur_task`](Self::cur_task): panics instead of
    /// risking a null deref if nothing is running. Crate-internal, see
    /// [`UltWorker::cur_task_ref`]; `#[doc(hidden)]` for the same reason as
    /// [`cur_task`](Self::cur_task).
    #[doc(hidden)]
    fn cur_task_ref(&self) -> &S::Desc;

    /// Mutable peek at the currently-running task's token, for callers with
    /// no explicit `RunningTaskToken` in scope. Crate-internal, see
    /// [`UltWorker::cur_task_token_mut`]; `#[doc(hidden)]` for the same
    /// reason as [`cur_task`](Self::cur_task).
    #[doc(hidden)]
    fn cur_task_token_mut(&self) -> &mut RunningTaskToken<S::Desc>;

    /// Where an off-pool `wake()` for a task created on this worker will
    /// deliver. Crate-internal, see [`UltWorker::external_queue`];
    /// `#[doc(hidden)]` for the same reason as [`cur_task`](Self::cur_task).
    #[doc(hidden)]
    fn external_queue(&self) -> &S::ExternalQueue;

    /// The descriptor `run_async_poll` is currently synchronously driving on
    /// this worker, or null. Crate-internal, see [`UltWorker`]'s
    /// `polling_async` field for the full invariant; `#[doc(hidden)]` for
    /// the same reason as [`cur_task`](Self::cur_task). Declared uniformly
    /// here (like [`PoolSystem::AsyncPool`])
    /// even though only stackless dispatch (`run_async_poll`,
    /// `JoinHandle::poll`'s fast path, `yield_now`) ever reads it — a
    /// stackful-only worker simply never sets it.
    #[doc(hidden)]
    fn polling_async(&self) -> *mut S::Desc;

    /// Set the [`polling_async`](Self::polling_async) marker.
    /// `#[doc(hidden)]` for the same reason.
    #[doc(hidden)]
    fn set_polling_async(&self, desc: *mut S::Desc);

    /// Take (read and reset to `false`) the yield-requested marker. See
    /// [`UltWorker`]'s `yield_requested` field for the full invariant.
    /// `#[doc(hidden)]` for the same reason.
    #[doc(hidden)]
    fn take_yield_requested(&self) -> bool;

    /// Set the yield-requested marker. `#[doc(hidden)]` for the same
    /// reason.
    #[doc(hidden)]
    fn set_yield_requested(&self, v: bool);
}

// ---------------------------------------------------------------------------
// Concrete implementation: UltWorker<S>
// ---------------------------------------------------------------------------

pub struct UltWorker<S: WorkerSystem + PoolSystem> {
    num: usize,
    pub(crate) deque: S::RunQueue,
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
    pub(crate) root_cont: Cell<Option<SuspendedTaskToken<S::Desc>>>,
    steal_seed: Cell<usize>,
    shared: Cell<*const Scheduler<S>>,
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
    /// Set by [`StacklessTaskSystem::yield_now`](crate::traits::stackless::StacklessTaskSystem::yield_now)'s
    /// first poll when it observes `polling_async` non-null (i.e. it is
    /// running as the task `run_async_poll` is synchronously driving right
    /// now), so that function's `Pending` arm knows to requeue via `defer`
    /// (fair) instead of the ordinary self-wake path's `push`. Read (via
    /// `take`, clearing it) exactly once per poll, unconditionally, in
    /// every `TaskPollResult` arm — not only the one that acts on it — so a
    /// `true` left behind by a task that then returned `Ready` (or parked
    /// properly instead of self-waking) can never leak into whatever gets
    /// polled next on this worker. See `run_async_poll`
    /// (`resumable/stackless/worker.rs`) and `yield_now`'s blanket impl
    /// (`resumable/stackless/system.rs`).
    pub(crate) yield_requested: Cell<bool>,
}

// `Cell` fields are only accessed by the owning base thread; `deque` is
// internally synchronized; `shared` is read-only after init. None of this
// (nor the inherent methods below) touches dispatch, so `WorkerSystem` is
// enough — no need for `SchedulerSystem`.
unsafe impl<S: WorkerSystem + PoolSystem> Send for UltWorker<S> {}
unsafe impl<S: WorkerSystem + PoolSystem> Sync for UltWorker<S> {}

impl<S: WorkerSystem + PoolSystem> UltWorker<S> {
    pub(crate) fn new(num: usize) -> Self {
        UltWorker {
            num,
            deque: S::RunQueue::default(),
            cur_task_cell: Cell::new(None),
            root_desc: S::Desc::new_root(),
            root_cont: Cell::new(None),
            steal_seed: Cell::new(num.wrapping_mul(0x9E37_79B9).wrapping_add(1)),
            shared: Cell::new(ptr::null()),
            polling_async: Cell::new(ptr::null_mut()),
            yield_requested: Cell::new(false),
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

    fn shared(&self) -> &Scheduler<S> {
        unsafe { &*self.shared.get() }
    }

    /// Called once per worker at pool construction, before any worker runs —
    /// the only write to the `shared` field. Every other access goes through
    /// a narrow accessor (e.g. [`external_queue`](Self::external_queue)) or
    /// stays private to this module.
    pub(crate) fn bind_scheduler(&self, sched: *const Scheduler<S>) {
        self.shared.set(sched);
    }

    /// The one component a task-creation path needs out of the scheduler:
    /// where an off-pool `wake()` for the task being created will deliver —
    /// see [`HasExternalQueue`](crate::resumable::common::desc::HasExternalQueue).
    pub(crate) fn external_queue(&self) -> &S::ExternalQueue {
        &self.shared().external_queue
    }

    /// Take the stored root (scheduler-loop) continuation. Shared by
    /// `pop_or_root_stackful`/`pop_or_root_dual`.
    pub(crate) fn take_root_cont(&self) -> SuspendedTaskToken<S::Desc> {
        self.root_cont.take()
            .unwrap_or_else(|| panic!("no runnable continuation on worker {}", self.num))
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
    ///
    /// The `debug_assert` compares thin raw pointers erased to `*const ()`
    /// rather than `self`/`cur` directly: `<S::Worker as
    /// WorkerOps<S>>::current()` hands back `&S::Worker`, not `&Self`, and
    /// nothing here needs those two types to be the same (that equality is
    /// the very `DescScheduler` pin this method used to require solely to
    /// make `std::ptr::eq` typecheck) — identity is still fully decided by
    /// comparing the two addresses.
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn cur_task_token_mut(&self) -> &mut RunningTaskToken<S::Desc> {
        debug_assert!(
            <S::Worker as WorkerOps<S>>::current()
                .is_some_and(|cur| cur as *const S::Worker as *const () == self as *const Self as *const ()),
            "cmpth: cur_task_token_mut called from a thread not currently running as this worker"
        );
        let opt: &mut Option<RunningTaskToken<S::Desc>> = unsafe { &mut *self.cur_task_cell.as_ptr() };
        opt.as_mut().expect("cmpth: no current task on worker")
    }
}

// --- TaskPool ---

impl<S: WorkerSystem + PoolSystem> TaskPool<S> for UltWorker<S> {
    fn alloc_task(&self, has_handle: bool) -> SuspendedTaskToken<S::Desc> {
        let shared = self.shared();
        let ptr = shared.task_pool.alloc(self.num, has_handle, shared.stack_size);
        // SAFETY: `ptr` was just freshly allocated by `DescPool::alloc`
        // (fresh memory, or a pool slot reinitialized with no live token
        // pointing at it) — trivially exclusive. This is the one place
        // that wraps it: `DescPool` itself stays raw-pointer-based since
        // it's a pluggable `traits/component/` trait that must not have to
        // name this crate's own `SuspendedTaskToken`.
        unsafe { SuspendedTaskToken::from_raw(ptr) }
    }

    unsafe fn free_task(&self, desc: *mut S::Desc) {
        unsafe { self.shared().task_pool.dealloc(self.num, desc) };
    }
}

// --- AsyncTaskPool ---

impl<S: WorkerSystem + PoolSystem> AsyncTaskPool<S> for UltWorker<S> {
    fn alloc_async_task(&self, has_handle: bool, size: usize) -> SuspendedTaskToken<S::Desc> {
        let ptr = self.shared().async_task_pool.alloc(self.num, has_handle, size);
        // SAFETY: same reasoning as `TaskPool::alloc_task` above.
        unsafe { SuspendedTaskToken::from_raw(ptr) }
    }

    unsafe fn free_async_task(&self, desc: *mut S::Desc) {
        unsafe { self.shared().async_task_pool.dealloc(self.num, desc) };
    }
}

// --- RecursionAlloc ---

impl<S: WorkerSystem + PoolSystem> RecursionAlloc for UltWorker<S> {
    fn alloc_recursion_frame(&self, layout: Layout) -> *mut u8 {
        self.shared().recursion_pool.alloc(self.num, layout)
    }

    unsafe fn free_recursion_frame(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.shared().recursion_pool.dealloc(self.num, ptr, layout) };
    }
}

// --- LocalQueue ---

// Bounded by bare `WorkerSystem`, not `SchedulerSystem`: `LocalQueue` speaks
// `S::SuspendedToken` rather than `SuspendedTaskToken<S::Desc>`,
// nothing here needs the two to be equal, nor dispatch capability. `self.deque`
// is already `S::RunQueue: WorkerRunQueue<S::Item>` and `shared.stealers`
// already yields `Steal<S::Item>`, so every body below type-checks against the
// opaque item alone. The equality is still needed one layer up, wherever a
// caller hands a concrete token to these methods — but that is the task
// layer, which legitimately knows the descriptor type.
impl<S: WorkerSystem + PoolSystem> LocalQueue<S> for UltWorker<S> {
    fn push(&self, c: S::SuspendedToken) {
        self.deque.push(c);
    }

    fn defer(&self, c: S::SuspendedToken) {
        self.deque.defer(c);
    }

    fn try_pop(&self) -> Option<S::SuspendedToken> {
        self.deque.try_pop()
    }

    fn try_steal(&self) -> Steal<S::SuspendedToken> {
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
            // Reached exclusively through `shared.stealers[victim]` -- a
            // plain, already-built stealer handle -- never through
            // `shared.workers[victim]` itself. See `Scheduler::stealers`'s
            // doc comment for why that's more than a style preference.
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

// --- WorkerOps ---

impl<S: WorkerSystem<Worker = UltWorker<S>> + PoolSystem> WorkerOps<S> for UltWorker<S> {
    fn current() -> Option<&'static Self> {
        <S::Lookup as crate::resumable::common::lookup::CurrentLookup<S>>::current()
    }
}

impl<S: WorkerSystem<Worker = UltWorker<S>> + PoolSystem> DescWorkerOps<S> for UltWorker<S> {
    fn cur_task(&self) -> *mut S::Desc {
        UltWorker::cur_task(self)
    }

    fn cur_task_ref(&self) -> &S::Desc {
        UltWorker::cur_task_ref(self)
    }

    fn cur_task_token_mut(&self) -> &mut RunningTaskToken<S::Desc> {
        UltWorker::cur_task_token_mut(self)
    }

    fn external_queue(&self) -> &S::ExternalQueue {
        UltWorker::external_queue(self)
    }

    fn polling_async(&self) -> *mut S::Desc {
        self.polling_async.get()
    }

    fn set_polling_async(&self, desc: *mut S::Desc) {
        self.polling_async.set(desc);
    }

    fn take_yield_requested(&self) -> bool {
        self.yield_requested.take()
    }

    fn set_yield_requested(&self, v: bool) {
        self.yield_requested.set(v);
    }
}

// ---------------------------------------------------------------------------
// Free function kept for call-site compatibility
// ---------------------------------------------------------------------------

pub fn current_worker<S>() -> Option<&'static UltWorker<S>>
where
    S: WorkerSystem<Worker = UltWorker<S>> + PoolSystem,
{
    UltWorker::<S>::current()
}
