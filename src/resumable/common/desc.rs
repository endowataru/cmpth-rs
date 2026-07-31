//! Task descriptors and continuations.
//!
//! A [`SuspendedTaskToken`] is an owning handle to a suspended task: exactly one
//! continuation exists per suspended task, and consuming it (switching into
//! the context) invalidates it.  This mirrors ComposableThreads'
//! `basic_sct_continuation` / `suspended_thread` ownership model and is what
//! removes the old `ctx_saving` / `TaskState::Suspending` handshake: a
//! continuation only comes into existence *after* the context is fully saved,
//! because it is created by the switch callback running on the next stack.
//!
//! # `TaskDesc`/`StackfulTaskDesc`/`AsyncTaskDesc`
//!
//! The field set lives behind named accessor traits rather than a single
//! hardcoded struct, mirroring [`crate::resumable::stackful::suspended::StackfulOnlyResumable`]
//! (implementors supply accessors; scheduler code only ever calls the
//! trait) — every direct `(*desc).field` touch across
//! `worker.rs`/`thread.rs`/`waker.rs`/`pool.rs`/`tls.rs` goes through a
//! named method, so a concrete descriptor type is a contract to implement,
//! not a fixed struct to match byte-for-byte.
//!
//! Three concrete types implement that contract, one per scheduler flavor:
//! [`StackfulOnlyTaskDesc`] (`UltIdentity` systems: a real ULT, no
//! `spawn_async` capability, no `poll_fn` slot at all),
//! [`StacklessOnlyTaskDesc`] (`UltAsyncIdentity` systems: a `spawn_async`
//! future, no real context switch, no `ctx` slot at all), and
//! [`BasicTaskDesc`] (dual systems: both capabilities on the *same*
//! descriptor, since a stackful sync joiner and a stackless async waker can
//! race to register on the same task — see `BasicTaskDesc`'s own doc
//! comment for why its `ctx`/`poll_fn` union needs the `commit_as_ctx`/
//! `commit_as_poll_fn` hooks the other two never touch). All three mirror
//! each other's field grouping for everything they share (join protocol /
//! TLS / stack fields stay in the same relative order) so the "hot fields
//! share a cache line" intent (see each struct's own
//! `#[repr(C)]` comment) survives the split instead of falling out of
//! struct-field order by accident.
//!
//! # Layering
//!
//! `TaskDesc`/`TaskDescAlloc`/`WakerTaskDesc`/`WakeOutcome`/`JoinState` live
//! here because they're genuinely shared: the join-protocol
//! (`join_state`/`JS_*`) applies to every task regardless of flavor, and
//! `WakerTaskDesc` (`waker_refs`/`JS_*`-adjacent state machine) is needed by
//! *both* `block_on` (stackful, via
//! [`stackful::waker::UltPoller`](crate::resumable::stackful::waker::UltPoller))
//! and `spawn_async` (stackless, via
//! `stackless::worker::run_async_poll`).
//! [`StackfulTaskDesc`]
//! (real saved-context handling) and
//! [`AsyncTaskDesc`]
//! (`poll_fn`) are the two genuinely flavor-specific extension traits, split
//! out to `stackful::desc`/`stackless::desc`.

use std::any::Any;
use std::cell::{Cell, UnsafeCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Waker;

use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::stackless::desc::{AsyncTaskDesc, TaskPollFn};

pub type TaskResult = Result<Box<dyn Any + Send>, Box<dyn Any + Send>>;

// ---------------------------------------------------------------------------
// waker_refs encoding
//
// bit 63:   EVER_SHARED — set on first clone of a waker; sticky forever.
// bits 2-62: ref count for SHARED wakers (0 in PRIVATE mode).
// bits 0-1:  state for PRIVATE mode, or preserved state for SHARED:
//   IDLE     = 0  — block_on not active
//   POLLING  = 1  — currently inside poll()
//   PARKED   = 2  — suspended, waiting for wake()
//   NOTIFIED = 3  — wake() called while polling; re-poll on next iteration
// ---------------------------------------------------------------------------
pub(crate) const IDLE:        usize = 0;
pub(crate) const POLLING:     usize = 1;
pub(crate) const PARKED:      usize = 2;
pub(crate) const NOTIFIED:    usize = 3;
pub(crate) const EVER_SHARED: usize = 1 << 63;
pub(crate) const STATE_MASK:  usize = 3;
pub(crate) const REF_ONE:     usize = 4; // one unit of ref count (bits 2+)

// ---------------------------------------------------------------------------
// join_state encoding
//
// One word replaces the old lock/finished/joiner triple; every transition is
// a single atomic operation, so nothing is ever held across a context switch.
//
//   RUNNING  = 0   — task alive, nobody waiting
//   FINISHED = 1   — result written (or task detached-and-cleaned)
//   DETACHED = 2   — JoinHandle dropped early; the exit path cleans up.
//                    Also the initial state of handle-less (root) tasks.
//   ptr            — a parked sync joiner (`*mut D`, aligned, > 7)
//   ptr | 1        — a registered async waker (`*mut Waker`, boxed) — used
//                    when the polling task's waker isn't verifiably one of
//                    ours (foreign executor, or no worker at all).
//   ptr | 2        — a registered async joiner (`*mut D`, unboxed) — the
//                    common case: the polling task is itself driven by this
//                    same system's `run_async_poll`, so its own descriptor
//                    is enough to reconstruct the wake without allocating a
//                    `Box<Waker>` (see `JoinHandle::poll`).
// ---------------------------------------------------------------------------
pub(crate) const JS_RUNNING: usize = 0;
pub(crate) const JS_FINISHED: usize = 1;
pub(crate) const JS_DETACHED: usize = 2;
pub(crate) const JS_ASYNC_TAG: usize = 1;
pub(crate) const JS_ASYNC_JOINER_TAG: usize = 2;

/// Decoded view of a `join_state` word.
pub enum JoinState<D> {
    Running,
    Finished,
    Detached,
    SyncJoiner(*mut D),
    AsyncWaker(*mut Waker),
    /// Same role as `AsyncWaker`, but unboxed: the polling task's own
    /// descriptor, reachable directly because its waker is known (by
    /// construction) to be this system's own `run_async_poll` waker.
    AsyncJoiner(*mut D),
}

pub(crate) fn decode_join_state<D>(v: usize) -> JoinState<D> {
    match v {
        JS_RUNNING => JoinState::Running,
        JS_FINISHED => JoinState::Finished,
        JS_DETACHED => JoinState::Detached,
        v if v & JS_ASYNC_TAG != 0 => JoinState::AsyncWaker((v & !JS_ASYNC_TAG) as *mut Waker),
        v if v & JS_ASYNC_JOINER_TAG != 0 => {
            JoinState::AsyncJoiner((v & !JS_ASYNC_JOINER_TAG) as *mut D)
        }
        v => JoinState::SyncJoiner(v as *mut D),
    }
}

/// Core per-task descriptor operations: every task, regardless of flavor,
/// needs these (pool linkage, join protocol, per-worker TLS, result slot).
///
/// Implementors are free to choose their own field layout, padding, and any
/// extra members — callers only ever go through these named accessors, the
/// same shape as [`crate::resumable::stackful::suspended::StackfulOnlyResumable`]'s `cont()`.
pub trait TaskDesc: Send + Sync + Sized + 'static {
    /// The join-protocol state word (see the `JS_*` encoding above).
    ///
    /// The exiting task publishes `FINISHED` with `Release` *after* writing
    /// the result; a joiner reading `FINISHED` with `Acquire` may take the
    /// result and free the descriptor immediately — the exit path never
    /// touches the descriptor after that store.
    fn join_state(&self) -> &AtomicUsize;

    /// Type-erased `*const UltWorker<S>`: the worker that most recently
    /// switched into this task, written by the switch shims alongside
    /// `cur_task`.  A task cannot migrate between its last resume and its
    /// next suspension, so the exit path reads this instead of doing a TLS
    /// lookup.  Only valid while the task is running.
    fn worker(&self) -> &Cell<*const ()>;

    /// Points at the arena cell's `[worker, system_id]` slot for arena
    /// stacks, or `None` for heap/root stacks.  The switch shims write the
    /// resuming worker pointer here when present.
    fn slot(&self) -> &Cell<Option<*mut crate::resumable::common::stack::CellSlot>>;

    /// True for the pseudo-descriptor representing a worker's scheduler-loop
    /// context (the "root continuation"). Fixed at construction.
    fn is_root(&self) -> bool;

    /// Written by the task itself before exiting; read by the joiner after
    /// `FINISHED` is observed.  (Root tasks only; spawned tasks put the
    /// result on their own stack.)
    fn result(&self) -> &UnsafeCell<Option<TaskResult>>;

    /// Used by nested schedulers for their per-worker pointer (`UltTls`).
    /// Only touched by the OS thread currently running this task.
    fn tls(&self) -> &UnsafeCell<Option<HashMap<usize, *mut ()>>>;

    /// Top of this task's stack allocation (`StackMem::None` for root
    /// pseudo-descriptors, in which case this must never be called).
    fn stack_top(&self) -> *mut u8;

    /// Type-erased `*const Scheduler<S>`.  Set at task-creation time —
    /// `spawn`, `spawn_async`, and `fork_parent_first` all record it,
    /// regardless of task flavor — so that `wake()` called from an external
    /// OS thread can reach the scheduler's `ExternalQueue` without going
    /// through worker TLS.  Null for root pseudo-descriptors. Only actually
    /// read by the `AsyncTaskDesc` wake path (`waker.rs::push_continuation`)
    /// today, but writing it doesn't need `AsyncTaskDesc` capability, so it
    /// lives on the base trait rather than gating every constructor on it.
    fn scheduler(&self) -> &Cell<*const ()>;

    /// Record that this task is now running on `worker_ptr` — called by
    /// every context-switch shim immediately after deciding `self` is the
    /// task being switched into.  Propagates to the arena cell slot too
    /// (when present), since every caller that sets `worker()` here has
    /// always also needed to update `slot()` in the same breath.
    #[inline]
    fn mark_resumed_on(&self, worker_ptr: *const ()) {
        self.worker().set(worker_ptr);
        if let Some(slot) = self.slot().get() {
            unsafe { (*slot).worker.set(worker_ptr) };
        }
    }

    // --- join-protocol operations, built on join_state() ------------------

    /// Read and decode the current join state (`Acquire`).
    #[inline]
    fn read_join_state(&self) -> JoinState<Self> {
        decode_join_state(self.join_state().load(Ordering::Acquire))
    }

    /// Fast check for the hot join path: is the task already finished?
    /// (`Acquire` — pairs with the `Release`/`AcqRel` publish in
    /// [`publish_finished`](Self::publish_finished)/
    /// [`commit_finished`](Self::commit_finished), making the written result
    /// visible.)
    #[inline]
    fn is_finished(&self) -> bool {
        self.join_state().load(Ordering::Acquire) == JS_FINISHED
    }

    /// Direct-handoff exit: the exiting task already switched straight into
    /// the parked sync joiner's continuation: this just publishes `FINISHED`
    /// (`Release`) so the joiner (now running) observes its result is ready.
    #[inline]
    fn commit_finished(&self) {
        self.join_state().store(JS_FINISHED, Ordering::Release);
    }

    /// General-case exit: publish `FINISHED` (`AcqRel` swap) and return
    /// whichever party the old state names, so the caller can settle it
    /// (wake a late-registered joiner/waker, or notice the handle was
    /// dropped).
    #[inline]
    fn publish_finished(&self) -> JoinState<Self> {
        decode_join_state(self.join_state().swap(JS_FINISHED, Ordering::AcqRel))
    }

    /// `JoinHandle::join`'s slow path: try to register `joiner` (a parked
    /// sync joiner's descriptor) as this task's waiter. Returns `false` if
    /// the task turned out to already be finished (caller should cancel its
    /// own suspension and proceed immediately) — otherwise commits `joiner`
    /// and drops any async waker it superseded (a sync join always wins over
    /// a previously-registered one).
    ///
    /// # Safety
    /// `joiner` must be a stable pointer to a currently-parked task
    /// descriptor for as long as it might be woken through this slot.
    unsafe fn try_register_sync_joiner(&self, joiner: *mut Self) -> bool {
        let mut cur = self.join_state().load(Ordering::Relaxed);
        loop {
            if cur == JS_FINISHED {
                return false;
            }
            match self.join_state().compare_exchange_weak(
                cur,
                joiner as usize,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if let JoinState::AsyncWaker(w) = decode_join_state::<Self>(cur) {
                        drop(unsafe { Box::from_raw(w) });
                    }
                    return true;
                }
                Err(c) => cur = c,
            }
        }
    }

    /// `JoinHandle::poll`'s fast-path registration: install `joiner` (the
    /// currently-polling task's own descriptor) directly, with no
    /// allocation. Returns `false` if the task turned out to already be
    /// finished (caller should proceed to take the result instead) —
    /// otherwise commits the tagged pointer and drops whichever *boxed*
    /// waker it superseded, if any (an old `AsyncJoiner` needs no cleanup:
    /// it never allocated).
    ///
    /// # Safety
    /// `joiner` must be a stable pointer to the currently-polling task's own
    /// descriptor for as long as it might be woken through this slot — the
    /// caller (`JoinHandle::poll`) only calls this when `joiner` came from
    /// `UltWorker::polling_async`, which is only ever set to the descriptor
    /// `run_async_poll` is synchronously driving right now (see that
    /// function's doc comment for why the ambient waker is then guaranteed
    /// to be `joiner`'s own).
    #[inline]
    unsafe fn try_register_async_joiner(&self, joiner: *mut Self) -> bool {
        debug_assert_eq!(
            joiner as usize & (JS_ASYNC_TAG | JS_ASYNC_JOINER_TAG),
            0,
            "cmpth: descriptor pointer not aligned enough to tag"
        );
        let mut cur = self.join_state().load(Ordering::Acquire);
        let new = (joiner as usize) | JS_ASYNC_JOINER_TAG;
        loop {
            if cur == JS_FINISHED {
                return false;
            }
            match self.join_state().compare_exchange_weak(
                cur, new, Ordering::Release, Ordering::Acquire,
            ) {
                Ok(_) => {
                    if let JoinState::AsyncWaker(w) = decode_join_state::<Self>(cur) {
                        drop(unsafe { Box::from_raw(w) });
                    }
                    return true;
                }
                Err(c) => cur = c,
            }
        }
    }

    /// `JoinHandle::poll`'s waker registration: try to install `waker` as
    /// this task's async waiter. Returns `false` if the task turned out to
    /// already be finished (caller should proceed to take the result
    /// instead) — otherwise commits the boxed, tagged waker and drops
    /// whichever waker it superseded, if any.
    fn try_register_waker(&self, waker: Waker) -> bool {
        let mut cur = self.join_state().load(Ordering::Acquire);
        if cur == JS_FINISHED {
            return false;
        }
        let new = Box::into_raw(Box::new(waker)) as usize | JS_ASYNC_TAG;
        loop {
            if cur == JS_FINISHED {
                drop(unsafe { Box::from_raw((new & !JS_ASYNC_TAG) as *mut Waker) });
                return false;
            }
            match self.join_state().compare_exchange_weak(
                cur, new, Ordering::Release, Ordering::Acquire,
            ) {
                Ok(_) => {
                    if let JoinState::AsyncWaker(w) = decode_join_state::<Self>(cur) {
                        drop(unsafe { Box::from_raw(w) });
                    }
                    // Left disabled: this is a generic TaskDesc default
                    // method with no `S` in scope, so there's no worker to
                    // reach a per-worker WorkerLog through. Also confirmed
                    // uninvolved in the Part 7 crash (zero wake events for
                    // that capture's sid).
                    return true;
                }
                Err(c) => cur = c,
            }
        }
    }

    /// `JoinHandle::drop`'s detach attempt: try to mark this task detached
    /// (no handle left to collect the result). Returns `true` if the task
    /// was already finished (caller now owns the result and the
    /// descriptor) — otherwise commits `DETACHED` and drops any registered
    /// async waker (nobody left to wake).
    fn try_mark_detached(&self) -> bool {
        let mut cur = self.join_state().load(Ordering::Acquire);
        loop {
            if cur == JS_FINISHED {
                return true;
            }
            match self.join_state().compare_exchange_weak(
                cur, JS_DETACHED, Ordering::AcqRel, Ordering::Acquire,
            ) {
                Ok(_) => {
                    if let JoinState::AsyncWaker(w) = decode_join_state::<Self>(cur) {
                        drop(unsafe { Box::from_raw(w) });
                    }
                    return false;
                }
                Err(c) => cur = c,
            }
        }
    }
}

/// Construction/lifecycle operations for a descriptor type, kept separate
/// from [`TaskDesc`] itself so generic pool/worker code (`DescPool`,
/// `UltWorker::new`, `spawn`/`spawn_async`) can allocate, free, and reset a
/// descriptor through `D::alloc_with`/`new_root`/`free`/`reinit` without
/// naming the concrete descriptor type — the same "trait owns the contract,
/// one struct satisfies it today" shape as `TaskDesc` itself. Every method
/// here mirrors an existing `BasicTaskDesc` inherent fn byte-for-byte; this
/// is a mechanical accessor split, not a behavior change.
pub trait TaskDescAlloc: TaskDesc + Sized {
    /// Construct a descriptor value whose stack storage is `stack` (heap or
    /// arena, per the caller's `StackAlloc` policy). Returns `Self` by
    /// value, not a boxed pointer: pool bookkeeping (the old `pool_next`/
    /// `alloc_wk`/`oversized` fields) no longer lives on the descriptor, so
    /// wrapping it in a heap allocation (bare `Box<Self>`, or a pool's
    /// `Node<Self>`) is entirely the caller's decision, not this trait's.
    /// Used by the pool and by `spawn`'s parent-first fork path.
    fn alloc_with(stack: crate::resumable::common::stack::StackMem, has_handle: bool) -> Self;

    /// Construct a descriptor value with a plain heap buffer of
    /// `stack_size` bytes, bypassing any arena/guard-page policy. Used by
    /// `spawn_async`, whose "stack" only ever stores a `Future` + result —
    /// no code runs on it, so it never needs the arena.
    fn alloc(stack_size: usize, has_handle: bool) -> Self;

    /// Pseudo-descriptor for a worker's own scheduler-loop context (the
    /// "root continuation"), embedded by value in `UltWorker`.
    fn new_root() -> Self;

    /// Reset a pooled descriptor for reuse (the stack allocation is kept).
    fn reinit(&mut self, has_handle: bool);
}

/// Descriptor operations needed by any task that can be driven via a real
/// [`std::task::Waker`] — both `block_on` (polling an arbitrary `Future`
/// from a real ULT) and `spawn_async` tasks need this `waker_refs` state
/// machine; `spawn_async` additionally needs
/// [`AsyncTaskDesc`] on
/// top for its `poll_fn` task representation. Kept separate from
/// `AsyncTaskDesc` so that `ThreadSystem::block_on` (and anything generic
/// over `S: ThreadSystem`, like `DelegatorConsumer`) doesn't drag in
/// `poll_fn`/spawn_async-specific machinery it never touches — a system that
/// supports `block_on` but not `spawn_async` is expressible this way.
pub trait WakerTaskDesc: TaskDesc {
    /// Encodes PRIVATE/SHARED mode, ref count, and POLLING/PARKED/NOTIFIED/
    /// IDLE state.  See the `waker_refs` constants at the top of this file.
    /// Zero (IDLE) when no `block_on` call is active on this task.
    fn waker_refs(&self) -> &AtomicUsize;

    // --- named waker_refs state-machine operations -------------------------

    /// Reset to POLLING unconditionally.  Called whenever a poll is about
    /// to begin (`UltPoller::new`, `run_async_poll`'s pre-poll mark).
    #[inline]
    fn mark_polling(&self) {
        self.waker_refs().store(POLLING, Ordering::Release);
    }

    /// Reset to IDLE unconditionally.  Called when a poll session ends
    /// (`UltPoller::drop`, `poll_spawned_task` invalidating the waker before
    /// publishing the result so a racing `wake()` becomes a no-op).
    #[inline]
    fn mark_idle(&self) {
        self.waker_refs().store(IDLE, Ordering::Release);
    }

    /// `UltPoller::wait`'s cond_suspend commit decision, run from inside
    /// the suspend shim after the context is already saved: read the
    /// current state and either commit to PARKED (`true`) or, if a wake
    /// already raced in during `poll()` (state == NOTIFIED), cancel by
    /// resetting to POLLING (`false`) so the caller re-polls immediately
    /// instead of parking.
    ///
    /// CAS loop, matching `park_after_poll`. A plain load-then-store here
    /// raced with `try_wake_state`'s CAS: if a concurrent `wake()` claimed
    /// POLLING -> NOTIFIED between this method's load and store, the store
    /// (computed from the stale POLLING read) clobbered NOTIFIED back to
    /// PARKED. `try_wake_state` had already returned `SetNotified` (task
    /// "still running, will notice on its own") and so never pushed a
    /// continuation -- the task committed to PARKED with nobody left to
    /// wake it, a permanent lost-wakeup livelock.
    fn decide_park(&self) -> bool {
        loop {
            let refs = self.waker_refs().load(Ordering::Acquire);
            let state = refs & STATE_MASK;
            if state == NOTIFIED {
                let new = (refs & !STATE_MASK) | POLLING;
                if self
                    .waker_refs()
                    .compare_exchange(refs, new, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return false;
                }
            } else {
                let new = (refs & !STATE_MASK) | PARKED;
                if self
                    .waker_refs()
                    .compare_exchange(refs, new, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return true;
                }
            }
        }
    }

    /// `run_async_poll`'s post-poll transition after a `Pending` result:
    /// try to park (POLLING -> PARKED, returns `true`). If a wake raced in
    /// during the poll (state == NOTIFIED), claims a re-poll instead
    /// (NOTIFIED -> POLLING, returns `false` so the caller re-queues the
    /// task immediately rather than parking it).
    ///
    /// CAS loop — unlike `decide_park`, concurrent wakers can race this
    /// transition (matches the original call site exactly).
    #[inline]
    fn park_after_poll(&self) -> bool {
        loop {
            let refs = self.waker_refs().load(Ordering::Acquire);
            let state = refs & STATE_MASK;
            if state == NOTIFIED {
                let new = (refs & !STATE_MASK) | POLLING;
                if self
                    .waker_refs()
                    .compare_exchange(refs, new, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return false;
                }
            } else {
                let new = (refs & !STATE_MASK) | PARKED;
                if self
                    .waker_refs()
                    .compare_exchange(refs, new, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return true;
                }
            }
        }
    }

    /// Core wake CAS loop, shared by the stackful (`try_wake`) and async
    /// (`try_wake_async`) wake paths in `waker.rs`.
    fn try_wake_state(&self) -> WakeOutcome {
        loop {
            let refs = self.waker_refs().load(Ordering::Acquire);
            let state = refs & STATE_MASK;
            match state {
                s if s == POLLING => {
                    // Task is running; request a re-poll by setting NOTIFIED.
                    let new = (refs & !STATE_MASK) | NOTIFIED;
                    match self.waker_refs().compare_exchange(
                        refs, new, Ordering::AcqRel, Ordering::Acquire,
                    ) {
                        Ok(_) => return WakeOutcome::SetNotified,
                        Err(_) => continue,
                    }
                }
                s if s == PARKED => {
                    // Task is suspended; claim it -- caller must deliver
                    // the continuation.
                    let new = (refs & !STATE_MASK) | POLLING;
                    match self.waker_refs().compare_exchange(
                        refs, new, Ordering::AcqRel, Ordering::Acquire,
                    ) {
                        Ok(_) => return WakeOutcome::ClaimedParked,
                        Err(_) => continue,
                    }
                }
                s if s == NOTIFIED => return WakeOutcome::NoOp, // already notified
                s if s == IDLE => return WakeOutcome::NoOp,     // stale wake
                _ => unreachable!("unexpected waker_refs: {:#x}", refs),
            }
        }
    }

    /// True once this waker has been cloned at least once: bits 2-62 hold
    /// a real ref count from that point on (state bits 0-1 keep their
    /// meaning either way). Sticky -- never clears back to false.
    fn is_ever_shared(&self) -> bool {
        self.waker_refs().load(Ordering::Relaxed) & EVER_SHARED != 0
    }

    /// First-clone transition: set `EVER_SHARED` and initialize the ref
    /// count to 2 (the original waker + this new clone), preserving
    /// whatever state bits are currently set. CAS loop: a concurrent
    /// `wake()` may change the state bits underneath.
    fn transition_to_shared(&self) {
        loop {
            let old = self.waker_refs().load(Ordering::Relaxed);
            let new = EVER_SHARED | (2 * REF_ONE) | (old & STATE_MASK);
            if self
                .waker_refs()
                .compare_exchange(old, new, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Increment the SHARED ref count (an existing SHARED waker being
    /// cloned again).
    fn incr_shared_ref(&self) {
        self.waker_refs().fetch_add(REF_ONE, Ordering::Relaxed);
    }

    /// Decrement the SHARED ref count (a SHARED waker being dropped).
    fn decr_shared_ref(&self) {
        self.waker_refs().fetch_sub(REF_ONE, Ordering::Release);
    }
}

/// Outcome of [`WakerTaskDesc::try_wake_state`].
pub enum WakeOutcome {
    /// Was POLLING; now NOTIFIED. The task will notice on its next
    /// `waker_refs` check and re-poll; there is no continuation to push.
    SetNotified,
    /// Was PARKED; now POLLING. The caller owns delivering the
    /// continuation (push to a worker deque or the external queue).
    ClaimedParked,
    /// Was already NOTIFIED, or IDLE (a stale wake after the poll session
    /// ended). Nothing to do.
    NoOp,
}

// ---------------------------------------------------------------------------
// StackfulOnlyTaskDesc — UltIdentity systems (real ULTs, no spawn_async)
// ---------------------------------------------------------------------------

/// Concrete descriptor for `UltIdentity`-based (stackful-only) systems: a
/// real ULT with no `spawn_async` capability, so no `poll_fn` slot exists
/// at all (contrast [`BasicTaskDesc`], which needs both on the same
/// struct). `repr(C)`: hot fields (touched on every spawn/exit/join) laid
/// out first, mirroring `BasicTaskDesc`'s grouping minus `poll_fn`.
#[repr(C)]
pub struct StackfulOnlyTaskDesc {
    // --- Hot: touched on every spawn/exit/join ----------------------------
    ctx: Cell<*mut u8>,
    join_state: AtomicUsize,
    worker: Cell<*const ()>,
    slot: Cell<Option<*mut crate::resumable::common::stack::CellSlot>>,
    is_root: bool,

    // --- Warm -------------------------------------------------------------
    result: UnsafeCell<Option<TaskResult>>,

    // --- Async waker (block_on) --------------------------------------------
    waker_refs: AtomicUsize,
    scheduler: Cell<*const ()>,

    // --- ULT-local storage --------------------------------------------------
    tls: UnsafeCell<Option<HashMap<usize, *mut ()>>>,

    // --- Stack ----------------------------------------------------------------
    stack: crate::resumable::common::stack::StackMem,
}

unsafe impl Send for StackfulOnlyTaskDesc {}
unsafe impl Sync for StackfulOnlyTaskDesc {}

impl TaskDesc for StackfulOnlyTaskDesc {
    fn join_state(&self) -> &AtomicUsize { &self.join_state }
    fn worker(&self) -> &Cell<*const ()> { &self.worker }
    fn slot(&self) -> &Cell<Option<*mut crate::resumable::common::stack::CellSlot>> { &self.slot }
    fn is_root(&self) -> bool { self.is_root }
    fn result(&self) -> &UnsafeCell<Option<TaskResult>> { &self.result }
    fn tls(&self) -> &UnsafeCell<Option<HashMap<usize, *mut ()>>> { &self.tls }
    fn stack_top(&self) -> *mut u8 { self.stack.top() }
    fn scheduler(&self) -> &Cell<*const ()> { &self.scheduler }
}

impl StackfulTaskDesc for StackfulOnlyTaskDesc {
    fn ctx(&self) -> &Cell<*mut u8> { &self.ctx }
}

impl WakerTaskDesc for StackfulOnlyTaskDesc {
    fn waker_refs(&self) -> &AtomicUsize { &self.waker_refs }
}

impl TaskDescAlloc for StackfulOnlyTaskDesc {
    fn alloc_with(stack: crate::resumable::common::stack::StackMem, has_handle: bool) -> Self {
        StackfulOnlyTaskDesc::alloc_with(stack, has_handle)
    }

    fn alloc(stack_size: usize, has_handle: bool) -> Self {
        StackfulOnlyTaskDesc::alloc(stack_size, has_handle)
    }

    fn new_root() -> Self {
        StackfulOnlyTaskDesc::new_root()
    }

    fn reinit(&mut self, has_handle: bool) {
        StackfulOnlyTaskDesc::reinit(self, has_handle)
    }
}

impl StackfulOnlyTaskDesc {
    /// Construct a descriptor value with a heap stack.
    pub(crate) fn alloc(stack_size: usize, has_handle: bool) -> StackfulOnlyTaskDesc {
        use crate::resumable::common::stack::{HeapStack, StackAlloc as _};
        Self::alloc_with(HeapStack::alloc_stack(stack_size).into(), has_handle)
    }

    /// Construct a descriptor value with a policy-allocated stack. For arena
    /// stacks, captures the cell slot pointer for use by the switch shims.
    pub(crate) fn alloc_with(stack: crate::resumable::common::stack::StackMem, has_handle: bool) -> StackfulOnlyTaskDesc {
        let slot = stack.cell_slot();
        StackfulOnlyTaskDesc {
            ctx: Cell::new(std::ptr::null_mut()),
            is_root: false,
            join_state: AtomicUsize::new(if has_handle { JS_RUNNING } else { JS_DETACHED }),
            result: UnsafeCell::new(None),
            waker_refs: AtomicUsize::new(0),
            scheduler: Cell::new(std::ptr::null()),
            worker: Cell::new(std::ptr::null()),
            slot: Cell::new(slot),
            tls: UnsafeCell::new(None),
            stack,
        }
    }

    /// Pseudo-descriptor for a worker's scheduler-loop context.
    pub(crate) fn new_root() -> StackfulOnlyTaskDesc {
        StackfulOnlyTaskDesc {
            ctx: Cell::new(std::ptr::null_mut()),
            is_root: true,
            join_state: AtomicUsize::new(JS_DETACHED),
            result: UnsafeCell::new(None),
            waker_refs: AtomicUsize::new(0),
            scheduler: Cell::new(std::ptr::null()),
            worker: Cell::new(std::ptr::null()),
            slot: Cell::new(None),
            tls: UnsafeCell::new(None),
            stack: crate::resumable::common::stack::StackMem::None,
        }
    }

    /// Reset a pooled descriptor for reuse (the stack allocation is kept).
    pub(crate) fn reinit(&mut self, has_handle: bool) {
        debug_assert!(!self.is_root);
        *self.ctx.get_mut() = std::ptr::null_mut();
        *self.join_state.get_mut() = if has_handle { JS_RUNNING } else { JS_DETACHED };
        *self.waker_refs.get_mut() = 0;
        *self.result.get_mut() = None;
        *self.tls.get_mut() = None;
    }
}

// ---------------------------------------------------------------------------
// StacklessOnlyTaskDesc — UltAsyncIdentity systems (spawn_async, no real
// context switch)
// ---------------------------------------------------------------------------

/// Concrete descriptor for `UltAsyncIdentity`-based (stackless-only)
/// systems: a `spawn_async` task with no real context switch, so no `ctx`
/// slot exists at all. `repr(C)`: hot fields laid out first, mirroring
/// `BasicTaskDesc`'s grouping minus `ctx`.
#[repr(C)]
pub struct StacklessOnlyTaskDesc {
    // --- Hot: touched on every spawn/exit/join ----------------------------
    join_state: AtomicUsize,
    worker: Cell<*const ()>,
    slot: Cell<Option<*mut crate::resumable::common::stack::CellSlot>>,
    poll_fn: Cell<Option<TaskPollFn<StacklessOnlyTaskDesc>>>,
    is_root: bool,

    // --- Warm -------------------------------------------------------------
    result: UnsafeCell<Option<TaskResult>>,

    // --- Async waker (spawn_async's own poll loop uses this too) ----------
    waker_refs: AtomicUsize,
    scheduler: Cell<*const ()>,

    // --- ULT-local storage --------------------------------------------------
    tls: UnsafeCell<Option<HashMap<usize, *mut ()>>>,

    // --- Stack (spawn_async's Future + result storage) -----------------------
    stack: crate::resumable::common::stack::StackMem,
}

unsafe impl Send for StacklessOnlyTaskDesc {}
unsafe impl Sync for StacklessOnlyTaskDesc {}

impl TaskDesc for StacklessOnlyTaskDesc {
    fn join_state(&self) -> &AtomicUsize { &self.join_state }
    fn worker(&self) -> &Cell<*const ()> { &self.worker }
    fn slot(&self) -> &Cell<Option<*mut crate::resumable::common::stack::CellSlot>> { &self.slot }
    fn is_root(&self) -> bool { self.is_root }
    fn result(&self) -> &UnsafeCell<Option<TaskResult>> { &self.result }
    fn tls(&self) -> &UnsafeCell<Option<HashMap<usize, *mut ()>>> { &self.tls }
    fn stack_top(&self) -> *mut u8 { self.stack.top() }
    fn scheduler(&self) -> &Cell<*const ()> { &self.scheduler }
}

impl WakerTaskDesc for StacklessOnlyTaskDesc {
    fn waker_refs(&self) -> &AtomicUsize { &self.waker_refs }
}

impl AsyncTaskDesc for StacklessOnlyTaskDesc {
    fn poll_fn(&self) -> &Cell<Option<TaskPollFn<Self>>> { &self.poll_fn }
}

impl TaskDescAlloc for StacklessOnlyTaskDesc {
    fn alloc_with(stack: crate::resumable::common::stack::StackMem, has_handle: bool) -> Self {
        StacklessOnlyTaskDesc::alloc_with(stack, has_handle)
    }

    fn alloc(stack_size: usize, has_handle: bool) -> Self {
        StacklessOnlyTaskDesc::alloc(stack_size, has_handle)
    }

    fn new_root() -> Self {
        StacklessOnlyTaskDesc::new_root()
    }

    fn reinit(&mut self, has_handle: bool) {
        StacklessOnlyTaskDesc::reinit(self, has_handle)
    }
}

impl StacklessOnlyTaskDesc {
    /// Construct a descriptor value with a heap stack. Used by
    /// `spawn_async` (whose "stack" only stores the future — no code runs
    /// on it, so it never needs the arena).
    pub(crate) fn alloc(stack_size: usize, has_handle: bool) -> StacklessOnlyTaskDesc {
        use crate::resumable::common::stack::{HeapStack, StackAlloc as _};
        Self::alloc_with(HeapStack::alloc_stack(stack_size).into(), has_handle)
    }

    /// Construct a descriptor value with a policy-allocated stack (e.g. an
    /// async arena). Captures the cell slot pointer for use by the wake
    /// path.
    pub(crate) fn alloc_with(stack: crate::resumable::common::stack::StackMem, has_handle: bool) -> StacklessOnlyTaskDesc {
        let slot = stack.cell_slot();
        StacklessOnlyTaskDesc {
            is_root: false,
            join_state: AtomicUsize::new(if has_handle { JS_RUNNING } else { JS_DETACHED }),
            result: UnsafeCell::new(None),
            waker_refs: AtomicUsize::new(0),
            scheduler: Cell::new(std::ptr::null()),
            worker: Cell::new(std::ptr::null()),
            slot: Cell::new(slot),
            poll_fn: Cell::new(None),
            tls: UnsafeCell::new(None),
            stack,
        }
    }

    /// Pseudo-descriptor for a worker's scheduler-loop context.
    pub(crate) fn new_root() -> StacklessOnlyTaskDesc {
        StacklessOnlyTaskDesc {
            is_root: true,
            join_state: AtomicUsize::new(JS_DETACHED),
            result: UnsafeCell::new(None),
            waker_refs: AtomicUsize::new(0),
            scheduler: Cell::new(std::ptr::null()),
            worker: Cell::new(std::ptr::null()),
            slot: Cell::new(None),
            poll_fn: Cell::new(None),
            tls: UnsafeCell::new(None),
            stack: crate::resumable::common::stack::StackMem::None,
        }
    }

    /// Reset a pooled descriptor for reuse (the stack allocation is kept).
    pub(crate) fn reinit(&mut self, has_handle: bool) {
        debug_assert!(!self.is_root);
        *self.join_state.get_mut() = if has_handle { JS_RUNNING } else { JS_DETACHED };
        *self.waker_refs.get_mut() = 0;
        *self.result.get_mut() = None;
        *self.tls.get_mut() = None;
        self.poll_fn.set(None);
    }
}

// ---------------------------------------------------------------------------
// BasicTaskDesc — dual systems (both capabilities on the same descriptor)
// ---------------------------------------------------------------------------

/// A dual task is never both a real ULT and a `spawn_async` future — this
/// enum makes that exclusivity a type-level fact instead of an implicit
/// "one of two nullable fields" convention.
///
/// Verified zero-cost (2026-07-29, `rustc -O --emit=asm` on AArch64): a
/// *safe* accessor that panics via `unreachable!()` on the wrong variant
/// compiles to a real branch, but `debug_assert!` + `unreachable_unchecked()`
/// (what [`BasicTaskDesc`]'s `ctx()`/`poll_fn()` use below) compiles to the
/// exact same code as a direct field access (`add x0, x0, #8; ret`) — no
/// discriminant check survives release codegen.
///
/// Naively, this looks like it should also save memory over two separate
/// fields — it doesn't: `size_of::<TaskDispatch<D>>() == 16`, same as
/// `Cell<*mut u8>` (8) + `Cell<Option<TaskPollFn<D>>>` (8) today, confirmed
/// via `size_of`. Raw pointers (unlike `&T`/`NonNull<T>`) have no spare
/// niche — `ctx` legitimately holds null as live data from construction
/// (`alloc_with`'s initial value) — so the compiler can't fold the
/// discriminant into an unused bit pattern of either variant; it costs a
/// real extra word. The enum's value here is exclusively the unrepresentable-
/// illegal-state guarantee, not size.
enum TaskDispatch<D> {
    Ctx(Cell<*mut u8>),
    PollFn(Cell<Option<TaskPollFn<D>>>),
}

/// The descriptor implementation for dual (stackful + stackless) systems:
/// implements every trait at once, since a stackful sync joiner and a
/// stackless async waker can race to register on the *same* task
/// regardless of which one the task itself turns out to be. `repr(C)`: the
/// fields touched by every spawn/exit/join round-trip (`dispatch`,
/// `join_state`, `worker`, `slot`, `is_root`) are laid out first so they
/// share one cache line, mirroring `StackfulOnlyTaskDesc`/
/// `StacklessOnlyTaskDesc`'s grouping.
#[repr(C)]
pub struct BasicTaskDesc {
    // --- Hot: touched on every spawn/exit/join ----------------------------
    // Cell-wrapped so commit_as_ctx/commit_as_poll_fn can overwrite the
    // variant through &self (Cell::set), matching every other field here.
    dispatch: Cell<TaskDispatch<BasicTaskDesc>>,
    join_state: AtomicUsize,
    worker: Cell<*const ()>,
    slot: Cell<Option<*mut crate::resumable::common::stack::CellSlot>>,
    is_root: bool,

    // --- Warm -------------------------------------------------------------
    result: UnsafeCell<Option<TaskResult>>,

    // --- Async waker (block_on) --------------------------------------------
    waker_refs: AtomicUsize,
    scheduler: Cell<*const ()>,

    // --- ULT-local storage --------------------------------------------------
    tls: UnsafeCell<Option<HashMap<usize, *mut ()>>>,

    // --- Stack ----------------------------------------------------------------
    stack: crate::resumable::common::stack::StackMem,
}

unsafe impl Send for BasicTaskDesc {}
unsafe impl Sync for BasicTaskDesc {}

impl TaskDesc for BasicTaskDesc {
    fn join_state(&self) -> &AtomicUsize { &self.join_state }
    fn worker(&self) -> &Cell<*const ()> { &self.worker }
    fn slot(&self) -> &Cell<Option<*mut crate::resumable::common::stack::CellSlot>> { &self.slot }
    fn is_root(&self) -> bool { self.is_root }
    fn result(&self) -> &UnsafeCell<Option<TaskResult>> { &self.result }
    fn tls(&self) -> &UnsafeCell<Option<HashMap<usize, *mut ()>>> { &self.tls }
    fn stack_top(&self) -> *mut u8 { self.stack.top() }
    fn scheduler(&self) -> &Cell<*const ()> { &self.scheduler }
}

impl StackfulTaskDesc for BasicTaskDesc {
    fn ctx(&self) -> &Cell<*mut u8> {
        // Safety: `dispatch`'s variant is fixed by `commit_as_ctx`/
        // `commit_as_poll_fn`, called exactly once by the allocating call
        // site (see their doc comments) before the descriptor is ever
        // shared with another thread or handed to code that would call
        // `ctx()`/`poll_fn()` — so by the time any accessor runs, the
        // variant is settled and stays settled for the descriptor's whole
        // life. `Cell::as_ptr` + deref is the standard way to borrow into
        // a `Cell`'s contents without requiring `T: Copy`.
        match unsafe { &*self.dispatch.as_ptr() } {
            TaskDispatch::Ctx(ctx) => ctx,
            TaskDispatch::PollFn(_) => {
                debug_assert!(false, "ctx() called on a descriptor committed to poll_fn dispatch");
                unsafe { std::hint::unreachable_unchecked() }
            }
        }
    }

    fn commit_as_ctx(&self) {
        self.dispatch.set(TaskDispatch::Ctx(Cell::new(std::ptr::null_mut())));
    }
}

impl WakerTaskDesc for BasicTaskDesc {
    fn waker_refs(&self) -> &AtomicUsize { &self.waker_refs }
}

impl AsyncTaskDesc for BasicTaskDesc {
    fn poll_fn(&self) -> &Cell<Option<TaskPollFn<Self>>> {
        // Safety: see StackfulTaskDesc::ctx's matching comment above.
        match unsafe { &*self.dispatch.as_ptr() } {
            TaskDispatch::PollFn(poll_fn) => poll_fn,
            TaskDispatch::Ctx(_) => {
                debug_assert!(false, "poll_fn() called on a descriptor committed to ctx dispatch");
                unsafe { std::hint::unreachable_unchecked() }
            }
        }
    }

    fn commit_as_poll_fn(&self) {
        self.dispatch.set(TaskDispatch::PollFn(Cell::new(None)));
    }

    fn is_poll_fn_dispatch(&self) -> bool {
        // Safety: see StackfulTaskDesc::ctx's matching comment above — this
        // one never panics on the "wrong" variant since that's the whole
        // point (it's the safe way to find out).
        matches!(unsafe { &*self.dispatch.as_ptr() }, TaskDispatch::PollFn(_))
    }
}

impl TaskDescAlloc for BasicTaskDesc {
    fn alloc_with(stack: crate::resumable::common::stack::StackMem, has_handle: bool) -> Self {
        BasicTaskDesc::alloc_with(stack, has_handle)
    }

    fn alloc(stack_size: usize, has_handle: bool) -> Self {
        BasicTaskDesc::alloc(stack_size, has_handle)
    }

    fn new_root() -> Self {
        BasicTaskDesc::new_root()
    }

    fn reinit(&mut self, has_handle: bool) {
        BasicTaskDesc::reinit(self, has_handle)
    }
}

impl BasicTaskDesc {
    /// Construct a descriptor value with a heap stack. Used (among other
    /// things) by `spawn_async` (whose "stack" only stores the future — no
    /// code runs on it, so it never needs the arena).
    ///
    /// `dispatch` starts as `Ctx` (an arbitrary placeholder — this
    /// constructor is shared by pooled allocation for *both* `S::Pool` and
    /// `S::AsyncPool`, so it cannot know its eventual role); the allocating
    /// call site is responsible for calling `commit_as_ctx`/
    /// `commit_as_poll_fn` immediately afterward, before anything else
    /// touches the descriptor. See `StackfulTaskDesc::commit_as_ctx`'s doc
    /// comment.
    pub(crate) fn alloc(stack_size: usize, has_handle: bool) -> BasicTaskDesc {
        use crate::resumable::common::stack::{HeapStack, StackAlloc as _};
        Self::alloc_with(HeapStack::alloc_stack(stack_size).into(), has_handle)
    }

    /// Construct a descriptor value with a policy-allocated stack.  For
    /// arena stacks, captures the cell slot pointer for use by the switch
    /// shims. See [`BasicTaskDesc::alloc`]'s doc comment for the `dispatch`
    /// placeholder-then-commit protocol this also follows.
    pub(crate) fn alloc_with(stack: crate::resumable::common::stack::StackMem, has_handle: bool) -> BasicTaskDesc {
        let slot = stack.cell_slot();
        BasicTaskDesc {
            dispatch: Cell::new(TaskDispatch::Ctx(Cell::new(std::ptr::null_mut()))),
            is_root: false,
            join_state: AtomicUsize::new(if has_handle { JS_RUNNING } else { JS_DETACHED }),
            result: UnsafeCell::new(None),
            waker_refs: AtomicUsize::new(0),
            scheduler: Cell::new(std::ptr::null()),
            worker: Cell::new(std::ptr::null()),
            slot: Cell::new(slot),
            tls: UnsafeCell::new(None),
            stack,
        }
    }

    /// Pseudo-descriptor for a worker's scheduler-loop context. Always
    /// `Ctx`: the root represents the OS-thread-level scheduler loop
    /// itself, resumed via a real context switch back into it — never a
    /// `spawn_async` future — so unlike `alloc`/`alloc_with` there is no
    /// per-call-site ambiguity to resolve here.
    pub(crate) fn new_root() -> BasicTaskDesc {
        BasicTaskDesc {
            dispatch: Cell::new(TaskDispatch::Ctx(Cell::new(std::ptr::null_mut()))),
            is_root: true,
            join_state: AtomicUsize::new(JS_DETACHED),
            result: UnsafeCell::new(None),
            waker_refs: AtomicUsize::new(0),
            scheduler: Cell::new(std::ptr::null()),
            worker: Cell::new(std::ptr::null()),
            slot: Cell::new(None),
            tls: UnsafeCell::new(None),
            stack: crate::resumable::common::stack::StackMem::None,
        }
    }

    /// Reset a pooled descriptor for reuse (the stack allocation is kept).
    ///
    /// Safe to reset everything: the exit path's *last* access to a
    /// descriptor is the `join_state` publication itself, so once a joiner
    /// has observed `FINISHED` and freed the descriptor, no stale stores
    /// from the previous task can be in flight.
    ///
    /// Does *not* touch `dispatch`'s variant: a pooled descriptor is always
    /// reused from the same pool (`S::Pool` or `S::AsyncPool`) it came
    /// from, so its role never changes across reuse — only reset the
    /// currently-active variant's own inner value. (The allocating call
    /// site still unconditionally calls `commit_as_ctx`/`commit_as_poll_fn`
    /// after this, same as for a fresh allocation; on a reused descriptor
    /// that is a harmless idempotent overwrite with an equivalent value.)
    pub(crate) fn reinit(&mut self, has_handle: bool) {
        debug_assert!(!self.is_root);
        match self.dispatch.get_mut() {
            TaskDispatch::Ctx(ctx) => *ctx.get_mut() = std::ptr::null_mut(),
            TaskDispatch::PollFn(poll_fn) => *poll_fn.get_mut() = None,
        }
        *self.join_state.get_mut() = if has_handle { JS_RUNNING } else { JS_DETACHED };
        *self.waker_refs.get_mut() = 0;
        *self.result.get_mut() = None;
        *self.tls.get_mut() = None;
    }
}

/// Owning handle to a suspended task.  Not `Clone`, not `Drop`: ownership is
/// linear and consuming the continuation (resuming it or storing it in a
/// waiter slot) is explicit.
///
/// Generic over the descriptor type `D` so a stackful-only or stackless-only
/// system plugs in a narrower descriptor (`StackfulOnlyTaskDesc`,
/// `StacklessOnlyTaskDesc`, or `BasicTaskDesc` for dual) without touching
/// every deque/pool/worker call site — they're all written generically
/// over `D: TaskDesc`.
pub struct SuspendedTaskToken<D: TaskDesc>(pub(crate) *mut D);

unsafe impl<D: TaskDesc> Send for SuspendedTaskToken<D> {}

impl<D: TaskDesc> SuspendedTaskToken<D> {
    pub(crate) fn desc(&self) -> *mut D {
        self.0
    }

    pub(crate) fn is_root(&self) -> bool {
        unsafe { (*self.0).is_root() }
    }

    pub(crate) fn into_raw(self) -> *mut D {
        self.0
    }

    /// A switch shim just resumed into this continuation: promote it from
    /// "suspended, sitting somewhere" to "running, held by the worker's
    /// `cur_task`/`polling_async` slot". See [`RunningTaskToken`] for why this
    /// is a distinct type rather than reusing `SuspendedTaskToken` for both —
    /// the name `SuspendedTaskToken` would be a lie for something that's
    /// actively executing.
    pub(crate) fn into_running(self) -> RunningTaskToken<D> {
        RunningTaskToken(self.into_raw())
    }
}

/// Owning handle to the task currently *running* on a worker (held in
/// `UltWorker::cur_task`/`polling_async`) — the running-task counterpart to
/// [`SuspendedTaskToken`]. Deliberately a separate type, not a reused
/// `SuspendedTaskToken`: `SuspendedTaskToken` means "not currently executing", which is
/// the opposite of what sits in `cur_task`/`polling_async`. Same move-only
/// discipline (no `Clone`): at most one `RunningTaskToken<D>` for a given
/// descriptor exists at a time, either held by whichever code is actively
/// driving it, or sitting in the worker's `cur_task`/`polling_async` cell
/// (never both at once — see `UltWorker::cur_task`'s doc comment).
pub struct RunningTaskToken<D: TaskDesc>(pub(crate) *mut D);

unsafe impl<D: TaskDesc> Send for RunningTaskToken<D> {}

impl<D: TaskDesc> RunningTaskToken<D> {
    pub(crate) fn desc(&self) -> *mut D {
        self.0
    }

    pub(crate) fn into_raw(self) -> *mut D {
        self.0
    }

    /// This task is being parked/hand back to a caller instead of
    /// continuing to run: demote it back to a suspended continuation. The
    /// counterpart to [`SuspendedTaskToken::into_running`].
    pub(crate) fn into_suspended(self) -> SuspendedTaskToken<D> {
        SuspendedTaskToken(self.into_raw())
    }
}

#[cfg(test)]
mod tests {
    use super::{BasicTaskDesc, StackfulOnlyTaskDesc, StacklessOnlyTaskDesc};

    /// Regression guard for the whole point of splitting the descriptor per
    /// flavor: a stackful-only/stackless-only system must not carry the
    /// unused half of `BasicTaskDesc`'s `ctx`/`poll_fn` union. If this ever
    /// fails, something added a field back (or grew one) without noticing
    /// it defeated the split.
    #[test]
    fn narrow_descriptors_are_smaller_than_basic() {
        let basic = std::mem::size_of::<BasicTaskDesc>();
        let stackful = std::mem::size_of::<StackfulOnlyTaskDesc>();
        let stackless = std::mem::size_of::<StacklessOnlyTaskDesc>();
        assert!(stackful < basic, "StackfulOnlyTaskDesc ({stackful}) should be smaller than BasicTaskDesc ({basic})");
        assert!(stackless < basic, "StacklessOnlyTaskDesc ({stackless}) should be smaller than BasicTaskDesc ({basic})");
    }
}
