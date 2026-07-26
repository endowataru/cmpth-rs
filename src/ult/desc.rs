//! Task descriptors and continuations.
//!
//! A [`SuspendedUlt`] is an owning handle to a suspended task: exactly one
//! continuation exists per suspended task, and consuming it (switching into
//! the context) invalidates it.  This mirrors ComposableThreads'
//! `basic_sct_continuation` / `suspended_thread` ownership model and is what
//! removes the old `ctx_saving` / `TaskState::Suspending` handshake: a
//! continuation only comes into existence *after* the context is fully saved,
//! because it is created by the switch callback running on the next stack.
//!
//! # `TaskDesc`/`StackfulTaskDesc`/`AsyncTaskDesc`
//!
//! Stage 1 of generalizing the descriptor (docs/sync-async-unification.md):
//! the field set is now behind named accessor traits rather than a single
//! hardcoded struct, mirroring [`crate::ult::suspended::UltSuspendedThread`]
//! (implementors supply accessors; scheduler code only ever calls the
//! trait). [`BasicTaskDesc`] is the one implementation that exists today —
//! same field layout, same types, same atomic orderings as the old
//! (non-generic) `UltDesc` — nothing about *runtime behavior* changed in
//! this pass. What it buys: every direct `(*desc).field` touch across
//! `worker.rs`/`thread.rs`/`waker.rs`/`pool.rs`/`tls.rs` now goes through a
//! named method, so a future descriptor type (e.g. a stackful-only one
//! without `ctx`, or a stackless-only one without a real stack) has a
//! contract to implement instead of a fixed struct to match byte-for-byte.
//! Actually letting the descriptor type vary per system (today every
//! concrete `S` still uses `BasicTaskDesc`) is deliberately deferred to a
//! later stage, to keep this pass's risk to "mechanical accessor
//! substitution, zero logic change" — this code is on every spawn/exit/join
//! hot path and has real atomic-ordering invariants (see each accessor's
//! callers), not something to restructure and re-derive semantics for in
//! the same pass.

use std::any::Any;
use std::cell::{Cell, UnsafeCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::task::{Context, Waker};

pub type TaskResult = Result<Box<dyn Any + Send>, Box<dyn Any + Send>>;

/// Result of driving one `spawn_async` task's poll to completion or a
/// suspend point (named `TaskPollResult`, not `PollResult`, to keep it out
/// of the way of `std::task::Poll` and `std::future::poll_fn` at a glance).
pub enum TaskPollResult<D> {
    /// The future finished; nothing left to do for this task.
    Ready,
    /// The future finished, and its completion claimed exclusive ownership
    /// of a waiting [`JoinState::AsyncJoiner`] — the caller's poll loop
    /// should continue directly into that descriptor next (symmetric
    /// transfer), instead of pushing it to a deque and waiting for some
    /// worker to pop it back out. Safe because `try_wake_state`'s
    /// `ClaimedParked` outcome (the only case this is constructed for)
    /// proves nobody else can be concurrently polling that descriptor.
    ReadyAndContinue(*mut D),
    /// The future returned `Poll::Pending`; the caller should park (or
    /// requeue immediately if a wake raced in during the poll).
    Pending,
}

/// Type-erased poll function stored on an async task's descriptor. Not
/// `PollFn` — that reads too much like `std::future::poll_fn` for a type
/// that has nothing to do with it.
pub type TaskPollFn<D> = for<'cx> unsafe fn(*mut D, &mut Context<'cx>) -> TaskPollResult<D>;

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
/// same shape as [`crate::ult::suspended::UltSuspendedThread`]'s `cont()`.
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
    fn slot(&self) -> &Cell<Option<*mut crate::ult::stack::CellSlot>>;

    /// True for the pseudo-descriptor representing a worker's scheduler-loop
    /// context (the "root continuation"). Fixed at construction.
    fn is_root(&self) -> bool;

    /// Written by the task itself before exiting; read by the joiner after
    /// `FINISHED` is observed.  (Root tasks only; spawned tasks put the
    /// result on their own stack.)
    fn result(&self) -> &UnsafeCell<Option<TaskResult>>;

    /// Intrusive linked-list pointer used when this descriptor sits in the
    /// task pool.  Undefined while the task is running.
    fn pool_next(&self) -> &Cell<*mut Self>;

    /// Index of the worker that allocated this descriptor.  Used by
    /// [`ReturnPool`](crate::ult::pool::ReturnPool) to route deallocation
    /// back to the home worker. Meaningless when [`oversized`](Self::oversized)
    /// is set (an oversized descriptor is never routed back to a free list).
    fn alloc_wk(&self) -> &Cell<usize>;

    /// True if this descriptor's storage is a one-off allocation that didn't
    /// fit a [`DescPool`](crate::ult::pool::DescPool)'s fixed slot size (see
    /// `DescPool::alloc`) — `dealloc` must free it directly rather than
    /// return it to the free list. Always false for descriptors that fit the
    /// pool's configured size, which is the common case for both fixed-size
    /// ULT stacks and most `spawn_async` futures.
    fn oversized(&self) -> &Cell<bool>;

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
    /// Allocate a descriptor whose stack storage is `stack` (heap or arena,
    /// per the caller's `StackAlloc` policy). Used by the pool and by
    /// `spawn`'s parent-first fork path.
    fn alloc_with(stack: crate::ult::stack::StackMem, has_handle: bool) -> *mut Self;

    /// Allocate a descriptor with a plain heap buffer of `stack_size` bytes,
    /// bypassing any arena/guard-page policy. Used by `spawn_async`, whose
    /// "stack" only ever stores a `Future` + result — no code runs on it, so
    /// it never needs the arena.
    fn alloc(stack_size: usize, has_handle: bool) -> *mut Self;

    /// Pseudo-descriptor for a worker's own scheduler-loop context (the
    /// "root continuation"), embedded by value in `UltWorker`.
    fn new_root() -> Self;

    /// # Safety
    /// Must be called exactly once, after no other references exist.
    unsafe fn free(ptr: *mut Self);

    /// Reset a pooled descriptor for reuse (the stack allocation is kept).
    fn reinit(&mut self, has_handle: bool);
}

/// Descriptor operations needed only by tasks with a real, switchable
/// execution stack (stackful ULTs). A pure-stackless descriptor type would
/// not implement this — there is no saved context to hand off, since
/// `run_async_poll` never does a context switch.
pub trait StackfulTaskDesc: TaskDesc {
    /// Saved context pointer; null while the task is running.
    ///
    /// Written with `Release` by the context-switch shim; claimed with
    /// `Acquire` or `AcqRel` by resumer or waker.
    fn ctx(&self) -> &AtomicPtr<u8>;

    /// Claim this task's saved context before switching into it (`Acquire`
    /// swap-to-null). The caller is expected to `debug_assert` the returned
    /// pointer is non-null (a null result means a double-resume — the exact
    /// diagnostic message differs per call site, so that check stays there).
    fn claim_saved_context(&self) -> *mut u8 {
        self.ctx().swap(std::ptr::null_mut(), std::sync::atomic::Ordering::Acquire)
    }

    /// Look at this task's saved context without consuming it (`Acquire`
    /// load) — used when the caller might not actually commit to switching
    /// (`cond_suspend_to_cont`).
    fn peek_saved_context(&self) -> *mut u8 {
        self.ctx().load(std::sync::atomic::Ordering::Acquire)
    }

    /// Publish a just-saved context (`Release` swap), making this task
    /// resumable. Returns the previous value so the caller can
    /// `debug_assert` it was null (overwriting a live context is a bug).
    fn publish_saved_context(&self, ptr: *mut u8) -> *mut u8 {
        self.ctx().swap(ptr, std::sync::atomic::Ordering::Release)
    }

    /// Initialize the context of a freshly allocated task that has never
    /// been suspended (`Release` store — cheaper than `publish_saved_context`
    /// since there is provably nothing to overwrite, so no swap-and-check
    /// is needed).
    fn init_saved_context(&self, ptr: *mut u8) {
        self.ctx().store(ptr, std::sync::atomic::Ordering::Release);
    }

    /// Clear this task's saved context (`Relaxed` store) when synchronization
    /// is already established by other means — used by `cond_suspend_shim`'s
    /// commit/cancel cleanup, after the ordering-relevant handoff already
    /// happened via the context switch itself.
    fn clear_saved_context(&self) {
        self.ctx().store(std::ptr::null_mut(), std::sync::atomic::Ordering::Relaxed);
    }
}

/// Descriptor operations needed by any task that can be driven via a real
/// [`std::task::Waker`] — both `block_on` (polling an arbitrary `Future`
/// from a real ULT) and `spawn_async` tasks need this `waker_refs` state
/// machine; `spawn_async` additionally needs [`AsyncTaskDesc`] on top for
/// its `poll_fn` task representation. Kept separate from `AsyncTaskDesc` so
/// that `ThreadSystem::block_on` (and anything generic over `S: ThreadSystem`,
/// like `DelegatorConsumer`) doesn't drag in `poll_fn`/spawn_async-specific
/// machinery it never touches — a system that supports `block_on` but not
/// `spawn_async` is expressible this way.
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

/// Descriptor operations needed only by tasks that represent a `spawn_async`
/// Future — the type-erased poll entry point. Builds on [`WakerTaskDesc`]
/// (a `spawn_async` task's own poll loop, `run_async_poll`, uses
/// `mark_polling`/`park_after_poll` on itself just like `block_on` does).
pub trait AsyncTaskDesc: WakerTaskDesc {
    /// Non-null for async tasks spawned via `spawn_async`; null for sync
    /// ULTs.
    ///
    /// When set, `Worker::execute` calls this instead of doing a context
    /// switch.  The function polls the Future stored in the task's "stack"
    /// buffer; see [`TaskPollResult`] for what it reports back (`Ready`:
    /// don't touch `desc` again; `Pending`: park it; `ReadyAndContinue`:
    /// poll the named descriptor next instead).
    fn poll_fn(&self) -> &Cell<Option<TaskPollFn<Self>>>;
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

/// The one descriptor implementation that exists today: same layout as the
/// pre-generalization `UltDesc`.  `repr(C)`: the fields touched by every
/// spawn/exit/join round-trip (`ctx`, `join_state`, `worker`, `slot`,
/// `poll_fn`, flags) are laid out first so they share one cache line.
#[repr(C)]
pub struct BasicTaskDesc {
    // --- Hot: touched on every spawn/exit/join ----------------------------
    ctx: AtomicPtr<u8>,
    join_state: AtomicUsize,
    worker: Cell<*const ()>,
    slot: Cell<Option<*mut crate::ult::stack::CellSlot>>,
    poll_fn: Cell<Option<TaskPollFn<BasicTaskDesc>>>,
    is_root: bool,

    // --- Warm ---------------------------------------------------------------
    result: UnsafeCell<Option<TaskResult>>,

    // --- Async waker (block_on) ------------------------------------------
    waker_refs: AtomicUsize,
    scheduler: Cell<*const ()>,

    // --- Pool metadata ---------------------------------------------------
    pool_next: Cell<*mut BasicTaskDesc>,
    alloc_wk: Cell<usize>,
    oversized: Cell<bool>,

    // --- ULT-local storage -----------------------------------------------
    tls: UnsafeCell<Option<HashMap<usize, *mut ()>>>,

    // --- Stack -----------------------------------------------------------
    stack: crate::ult::stack::StackMem,
}

unsafe impl Send for BasicTaskDesc {}
unsafe impl Sync for BasicTaskDesc {}

impl TaskDesc for BasicTaskDesc {
    fn join_state(&self) -> &AtomicUsize { &self.join_state }
    fn worker(&self) -> &Cell<*const ()> { &self.worker }
    fn slot(&self) -> &Cell<Option<*mut crate::ult::stack::CellSlot>> { &self.slot }
    fn is_root(&self) -> bool { self.is_root }
    fn result(&self) -> &UnsafeCell<Option<TaskResult>> { &self.result }
    fn pool_next(&self) -> &Cell<*mut Self> { &self.pool_next }
    fn alloc_wk(&self) -> &Cell<usize> { &self.alloc_wk }
    fn oversized(&self) -> &Cell<bool> { &self.oversized }
    fn tls(&self) -> &UnsafeCell<Option<HashMap<usize, *mut ()>>> { &self.tls }
    fn stack_top(&self) -> *mut u8 { self.stack.top() }
    fn scheduler(&self) -> &Cell<*const ()> { &self.scheduler }
}

impl StackfulTaskDesc for BasicTaskDesc {
    fn ctx(&self) -> &AtomicPtr<u8> { &self.ctx }
}

impl WakerTaskDesc for BasicTaskDesc {
    fn waker_refs(&self) -> &AtomicUsize { &self.waker_refs }
}

impl AsyncTaskDesc for BasicTaskDesc {
    fn poll_fn(&self) -> &Cell<Option<TaskPollFn<Self>>> { &self.poll_fn }
}

impl TaskDescAlloc for BasicTaskDesc {
    fn alloc_with(stack: crate::ult::stack::StackMem, has_handle: bool) -> *mut Self {
        BasicTaskDesc::alloc_with(stack, has_handle)
    }

    fn alloc(stack_size: usize, has_handle: bool) -> *mut Self {
        BasicTaskDesc::alloc(stack_size, has_handle)
    }

    fn new_root() -> Self {
        BasicTaskDesc::new_root()
    }

    unsafe fn free(ptr: *mut Self) {
        unsafe { BasicTaskDesc::free(ptr) }
    }

    fn reinit(&mut self, has_handle: bool) {
        BasicTaskDesc::reinit(self, has_handle)
    }
}

impl BasicTaskDesc {
    /// Allocate a descriptor with a heap stack.  Freed with
    /// [`BasicTaskDesc::free`].  Used by `spawn_async` (whose "stack" only
    /// stores the future — no code runs on it, so it never needs the arena).
    pub(crate) fn alloc(stack_size: usize, has_handle: bool) -> *mut BasicTaskDesc {
        use crate::ult::stack::{HeapStack, StackAlloc as _};
        Self::alloc_with(HeapStack::alloc_stack(stack_size).into(), has_handle)
    }

    /// Allocate a descriptor with a policy-allocated stack.  For arena
    /// stacks, captures the cell slot pointer for use by the switch shims.
    pub(crate) fn alloc_with(stack: crate::ult::stack::StackMem, has_handle: bool) -> *mut BasicTaskDesc {
        // Compute slot before moving `stack` into the Box.
        let slot = stack.cell_slot();
        Box::into_raw(Box::new(BasicTaskDesc {
            ctx: AtomicPtr::new(std::ptr::null_mut()),
            is_root: false,
            join_state: AtomicUsize::new(if has_handle { JS_RUNNING } else { JS_DETACHED }),
            result: UnsafeCell::new(None),
            waker_refs: AtomicUsize::new(0),
            scheduler: Cell::new(std::ptr::null()),
            worker: Cell::new(std::ptr::null()),
            slot: Cell::new(slot),
            poll_fn: Cell::new(None),
            pool_next: Cell::new(std::ptr::null_mut()),
            alloc_wk: Cell::new(0),
            oversized: Cell::new(false),
            tls: UnsafeCell::new(None),
            stack,
        }))
    }

    /// Pseudo-descriptor for a worker's scheduler-loop context.
    pub(crate) fn new_root() -> BasicTaskDesc {
        BasicTaskDesc {
            ctx: AtomicPtr::new(std::ptr::null_mut()),
            is_root: true,
            join_state: AtomicUsize::new(JS_DETACHED),
            result: UnsafeCell::new(None),
            waker_refs: AtomicUsize::new(0),
            scheduler: Cell::new(std::ptr::null()),
            worker: Cell::new(std::ptr::null()),
            slot: Cell::new(None),
            poll_fn: Cell::new(None),
            pool_next: Cell::new(std::ptr::null_mut()),
            alloc_wk: Cell::new(0),
            oversized: Cell::new(false),
            tls: UnsafeCell::new(None),
            stack: crate::ult::stack::StackMem::None,
        }
    }

    /// # Safety
    /// Must be called exactly once, after no other references exist.
    pub(crate) unsafe fn free(ptr: *mut BasicTaskDesc) {
        unsafe { drop(Box::from_raw(ptr)) };
    }

    /// Reset a pooled descriptor for reuse (the stack allocation is kept).
    ///
    /// Safe to reset everything: the exit path's *last* access to a
    /// descriptor is the `join_state` publication itself, so once a joiner
    /// has observed `FINISHED` and freed the descriptor, no stale stores
    /// from the previous task can be in flight.
    pub(crate) fn reinit(&mut self, has_handle: bool) {
        debug_assert!(!self.is_root);
        *self.ctx.get_mut() = std::ptr::null_mut();
        *self.join_state.get_mut() = if has_handle { JS_RUNNING } else { JS_DETACHED };
        *self.waker_refs.get_mut() = 0;
        *self.result.get_mut() = None;
        *self.tls.get_mut() = None;
        self.poll_fn.set(None);
    }
}

/// Owning handle to a suspended task.  Not `Clone`, not `Drop`: ownership is
/// linear and consuming the continuation (resuming it or storing it in a
/// waiter slot) is explicit.
///
/// Generic over the descriptor type `D` so a stackful-only or stackless-only
/// system can plug in a narrower descriptor later without touching every
/// deque/pool/worker call site again — for now every concrete system still
/// sets `D = BasicTaskDesc`.
pub struct SuspendedUlt<D: TaskDesc>(pub(crate) *mut D);

unsafe impl<D: TaskDesc> Send for SuspendedUlt<D> {}

impl<D: TaskDesc> SuspendedUlt<D> {
    pub(crate) fn desc(&self) -> *mut D {
        self.0
    }

    pub(crate) fn is_root(&self) -> bool {
        unsafe { (*self.0).is_root() }
    }

    pub(crate) fn into_raw(self) -> *mut D {
        self.0
    }
}
