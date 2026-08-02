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
//! # `TaskDesc`/`Owned`/`TaskDescAlloc`
//!
//! The field set lives behind named accessor traits/associated types rather
//! than a single hardcoded struct, mirroring [`crate::resumable::stackful::suspended::StackfulOnlyResumable`]
//! (implementors supply accessors; scheduler code only ever calls the
//! trait) — a concrete descriptor type is a contract to implement, not a
//! fixed struct to match byte-for-byte. Owner-exclusive fields
//! (`worker`/`slot`/`result`/`tls`/`scheduler`, plus each flavor's own
//! `ctx`/`poll_fn`) live in a per-flavor [`TaskDesc::Owned`] struct, reached
//! only through a [`SuspendedTaskToken`]/[`RunningTaskToken`]'s `Deref`/
//! `DerefMut` — see [`BaseOwned`]/[`HasBaseOwned`]'s doc comments for why.
//!
//! Only the shared machinery lives in this module. The three concrete
//! descriptor types, one per scheduler flavor, live alongside their own
//! flavor's other descriptor traits: `StackfulOnlyTaskDesc`
//! (`resumable::stackful::desc` — `UltIdentity` systems: a real ULT, no
//! `spawn_async` capability, no `poll_fn` slot at all), `StacklessOnlyTaskDesc`
//! (`resumable::stackless::desc` — `UltAsyncIdentity` systems: a
//! `spawn_async` future, no real context switch, no `ctx` slot at all), and
//! `DualTaskDesc` (`resumable::dual::desc` — dual systems: both capabilities
//! on the *same* descriptor, since a stackful sync joiner and a stackless
//! async waker can race to register on the same task — see that type's own
//! doc comment for why its `ctx`/`poll_fn` union needs the `commit_as_ctx`/
//! `commit_as_poll_fn` hooks the other two never touch).
//!
//! # Layering
//!
//! `TaskDesc`/`TaskDescAlloc`/`JoinState`/`BaseOwned`/`HasBaseOwned`/
//! `SuspendedTaskToken`/`RunningTaskToken` live here because they're
//! genuinely shared: the join-protocol (`join_state`/`JS_*`) applies to
//! every task regardless of flavor, and the tokens are generic over `D:
//! TaskDesc` without knowing which flavor `D` is.
//!
//! [`StackfulTaskDesc`](crate::resumable::stackful::desc::StackfulTaskDesc)/[`HasCtx`](crate::resumable::stackful::desc::HasCtx)
//! (real saved-context handling) and
//! [`WakerTaskDesc`](crate::resumable::stackless::desc::WakerTaskDesc)/[`AsyncTaskDesc`](crate::resumable::stackless::desc::AsyncTaskDesc)/[`HasPollFn`](crate::resumable::stackless::desc::HasPollFn)
//! are flavor-specific extension traits, split out to `stackful::desc`/
//! `stackless::desc` alongside the concrete descriptor type(s) that need
//! them. `WakerTaskDesc` moved to `stackless::desc` specifically because
//! `spawn_async` has no stack to resume — its wake state has nowhere to
//! live but the descriptor itself. `block_on` (stackful) needed the same
//! shape of state machine but not the descriptor: it uses
//! [`stackful::waker::ResumablePoller`](crate::resumable::stackful::waker::ResumablePoller),
//! a block_on-call-scoped box, driven by the same core CAS logic (factored
//! out to [`common::waker`](crate::resumable::common::waker) so both share
//! it) instead of a per-task field.

use std::any::Any;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Waker;

pub type TaskResult = Result<Box<dyn Any + Send>, Box<dyn Any + Send>>;

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

/// The owner-exclusive fields every task descriptor has, regardless of
/// flavor: touched only by whoever holds a live [`RunningTaskToken`]/
/// [`SuspendedTaskToken`] for this descriptor, never concurrently (that's
/// exactly the invariant those tokens' move-only discipline proves) — so
/// unlike `join_state`/`waker_refs` (genuinely racy, touched by an external
/// `wake()` at arbitrary times, and so left as plain `AtomicUsize` fields
/// on the descriptor itself), these need no `Cell`/`UnsafeCell` wrapping at
/// all. Reached only through a token's [`Deref`]/[`DerefMut`] (`Target =
/// D::Owned`), the same "the token proves the precondition, `Deref` cashes
/// it in" pattern as `MutexGuard`/`RefMut`.
pub struct BaseOwned {
    /// Type-erased `*const UltWorker<S>`: the worker that most recently
    /// switched into this task, written by the switch shims alongside
    /// `cur_task`.  A task cannot migrate between its last resume and its
    /// next suspension, so the exit path reads this instead of doing a TLS
    /// lookup.  Only valid while the task is running.
    pub(crate) worker: *const (),

    /// Points at the arena cell's `[worker, system_id]` slot for arena
    /// stacks, or `None` for heap/root stacks.  The switch shims write the
    /// resuming worker pointer here when present.
    pub(crate) slot: Option<*mut crate::resumable::common::stack::CellSlot>,

    /// Written by the task itself before exiting; read by the joiner after
    /// `FINISHED` is observed.  (Root tasks only; spawned tasks put the
    /// result on their own stack.)
    pub(crate) result: Option<TaskResult>,

    /// Used by nested schedulers for their per-worker pointer (`UltTls`).
    /// Only touched by the OS thread currently running this task.
    pub(crate) tls: Option<HashMap<usize, *mut ()>>,

    /// Type-erased `*const Scheduler<S>`.  Set at task-creation time —
    /// `spawn`, `spawn_async`, and `fork_parent_first` all record it,
    /// regardless of task flavor — so that `wake()` called from an external
    /// OS thread can reach the scheduler's `ExternalQueue` without going
    /// through worker TLS.  Null for root pseudo-descriptors. Only actually
    /// read by the `AsyncTaskDesc` wake path (`waker.rs::push_continuation`)
    /// today, but writing it doesn't need `AsyncTaskDesc` capability, so it
    /// lives on the base fields rather than gating every constructor on it.
    pub(crate) scheduler: *const (),
}

impl BaseOwned {
    pub(crate) const fn new() -> Self {
        BaseOwned { worker: std::ptr::null(), slot: None, result: None, tls: None, scheduler: std::ptr::null() }
    }
}

/// Implemented by every [`TaskDesc::Owned`] type: gives generic code (e.g.
/// `RunningTaskToken::mark_resumed_on`) access to the fields every flavor
/// shares, regardless of what flavor-specific fields (`ctx`, `poll_fn`,
/// `dispatch`) the concrete `Owned` type adds alongside `base`.
pub trait HasBaseOwned {
    fn base(&self) -> &BaseOwned;
    fn base_mut(&mut self) -> &mut BaseOwned;
}

/// Core per-task descriptor operations: every task, regardless of flavor,
/// needs these (join protocol, owner-exclusive field storage).
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

    /// True for the pseudo-descriptor representing a worker's scheduler-loop
    /// context (the "root continuation"). Fixed at construction.
    fn is_root(&self) -> bool;

    /// Top of this task's stack allocation (`StackMem::None` for root
    /// pseudo-descriptors, in which case this must never be called).
    fn stack_top(&self) -> *mut u8;

    /// This descriptor's owner-exclusive fields (see [`BaseOwned`]/
    /// [`HasBaseOwned`]).  Reached only through a live
    /// [`RunningTaskToken`]/[`SuspendedTaskToken`]'s `Deref`/`DerefMut` —
    /// never called directly outside this module.
    type Owned: HasBaseOwned;
    fn owned_cell(&self) -> &UnsafeCell<Self::Owned>;

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
/// here mirrors an existing `DualTaskDesc` inherent fn byte-for-byte; this
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

/// Peek at `desc`'s owner-exclusive `worker` field without going through a
/// token. For the handful of call sites that know, by construction, that
/// they're currently executing as `desc`'s own task, but have no token
/// value in local scope to route through — typically because they're using
/// `worker` to *rediscover* which `UltWorker` they're now running on after
/// a possible cross-worker migration, so there's no `wk` handy yet either.
/// Same peek discipline as [`UltWorker::cur_task`](crate::resumable::common::worker::UltWorker::cur_task):
/// sound because the caller is the only thread that could possibly be
/// touching this descriptor right now.
///
/// # Safety
/// The calling OS thread must currently be driving `desc` (mid-execution of
/// its body, between resume and suspend/exit).
pub(crate) unsafe fn peek_worker<D: TaskDesc>(desc: *mut D) -> *const () {
    unsafe { (*(*desc).owned_cell().get()).base().worker }
}

/// Owning handle to a suspended task.  Not `Clone`, not `Drop`: ownership is
/// linear and consuming the continuation (resuming it or storing it in a
/// waiter slot) is explicit.
///
/// Generic over the descriptor type `D` so a stackful-only or stackless-only
/// system plugs in a narrower descriptor (`StackfulOnlyTaskDesc`,
/// `StacklessOnlyTaskDesc`, or `DualTaskDesc` for dual) without touching
/// every deque/pool/worker call site — they're all written generically
/// over `D: TaskDesc`.
pub struct SuspendedTaskToken<D: TaskDesc>(pub(crate) *mut D);

unsafe impl<D: TaskDesc> Send for SuspendedTaskToken<D> {}

/// Cashes in the token's proof of exclusive ownership: sound because a live
/// `SuspendedTaskToken<D>` is the only handle able to reach `D::Owned`
/// while it exists (move-only, no `Clone` — see the struct's own doc
/// comment). `join_state`/`waker_refs` live outside `Owned` specifically so
/// this never claims exclusivity over the genuinely-shared fields `wake()`
/// touches concurrently.
impl<D: TaskDesc> Deref for SuspendedTaskToken<D> {
    type Target = D::Owned;
    fn deref(&self) -> &D::Owned {
        unsafe { &*(*self.0).owned_cell().get() }
    }
}

impl<D: TaskDesc> DerefMut for SuspendedTaskToken<D> {
    fn deref_mut(&mut self) -> &mut D::Owned {
        unsafe { &mut *(*self.0).owned_cell().get() }
    }
}

impl<D: TaskDesc> SuspendedTaskToken<D> {
    pub(crate) fn desc(&self) -> *mut D {
        self.0
    }

    pub(crate) fn is_root(&self) -> bool {
        unsafe { (*self.0).is_root() }
    }

    /// Safe access to the descriptor's own `&self` methods (join-protocol,
    /// waker state machine) — the token's existence is itself the proof
    /// `self.0` is live, so this is the one place that proof gets cashed in
    /// for `D` rather than `D::Owned`. Never conflicts with `DerefMut`'s
    /// exclusivity claim above: `TaskDesc`/`WakerTaskDesc`'s own methods
    /// only ever touch `join_state`/`waker_refs`, which live outside
    /// `Owned` for exactly this reason.
    pub(crate) fn as_desc(&self) -> &D {
        unsafe { &*self.0 }
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

/// See [`SuspendedTaskToken`]'s matching impl — identical reasoning, same
/// move-only exclusivity proof.
impl<D: TaskDesc> Deref for RunningTaskToken<D> {
    type Target = D::Owned;
    fn deref(&self) -> &D::Owned {
        unsafe { &*(*self.0).owned_cell().get() }
    }
}

impl<D: TaskDesc> DerefMut for RunningTaskToken<D> {
    fn deref_mut(&mut self) -> &mut D::Owned {
        unsafe { &mut *(*self.0).owned_cell().get() }
    }
}

impl<D: TaskDesc> RunningTaskToken<D> {
    pub(crate) fn desc(&self) -> *mut D {
        self.0
    }

    /// See [`SuspendedTaskToken::as_desc`] — identical reasoning.
    pub(crate) fn as_desc(&self) -> &D {
        unsafe { &*self.0 }
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

    /// Record that this task is now running on `worker_ptr` — called by
    /// every context-switch shim immediately after promoting `self` to
    /// `RunningTaskToken`. Propagates to the arena cell slot too (when
    /// present), since every caller that sets `worker` here has always also
    /// needed to update `slot` in the same breath.
    #[inline]
    pub(crate) fn mark_resumed_on(&mut self, worker_ptr: *const ()) {
        let base = self.base_mut();
        base.worker = worker_ptr;
        if let Some(slot) = base.slot {
            unsafe { (*slot).worker.set(worker_ptr) };
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::resumable::dual::desc::DualTaskDesc;
    use crate::resumable::stackful::desc::StackfulOnlyTaskDesc;
    use crate::resumable::stackless::desc::StacklessOnlyTaskDesc;

    /// Regression guard for the whole point of splitting the descriptor per
    /// flavor: a stackful-only/stackless-only system must not carry the
    /// unused half of `DualTaskDesc`'s `ctx`/`poll_fn` union. If this ever
    /// fails, something added a field back (or grew one) without noticing
    /// it defeated the split.
    #[test]
    fn narrow_descriptors_are_smaller_than_dual() {
        let dual = std::mem::size_of::<DualTaskDesc>();
        let stackful = std::mem::size_of::<StackfulOnlyTaskDesc>();
        let stackless = std::mem::size_of::<StacklessOnlyTaskDesc>();
        assert!(stackful < dual, "StackfulOnlyTaskDesc ({stackful}) should be smaller than DualTaskDesc ({dual})");
        assert!(stackless < dual, "StacklessOnlyTaskDesc ({stackless}) should be smaller than DualTaskDesc ({dual})");
    }
}
