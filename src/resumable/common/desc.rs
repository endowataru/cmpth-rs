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
//! than a single hardcoded struct, mirroring [`crate::resumable::stackful::suspended::StackfulOnlyResumableCore`]
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

pub use crate::traits::common::{JoinState, TaskDesc};

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

/// Raw per-task descriptor storage: this crate's own concrete field layout
/// (a single `AtomicUsize` join-state word, per the `JS_*` encoding above,
/// plus owner-exclusive field storage). Implementing this trait opts a
/// descriptor into the join-protocol algorithm below for free (via the
/// blanket [`TaskDesc`] impl); a descriptor that wants a completely
/// different internal representation implements [`TaskDesc`] directly
/// instead — the same two-tier relationship as [`MutexCore`](crate::resumable::stackful::sync::MutexCore)/[`StackfulMutex`](crate::traits::StackfulMutex).
pub trait TaskDescCore: Send + Sync + Sized + 'static {
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
}

/// Blanket [`TaskDesc`] for any descriptor
/// using this crate's own word-based join-state encoding: the actual
/// join-protocol algorithm lives here (not as trait defaults on
/// [`TaskDescCore`]) so that trait stays a pure accessor contract. The two
/// token types this crate provides ([`SuspendedTaskToken`]/
/// [`RunningTaskToken`]) are the `Suspended`/`Running` witnesses — their
/// `DerefMut` (via `owned_cell()`/`UnsafeCell`) is exactly the kind of
/// implementation detail `TaskDesc` itself never mentions.
impl<D: TaskDescCore> TaskDesc for D {
    type Owned = <D as TaskDescCore>::Owned;
    type Suspended = SuspendedTaskToken<D>;
    type Running = RunningTaskToken<D>;

    #[inline]
    fn read_join_state(&self) -> JoinState<Self> {
        decode_join_state(self.join_state().load(Ordering::Acquire))
    }

    #[inline]
    fn is_finished(&self) -> bool {
        self.join_state().load(Ordering::Acquire) == JS_FINISHED
    }

    #[inline]
    fn commit_finished(&self) {
        self.join_state().store(JS_FINISHED, Ordering::Release);
    }

    #[inline]
    fn publish_finished(&self) -> JoinState<Self> {
        decode_join_state(self.join_state().swap(JS_FINISHED, Ordering::AcqRel))
    }

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
pub trait TaskDescAlloc: TaskDescCore + Sized {
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
pub(crate) unsafe fn peek_worker<D: TaskDescCore>(desc: *mut D) -> *const () {
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
pub struct SuspendedTaskToken<D: TaskDescCore>(*mut D);

unsafe impl<D: TaskDescCore> Send for SuspendedTaskToken<D> {}

/// Cashes in the token's proof of exclusive ownership: sound because a live
/// `SuspendedTaskToken<D>` is the only handle able to reach `D::Owned`
/// while it exists (move-only, no `Clone` — see the struct's own doc
/// comment). `join_state`/`waker_refs` live outside `Owned` specifically so
/// this never claims exclusivity over the genuinely-shared fields `wake()`
/// touches concurrently.
impl<D: TaskDescCore> Deref for SuspendedTaskToken<D> {
    type Target = D::Owned;
    fn deref(&self) -> &D::Owned {
        unsafe { &*(*self.0).owned_cell().get() }
    }
}

impl<D: TaskDescCore> DerefMut for SuspendedTaskToken<D> {
    fn deref_mut(&mut self) -> &mut D::Owned {
        unsafe { &mut *(*self.0).owned_cell().get() }
    }
}

impl<D: TaskDescCore> SuspendedTaskToken<D> {
    /// The one sanctioned way to conjure a token from a raw descriptor
    /// pointer. Every call site must justify, in its own `// SAFETY:`
    /// comment, why it alone holds exclusive access to `*ptr` right now
    /// (freshly allocated and never wrapped before; recovered from a
    /// slot/`join_state` word that only ever holds a pointer produced by a
    /// real token's `into_raw()`, whose own publish/consume protocol
    /// already proves single-consumer; or an FFI-boundary handoff of an
    /// already-linear pointer).
    ///
    /// # Safety
    /// The caller must hold exclusive access to `*ptr` for the lifetime of
    /// the returned token.
    pub(crate) unsafe fn from_raw(ptr: *mut D) -> Self {
        SuspendedTaskToken(ptr)
    }

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
        // SAFETY: `self` is itself the exclusivity proof; converting it to
        // the running-task counterpart for the same pointer transfers that
        // proof, it doesn't fabricate a new one.
        unsafe { RunningTaskToken::from_raw(self.into_raw()) }
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
pub struct RunningTaskToken<D: TaskDescCore>(*mut D);

unsafe impl<D: TaskDescCore> Send for RunningTaskToken<D> {}

/// See [`SuspendedTaskToken`]'s matching impl — identical reasoning, same
/// move-only exclusivity proof.
impl<D: TaskDescCore> Deref for RunningTaskToken<D> {
    type Target = D::Owned;
    fn deref(&self) -> &D::Owned {
        unsafe { &*(*self.0).owned_cell().get() }
    }
}

impl<D: TaskDescCore> DerefMut for RunningTaskToken<D> {
    fn deref_mut(&mut self) -> &mut D::Owned {
        unsafe { &mut *(*self.0).owned_cell().get() }
    }
}

impl<D: TaskDescCore> RunningTaskToken<D> {
    /// See [`SuspendedTaskToken::from_raw`] — identical contract.
    ///
    /// # Safety
    /// The caller must hold exclusive access to `*ptr` for the lifetime of
    /// the returned token.
    pub(crate) unsafe fn from_raw(ptr: *mut D) -> Self {
        RunningTaskToken(ptr)
    }

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
        // SAFETY: `self` is itself the exclusivity proof; converting it to
        // the suspended counterpart for the same pointer transfers that
        // proof, it doesn't fabricate a new one.
        unsafe { SuspendedTaskToken::from_raw(self.into_raw()) }
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

// ---------------------------------------------------------------------------
// PointerInterchangeable / Transferred
// ---------------------------------------------------------------------------

/// Implemented by move-only/linear values that can be losslessly flattened
/// to a raw pointer and reconstructed from one. Generalizes the
/// `into_raw`/`from_raw` naming convention `Box`/`Rc`/`Arc` each hand-roll
/// independently (the standard library has no shared trait for it) so
/// generic code — e.g. an FFI payload carrying "some interchangeable
/// value" across a context switch — can convert without knowing which
/// concrete linear type it's holding.
pub(crate) trait PointerInterchangeable: Sized {
    type Pointee;

    /// Consume `self`, discarding the wrapper but keeping the pointer.
    /// Always safe: the caller already owned `self`, this just changes its
    /// representation.
    fn into_ptr(self) -> *mut Self::Pointee;

    /// Reconstruct `Self` from a pointer previously produced by a matching
    /// [`into_ptr`](Self::into_ptr) (of this or a compatible type sharing
    /// the same `Pointee`), whose resulting claim hasn't been reclaimed
    /// since.
    ///
    /// # Safety
    /// The caller must hold exclusive access to `*ptr` for the lifetime of
    /// the returned value.
    unsafe fn from_ptr(ptr: *mut Self::Pointee) -> Self;
}

impl<D: TaskDescCore> PointerInterchangeable for SuspendedTaskToken<D> {
    type Pointee = D;
    fn into_ptr(self) -> *mut D { self.into_raw() }
    unsafe fn from_ptr(ptr: *mut D) -> Self { unsafe { Self::from_raw(ptr) } }
}

impl<D: TaskDescCore> PointerInterchangeable for RunningTaskToken<D> {
    type Pointee = D;
    fn into_ptr(self) -> *mut D { self.into_raw() }
    unsafe fn from_ptr(ptr: *mut D) -> Self { unsafe { Self::from_raw(ptr) } }
}

/// A [`PointerInterchangeable`] value, flattened so it can cross an
/// `extern "C"` context-switch boundary — a move-only Rust value can't
/// survive the actual assembly switch, so the shims carry this instead.
///
/// Constructing one from an already-owned value is safe (the caller
/// already holds whatever claim the value represented; this just reshapes
/// it for the FFI hop). Unpacking it back out the other side
/// ([`into_inner`](Self::into_inner), possibly as a *different*
/// `PointerInterchangeable` type sharing the same `Pointee` — e.g.
/// `SuspendedTaskToken` in, `RunningTaskToken` out) is therefore also safe
/// — a live `Transferred<T>` is itself the proof a real value was
/// flattened to make it, the same "the type's own existence is the proof"
/// pattern the tokens already use for `Owned` access. `from_raw` remains
/// as the one unavoidable exception (no predecessor value exists yet —
/// freshly allocated stacks), and is now the *only* unsafe surface left in
/// the whole FFI-crossing family, instead of being re-derived
/// independently at both ends of every shim.
#[repr(transparent)]
pub(crate) struct Transferred<T: PointerInterchangeable>(*mut T::Pointee);

impl<T: PointerInterchangeable> Transferred<T> {
    pub(crate) fn new(t: T) -> Self {
        Transferred(t.into_ptr())
    }

    /// # Safety
    /// The caller must hold exclusive access to `*ptr` for the lifetime of
    /// the returned value.
    pub(crate) unsafe fn from_raw(ptr: *mut T::Pointee) -> Self {
        Transferred(ptr)
    }

    pub(crate) fn into_inner<U>(self) -> U
    where
        U: PointerInterchangeable<Pointee = T::Pointee>,
    {
        // SAFETY: every `Transferred` either came from a real
        // `PointerInterchangeable` value (`new`) or from a call site that
        // independently justified `from_raw` — this doesn't add a new
        // claim, it just un-flattens the one already made.
        unsafe { U::from_ptr(self.0) }
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
