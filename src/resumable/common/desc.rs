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
//! (`result`/`tls`/`scheduler`, plus each flavor's own `ctx`/`poll_fn`) live
//! in a per-flavor [`TaskDesc::Owned`] struct, reached only through a
//! [`SuspendedTaskToken`]/[`RunningTaskToken`]'s `Deref`/`DerefMut` — see
//! [`DescOwned`]/[`HasDescOwned`]'s doc comments for why.
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
//! `TaskDesc`/`TaskDescAlloc`/`JoinState`/`DescOwned`/`HasDescOwned`/
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
pub use crate::traits::stackful::SyncJoinerTaskDesc;
use crate::interchange::PointerInterchangeable;

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
pub struct DescOwned {
    /// Written by the task itself before exiting; read by the joiner after
    /// `FINISHED` is observed.  (Root tasks only; spawned tasks put the
    /// result on their own stack.)
    pub(crate) result: Option<TaskResult>,

    /// Used by nested schedulers for their per-worker pointer (`UltTls`).
    /// Only touched by the OS thread currently running this task.
    pub(crate) tls: Option<HashMap<usize, *mut ()>>,
}

impl DescOwned {
    pub(crate) const fn new() -> Self {
        DescOwned { result: None, tls: None }
    }
}

/// Implemented by a [`TaskDescCore::Owned`] type that can hold a pointer
/// back to the [`Scheduler<S>`](crate::resumable::common::scheduler::Scheduler)
/// that owns this task — genuinely per-`S`-instance, not per-`S`-type: a
/// system's `run::<S>()` can have multiple live `Scheduler<S>` instances at
/// once (nothing prevents two independent `run::<S>()` calls on different
/// threads), so this can't be resolved through a single `S`-keyed static the
/// way `S::worker_tls()` is. Set at task-creation time (`spawn`/
/// `spawn_async`/`fork_parent_first`, regardless of task flavor) so `wake()`
/// called from an external OS thread — with no worker TLS to consult at all —
/// can still reach the right instance's `ExternalQueue`. Null for root
/// pseudo-descriptors.
///
/// Same shape as [`HasCtx`](crate::resumable::stackful::desc::HasCtx)/
/// [`HasPollFn`](crate::resumable::stackless::desc::HasPollFn): a capability
/// trait implemented by each flavor's own `Owned` struct. `System` (not a
/// generic parameter) names which `SchedulerSystem` this `Owned` belongs to
/// — this crate's three concrete `Owned` types are each already
/// parameterized by their own `S`, so `System = S` is a trivial projection,
/// not an extra type to track.
///
/// Folded in as a supertrait bound (unpinned `Owned: HasScheduler`, no
/// `System` equality yet) on
/// [`StackfulTaskDesc`](crate::resumable::stackful::desc::StackfulTaskDesc)
/// and [`AsyncTaskDesc`](crate::resumable::stackless::desc::AsyncTaskDesc).
/// The equality that actually matters for call sites —
/// `HasScheduler<System = S>` for the exact `S` in scope — is instead pinned
/// on [`StackfulSchedulerSystem`](crate::resumable::stackful::system::StackfulSchedulerSystem)/
/// [`StacklessSchedulerSystem`](crate::resumable::stackless::system::StacklessSchedulerSystem)
/// (each `S`-aware, unlike the `D`-only marker traits above), nested
/// directly in their own supertrait bound list
/// (`SchedulerSystem<Desc: ... + TaskDescCore<Owned: HasScheduler<System =
/// Self>>>`), not a separate `where`-clause. This distinction is load-bearing,
/// not stylistic: a `where`-clause attached to an associated type's own
/// declaration (whether on `SchedulerSystem::Desc` itself, or as a
/// *separate* `where`-clause on `StackfulSchedulerSystem`'s trait
/// declaration) is *not* an implied bound at call sites merely bounded by
/// the trait — verified empirically, twice, the hard way. Only an
/// associated-type bound nested inside a supertrait's own bound list
/// propagates as a real implied bound. `push_continuation` is the one
/// genuine leaf that reads `scheduler` without either flavor trait in scope
/// (it's only ever bounded on bare `SchedulerSystem`), so it restates
/// `HasScheduler<System = S>` explicitly instead.
pub trait HasScheduler {
    type System: crate::resumable::common::system::SchedulerSystem;
    fn scheduler(&self) -> *const crate::resumable::common::scheduler::Scheduler<Self::System>;
    fn set_scheduler(&mut self, scheduler: *const crate::resumable::common::scheduler::Scheduler<Self::System>);
}

/// Implemented by every [`TaskDesc::Owned`] type: gives generic code access
/// to the fields every flavor shares, regardless of what flavor-specific
/// fields (`ctx`, `poll_fn`, `dispatch`) the concrete `Owned` type adds
/// alongside `desc_owned`.
pub trait HasDescOwned {
    fn desc_owned(&self) -> &DescOwned;
    fn desc_owned_mut(&mut self) -> &mut DescOwned;
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

    /// This descriptor's owner-exclusive fields (see [`DescOwned`]/
    /// [`HasDescOwned`]).  Reached only through a live
    /// [`RunningTaskToken`]/[`SuspendedTaskToken`]'s `Deref`/`DerefMut` —
    /// never called directly outside this module.
    type Owned: HasDescOwned;
    fn owned_cell(&self) -> &UnsafeCell<Self::Owned>;

    /// Per-flavor "who might be waiting" outcome type — see
    /// [`TaskDesc::JoinOutcome`].
    /// Hand-specified per concrete flavor (same as `Owned` above), not
    /// blanket-derived, since the variant set genuinely differs.
    type JoinOutcome;

    /// Decode a raw `join_state` word (per the `JS_*` encoding above) into
    /// this flavor's own [`JoinOutcome`](Self::JoinOutcome).
    fn decode_join(word: usize) -> Self::JoinOutcome;
}

/// Internal, always-full-union decode — used only by this crate's own
/// resumable/-layer exit-completion code (`exit_with_result`/`exit`/
/// `poll_spawned_task`), which is already generic over a *concrete*
/// capability-bound flavor combination (`StackfulTaskDesc`/`AsyncTaskDesc`)
/// and needs to handle whichever waiter kind a `Dual` instantiation might
/// actually produce, regardless of which narrower
/// [`TaskDescCore::JoinOutcome`] the same generic code's *other*
/// instantiations (`StackfulOnlyTaskDesc`/`StacklessOnlyTaskDesc`) declare.
/// Deliberately bypasses the abstract [`TaskDesc::read_join_state`]/
/// [`TaskDesc::publish_finished`] (whose return type is properly narrowed
/// per flavor for external implementors) — this is the crate's own
/// implementation detail, not part of that public contract.
pub(crate) fn read_join_state_raw<D: TaskDescCore>(desc: &D) -> JoinState<D> {
    decode_join_state(desc.join_state().load(Ordering::Acquire))
}

/// See [`read_join_state_raw`] — same reasoning, the swap-and-decode
/// counterpart of [`TaskDesc::publish_finished`].
pub(crate) fn publish_finished_raw<D: TaskDescCore>(desc: &D) -> JoinState<D> {
    decode_join_state(desc.join_state().swap(JS_FINISHED, Ordering::AcqRel))
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
    type JoinOutcome = <D as TaskDescCore>::JoinOutcome;

    #[inline]
    fn read_join_state(&self) -> Self::JoinOutcome {
        D::decode_join(self.join_state().load(Ordering::Acquire))
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
    fn publish_finished(&self) -> Self::JoinOutcome {
        D::decode_join(self.join_state().swap(JS_FINISHED, Ordering::AcqRel))
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

/// Blanket [`SyncJoinerTaskDesc`] for any descriptor using this crate's own
/// word-based join-state encoding — same algorithm this used to be, just
/// relocated off the base [`TaskDesc`] (see that trait's doc comment for
/// why: only stackful call sites ever invoke it).
impl<D: TaskDescCore> SyncJoinerTaskDesc for D {
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
    /// Construct a descriptor value whose stack storage is `stack`, per the
    /// caller's `StackAlloc` policy. Returns `Self` by value, not a boxed
    /// pointer: pool bookkeeping (the old `pool_next`/`alloc_wk`/`oversized`
    /// fields) no longer lives on the descriptor, so wrapping it in a heap
    /// allocation (bare `Box<Self>`, or a pool's `Node<Self>`) is entirely
    /// the caller's decision, not this trait's. Used by the pool and by
    /// `spawn`'s parent-first fork path.
    fn alloc_with(stack: crate::resumable::common::stack::StackMem, has_handle: bool) -> Self;

    /// Construct a descriptor value with a plain heap buffer of
    /// `stack_size` bytes. Used by `spawn_async`, whose "stack" only ever
    /// stores a `Future` + result — no code runs on it, but it's allocated
    /// through the same `HeapStack` policy as a real stack regardless.
    fn alloc(stack_size: usize, has_handle: bool) -> Self;

    /// Pseudo-descriptor for a worker's own scheduler-loop context (the
    /// "root continuation"), embedded by value in `UltWorker`.
    fn new_root() -> Self;

    /// Reset a pooled descriptor for reuse (the stack allocation is kept).
    fn reinit(&mut self, has_handle: bool);
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

}

// ---------------------------------------------------------------------------
// PointerInterchangeable impls for this crate's own token types
// ---------------------------------------------------------------------------
//
// The trait itself, [`Transferred`], and the atomic hand-off slots built on
// it now live in [`crate::interchange`] — none of that machinery mentions
// any cmpth-specific type. These two impls are the bridge from that generic
// machinery to this crate's own linear token types.

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

#[cfg(test)]
mod tests {
    use crate::resumable::dual::desc::DualTaskDesc;
    use crate::resumable::stackful::desc::StackfulOnlyTaskDesc;
    use crate::resumable::stackless::desc::StacklessOnlyTaskDesc;
    use crate::{DefaultDualTaskSystem, DefaultStackfulOnlyTaskSystem, DefaultStacklessOnlyTaskSystem};

    /// Regression guard for the whole point of splitting the descriptor per
    /// flavor: a stackful-only/stackless-only system must not carry the
    /// unused half of `DualTaskDesc`'s `ctx`/`poll_fn` union. If this ever
    /// fails, something added a field back (or grew one) without noticing
    /// it defeated the split.
    #[test]
    fn narrow_descriptors_are_smaller_than_dual() {
        let dual = std::mem::size_of::<DualTaskDesc<DefaultDualTaskSystem>>();
        let stackful = std::mem::size_of::<StackfulOnlyTaskDesc<DefaultStackfulOnlyTaskSystem>>();
        let stackless = std::mem::size_of::<StacklessOnlyTaskDesc<DefaultStacklessOnlyTaskSystem>>();
        assert!(stackful < dual, "StackfulOnlyTaskDesc ({stackful}) should be smaller than DualTaskDesc ({dual})");
        assert!(stackless < dual, "StacklessOnlyTaskDesc ({stackless}) should be smaller than DualTaskDesc ({dual})");
    }
}
