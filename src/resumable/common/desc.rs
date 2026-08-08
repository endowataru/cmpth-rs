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
//! (`tls`/`external_queue`, plus each flavor's own `ctx`/`poll_fn`) live
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

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Waker;

pub use crate::traits::common::{TaskDesc, TaskExitSink};
pub use crate::traits::stackful::HandoffTaskDesc;
use crate::interchange::PointerInterchangeable;

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

/// Decoded view of a task descriptor's join-protocol state — who (if
/// anyone) is waiting on this task, or whether it has already finished.
/// Crate-private: [`TaskDesc`] itself deliberately says nothing about who
/// might be waiting (see that trait's own doc comment) — this is purely
/// this crate's own internal decoding of the `join_state` word, used by
/// [`decode_join_state`] and the blanket [`TaskDesc`]/[`HandoffTaskDesc`]/
/// [`WakerTaskDesc`](crate::traits::stackless::WakerTaskDesc) impls below.
pub(crate) enum JoinState<D> {
    /// Task alive, nobody waiting.
    Running,
    /// Result written (or the task was detached-and-cleaned).
    Finished,
    /// The `JoinHandle` was dropped early; the exit path cleans up.
    Detached,
    /// A parked sync joiner, registered via
    /// [`HandoffTaskDesc::try_register_joiner`].
    SyncJoiner(*mut D),
    /// A registered async waker — used when the polling task's waker isn't
    /// verifiably one of this system's own (foreign executor, or no worker
    /// at all).
    AsyncWaker(*mut Waker),
    /// Same role as `AsyncWaker`, but unboxed: the polling task's own
    /// descriptor, reachable directly because its waker is known (by
    /// construction) to be this system's own poll-loop waker.
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
pub struct DescOwned {
    /// Used by nested schedulers for their per-worker pointer (`UltTls`).
    /// Only touched by the OS thread currently running this task.
    pub(crate) tls: Option<HashMap<usize, *mut ()>>,
}

impl DescOwned {
    pub(crate) const fn new() -> Self {
        DescOwned { tls: None }
    }
}

/// Implemented by a [`TaskDescCore::Owned`] type that records the capability
/// "this task can be woken by a thread that is not one of its own pool's
/// workers" — the descriptor remembers where such a wake should deliver its
/// continuation.
///
/// The target is the queue, not the whole
/// [`Scheduler<S>`](crate::resumable::common::scheduler::Scheduler): the
/// only reader, `push_continuation`, only ever pushes onto it. Naming
/// `Scheduler<S>` here — as this used to (`HasScheduler`) — forced the
/// descriptor layer to depend on the scheduler layer for no reason beyond
/// reaching one field, and forced an extra `System` associated type that
/// existed purely so `Scheduler<Self::System>` had something to name. A
/// pointer straight to the
/// [`ExternalWakeQueue`](crate::resumable::common::external_queue::ExternalWakeQueue)
/// the descriptor is actually allowed to see removes both: the descriptor
/// layer never needs to know `Scheduler` exists at all.
///
/// A raw pointer, not a reference: the queue outlives the descriptor, but
/// that is an invariant of the pool's lifecycle (the queue lives in the
/// `Scheduler`, which outlives every task it schedules), not something the
/// borrow checker can carry across a token whose own lifetime is tied to the
/// pool, not to this field. A reference would also be the wrong shape at the
/// one read site regardless: `push_continuation` must copy the pointer out
/// *before* it moves the token that borrows it (the push consumes the
/// token), so a borrow that outlived the copy would be self-defeating.
/// Null for root pseudo-descriptors, and for any task never handed to an
/// external waker.
///
/// Set at task-creation time (`spawn`/`spawn_async`/`fork_parent_first`/
/// `init`, regardless of task flavor) rather than resolved through a single
/// `S`-keyed static the way `S::worker_tls()` is: a system's `run::<S>()`
/// can have multiple live `Scheduler<S>` instances at once (nothing
/// prevents two independent `run::<S>()` calls on different threads, each
/// with its own external queue), so a process-wide per-`S` static would
/// pick the wrong instance's queue as often as the right one.
///
/// Same shape as [`HasCtx`](crate::resumable::stackful::desc::HasCtx)/
/// [`HasPollFn`](crate::resumable::stackless::desc::HasPollFn): a capability
/// trait implemented by each flavor's own `Owned` struct, parameterized by
/// `D` (not a generic type parameter of `Self`) for the same reason
/// [`HasPollFn<D>`](crate::resumable::stackless::desc::HasPollFn) is — this
/// crate's three concrete `Owned` types are each already parameterized by
/// their own `S`, so naming `D` here is what lets the *equality*
/// (`Queue = S::ExternalQueue`) be pinned independently of which `Owned`
/// happens to implement the trait.
///
/// Folded in as a supertrait bound (unpinned `Owned: HasExternalQueue<Self>`,
/// no `Queue` equality yet) on
/// [`StackfulTaskDesc`](crate::resumable::stackful::desc::StackfulTaskDesc)
/// and [`AsyncTaskDesc`](crate::resumable::stackless::desc::AsyncTaskDesc).
/// The equality that actually matters for call sites —
/// `HasExternalQueue<S::Desc, Queue = S::ExternalQueue>` for the exact `S` in
/// scope — is instead pinned on
/// [`StackfulSchedulerSystem`](crate::resumable::stackful::system::StackfulSchedulerSystem)/
/// [`StacklessSchedulerSystem`](crate::resumable::stackless::system::StacklessSchedulerSystem)
/// (each `S`-aware, unlike the `D`-only marker traits above), nested
/// directly in their own supertrait bound list (`SchedulerSystem<Desc: ... +
/// TaskDescCore<Owned: HasExternalQueue<Self::Desc, Queue =
/// Self::ExternalQueue>>>`), not a separate `where`-clause. This distinction
/// is load-bearing, not stylistic: a `where`-clause attached to an
/// associated type's own declaration (whether on `SchedulerSystem::Desc`
/// itself, or as a *separate* `where`-clause on `StackfulSchedulerSystem`'s
/// trait declaration) is *not* an implied bound at call sites merely bounded
/// by the trait — verified empirically, twice, the hard way. Only an
/// associated-type bound nested inside a supertrait's own bound list
/// propagates as a real implied bound. `push_continuation` is the one
/// genuine leaf that reads `external_queue` without either flavor trait in
/// scope (it's only ever bounded on bare `SchedulerSystem`), so it restates
/// `HasExternalQueue<S::Desc>` explicitly instead.
pub trait HasExternalQueue<D: TaskDescCore> {
    type Queue: crate::resumable::common::external_queue::ExternalWakeQueue<D>;
    fn external_queue(&self) -> *const Self::Queue;
    fn set_external_queue(&mut self, queue: *const Self::Queue);
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

    /// Try to claim a same-system async joiner registered via
    /// [`WakerTaskDesc::try_register_async_joiner`](crate::traits::stackless::WakerTaskDesc::try_register_async_joiner)
    /// — the async-joiner counterpart of [`TaskDesc::finish_and_settle`]'s
    /// `AsyncWaker` handling, letting the completing task hand off
    /// straight to a same-system polling task's own continuation instead
    /// of waking a boxed `Waker`.
    ///
    /// No-op default (`None`, nothing to claim): correct as-is for any
    /// descriptor type with no async capability at all — nothing can ever
    /// write this join-state tag without
    /// [`WakerTaskDesc`](crate::traits::stackless::WakerTaskDesc), which such a
    /// type doesn't implement, so this default is never actually invoked
    /// for it. The flavors that *do* implement
    /// [`WakerTaskDescCore`](crate::resumable::stackless::desc::WakerTaskDescCore)
    /// (`StacklessOnlyTaskDesc`, `DualTaskDesc`) override this to delegate
    /// to their own `WakerTaskDesc::try_claim_parked`. Same shape as
    /// [`HasCtx::commit_as_ctx`](crate::resumable::stackful::desc::HasCtx::commit_as_ctx)'s
    /// no-op default — a capability a type doesn't have has nothing to do
    /// here, so there is no async branch to reach for it, structurally
    /// rather than by a runtime check.
    fn try_claim_async_joiner(joiner: *mut Self) -> Option<SuspendedTaskToken<Self>> {
        let _ = joiner;
        None
    }
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
    fn is_finished(&self) -> bool {
        self.join_state().load(Ordering::Acquire) == JS_FINISHED
    }

    #[inline]
    fn commit_finished(&self) {
        self.join_state().store(JS_FINISHED, Ordering::Release);
    }

    fn finish_and_settle<K: TaskExitSink<Self>>(&self, sink: &K) {
        match decode_join_state::<D>(self.join_state().swap(JS_FINISHED, Ordering::AcqRel)) {
            JoinState::Running => {}
            JoinState::SyncJoiner(j) => {
                // SAFETY: `j` was published by `HandoffTaskDesc::try_register_joiner`'s
                // caller, which only commits it after consuming a real
                // `Suspended` token it exclusively held — this decode is
                // the sole consumer of that publish.
                sink.resume(unsafe { SuspendedTaskToken::from_raw(j) });
            }
            // No sink call: a foreign waker isn't this scheduler's
            // business to hand off through, unlike a same-system
            // continuation.
            JoinState::AsyncWaker(w) => unsafe { Box::from_raw(w) }.wake(),
            JoinState::AsyncJoiner(j) => {
                if let Some(token) = D::try_claim_async_joiner(j) {
                    sink.resume(token);
                }
                // else: genuinely POLLING/NOTIFIED right now — it will
                // notice FINISHED on its own next poll.
            }
            JoinState::Detached => sink.reclaim(),
            JoinState::Finished => unreachable!("cmpth: double task exit"),
        }
    }

    fn try_abandon(&self) -> bool {
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

    #[inline]
    fn is_abandoned(&self) -> bool {
        self.join_state().load(Ordering::Acquire) == JS_DETACHED
    }
}

/// Blanket [`HandoffTaskDesc`] for any descriptor using this crate's own
/// word-based join-state encoding — same algorithm this used to be, just
/// relocated off the base [`TaskDesc`] (see that trait's doc comment for
/// why: only stackful call sites ever invoke it).
impl<D: TaskDescCore> HandoffTaskDesc for D {
    fn try_take_handoff_target(&self) -> Option<Self::Suspended> {
        match decode_join_state::<D>(self.join_state().load(Ordering::Acquire)) {
            // SAFETY: same provenance as `TaskDesc::finish_and_settle`'s
            // `SyncJoiner` arm — `j` was published via a real `Suspended`
            // token consumed by `try_register_joiner`'s caller.
            JoinState::SyncJoiner(j) => Some(unsafe { SuspendedTaskToken::from_raw(j) }),
            JoinState::Running
            | JoinState::Finished
            | JoinState::Detached
            | JoinState::AsyncWaker(_)
            | JoinState::AsyncJoiner(_) => None,
        }
    }

    fn try_register_joiner(&self, joiner: Self::Suspended) -> Result<(), Self::Suspended> {
        let joiner_ptr = joiner.desc();
        let mut cur = self.join_state().load(Ordering::Relaxed);
        loop {
            if cur == JS_FINISHED {
                return Err(joiner);
            }
            match self.join_state().compare_exchange_weak(
                cur,
                joiner_ptr as usize,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if let JoinState::AsyncWaker(w) = decode_join_state::<Self>(cur) {
                        drop(unsafe { Box::from_raw(w) });
                    }
                    // Ownership now lives in the join_state word, reachable
                    // again only via `try_take_handoff_target`/
                    // `finish_and_settle`'s matching `from_raw` above.
                    let _ = joiner.into_raw();
                    return Ok(());
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
