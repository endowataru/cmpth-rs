//! Stackless-only descriptor operations: the `spawn_async` poll entry
//! point, and [`StacklessOnlyTaskDesc`] — the concrete descriptor for
//! `UltAsyncIdentity` (stackless-only) systems.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Waker};

use crate::resumable::common::desc::{BaseOwned, HasBaseOwned, JoinState, RunningTaskToken, SuspendedTaskToken, TaskDesc, TaskDescCore, TaskDescAlloc, decode_join_state, JS_ASYNC_JOINER_TAG, JS_ASYNC_TAG, JS_DETACHED, JS_FINISHED, JS_RUNNING};
use crate::resumable::common::waker::{self, WakeOutcome, EVER_SHARED, STATE_MASK};

/// Raw waker-state storage: this crate's own `AtomicUsize`-encoded
/// POLLING/PARKED/NOTIFIED/IDLE state machine (see
/// [`common::waker`](crate::resumable::common::waker)'s module doc comment
/// for the encoding). Implementing this (together with [`TaskDescCore`])
/// opts a descriptor into the [`WakerTaskDesc`] operations below for free
/// via the blanket impl — the same two-tier relationship as
/// [`TaskDescCore`]/[`TaskDesc`].
pub trait WakerTaskDescCore: TaskDescCore {
    /// Zero (IDLE) when no poll session is active on this task.
    fn waker_refs(&self) -> &AtomicUsize;
}

/// Descriptor operations needed by a task driven via a real
/// [`std::task::Waker`] whose wake state must live on the descriptor
/// itself — currently only `spawn_async` (see [`AsyncTaskDesc`]'s doc
/// comment for why: no stack to anchor the state anywhere else). Bodyless —
/// implement directly for a custom representation, or implement
/// [`WakerTaskDescCore`] instead to get this crate's own algorithm for free.
///
/// `block_on` (stackful) used to share this trait too, but only ever
/// needed the state machine, not a descriptor field — it now drives the
/// same core CAS logic (factored out to
/// [`common::waker`](crate::resumable::common::waker) so both share it)
/// against a block_on-call-scoped box instead; see
/// [`ResumablePoller`](crate::resumable::stackful::waker::ResumablePoller).
pub trait WakerTaskDesc: TaskDesc {
    // --- named waker_refs state-machine operations, delegating to the
    // shared core in `common::waker` -----------------------------------

    fn mark_polling(&self);
    fn mark_idle(&self);
    fn decide_park(&self) -> bool;
    fn park_after_poll(&self) -> bool;

    /// Core wake CAS loop, shared by the stackful (`try_wake`) and async
    /// (`try_wake_async`) wake paths in `waker.rs`.
    fn try_wake_state(&self) -> WakeOutcome;

    /// True once this waker has been cloned at least once. Sticky — never
    /// clears back to false.
    fn is_ever_shared(&self) -> bool;

    /// First-clone transition: set `EVER_SHARED`, preserving whatever state
    /// bits are currently set. CAS loop: a concurrent `wake()` may change
    /// the state bits underneath. No ref count to seed — nothing here ever
    /// frees based on one (see `common::waker::drop_shared`'s doc comment).
    fn transition_to_shared(&self);

    // --- async join-registration, sharing join_state() with sync joiners.
    // Lives here (not on the base TaskDesc) so a descriptor with no async
    // capability at all (StackfulOnlyTaskDesc) never gets these — they are
    // only ever called from `JoinHandle::poll` (`stackless/thread.rs`),
    // which itself requires `S::Desc: AsyncTaskDesc: WakerTaskDesc`. -------

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
    unsafe fn try_register_async_joiner(&self, joiner: *mut Self) -> bool;

    /// `JoinHandle::poll`'s waker registration: try to install `waker` as
    /// this task's async waiter. Returns `false` if the task turned out to
    /// already be finished (caller should proceed to take the result
    /// instead) — otherwise commits the boxed, tagged waker and drops
    /// whichever waker it superseded, if any.
    fn try_register_waker(&self, waker: Waker) -> bool;
}

/// Blanket [`WakerTaskDesc`] for any descriptor using this crate's own
/// word-based encodings for both join-state and waker-refs.
impl<D: TaskDescCore + WakerTaskDescCore> WakerTaskDesc for D {
    #[inline]
    fn mark_polling(&self) {
        waker::mark_polling(self.waker_refs())
    }

    #[inline]
    fn mark_idle(&self) {
        waker::mark_idle(self.waker_refs())
    }

    fn decide_park(&self) -> bool {
        waker::decide_park(self.waker_refs())
    }

    #[inline]
    fn park_after_poll(&self) -> bool {
        waker::park_after_poll(self.waker_refs())
    }

    fn try_wake_state(&self) -> WakeOutcome {
        waker::try_wake_state(self.waker_refs())
    }

    fn is_ever_shared(&self) -> bool {
        self.waker_refs().load(Ordering::Relaxed) & EVER_SHARED != 0
    }

    fn transition_to_shared(&self) {
        loop {
            let old = self.waker_refs().load(Ordering::Relaxed);
            let new = EVER_SHARED | (old & STATE_MASK);
            if self
                .waker_refs()
                .compare_exchange(old, new, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
        }
    }

    #[inline]
    unsafe fn try_register_async_joiner(&self, joiner: *mut Self) -> bool {
        debug_assert_eq!(
            joiner as usize & (JS_ASYNC_TAG | JS_ASYNC_JOINER_TAG),
            0,
            "cmpth: descriptor pointer not aligned enough to tag"
        );
        let mut cur = TaskDescCore::join_state(self).load(Ordering::Acquire);
        let new = (joiner as usize) | JS_ASYNC_JOINER_TAG;
        loop {
            if cur == JS_FINISHED {
                return false;
            }
            match TaskDescCore::join_state(self).compare_exchange_weak(
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

    fn try_register_waker(&self, waker: Waker) -> bool {
        let mut cur = TaskDescCore::join_state(self).load(Ordering::Acquire);
        if cur == JS_FINISHED {
            return false;
        }
        let new = Box::into_raw(Box::new(waker)) as usize | JS_ASYNC_TAG;
        loop {
            if cur == JS_FINISHED {
                drop(unsafe { Box::from_raw((new & !JS_ASYNC_TAG) as *mut Waker) });
                return false;
            }
            match TaskDescCore::join_state(self).compare_exchange_weak(
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
}

/// Result of driving one `spawn_async` task's poll to completion or a
/// suspend point (named `TaskPollResult`, not `PollResult`, to keep it out
/// of the way of `std::task::Poll` and `std::future::poll_fn` at a glance).
pub enum TaskPollResult<D> {
    /// The future finished; nothing left to do for this task.
    Ready,
    /// The future finished, and its completion claimed exclusive ownership
    /// of a waiting [`JoinState::AsyncJoiner`]
    /// — the caller's poll loop should continue directly into that
    /// descriptor next (symmetric transfer), instead of pushing it to a
    /// deque and waiting for some worker to pop it back out. Safe because
    /// `try_wake_state`'s `ClaimedParked` outcome (the only case this is
    /// constructed for) proves nobody else can be concurrently polling that
    /// descriptor.
    ReadyAndContinue(*mut D),
    /// The future returned `Poll::Pending`; the caller should park (or
    /// requeue immediately if a wake raced in during the poll).
    Pending,
}

/// Type-erased poll function stored on an async task's descriptor. Not
/// `PollFn` — that reads too much like `std::future::poll_fn` for a type
/// that has nothing to do with it.
pub type TaskPollFn<D> = for<'cx> unsafe fn(*mut D, &mut Context<'cx>) -> TaskPollResult<D>;

/// Implemented by a [`TaskDesc::Owned`] type that can hold a poll_fn entry
/// point — either directly ([`StacklessOnlyTaskDesc`]'s
/// `Owned`) or as one variant of a `ctx`/`poll_fn` union
/// ([`DualTaskDesc`](crate::resumable::dual::desc::DualTaskDesc)'s
/// `Owned`, via `TaskDispatch`).
/// Plain field, not `Cell`: same reasoning as
/// [`HasCtx`](crate::resumable::stackful::desc::HasCtx).
pub trait HasPollFn<D> {
    fn poll_fn(&self) -> Option<TaskPollFn<D>>;
    fn set_poll_fn(&mut self, f: Option<TaskPollFn<D>>);

    /// Ensure this `Owned` is configured for poll_fn dispatch. Called once
    /// by the allocating call site (`spawn_now`, `fork_async_parent_first`)
    /// right after allocation, before `set_poll_fn` ever runs — the async
    /// analogue of [`HasCtx::commit_as_ctx`](crate::resumable::stackful::desc::HasCtx::commit_as_ctx);
    /// see that method's doc comment for why this exists and which `Owned`
    /// type actually needs it.
    fn commit_as_poll_fn(&mut self) {}

    /// Is this `Owned` currently committed to poll_fn dispatch? The safe
    /// way to ask "is this a `spawn_async` task or a real ULT" — unlike
    /// calling `poll_fn()` directly, this never panics regardless of which
    /// way the answer comes out. Used by dual dispatch
    /// (`resumable::dual::worker`) wherever a popped continuation could
    /// legitimately be either kind.
    ///
    /// Default `true`: correct unconditionally for an `Owned` type that
    /// implements `HasPollFn` but not
    /// [`HasCtx`](crate::resumable::stackful::desc::HasCtx) (e.g.
    /// `StacklessOnlyTaskDesc`'s) — every task on such a system is
    /// poll_fn, there is no other possibility. `DualTaskDesc`'s `Owned`
    /// overrides this to check its `ctx`/`poll_fn` union's actual current
    /// variant.
    fn is_poll_fn_dispatch(&self) -> bool { true }
}

/// Descriptor operations needed only by tasks that represent a `spawn_async`
/// Future — the type-erased poll entry point. Builds on [`WakerTaskDesc`]
/// (a `spawn_async` task's own poll loop, `run_async_poll`, uses
/// `mark_polling`/`park_after_poll` on itself just like `block_on` does).
pub trait AsyncTaskDesc: WakerTaskDesc + TaskDesc<Owned: HasPollFn<Self>> {}

impl<D: WakerTaskDesc + TaskDesc<Owned: HasPollFn<D>>> AsyncTaskDesc for D {}

impl<D: TaskDesc<Owned: HasPollFn<D>>> SuspendedTaskToken<D> {
    /// The type-erased poll entry point, non-null once `spawn_now`/
    /// `fork_async_parent_first` finish setting up a `spawn_async` task.
    ///
    /// **Only call this once [`is_poll_fn_dispatch`](Self::is_poll_fn_dispatch)
    /// has confirmed this descriptor is actually committed to poll_fn
    /// dispatch** (or the caller otherwise already knows that, e.g. it's
    /// working with a `StacklessOnlyTaskDesc`-like type where that's the
    /// only possibility). On a type with more than one dispatch mode on the
    /// same struct (i.e. `DualTaskDesc`), calling this on a
    /// `ctx`-committed descriptor is a logic error (`debug_assert`s in
    /// debug builds, UB in release — see that type's `HasPollFn` impl).
    ///
    /// When set, `Worker::execute` calls this instead of doing a context
    /// switch.  The function polls the Future stored in the task's "stack"
    /// buffer; see [`TaskPollResult`] for what it reports back (`Ready`:
    /// don't touch `desc` again; `Pending`: park it; `ReadyAndContinue`:
    /// poll the named descriptor next instead).
    pub(crate) fn poll_fn(&self) -> Option<TaskPollFn<D>> {
        (**self).poll_fn()
    }

    pub(crate) fn is_poll_fn_dispatch(&self) -> bool {
        (**self).is_poll_fn_dispatch()
    }
}

impl<D: TaskDesc<Owned: HasPollFn<D>>> RunningTaskToken<D> {
    pub(crate) fn poll_fn(&self) -> Option<TaskPollFn<D>> {
        (**self).poll_fn()
    }
}

// ---------------------------------------------------------------------------
// StacklessOnlyTaskDesc — UltAsyncIdentity systems (spawn_async, no real
// context switch)
// ---------------------------------------------------------------------------

/// Owner-exclusive fields for [`StacklessOnlyTaskDesc`]: [`BaseOwned`] plus
/// the poll_fn entry point (no `ctx` slot — this flavor never does a real
/// context switch).
pub struct StacklessOnlyOwned {
    base: BaseOwned,
    poll_fn: Option<TaskPollFn<StacklessOnlyTaskDesc>>,
}

impl HasBaseOwned for StacklessOnlyOwned {
    fn base(&self) -> &BaseOwned { &self.base }
    fn base_mut(&mut self) -> &mut BaseOwned { &mut self.base }
}

impl HasPollFn<StacklessOnlyTaskDesc> for StacklessOnlyOwned {
    fn poll_fn(&self) -> Option<TaskPollFn<StacklessOnlyTaskDesc>> { self.poll_fn }
    fn set_poll_fn(&mut self, f: Option<TaskPollFn<StacklessOnlyTaskDesc>>) { self.poll_fn = f; }
}

/// Concrete descriptor for `UltAsyncIdentity`-based (stackless-only)
/// systems: a `spawn_async` task with no real context switch, so no `ctx`
/// slot exists at all.
pub struct StacklessOnlyTaskDesc {
    owned: UnsafeCell<StacklessOnlyOwned>,
    join_state: AtomicUsize,
    is_root: bool,
    waker_refs: AtomicUsize,
    stack: crate::resumable::common::stack::StackMem,
}

unsafe impl Send for StacklessOnlyTaskDesc {}
unsafe impl Sync for StacklessOnlyTaskDesc {}

impl TaskDescCore for StacklessOnlyTaskDesc {
    fn join_state(&self) -> &AtomicUsize { &self.join_state }
    fn is_root(&self) -> bool { self.is_root }
    fn stack_top(&self) -> *mut u8 { self.stack.top() }
    type Owned = StacklessOnlyOwned;
    fn owned_cell(&self) -> &UnsafeCell<StacklessOnlyOwned> { &self.owned }
}

impl WakerTaskDescCore for StacklessOnlyTaskDesc {
    fn waker_refs(&self) -> &AtomicUsize { &self.waker_refs }
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
        let mut base = BaseOwned::new();
        base.slot = stack.cell_slot();
        StacklessOnlyTaskDesc {
            owned: UnsafeCell::new(StacklessOnlyOwned { base, poll_fn: None }),
            is_root: false,
            join_state: AtomicUsize::new(if has_handle { JS_RUNNING } else { JS_DETACHED }),
            waker_refs: AtomicUsize::new(0),
            stack,
        }
    }

    /// Pseudo-descriptor for a worker's scheduler-loop context.
    pub(crate) fn new_root() -> StacklessOnlyTaskDesc {
        StacklessOnlyTaskDesc {
            owned: UnsafeCell::new(StacklessOnlyOwned { base: BaseOwned::new(), poll_fn: None }),
            is_root: true,
            join_state: AtomicUsize::new(JS_DETACHED),
            waker_refs: AtomicUsize::new(0),
            stack: crate::resumable::common::stack::StackMem::None,
        }
    }

    /// Reset a pooled descriptor for reuse (the stack allocation is kept).
    pub(crate) fn reinit(&mut self, has_handle: bool) {
        debug_assert!(!self.is_root);
        let owned = self.owned.get_mut();
        owned.poll_fn = None;
        owned.base.result = None;
        owned.base.tls = None;
        *self.join_state.get_mut() = if has_handle { JS_RUNNING } else { JS_DETACHED };
        *self.waker_refs.get_mut() = 0;
    }
}
