//! Stackless-only descriptor operations: the `spawn_async` poll entry
//! point, and [`StacklessOnlyTaskDesc`] — the concrete descriptor for
//! `UltAsyncIdentity` (stackless-only) systems.

use std::cell::UnsafeCell;
use std::sync::atomic::AtomicUsize;
use std::task::Context;

use crate::resumable::common::desc::{BaseOwned, HasBaseOwned, RunningTaskToken, SuspendedTaskToken, TaskDesc, TaskDescAlloc, WakerTaskDesc, JS_DETACHED, JS_RUNNING};

/// Result of driving one `spawn_async` task's poll to completion or a
/// suspend point (named `TaskPollResult`, not `PollResult`, to keep it out
/// of the way of `std::task::Poll` and `std::future::poll_fn` at a glance).
pub enum TaskPollResult<D> {
    /// The future finished; nothing left to do for this task.
    Ready,
    /// The future finished, and its completion claimed exclusive ownership
    /// of a waiting [`JoinState::AsyncJoiner`](crate::resumable::common::desc::JoinState::AsyncJoiner)
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

impl TaskDesc for StacklessOnlyTaskDesc {
    fn join_state(&self) -> &AtomicUsize { &self.join_state }
    fn is_root(&self) -> bool { self.is_root }
    fn stack_top(&self) -> *mut u8 { self.stack.top() }
    type Owned = StacklessOnlyOwned;
    fn owned_cell(&self) -> &UnsafeCell<StacklessOnlyOwned> { &self.owned }
}

impl WakerTaskDesc for StacklessOnlyTaskDesc {
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
