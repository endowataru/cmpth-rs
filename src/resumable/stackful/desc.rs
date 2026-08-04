//! Stackful-only descriptor operations: a real, switchable saved context,
//! and [`StackfulOnlyTaskDesc`] — the concrete descriptor for `UltIdentity`
//! (stackful-only) systems.

use std::cell::UnsafeCell;
use std::sync::atomic::AtomicUsize;

use crate::resumable::common::desc::{DescOwned, HasDescOwned, HasScheduler, RunningTaskToken, SuspendedTaskToken, TaskDescCore, TaskDescAlloc, decode_join_state, JS_DETACHED, JS_RUNNING};
use crate::resumable::common::scheduler::Scheduler;
use crate::resumable::common::system::SchedulerSystem;
use crate::traits::common::JoinState;
use crate::traits::stackful::{SyncJoinState, SyncJoinerTaskDesc};

/// Implemented by a [`TaskDescCore::Owned`] type that can hold a saved-context
/// pointer — either directly ([`StackfulOnlyTaskDesc`]'s
/// `Owned`) or as one variant of a `ctx`/`poll_fn` union
/// ([`DualTaskDesc`](crate::resumable::dual::desc::DualTaskDesc)'s
/// `Owned`, via `TaskDispatch`).
///
/// Deliberately plain fields, not `Cell`: `ctx` carries no ordering of its
/// own, and mutation only ever happens through a token's `&mut Owned`
/// (`DerefMut`), which already proves exclusivity. Its soundness rests
/// entirely on two invariants holding everywhere in the codebase, verified
/// once (2026-07-28) rather than re-proven per call site:
///
/// 1. Every suspend goes through `suspend_shim`/`cond_suspend_shim`
///    (`resumable::stackful::worker`), which write `ctx` (via
///    `RunningTaskToken::publish_saved_context`) *before*
///    running the caller-supplied closure that actually makes the
///    continuation reachable by another thread — this ordering is
///    structural (baked into the shim), not caller discipline.
/// 2. Whatever that closure uses to publish the continuation (a
///    wait-slot's `AtomicPtr`, an MCS queue link, `join_state`,
///    `waker_refs`, a deque push, an external queue) is itself a
///    genuine atomic `Release` write, observed via a genuine `Acquire`
///    on that *same* location by the resuming thread before it ever
///    calls `SuspendedTaskToken::claim_saved_context`/
///    `SuspendedTaskToken::peek_saved_context`. By program
///    order + release/acquire transitivity, that Acquire already makes
///    `ctx`'s plain write visible — the same reason `Mutex<T>`'s
///    guarded `T` needs no atomicity of its own.
///
/// **Any new suspend/resume path or wait-primitive must preserve
/// invariant 2** (publish via a real `Release`, consume via a real
/// `Acquire` on that location, before touching `ctx`) or this needs to
/// go back to being an `AtomicPtr` with its own `Release`/`Acquire`.
/// This exact subsystem has already produced one ARM-only, CI-invisible
/// weak-memory race from getting a nearly identical invariant wrong
/// (a wait-slot published with a plain store racing a `Release`d
/// context save) — don't relax this without the same stress-test rigor
/// that caught it (`taskpolicy -c background`-pinned E-core runs, not
/// just `cargo test`).
pub trait HasCtx {
    fn ctx(&self) -> *mut u8;
    fn set_ctx(&mut self, ptr: *mut u8);

    /// Ensure this `Owned` is configured for real-context-switch dispatch.
    /// Called once by the allocating call site (`spawn`, `fork_parent_first`)
    /// right after allocation, before `init_saved_context`/
    /// `publish_saved_context` ever runs.
    ///
    /// No-op default: only meaningful for an `Owned` type that also
    /// implements [`HasPollFn`](crate::resumable::stackless::desc::HasPollFn)
    /// (i.e. [`DualTaskDesc`](crate::resumable::dual::desc::DualTaskDesc)'s),
    /// which overrides this to commit its `ctx`/`poll_fn` union to the
    /// `Ctx` variant — the shared pool/`alloc_with` machinery that
    /// constructs it also serves `spawn_async`'s non-oversized-future path
    /// and can't tell from inside itself which role a given call is for.
    /// An `Owned` type with only one possible role has nothing to commit to.
    fn commit_as_ctx(&mut self) {}
}

/// Descriptor operations needed only by tasks with a real, switchable
/// execution stack (stackful ULTs). A pure-stackless descriptor type would
/// not implement this — there is no saved context to hand off, since
/// `run_async_poll` never does a context switch.
pub trait StackfulTaskDesc: SyncJoinerTaskDesc + TaskDescCore<Owned: HasCtx + HasScheduler> {}

impl<D: SyncJoinerTaskDesc + TaskDescCore<Owned: HasCtx + HasScheduler>> StackfulTaskDesc for D {}

impl<D: TaskDescCore<Owned: HasCtx>> SuspendedTaskToken<D> {
    /// Claim this task's saved context before switching into it (swap to
    /// null). The caller is expected to `debug_assert` the returned pointer
    /// is non-null (a null result means a double-resume — the exact
    /// diagnostic message differs per call site, so that check stays there).
    pub(crate) fn claim_saved_context(&mut self) -> *mut u8 {
        let ptr = self.ctx();
        self.set_ctx(std::ptr::null_mut());
        ptr
    }

    /// Look at this task's saved context without consuming it — used when
    /// the caller might not actually commit to switching
    /// (`cond_suspend_to_cont`).
    pub(crate) fn peek_saved_context(&self) -> *mut u8 {
        self.ctx()
    }

    /// Initialize the context of a freshly allocated task that has never
    /// been suspended.
    pub(crate) fn init_saved_context(&mut self, ptr: *mut u8) {
        self.set_ctx(ptr);
    }
}

impl<D: TaskDescCore<Owned: HasCtx>> RunningTaskToken<D> {
    /// Publish a just-saved context, making this (about-to-be-suspended)
    /// task resumable. Returns the previous value so the caller can
    /// `debug_assert` it was null (overwriting a live context is a bug).
    /// Called while `self` is still typed `RunningTaskToken` — the switch
    /// shims publish `ctx` before converting to `SuspendedTaskToken`.
    pub(crate) fn publish_saved_context(&mut self, ptr: *mut u8) -> *mut u8 {
        let old = self.ctx();
        self.set_ctx(ptr);
        old
    }

    /// Clear this task's saved context — used by `cond_suspend_shim`'s
    /// commit/cancel cleanup, after the ordering-relevant handoff already
    /// happened via the context switch itself.
    pub(crate) fn clear_saved_context(&mut self) {
        self.set_ctx(std::ptr::null_mut());
    }
}

// ---------------------------------------------------------------------------
// StackfulOnlyTaskDesc — UltIdentity systems (real ULTs, no spawn_async)
// ---------------------------------------------------------------------------

/// Owner-exclusive fields for [`StackfulOnlyTaskDesc`]: [`DescOwned`] plus
/// the real saved-context pointer (no `poll_fn` slot — this flavor never
/// has one).
pub struct StackfulOnlyOwned<S: SchedulerSystem> {
    desc_owned: DescOwned,
    scheduler: *const Scheduler<S>,
    ctx: *mut u8,
}

impl<S: SchedulerSystem> HasDescOwned for StackfulOnlyOwned<S> {
    fn desc_owned(&self) -> &DescOwned { &self.desc_owned }
    fn desc_owned_mut(&mut self) -> &mut DescOwned { &mut self.desc_owned }
}

impl<S: SchedulerSystem> HasScheduler for StackfulOnlyOwned<S> {
    type System = S;
    fn scheduler(&self) -> *const Scheduler<S> { self.scheduler }
    fn set_scheduler(&mut self, scheduler: *const Scheduler<S>) { self.scheduler = scheduler; }
}

impl<S: SchedulerSystem> HasCtx for StackfulOnlyOwned<S> {
    fn ctx(&self) -> *mut u8 { self.ctx }
    fn set_ctx(&mut self, ptr: *mut u8) { self.ctx = ptr; }
}

/// Concrete descriptor for `UltIdentity`-based (stackful-only) systems: a
/// real ULT with no `spawn_async` capability, so no `poll_fn` slot exists
/// at all (contrast [`DualTaskDesc`](crate::resumable::dual::desc::DualTaskDesc),
/// which needs both on the same struct).
pub struct StackfulOnlyTaskDesc<S: SchedulerSystem> {
    owned: UnsafeCell<StackfulOnlyOwned<S>>,
    join_state: AtomicUsize,
    is_root: bool,
    stack: crate::resumable::common::stack::StackMem,
}

unsafe impl<S: SchedulerSystem> Send for StackfulOnlyTaskDesc<S> {}
unsafe impl<S: SchedulerSystem> Sync for StackfulOnlyTaskDesc<S> {}

impl<S: SchedulerSystem> TaskDescCore for StackfulOnlyTaskDesc<S> {
    fn join_state(&self) -> &AtomicUsize { &self.join_state }
    fn is_root(&self) -> bool { self.is_root }
    fn stack_top(&self) -> *mut u8 { self.stack.top() }
    type Owned = StackfulOnlyOwned<S>;
    fn owned_cell(&self) -> &UnsafeCell<StackfulOnlyOwned<S>> { &self.owned }

    /// No async capability at all, so `AsyncWaker`/`AsyncJoiner` can never
    /// actually be published (the only writers,
    /// `WakerTaskDesc::try_register_waker`/`try_register_async_joiner`,
    /// don't exist for this type) — narrow to `SyncJoinState`.
    type JoinOutcome = SyncJoinState<Self>;
    fn decode_join(word: usize) -> SyncJoinState<Self> {
        match decode_join_state::<Self>(word) {
            JoinState::Running => SyncJoinState::Running,
            JoinState::Finished => SyncJoinState::Finished,
            JoinState::Detached => SyncJoinState::Detached,
            JoinState::SyncJoiner(j) => SyncJoinState::SyncJoiner(j),
            JoinState::AsyncWaker(_) | JoinState::AsyncJoiner(_) => {
                unreachable!("cmpth: async join state on a system with no async capability")
            }
        }
    }
}

impl<S: SchedulerSystem> TaskDescAlloc for StackfulOnlyTaskDesc<S> {
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

impl<S: SchedulerSystem> StackfulOnlyTaskDesc<S> {
    /// Construct a descriptor value with a heap stack.
    pub(crate) fn alloc(stack_size: usize, has_handle: bool) -> StackfulOnlyTaskDesc<S> {
        use crate::resumable::common::stack::{HeapStack, StackAlloc as _};
        Self::alloc_with(HeapStack::alloc_stack(stack_size).into(), has_handle)
    }

    /// Construct a descriptor value with a policy-allocated stack.
    pub(crate) fn alloc_with(stack: crate::resumable::common::stack::StackMem, has_handle: bool) -> StackfulOnlyTaskDesc<S> {
        let desc_owned = DescOwned::new();
        StackfulOnlyTaskDesc {
            owned: UnsafeCell::new(StackfulOnlyOwned { desc_owned, scheduler: std::ptr::null(), ctx: std::ptr::null_mut() }),
            is_root: false,
            join_state: AtomicUsize::new(if has_handle { JS_RUNNING } else { JS_DETACHED }),
            stack,
        }
    }

    /// Pseudo-descriptor for a worker's scheduler-loop context.
    pub(crate) fn new_root() -> StackfulOnlyTaskDesc<S> {
        StackfulOnlyTaskDesc {
            owned: UnsafeCell::new(StackfulOnlyOwned { desc_owned: DescOwned::new(), scheduler: std::ptr::null(), ctx: std::ptr::null_mut() }),
            is_root: true,
            join_state: AtomicUsize::new(JS_DETACHED),
            stack: crate::resumable::common::stack::StackMem::None,
        }
    }

    /// Reset a pooled descriptor for reuse (the stack allocation is kept).
    pub(crate) fn reinit(&mut self, has_handle: bool) {
        debug_assert!(!self.is_root);
        let owned = self.owned.get_mut();
        owned.ctx = std::ptr::null_mut();
        owned.desc_owned.tls = None;
        *self.join_state.get_mut() = if has_handle { JS_RUNNING } else { JS_DETACHED };
    }
}
