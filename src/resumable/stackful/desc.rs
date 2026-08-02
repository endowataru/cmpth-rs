//! Stackful-only descriptor operations: a real, switchable saved context,
//! and [`StackfulOnlyTaskDesc`] — the concrete descriptor for `UltIdentity`
//! (stackful-only) systems.

use std::cell::UnsafeCell;
use std::sync::atomic::AtomicUsize;

use crate::resumable::common::desc::{BaseOwned, HasBaseOwned, RunningTaskToken, SuspendedTaskToken, TaskDesc, TaskDescCore, TaskDescAlloc, JS_DETACHED, JS_RUNNING};

/// Implemented by a [`TaskDesc::Owned`] type that can hold a saved-context
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
pub trait StackfulTaskDesc: TaskDesc<Owned: HasCtx> {}

impl<D: TaskDesc<Owned: HasCtx>> StackfulTaskDesc for D {}

impl<D: TaskDesc<Owned: HasCtx>> SuspendedTaskToken<D> {
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

impl<D: TaskDesc<Owned: HasCtx>> RunningTaskToken<D> {
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

/// Owner-exclusive fields for [`StackfulOnlyTaskDesc`]: [`BaseOwned`] plus
/// the real saved-context pointer (no `poll_fn` slot — this flavor never
/// has one).
pub struct StackfulOnlyOwned {
    base: BaseOwned,
    ctx: *mut u8,
}

impl HasBaseOwned for StackfulOnlyOwned {
    fn base(&self) -> &BaseOwned { &self.base }
    fn base_mut(&mut self) -> &mut BaseOwned { &mut self.base }
}

impl HasCtx for StackfulOnlyOwned {
    fn ctx(&self) -> *mut u8 { self.ctx }
    fn set_ctx(&mut self, ptr: *mut u8) { self.ctx = ptr; }
}

/// Concrete descriptor for `UltIdentity`-based (stackful-only) systems: a
/// real ULT with no `spawn_async` capability, so no `poll_fn` slot exists
/// at all (contrast [`DualTaskDesc`](crate::resumable::dual::desc::DualTaskDesc),
/// which needs both on the same struct).
pub struct StackfulOnlyTaskDesc {
    owned: UnsafeCell<StackfulOnlyOwned>,
    join_state: AtomicUsize,
    is_root: bool,
    stack: crate::resumable::common::stack::StackMem,
}

unsafe impl Send for StackfulOnlyTaskDesc {}
unsafe impl Sync for StackfulOnlyTaskDesc {}

impl TaskDescCore for StackfulOnlyTaskDesc {
    fn join_state(&self) -> &AtomicUsize { &self.join_state }
    fn is_root(&self) -> bool { self.is_root }
    fn stack_top(&self) -> *mut u8 { self.stack.top() }
    type Owned = StackfulOnlyOwned;
    fn owned_cell(&self) -> &UnsafeCell<StackfulOnlyOwned> { &self.owned }
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
        let mut base = BaseOwned::new();
        base.slot = stack.cell_slot();
        StackfulOnlyTaskDesc {
            owned: UnsafeCell::new(StackfulOnlyOwned { base, ctx: std::ptr::null_mut() }),
            is_root: false,
            join_state: AtomicUsize::new(if has_handle { JS_RUNNING } else { JS_DETACHED }),
            stack,
        }
    }

    /// Pseudo-descriptor for a worker's scheduler-loop context.
    pub(crate) fn new_root() -> StackfulOnlyTaskDesc {
        StackfulOnlyTaskDesc {
            owned: UnsafeCell::new(StackfulOnlyOwned { base: BaseOwned::new(), ctx: std::ptr::null_mut() }),
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
        owned.base.result = None;
        owned.base.tls = None;
        *self.join_state.get_mut() = if has_handle { JS_RUNNING } else { JS_DETACHED };
    }
}
