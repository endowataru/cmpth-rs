//! Dual-only descriptor: [`DualTaskDesc`], the concrete descriptor for
//! systems that need both a real ULT's saved context and a `spawn_async`
//! task's poll_fn on the same struct (a stackful sync joiner and a
//! stackless async waker can race to register on the *same* task
//! regardless of which one it turns out to be).

use std::cell::UnsafeCell;
use std::sync::atomic::AtomicUsize;

use crate::resumable::common::desc::{BaseOwned, HasBaseOwned, TaskDesc, TaskDescAlloc, JS_DETACHED, JS_RUNNING};
use crate::resumable::stackless::desc::{TaskPollFn, WakerTaskDesc};

/// A dual task is never both a real ULT and a `spawn_async` future — this
/// enum makes that exclusivity a type-level fact instead of an implicit
/// "one of two nullable fields" convention.
///
/// Verified zero-cost (2026-07-29, `rustc -O --emit=asm` on AArch64): a
/// *safe* accessor that panics via `unreachable!()` on the wrong variant
/// compiles to a real branch, but `debug_assert!` + `unreachable_unchecked()`
/// (what [`DualOwned`]'s `HasCtx`/`HasPollFn` impls use below) compiles to
/// the exact same code as a direct field access (`add x0, x0, #8; ret`) —
/// no discriminant check survives release codegen. Plain values now (not
/// `Cell`-wrapped): `Owned`'s own mutation is already gated by a token's
/// `&mut self`, so the variant fields need no interior mutability of their
/// own.
enum TaskDispatch<D> {
    Ctx(*mut u8),
    PollFn(Option<TaskPollFn<D>>),
}

/// Owner-exclusive fields for [`DualTaskDesc`]: [`BaseOwned`] plus the
/// `ctx`/`poll_fn` union — a dual task is never both a real ULT and a
/// `spawn_async` future at once, but which one it is isn't known until the
/// allocating call site commits (see [`HasCtx::commit_as_ctx`](crate::resumable::stackful::desc::HasCtx::commit_as_ctx)/
/// [`HasPollFn::commit_as_poll_fn`](crate::resumable::stackless::desc::HasPollFn::commit_as_poll_fn)).
pub struct DualOwned {
    base: BaseOwned,
    dispatch: TaskDispatch<DualTaskDesc>,
}

impl HasBaseOwned for DualOwned {
    fn base(&self) -> &BaseOwned { &self.base }
    fn base_mut(&mut self) -> &mut BaseOwned { &mut self.base }
}

impl crate::resumable::stackful::desc::HasCtx for DualOwned {
    fn ctx(&self) -> *mut u8 {
        match self.dispatch {
            TaskDispatch::Ctx(ctx) => ctx,
            TaskDispatch::PollFn(_) => {
                debug_assert!(false, "ctx() called on a descriptor committed to poll_fn dispatch");
                unsafe { std::hint::unreachable_unchecked() }
            }
        }
    }

    fn set_ctx(&mut self, ptr: *mut u8) {
        match &mut self.dispatch {
            TaskDispatch::Ctx(ctx) => *ctx = ptr,
            TaskDispatch::PollFn(_) => {
                debug_assert!(false, "set_ctx() called on a descriptor committed to poll_fn dispatch");
                unsafe { std::hint::unreachable_unchecked() }
            }
        }
    }

    fn commit_as_ctx(&mut self) {
        self.dispatch = TaskDispatch::Ctx(std::ptr::null_mut());
    }
}

impl crate::resumable::stackless::desc::HasPollFn<DualTaskDesc> for DualOwned {
    fn poll_fn(&self) -> Option<TaskPollFn<DualTaskDesc>> {
        match self.dispatch {
            TaskDispatch::PollFn(poll_fn) => poll_fn,
            TaskDispatch::Ctx(_) => {
                debug_assert!(false, "poll_fn() called on a descriptor committed to ctx dispatch");
                unsafe { std::hint::unreachable_unchecked() }
            }
        }
    }

    fn set_poll_fn(&mut self, f: Option<TaskPollFn<DualTaskDesc>>) {
        match &mut self.dispatch {
            TaskDispatch::PollFn(poll_fn) => *poll_fn = f,
            TaskDispatch::Ctx(_) => {
                debug_assert!(false, "set_poll_fn() called on a descriptor committed to ctx dispatch");
                unsafe { std::hint::unreachable_unchecked() }
            }
        }
    }

    fn commit_as_poll_fn(&mut self) {
        self.dispatch = TaskDispatch::PollFn(None);
    }

    fn is_poll_fn_dispatch(&self) -> bool {
        matches!(self.dispatch, TaskDispatch::PollFn(_))
    }
}

/// The descriptor implementation for dual (stackful + stackless) systems:
/// implements every trait at once, since a stackful sync joiner and a
/// stackless async waker can race to register on the *same* task
/// regardless of which one the task itself turns out to be.
pub struct DualTaskDesc {
    owned: UnsafeCell<DualOwned>,
    join_state: AtomicUsize,
    is_root: bool,
    waker_refs: AtomicUsize,
    stack: crate::resumable::common::stack::StackMem,
}

unsafe impl Send for DualTaskDesc {}
unsafe impl Sync for DualTaskDesc {}

impl TaskDesc for DualTaskDesc {
    fn join_state(&self) -> &AtomicUsize { &self.join_state }
    fn is_root(&self) -> bool { self.is_root }
    fn stack_top(&self) -> *mut u8 { self.stack.top() }
    type Owned = DualOwned;
    fn owned_cell(&self) -> &UnsafeCell<DualOwned> { &self.owned }
}

impl WakerTaskDesc for DualTaskDesc {
    fn waker_refs(&self) -> &AtomicUsize { &self.waker_refs }
}

impl TaskDescAlloc for DualTaskDesc {
    fn alloc_with(stack: crate::resumable::common::stack::StackMem, has_handle: bool) -> Self {
        DualTaskDesc::alloc_with(stack, has_handle)
    }

    fn alloc(stack_size: usize, has_handle: bool) -> Self {
        DualTaskDesc::alloc(stack_size, has_handle)
    }

    fn new_root() -> Self {
        DualTaskDesc::new_root()
    }

    fn reinit(&mut self, has_handle: bool) {
        DualTaskDesc::reinit(self, has_handle)
    }
}

impl DualTaskDesc {
    /// Construct a descriptor value with a heap stack. Used (among other
    /// things) by `spawn_async` (whose "stack" only stores the future — no
    /// code runs on it, so it never needs the arena).
    ///
    /// `dispatch` starts as `Ctx` (an arbitrary placeholder — this
    /// constructor is shared by pooled allocation for *both* `S::Pool` and
    /// `S::AsyncPool`, so it cannot know its eventual role); the allocating
    /// call site is responsible for calling `commit_as_ctx`/
    /// `commit_as_poll_fn` immediately afterward, before anything else
    /// touches the descriptor. See `HasCtx::commit_as_ctx`'s doc comment.
    pub(crate) fn alloc(stack_size: usize, has_handle: bool) -> DualTaskDesc {
        use crate::resumable::common::stack::{HeapStack, StackAlloc as _};
        Self::alloc_with(HeapStack::alloc_stack(stack_size).into(), has_handle)
    }

    /// Construct a descriptor value with a policy-allocated stack.  For
    /// arena stacks, captures the cell slot pointer for use by the switch
    /// shims. See [`DualTaskDesc::alloc`]'s doc comment for the `dispatch`
    /// placeholder-then-commit protocol this also follows.
    pub(crate) fn alloc_with(stack: crate::resumable::common::stack::StackMem, has_handle: bool) -> DualTaskDesc {
        let mut base = BaseOwned::new();
        base.slot = stack.cell_slot();
        DualTaskDesc {
            owned: UnsafeCell::new(DualOwned { base, dispatch: TaskDispatch::Ctx(std::ptr::null_mut()) }),
            is_root: false,
            join_state: AtomicUsize::new(if has_handle { JS_RUNNING } else { JS_DETACHED }),
            waker_refs: AtomicUsize::new(0),
            stack,
        }
    }

    /// Pseudo-descriptor for a worker's scheduler-loop context. Always
    /// `Ctx`: the root represents the OS-thread-level scheduler loop
    /// itself, resumed via a real context switch back into it — never a
    /// `spawn_async` future — so unlike `alloc`/`alloc_with` there is no
    /// per-call-site ambiguity to resolve here.
    pub(crate) fn new_root() -> DualTaskDesc {
        DualTaskDesc {
            owned: UnsafeCell::new(DualOwned { base: BaseOwned::new(), dispatch: TaskDispatch::Ctx(std::ptr::null_mut()) }),
            is_root: true,
            join_state: AtomicUsize::new(JS_DETACHED),
            waker_refs: AtomicUsize::new(0),
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
        let owned = self.owned.get_mut();
        match &mut owned.dispatch {
            TaskDispatch::Ctx(ctx) => *ctx = std::ptr::null_mut(),
            TaskDispatch::PollFn(poll_fn) => *poll_fn = None,
        }
        owned.base.result = None;
        owned.base.tls = None;
        *self.join_state.get_mut() = if has_handle { JS_RUNNING } else { JS_DETACHED };
        *self.waker_refs.get_mut() = 0;
    }
}
