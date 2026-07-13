//! Task descriptors and continuations.
//!
//! A [`SuspendedUlt`] is an owning handle to a suspended task: exactly one
//! continuation exists per suspended task, and consuming it (switching into
//! the context) invalidates it.  This mirrors ComposableThreads'
//! `basic_sct_continuation` / `suspended_thread` ownership model and is what
//! removes the old `ctx_saving` / `TaskState::Suspending` handshake: a
//! continuation only comes into existence *after* the context is fully saved,
//! because it is created by the switch callback running on the next stack.

use std::any::Any;
use std::cell::{Cell, UnsafeCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, AtomicUsize};
use std::task::{Context, Waker};

pub type TaskResult = Result<Box<dyn Any + Send>, Box<dyn Any + Send>>;

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
//   ptr            — a parked sync joiner (`*mut UltDesc`, aligned, > 7)
//   ptr | 1        — a registered async waker (`*mut Waker`, boxed)
// ---------------------------------------------------------------------------
pub(crate) const JS_RUNNING: usize = 0;
pub(crate) const JS_FINISHED: usize = 1;
pub(crate) const JS_DETACHED: usize = 2;
pub(crate) const JS_ASYNC_TAG: usize = 1;

/// Decoded view of a `join_state` word.
pub(crate) enum JoinState {
    Running,
    Finished,
    Detached,
    SyncJoiner(*mut UltDesc),
    AsyncWaker(*mut Waker),
}

pub(crate) fn decode_join_state(v: usize) -> JoinState {
    match v {
        JS_RUNNING => JoinState::Running,
        JS_FINISHED => JoinState::Finished,
        JS_DETACHED => JoinState::Detached,
        v if v & JS_ASYNC_TAG != 0 => JoinState::AsyncWaker((v & !JS_ASYNC_TAG) as *mut Waker),
        v => JoinState::SyncJoiner(v as *mut UltDesc),
    }
}

/// Per-task descriptor.  Intentionally not generic: every scheduler level
/// uses the same descriptor layout.
///
/// `repr(C)`: the fields touched by every spawn/exit/join round-trip
/// (`ctx`, `join_state`, `worker`, `slot`, `poll_fn`, flags) are laid out
/// first so they share one cache line.
#[repr(C)]
pub struct UltDesc {
    // --- Hot: touched on every spawn/exit/join ----------------------------

    /// Saved context pointer; null while the task is running.
    ///
    /// Written with `Release` by the context-switch shim; claimed with
    /// `Acquire` or `AcqRel` by resumer or waker.
    pub(crate) ctx: AtomicPtr<u8>,

    /// The join-protocol state word (see the `JS_*` encoding above).
    ///
    /// The exiting task publishes `FINISHED` with `Release` *after* writing
    /// the result; a joiner reading `FINISHED` with `Acquire` may take the
    /// result and free the descriptor immediately — the exit path never
    /// touches the descriptor after that store.
    pub(crate) join_state: AtomicUsize,

    /// Type-erased `*const UltWorker<S>`: the worker that most recently
    /// switched into this task, written by the switch shims alongside
    /// `cur_task`.  A task cannot migrate between its last resume and its
    /// next suspension, so the exit path reads this instead of doing a TLS
    /// lookup.  Only valid while the task is running.
    pub(crate) worker: Cell<*const ()>,

    /// Points at the arena cell's `[worker, system_id]` slot for arena
    /// stacks, or `None` for heap/root stacks.  The switch shims write the
    /// resuming worker pointer here when present.
    pub(crate) slot: Option<*mut crate::ult::stack::CellSlot>,

    /// Non-null for async tasks spawned via `spawn_async`; null for sync ULTs.
    ///
    /// When `Some`, `Worker::execute` calls this instead of doing a context
    /// switch.  The function polls the Future stored in the task's "stack"
    /// buffer and returns `true` when the Future returned `Poll::Ready`
    /// (caller must not touch `desc` after that).
    pub(crate) poll_fn: Option<for<'cx> unsafe fn(*mut UltDesc, &mut Context<'cx>) -> bool>,

    /// True for the pseudo-descriptor representing a worker's scheduler-loop
    /// context (the "root continuation").
    pub(crate) is_root: bool,

    // --- Warm ---------------------------------------------------------------

    /// Written by the task itself before exiting; read by the joiner after
    /// `FINISHED` is observed.  (Root tasks only; spawned tasks put the
    /// result on their own stack.)
    pub(crate) result: UnsafeCell<Option<TaskResult>>,

    // --- Async waker (block_on) ------------------------------------------

    /// Encodes PRIVATE/SHARED mode, ref count, and POLLING/PARKED/NOTIFIED/
    /// IDLE state.  See the `waker_refs` constants at the top of this file.
    /// Zero (IDLE) when no `block_on` call is active on this task.
    pub(crate) waker_refs: AtomicUsize,
    /// Type-erased `*const Scheduler<S>`.  Set at task-creation time so that
    /// `wake()` called from an external OS thread can reach the scheduler's
    /// `ExternalQueue` without going through worker TLS.  Null for root
    /// pseudo-descriptors.
    pub(crate) scheduler: *const (),

    // --- Pool metadata ---------------------------------------------------

    /// Intrusive linked-list pointer used when this descriptor sits in the
    /// task pool.  Undefined while the task is running.
    pub(crate) pool_next: *mut UltDesc,
    /// Index of the worker that allocated this descriptor.  Used by
    /// [`ReturnPool`](crate::ult::pool::ReturnPool) to route deallocation
    /// back to the home worker.
    pub(crate) alloc_wk: usize,

    // --- ULT-local storage -----------------------------------------------

    /// Used by nested schedulers for their per-worker pointer (`UltTls`).
    /// Only touched by the OS thread currently running this task.
    pub(crate) tls: UnsafeCell<Option<HashMap<usize, *mut ()>>>,

    // --- Stack -----------------------------------------------------------

    /// Stack allocation (`StackMem::None` for root pseudo-descriptors).
    stack: crate::ult::stack::StackMem,
}

unsafe impl Send for UltDesc {}
unsafe impl Sync for UltDesc {}

impl UltDesc {
    /// Allocate a descriptor with a heap stack.  Freed with
    /// [`UltDesc::free`].  Used by `spawn_async` (whose "stack" only stores
    /// the future — no code runs on it, so it never needs the arena).
    pub(crate) fn alloc(stack_size: usize, has_handle: bool) -> *mut UltDesc {
        use crate::ult::stack::{HeapStack, StackAlloc as _};
        Self::alloc_with(HeapStack::alloc_stack(stack_size).into(), has_handle)
    }

    /// Allocate a descriptor with a policy-allocated stack.  For arena
    /// stacks, captures the cell slot pointer for use by the switch shims.
    pub(crate) fn alloc_with(stack: crate::ult::stack::StackMem, has_handle: bool) -> *mut UltDesc {
        // Compute slot before moving `stack` into the Box.
        let slot = crate::ult::stack::cell_slot(&stack);
        Box::into_raw(Box::new(UltDesc {
            ctx: AtomicPtr::new(std::ptr::null_mut()),
            is_root: false,
            join_state: AtomicUsize::new(if has_handle { JS_RUNNING } else { JS_DETACHED }),
            result: UnsafeCell::new(None),
            waker_refs: AtomicUsize::new(0),
            scheduler: std::ptr::null(),
            worker: Cell::new(std::ptr::null()),
            slot,
            poll_fn: None,
            pool_next: std::ptr::null_mut(),
            alloc_wk: 0,
            tls: UnsafeCell::new(None),
            stack,
        }))
    }

    /// Pseudo-descriptor for a worker's scheduler-loop context.
    pub(crate) fn new_root() -> UltDesc {
        UltDesc {
            ctx: AtomicPtr::new(std::ptr::null_mut()),
            is_root: true,
            join_state: AtomicUsize::new(JS_DETACHED),
            result: UnsafeCell::new(None),
            waker_refs: AtomicUsize::new(0),
            scheduler: std::ptr::null(),
            worker: Cell::new(std::ptr::null()),
            slot: None,
            poll_fn: None,
            pool_next: std::ptr::null_mut(),
            alloc_wk: 0,
            tls: UnsafeCell::new(None),
            stack: crate::ult::stack::StackMem::None,
        }
    }

    pub(crate) fn stack_top(&self) -> *mut u8 {
        self.stack.top()
    }

    /// # Safety
    /// Must be called exactly once, after no other references exist.
    pub(crate) unsafe fn free(ptr: *mut UltDesc) {
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
        self.poll_fn = None;
    }
}

/// Owning handle to a suspended task.  Not `Clone`, not `Drop`: ownership is
/// linear and consuming the continuation (resuming it or storing it in a
/// waiter slot) is explicit.
pub struct SuspendedUlt(pub(crate) *mut UltDesc);

unsafe impl Send for SuspendedUlt {}

impl SuspendedUlt {
    pub(crate) fn desc(&self) -> *mut UltDesc {
        self.0
    }

    pub(crate) fn is_root(&self) -> bool {
        unsafe { (*self.0).is_root }
    }

    pub(crate) fn into_raw(self) -> *mut UltDesc {
        self.0
    }
}
