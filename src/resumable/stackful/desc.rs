//! Stackful-only descriptor operations: a real, switchable saved context.

use crate::resumable::common::desc::{RunningTaskToken, SuspendedTaskToken, TaskDesc};

/// Implemented by a [`TaskDesc::Owned`] type that can hold a saved-context
/// pointer — either directly ([`StackfulOnlyTaskDesc`](crate::resumable::common::desc::StackfulOnlyTaskDesc)'s
/// `Owned`) or as one variant of a `ctx`/`poll_fn` union
/// ([`BasicTaskDesc`](crate::resumable::common::desc::BasicTaskDesc)'s
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
    /// (i.e. [`BasicTaskDesc`](crate::resumable::common::desc::BasicTaskDesc)'s),
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
