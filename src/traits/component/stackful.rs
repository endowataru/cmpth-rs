use std::task::{Context as TaskContext, RawWaker, RawWakerVTable, Waker};

use crate::traits::component::task::TaskDesc;
use crate::traits::component::wait::Resumable;

// ---------------------------------------------------------------------------
// HandoffTaskDesc
// ---------------------------------------------------------------------------

/// Descriptor operations needed by a blocking `.join()`: registering the
/// parked joiner, and taking a waiter the exiting task may switch straight
/// into. Bodyless — pure behavior, same spirit as [`TaskDesc`]. Implement
/// directly for a custom representation, or implement
/// [`TaskDescCore`](crate::resumable::common::desc::TaskDescCore) instead to
/// get this crate's own word-based algorithm for free via a blanket impl.
///
/// Lives here rather than on the base `TaskDesc` specifically so a
/// descriptor with no stackful capability at all never gets it — the only
/// callers in this crate are `JoinHandle::join`'s slow path and the exit
/// path's fast direct-handoff check, both of which require `S::Desc:
/// StackfulTaskDesc: HandoffTaskDesc` — mirrors why
/// `try_register_async_joiner`/`try_register_waker` live on
/// [`WakerTaskDesc`](crate::traits::component::stackless::WakerTaskDesc) instead of
/// `TaskDesc`.
pub trait HandoffTaskDesc: TaskDesc {
    /// Take a waiter this task's exit may switch straight into instead of
    /// going through the scheduler — `Some` means the caller should exit
    /// via a direct context-switch handoff (e.g.
    /// `StackfulWorker::exit_to_cont`) into the returned continuation.
    /// Stable once observed (a parked joiner cannot act until resumed), so
    /// this is safe to read before a context switch and act on afterward.
    fn try_take_handoff_target(&self) -> Option<Self::Suspended>;

    /// Try to register `joiner` (a parked sync joiner's continuation) as
    /// this task's waiter. `Err` means the task turned out to already be
    /// finished — hands `joiner` straight back so it cannot leak; the
    /// caller should cancel its own suspension and proceed immediately.
    /// `Ok` commits `joiner`.
    fn try_register_joiner(&self, joiner: Self::Suspended) -> Result<(), Self::Suspended>;
}

/// Drives a single `block_on` invocation.
///
/// `Poller` is to `block_on` what a wait-slot (`StackfulResumable`) is to `wait_with`: a thin
/// type that encapsulates the system-specific park/wake mechanism, leaving the
/// poll loop itself as a generic default on [`SpawnableStackfulTaskSystem`](crate::traits::system::stackful::SpawnableStackfulTaskSystem).
///
/// Implementations are always stack-local inside `block_on`.  They are `!Send`
/// by convention — bound to the same ULT, not to a specific OS thread.
///
/// In cmpth, `!Send` because of a **data-race** hazard (`Rc`, a non-atomic
/// refcount, anything whose invariant is "no two threads touch this
/// concurrently") means "bound to the same ULT", not "bound to the same OS
/// thread": work-stealing moves the entire ULT stack atomically, and the
/// deque's own atomics supply the happens-before a migration needs, so such
/// a value is safe across `yield_now` even when the ULT migrates to a
/// different OS thread.
///
/// That does **not** extend to a value that's `!Send` because its invariant
/// is tied to *OS-thread identity* itself — the canonical case is
/// [`std::sync::MutexGuard`] (some platforms require the same OS thread
/// that locked a mutex to be the one that unlocks it). Migrating a ULT
/// holding one of those across a suspension point is unsound: the unlock
/// may run on a different OS thread than the lock did. This is a user
/// obligation this crate cannot check for you, in the same spirit as the
/// TLS-caching hazard documented on [`crate::traits::common::TlsSlot`] —
/// **do not hold a `std::sync::MutexGuard` (or anything else OS-thread-
/// bound) across a suspension point; use this crate's own ULT `Mutex`
/// instead**, whose guard has no such requirement. See also
/// [`StackfulBuilder::init`](crate::traits::system::stackful::StackfulBuilder::init)'s
/// doc comment, which restates this for `main`'s own stack once a
/// standalone initializer is in play.
///
/// [`Drop`] performs cleanup (e.g. resetting `waker_refs` to `IDLE`).
pub trait Poller {
    /// Initialise for the current thread/ULT.
    fn new() -> Self;

    /// Return a [`TaskContext`] whose [`Waker`] resumes the current thread/ULT.
    fn context<'a>(&'a self) -> TaskContext<'a>;

    /// Suspend until the waker fires.
    ///
    /// For ULT systems this uses `cond_suspend_to_sched` and handles the race
    /// where `wake()` fires between `poll()` returning `Pending` and the actual
    /// suspension (NOTIFIED → re-poll without parking).
    fn wait(&self);
}

/// A waker whose `wake()` is a no-op.  Used for busy-polling fallbacks where
/// the poll loop drives re-polling itself (via `yield_now`). Shared by every
/// [`Poller`] implementation that busy-polls (`OsPoller`, `UltPoller`'s
/// fallback).
pub(crate) fn noop_waker() -> Waker {
    static VTABLE: RawWakerVTable =
        RawWakerVTable::new(|p| RawWaker::new(p, &VTABLE), |_| {}, |_| {}, |_| {});
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

/// Stackful (real-context-switch) flavor of parking. These do a real
/// context switch and must only be called from a genuine ULT stack —
/// checked dynamically via `cur_task.is_root` (see
/// `docs/sync-async-unification.md`), not via an explicit capability token.
pub trait StackfulResumable<S>: Resumable<S> {
    /// Suspend the current ULT into this slot. `f` runs after the context
    /// is fully saved (release any spinlock protecting this slot inside
    /// it).
    fn wait_with<F: FnOnce()>(&self, f: F);

    /// Like [`wait_with`](Self::wait_with), but `f` may cancel the
    /// suspension by returning `false`.
    fn wait_with_cond<F: FnOnce() -> bool>(&self, f: F);

    /// Switch directly to the parked continuation, pushing the caller's own
    /// continuation to the local deque. If the slot didn't hold a real
    /// continuation — only possible when `Self` also admits async waiters
    /// (e.g. `DualResumable`) — falls back to waking it the
    /// [`Resumable::notify`] way internally, so callers never need to
    /// branch on whether a real switch happened.
    fn enter(&self);

    /// Symmetric handoff: park the current ULT here and switch to `next`.
    /// Same async-target fallback as [`enter`](Self::enter).
    fn swap(&self, next: &Self);
}

// ---------------------------------------------------------------------------
// ContextPolicy
//
// All switch functions take a plain function pointer that is executed *after*
// the stack switch, on the stack of the destination context.  This is the key
// optimization inherited from ComposableThreads: the code that publishes the
// suspended continuation (pushing it to a deque, storing it in a waiter list,
// releasing a lock) runs when the context is already fully saved, so no
// "saving in progress" handshake (flags, spin loops, post-swap states) is
// needed anywhere in the scheduler.
// ---------------------------------------------------------------------------

/// Pointer to a saved context frame, located on the suspended thread's stack.
///
/// A null context means "not saved" — a context is only valid between the
/// switch that saved it and the switch that resumes it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Context(pub *mut u8);

impl Context {
    pub const NULL: Context = Context(std::ptr::null_mut());

    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
}

/// Value forwarded from the switch callback to the resumed context.
/// By convention cmpth passes the current `Worker` pointer here, so the
/// resumed side always knows which worker it woke up on.
#[repr(C)]
pub struct Transfer(pub *mut ());

/// Return value of a conditional-switch callback: `flag != 0` commits the
/// switch, `flag == 0` cancels it and resumes the caller immediately.
#[repr(C)]
pub struct CondTransfer {
    pub value: *mut (),
    pub flag: isize,
}

/// Callback run on the destination stack after `swap_context`/`save_context`,
/// named via a type rather than passed as a runtime function-pointer value.
///
/// Implementations must be pooled onto a zero-sized marker type (e.g.
/// `struct SuspendShim<S, F>(PhantomData<(S, F)>)`); one exists per distinct
/// callback. Passing the callback as a *type* parameter to
/// [`ContextPolicy::swap_context`]/[`save_context`](ContextPolicy::save_context)
/// rather than as a value lets the implementation reference `Self::call` with
/// `asm!`'s `sym` operand — a direct branch resolved at compile time — instead
/// of loading a function-pointer value into a register for an indirect branch.
/// `sym` requires a path to an item, not an arbitrary (even `const`) value of
/// function-pointer type, which is why this has to be a trait rather than a
/// `const F: SwitchFn` generic.
pub trait SwitchFnLike {
    /// `prev` is the context that was just saved.
    unsafe extern "C" fn call(prev: Context, a1: *mut (), a2: *mut ()) -> Transfer;
}

/// Callback run on the destination stack after `cond_swap_context`, named via
/// a type for the same reason as [`SwitchFnLike`].
pub trait CondSwitchFnLike {
    unsafe extern "C" fn call(prev: Context, a1: *mut (), a2: *mut ()) -> CondTransfer;
}

/// Callback run on the destination stack after `restore_context`, named via a
/// type for the same reason as [`SwitchFnLike`]. There is no `prev`: the
/// calling context is abandoned, not saved.
pub trait RestoreFnLike {
    unsafe extern "C" fn call(a1: *mut (), a2: *mut ()) -> Transfer;
}

/// Entry point of a context created with `make_context`.  `transfer` is the
/// value returned by the first switcher's callback.
///
/// Unlike the switch callbacks above, this stays a plain function-pointer
/// value: it is read out of the ctx frame by the (shared, non-generic)
/// entry trampoline on the *first* resume of a freshly `make_context`-built
/// task — a cold, once-per-task path, not a hot per-switch one — and the
/// trampoline has no type parameter to hang a `SwitchFnLike`-style `sym`
/// reference on.
pub type EntryFn = unsafe extern "C" fn(transfer: Transfer, arg: *mut ()) -> !;

/// Swappable context-switch implementation (the Rust counterpart of
/// ComposableThreads' `context_policy`).
///
/// # Safety
/// Implementations must uphold the save/resume contract described above:
/// a context saved by `swap`/`save`/`cond_swap` must be resumable exactly once
/// and must return to its caller with the resumer's `Transfer` value.
pub unsafe trait ContextPolicy: 'static {
    /// Save the current context, switch to `to`, run `F::call` there.
    ///
    /// # Safety
    /// `to` must be a live, never-yet-resumed context; `a1`/`a2` must satisfy
    /// whatever `func` requires of them.
    unsafe fn swap_context<F: SwitchFnLike>(to: Context, a1: *mut (), a2: *mut ()) -> Transfer;

    /// Save the current context, switch to the fresh stack `new_sp`, run
    /// `F::call` there.  If it returns, the saved context resumes at once.
    ///
    /// # Safety
    /// `new_sp` must be the top of a stack that is unused and large enough
    /// for everything `F::call` executes.
    unsafe fn save_context<F: SwitchFnLike>(new_sp: *mut u8, a1: *mut (), a2: *mut ())
    -> Transfer;

    /// Like `swap_context`, but `F::call` may cancel the switch by returning
    /// `flag == 0`, in which case the caller resumes immediately and the
    /// destination context stays saved.
    ///
    /// # Safety
    /// As for [`swap_context`](Self::swap_context); additionally, on the
    /// cancel path `F::call` must leave the destination context untouched.
    unsafe fn cond_swap_context<F: CondSwitchFnLike>(
        to: Context,
        a1: *mut (),
        a2: *mut (),
    ) -> Transfer;

    /// Abandon the current context, switch to `to`, run `F::call` there.
    ///
    /// # Safety
    /// As for [`swap_context`](Self::swap_context); the current stack is
    /// abandoned without unwinding, so no live destructors may remain on it.
    unsafe fn restore_context<F: RestoreFnLike>(to: Context, a1: *mut (), a2: *mut ()) -> !;

    /// Prepare a context on a fresh stack that enters `entry` when first
    /// switched to.
    ///
    /// # Safety
    /// `stack_top` must be the top of an unused stack that stays alive until
    /// the task completes.
    unsafe fn make_context(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context;
}
