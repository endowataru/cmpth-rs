//! Context-switch policy (the Rust counterpart of ComposableThreads'
//! `context_policy`).
//!
//! All switch functions take a plain function pointer that is executed *after*
//! the stack switch, on the stack of the destination context.  This is the key
//! optimization inherited from ComposableThreads: the code that publishes the
//! suspended continuation (pushing it to a deque, storing it in a waiter list,
//! releasing a lock) runs when the context is already fully saved, so no
//! "saving in progress" handshake (flags, spin loops, post-swap states) is
//! needed anywhere in the scheduler.

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

/// Callback run on the destination stack after `swap_context`/`save_context`.
/// `prev` is the context that was just saved.
pub type SwitchFn = unsafe extern "C" fn(prev: Context, a1: *mut (), a2: *mut ()) -> Transfer;

/// Callback run on the destination stack after `cond_swap_context`.
pub type CondSwitchFn =
    unsafe extern "C" fn(prev: Context, a1: *mut (), a2: *mut ()) -> CondTransfer;

/// Callback run on the destination stack after `restore_context`.
/// There is no `prev`: the calling context is abandoned, not saved.
pub type RestoreFn = unsafe extern "C" fn(a1: *mut (), a2: *mut ()) -> Transfer;

/// Entry point of a context created with `make_context`.  `transfer` is the
/// value returned by the first switcher's callback.
pub type EntryFn = unsafe extern "C" fn(transfer: Transfer, arg: *mut ()) -> !;

/// Swappable context-switch implementation.
///
/// # Safety
/// Implementations must uphold the save/resume contract described above:
/// a context saved by `swap`/`save`/`cond_swap` must be resumable exactly once
/// and must return to its caller with the resumer's `Transfer` value.
pub unsafe trait ContextPolicy: 'static {
    /// Save the current context, switch to `to`, run `func` there.
    ///
    /// # Safety
    /// `to` must be a live, never-yet-resumed context; `a1`/`a2` must satisfy
    /// whatever `func` requires of them.
    unsafe fn swap_context(to: Context, func: SwitchFn, a1: *mut (), a2: *mut ()) -> Transfer;

    /// Save the current context, switch to the fresh stack `new_sp`, run
    /// `func` there.  If `func` returns, the saved context resumes at once.
    ///
    /// # Safety
    /// `new_sp` must be the top of a stack that is unused and large enough
    /// for everything `func` executes.
    unsafe fn save_context(new_sp: *mut u8, func: SwitchFn, a1: *mut (), a2: *mut ())
    -> Transfer;

    /// Like `swap_context`, but `func` may cancel the switch by returning
    /// `flag == 0`, in which case the caller resumes immediately and the
    /// destination context stays saved.
    ///
    /// # Safety
    /// As for [`swap_context`](Self::swap_context); additionally, on the
    /// cancel path `func` must leave the destination context untouched.
    unsafe fn cond_swap_context(
        to: Context,
        func: CondSwitchFn,
        a1: *mut (),
        a2: *mut (),
    ) -> Transfer;

    /// Abandon the current context, switch to `to`, run `func` there.
    ///
    /// # Safety
    /// As for [`swap_context`](Self::swap_context); the current stack is
    /// abandoned without unwinding, so no live destructors may remain on it.
    unsafe fn restore_context(to: Context, func: RestoreFn, a1: *mut (), a2: *mut ()) -> !;

    /// Prepare a context on a fresh stack that enters `entry` when first
    /// switched to.
    ///
    /// # Safety
    /// `stack_top` must be the top of an unused stack that stays alive until
    /// the task completes.
    unsafe fn make_context(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context;
}

// ---------------------------------------------------------------------------
// Native (assembly) implementation
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn cmpth_swap_context(to: Context, func: SwitchFn, a1: *mut (), a2: *mut ()) -> Transfer;
    fn cmpth_save_context(new_sp: *mut u8, func: SwitchFn, a1: *mut (), a2: *mut ()) -> Transfer;
    fn cmpth_cond_swap_context(
        to: Context,
        func: CondSwitchFn,
        a1: *mut (),
        a2: *mut (),
    ) -> Transfer;
    fn cmpth_restore_context(to: Context, func: RestoreFn, a1: *mut (), a2: *mut ()) -> !;
    fn cmpth_make_context(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context;
}

/// Default `ContextPolicy` backed by the hand-written assembly in `asm/`.
pub struct NativeContext;

#[cfg(not(target_arch = "aarch64"))]
unsafe impl ContextPolicy for NativeContext {
    unsafe fn swap_context(to: Context, func: SwitchFn, a1: *mut (), a2: *mut ()) -> Transfer {
        unsafe { cmpth_swap_context(to, func, a1, a2) }
    }

    unsafe fn save_context(
        new_sp: *mut u8,
        func: SwitchFn,
        a1: *mut (),
        a2: *mut (),
    ) -> Transfer {
        unsafe { cmpth_save_context(new_sp, func, a1, a2) }
    }

    unsafe fn cond_swap_context(
        to: Context,
        func: CondSwitchFn,
        a1: *mut (),
        a2: *mut (),
    ) -> Transfer {
        unsafe { cmpth_cond_swap_context(to, func, a1, a2) }
    }

    unsafe fn restore_context(to: Context, func: RestoreFn, a1: *mut (), a2: *mut ()) -> ! {
        unsafe { cmpth_restore_context(to, func, a1, a2) }
    }

    unsafe fn make_context(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context {
        unsafe { cmpth_make_context(stack_top, entry, arg) }
    }
}

// On AArch64 the C ABI makes v8–v15 (lower halves) callee-saved, but the
// switch routines in `asm/aarch64.s` save only the general-purpose set — a
// task suspended with live floating-point state could resume with another
// task's register contents.  Saving v8–v15 unconditionally in the assembly
// would cost 8 extra stores + 8 loads on every switch, even though integer
// code (the common case for a scheduler hot path) has nothing live there.
//
// Instead the routines are invoked through inline-asm stubs that declare
// v8–v15 as clobbered: the *compiler* spills exactly the live ones, which is
// free for integer code and correct for floating-point code.  `clobber_abi`
// covers the ordinary caller-saved set; x19–x28 stay with the callee (the
// assembly saves them, as the C ABI promises).
#[cfg(target_arch = "aarch64")]
macro_rules! call_switch {
    ($sym:ident, $x0:expr, $x1:expr, $x2:expr, $x3:expr) => {{
        let ret: *mut ();
        core::arch::asm!(
            "bl {f}",
            f = sym $sym,
            inout("x0") $x0 => ret,
            in("x1") $x1,
            in("x2") $x2,
            in("x3") $x3,
            lateout("v8") _, lateout("v9") _, lateout("v10") _, lateout("v11") _,
            lateout("v12") _, lateout("v13") _, lateout("v14") _, lateout("v15") _,
            clobber_abi("C"),
        );
        ret
    }};
}

#[cfg(target_arch = "aarch64")]
unsafe impl ContextPolicy for NativeContext {
    #[inline(always)]
    unsafe fn swap_context(to: Context, func: SwitchFn, a1: *mut (), a2: *mut ()) -> Transfer {
        unsafe { Transfer(call_switch!(cmpth_swap_context, to.0, func, a1, a2)) }
    }

    #[inline(always)]
    unsafe fn save_context(
        new_sp: *mut u8,
        func: SwitchFn,
        a1: *mut (),
        a2: *mut (),
    ) -> Transfer {
        unsafe { Transfer(call_switch!(cmpth_save_context, new_sp, func, a1, a2)) }
    }

    #[inline(always)]
    unsafe fn cond_swap_context(
        to: Context,
        func: CondSwitchFn,
        a1: *mut (),
        a2: *mut (),
    ) -> Transfer {
        unsafe { Transfer(call_switch!(cmpth_cond_swap_context, to.0, func, a1, a2)) }
    }

    #[inline(always)]
    unsafe fn restore_context(to: Context, func: RestoreFn, a1: *mut (), a2: *mut ()) -> ! {
        // The current context is abandoned: nothing needs preserving, so a
        // plain tail-jump without clobber bookkeeping is enough.
        unsafe {
            core::arch::asm!(
                "b {f}",
                f = sym cmpth_restore_context,
                in("x0") to.0,
                in("x1") func,
                in("x2") a1,
                in("x3") a2,
                options(noreturn),
            );
        }
    }

    unsafe fn make_context(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context {
        // Ordinary function: no context is switched, the plain call is fine.
        unsafe { cmpth_make_context(stack_top, entry, arg) }
    }
}

