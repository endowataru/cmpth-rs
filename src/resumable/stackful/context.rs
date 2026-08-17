//! [`NativeContext`]: the default [`ContextPolicy`] implementation, backed
//! by the hand-written assembly in `asm/`.

use crate::traits::stackful::{CondSwitchFn, Context, ContextPolicy, EntryFn, RestoreFn, SwitchFn, Transfer};

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
        // Fully inlined: the return address pushed into the saved frame is
        // the local label `3:` at the end of *this* block, so a resume lands
        // straight back in the caller instead of in a separate symbol.  Only
        // numeric local labels may be used — the compiler is free to
        // duplicate an `asm!` block, and a named label would then be a
        // duplicate symbol.
        let ret: *mut ();
        unsafe {
            core::arch::asm!(
                "sub  sp, sp, #96",
                "stp  x19, x20, [sp,  #0]",
                "stp  x21, x22, [sp, #16]",
                "stp  x23, x24, [sp, #32]",
                "stp  x25, x26, [sp, #48]",
                "stp  x27, x28, [sp, #64]",
                "adr  x30, 3f",          // resume address = end of this block
                "stp  x29, x30, [sp, #80]",

                "mov  x9,  x0",          // x9  = destination context
                "mov  x10, x1",          // x10 = func
                "mov  x0,  sp",          // arg0 = prev_ctx (our saved frame)
                "mov  x1,  x2",          // arg1 = a1
                "mov  x2,  x3",          // arg2 = a2

                "ldp  x19, x20, [x9,  #0]",
                "ldp  x21, x22, [x9, #16]",
                "ldp  x23, x24, [x9, #32]",
                "ldp  x25, x26, [x9, #48]",
                "ldp  x27, x28, [x9, #64]",
                "ldp  x29, x30, [x9, #80]",
                "add  sp,  x9, #96",     // pop the destination frame
                "br   x10",              // func(prev_ctx, a1, a2); ret -> destination
                "3:",                    // resumed: x0 = the resumer's Transfer
                inout("x0") to.0 => ret,
                inout("x1") func => _,
                inout("x2") a1 => _,
                inout("x3") a2 => _,
                lateout("v8") _, lateout("v9") _, lateout("v10") _, lateout("v11") _,
                lateout("v12") _, lateout("v13") _, lateout("v14") _, lateout("v15") _,
                clobber_abi("C"),
            );
        }
        Transfer(ret)
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
