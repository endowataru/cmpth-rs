//! [`NativeContext`]: the default [`ContextPolicy`] implementation, backed
//! by the hand-written assembly in `asm/`.

use crate::traits::stackful::{CondSwitchFn, Context, ContextPolicy, EntryFn, RestoreFn, SwitchFn, Transfer};

// ---------------------------------------------------------------------------
// Native (assembly) implementation
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "aarch64"))]
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
}

/// Default `ContextPolicy` backed by the hand-written assembly in `asm/`.
pub struct NativeContext;

// ---------------------------------------------------------------------------
// make_context
// ---------------------------------------------------------------------------
//
// Unlike the switch primitives, `make_context` performs no context switch at
// all: it never reads or writes a live CPU register, it only stores pointers
// at fixed offsets of a fresh stack.  So it needs no assembly — plain Rust
// stores build the very same frame the switch routines save and restore, and
// a resuming switch cannot tell the two apart (the frame format is identical
// by construction; see `docs/scoped-ult-promotion.md` §9.14.2).
//
// The one part that *must* be assembly is the entry trampoline, because it is
// reached by `ret`/`br` rather than by a call: the switcher's `func` leaves
// its Transfer in the return-value register while `entry`/`arg` sit in the
// callee-saved registers the frame just restored, which is not a shape any
// Rust function signature can describe.  Bridging that to the ordinary C ABI
// takes three instructions.
//
// Both Mach-O (`_cmpth_*`) and ELF (`cmpth_*`) symbol names are provided, the
// same way `asm/*.s` does it, so no `cfg` on the symbol name is needed.

#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    ".p2align 2",
    ".global _cmpth_entry_trampoline",
    ".global cmpth_entry_trampoline",
    "_cmpth_entry_trampoline:",
    "cmpth_entry_trampoline:",
    // x0 = Transfer (the switcher's func returned it), x19 = entry, x20 = arg.
    "mov x1, x20",
    "blr x19", // entry(transfer, arg) -> !
    "brk #0",  // entry must never return
);

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    ".p2align 4",
    ".global _cmpth_entry_trampoline",
    ".global cmpth_entry_trampoline",
    "_cmpth_entry_trampoline:",
    "cmpth_entry_trampoline:",
    // rax = Transfer, r12 = entry, r13 = arg; rsp is 16-byte aligned.
    "mov rdi, rax",
    "mov rsi, r13",
    "call r12", // entry(transfer, arg) -> !
    "ud2",      // entry must never return
);

unsafe extern "C" {
    fn cmpth_entry_trampoline();
}

/// Build the context frame that [`ContextPolicy::make_context`] promises.
///
/// # Safety
/// `stack_top` must be the top of an unused stack with at least one frame's
/// worth of space below it (96 bytes on AArch64, 56 on x86-64).
#[inline(always)]
unsafe fn native_make_context(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context {
    let aligned = (stack_top as usize & !0xF) as *mut u8;
    let trampoline = cmpth_entry_trampoline as *const () as usize;

    #[cfg(target_arch = "aarch64")]
    {
        // 96-byte frame: x19..x28, x29 (fp), x30 (resume address).
        let ctx = aligned.wrapping_sub(96);
        let w = ctx as *mut usize;
        unsafe {
            w.add(0).write(entry as usize); // x19 = entry
            w.add(1).write(arg as usize); // x20 = arg
            for i in 2..11 {
                w.add(i).write(0); // x21..x28, x29 = 0 (fp terminates backtraces)
            }
            w.add(11).write(trampoline); // x30 = trampoline
        }
        Context(ctx)
    }

    #[cfg(target_arch = "x86_64")]
    {
        // 56-byte frame: rbx, rbp, r12, r13, r14, r15, return address.
        let ctx = aligned.wrapping_sub(56);
        let w = ctx as *mut usize;
        unsafe {
            w.add(0).write(0); // rbx
            w.add(1).write(0); // rbp = 0 (terminates backtraces)
            w.add(2).write(entry as usize); // r12 = entry
            w.add(3).write(arg as usize); // r13 = arg
            w.add(4).write(0); // r14
            w.add(5).write(0); // r15
            w.add(6).write(trampoline); // resume address = trampoline
        }
        Context(ctx)
    }
}

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

    #[inline(always)]
    unsafe fn make_context(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context {
        unsafe { native_make_context(stack_top, entry, arg) }
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
        // Two local labels: `4:` is where `func` returns to (the switch was
        // never committed to a destination context, so the previous one is
        // resumed at once), `3:` is where a genuine resume lands.
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

                "mov  x9,  x0",          // x9  = new stack top
                "mov  x10, x1",          // x10 = func
                "mov  x0,  sp",          // arg0 = prev_ctx
                "mov  x1,  x2",          // arg1 = a1
                "mov  x2,  x3",          // arg2 = a2

                "bic  x9, x9, #15",      // align the new stack top
                "mov  x11, sp",          // keep prev frame for the return path
                "mov  sp, x9",
                "str  x11, [sp, #-16]!", // push prev frame on the new stack
                "adr  x30, 4f",
                "br   x10",              // func(prev_ctx, a1, a2) on the new stack

                "4:",                    // func returned: resume the saved context
                "ldr  x9, [sp]",
                "ldp  x19, x20, [x9,  #0]",
                "ldp  x21, x22, [x9, #16]",
                "ldp  x23, x24, [x9, #32]",
                "ldp  x25, x26, [x9, #48]",
                "ldp  x27, x28, [x9, #64]",
                "ldp  x29, x30, [x9, #80]",
                "add  sp,  x9, #96",
                "ret",                   // -> `3:` (the address stored above)
                "3:",
                inout("x0") new_sp => ret,
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
    unsafe fn cond_swap_context(
        to: Context,
        func: CondSwitchFn,
        a1: *mut (),
        a2: *mut (),
    ) -> Transfer {
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

                // x19/x20 are ours to use now: the previous values are in the
                // frame and both exit paths restore every callee-saved
                // register from a frame.
                "mov  x19, x0",          // x19 = destination context
                "mov  x20, sp",          // x20 = previous frame
                "mov  x10, x1",          // x10 = func
                "mov  x0,  sp",          // arg0 = prev_ctx
                "mov  x1,  x2",          // arg1 = a1
                "mov  x2,  x3",          // arg2 = a2

                "mov  sp,  x19",         // run func on the destination stack,
                                         // below the destination's frame
                "blr  x10",              // (x0, x1) = func(prev_ctx, a1, a2)

                "cbnz x1, 4f",
                "mov  x9, x20",          // cancel: restore the previous context
                "b    5f",
                "4:",
                "mov  x9, x19",          // commit: restore the destination
                "5:",
                "ldp  x19, x20, [x9,  #0]",
                "ldp  x21, x22, [x9, #16]",
                "ldp  x23, x24, [x9, #32]",
                "ldp  x25, x26, [x9, #48]",
                "ldp  x27, x28, [x9, #64]",
                "ldp  x29, x30, [x9, #80]",
                "add  sp,  x9, #96",
                "ret",                   // cancel -> `3:`; commit -> the
                                         // destination's own resume label
                "3:",
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
    unsafe fn restore_context(to: Context, func: RestoreFn, a1: *mut (), a2: *mut ()) -> ! {
        // The current context is abandoned: nothing needs saving, and this
        // block never returns, so no clobber bookkeeping is needed either.
        unsafe {
            core::arch::asm!(
                "mov  x9,  x0",          // x9  = destination context
                "mov  x10, x1",          // x10 = func
                "mov  x0,  x2",          // arg0 = a1
                "mov  x1,  x3",          // arg1 = a2

                "ldp  x19, x20, [x9,  #0]",
                "ldp  x21, x22, [x9, #16]",
                "ldp  x23, x24, [x9, #32]",
                "ldp  x25, x26, [x9, #48]",
                "ldp  x27, x28, [x9, #64]",
                "ldp  x29, x30, [x9, #80]",
                "add  sp,  x9, #96",
                "br   x10",              // func(a1, a2); ret -> destination
                in("x0") to.0,
                in("x1") func,
                in("x2") a1,
                in("x3") a2,
                options(noreturn),
            );
        }
    }

    #[inline(always)]
    unsafe fn make_context(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context {
        unsafe { native_make_context(stack_top, entry, arg) }
    }
}
