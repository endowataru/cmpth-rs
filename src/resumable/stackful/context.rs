//! [`NativeContext`]: the default [`ContextPolicy`] implementation.
//!
//! The switch primitives are hand-written assembly, but they are *inlined*
//! into their call sites with `asm!` rather than called out of line: the
//! return address stored in a saved frame is a label inside the caller, so
//! resuming a context lands straight back in the caller's own code.
//!
//! Inlining does not, by itself, reduce what the caller spills around a
//! switch: the frame format is fixed (every switch must produce a frame any
//! other switch can resume), so the callee-saved set is still saved
//! unconditionally, and declaring the caller-saved set as clobbers achieves
//! the same spill decisions an ordinary call already did.  What it does
//! remove is the call/return pair, the argument shuffling across the ABI
//! boundary, and — on AArch64 — the need for a separate stub whose only job
//! was to carry the `v8`–`v15` clobber list.
//!
//! `make_context` needs no assembly at all (it switches nothing, it only
//! writes a frame), and the only symbol left is the entry trampoline, which
//! has to be assembly because it is entered by `ret` rather than by a call.

use crate::traits::stackful::{CondSwitchFn, Context, ContextPolicy, EntryFn, RestoreFn, SwitchFn, Transfer};

/// Default `ContextPolicy`, implemented with inline assembly.
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
// Both the Mach-O (`_cmpth_*`) and the ELF (`cmpth_*`) spelling are defined,
// so no `cfg` is needed to pick the right one for the platform.

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

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
compile_error!("cmpth: no ContextPolicy implementation for this architecture");

// x86-64 (System V).  Same shape as the AArch64 implementation below; the two
// differences worth calling out:
//
// * There is no `call` any more, so the return address a `call` used to push
//   is pushed explicitly: 7 pushes from a 16-byte-aligned rsp put ctx at
//   8 (mod 16), the alignment the rest of the protocol assumes.  Getting
//   this right matters because a resuming switch must not be able to tell
//   frames of different origins apart.
// * No vector register needs declaring: every xmm/ymm/zmm is caller-saved
//   under System V, so `clobber_abi("C")` already covers them (LLVM handles
//   the aliasing of the wider names by itself).  The `v8`–`v15` `lateout`
//   list in the AArch64 impl exists only because those are callee-saved
//   there.
//
// Windows x64 is *not* covered by this: xmm6–xmm15 are callee-saved in its
// ABI, so it would need the AArch64 treatment.
#[cfg(target_arch = "x86_64")]
unsafe impl ContextPolicy for NativeContext {
    #[inline(always)]
    unsafe fn swap_context(to: Context, func: SwitchFn, a1: *mut (), a2: *mut ()) -> Transfer {
        let ret: *mut ();
        unsafe {
            core::arch::asm!(
                "lea  r11, [rip + 3f]",  // resume address = end of this block
                "push r11",              // ...where `call` would have pushed it
                "push r15",
                "push r14",
                "push r13",
                "push r12",
                "push rbp",
                "push rbx",
                // rsp = prev_ctx
                "mov  r8, rdi",          // r8 = destination context
                "mov  r9, rsi",          // r9 = func
                "mov  rdi, rsp",         // arg0 = prev_ctx
                "mov  rsi, rdx",         // arg1 = a1
                "mov  rdx, rcx",         // arg2 = a2

                "mov  rsp, r8",          // switch to the destination
                "pop  rbx",
                "pop  rbp",
                "pop  r12",
                "pop  r13",
                "pop  r14",
                "pop  r15",
                "jmp  r9",               // func(prev_ctx, a1, a2); ret -> destination
                "3:",                    // resumed: rax = the resumer's Transfer
                inout("rdi") to.0 => _,
                inout("rsi") func => _,
                inout("rdx") a1 => _,
                inout("rcx") a2 => _,
                lateout("rax") ret,
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
        let ret: *mut ();
        unsafe {
            core::arch::asm!(
                "lea  r11, [rip + 3f]",
                "push r11",
                "push r15",
                "push r14",
                "push r13",
                "push r12",
                "push rbp",
                "push rbx",
                // rsp = prev_ctx
                "mov  r8, rdi",          // r8  = new stack top
                "mov  r9, rsi",          // r9  = func
                "mov  r10, rsp",         // r10 = prev_ctx
                "mov  rdi, rsp",         // arg0 = prev_ctx
                "mov  rsi, rdx",         // arg1 = a1
                "mov  rdx, rcx",         // arg2 = a2

                "and  r8, -16",          // align the new stack top
                "mov  [r8 - 8], r10",    // prev_ctx, for the return path below
                "lea  r11, [rip + 4f]",
                "mov  [r8 - 24], r11",   // func's return address
                "lea  rsp, [r8 - 24]",   // == 8 (mod 16): func enters as if called
                "jmp  r9",               // func(prev_ctx, a1, a2) on the new stack

                "4:",                    // func returned: resume the saved context
                "mov  r9, [rsp + 8]",    // r9 = prev_ctx
                "mov  rsp, r9",
                "pop  rbx",
                "pop  rbp",
                "pop  r12",
                "pop  r13",
                "pop  r14",
                "pop  r15",
                "ret",                   // -> `3:` (the address pushed above)
                "3:",
                inout("rdi") new_sp => _,
                inout("rsi") func => _,
                inout("rdx") a1 => _,
                inout("rcx") a2 => _,
                lateout("rax") ret,
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
                "lea  r11, [rip + 3f]",
                "push r11",
                "push r15",
                "push r14",
                "push r13",
                "push r12",
                "push rbp",
                "push rbx",
                // rsp = prev_ctx.  r12/r13/r14 are ours to use now: the
                // previous values are in the frame and both exit paths
                // restore every callee-saved register from a frame.
                "mov  r12, rdi",         // r12 = destination context
                "mov  r13, rsp",         // r13 = prev_ctx
                "mov  r14, rsi",         // r14 = func
                "mov  rdi, rsp",         // arg0 = prev_ctx
                "mov  rsi, rdx",         // arg1 = a1
                "mov  rdx, rcx",         // arg2 = a2

                // Run func on the destination stack, below its intact frame.
                // ctx == 8 (mod 16), so sub 8 aligns rsp for the call.
                "mov  rsp, r12",
                "sub  rsp, 8",
                "call r14",              // rax = value, rdx = flag
                "add  rsp, 8",

                "test rdx, rdx",
                "jnz  4f",
                "mov  rsp, r13",         // cancel: restore the previous context
                "jmp  5f",
                "4:",
                "mov  rsp, r12",         // commit: restore the destination
                "5:",
                "pop  rbx",
                "pop  rbp",
                "pop  r12",
                "pop  r13",
                "pop  r14",
                "pop  r15",
                "ret",                   // cancel -> `3:`; commit -> the
                                         // destination's own resume label
                "3:",
                inout("rdi") to.0 => _,
                inout("rsi") func => _,
                inout("rdx") a1 => _,
                inout("rcx") a2 => _,
                lateout("rax") ret,
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
                "mov  r8, rdi",          // r8 = destination context
                "mov  r9, rsi",          // r9 = func
                "mov  rdi, rdx",         // arg0 = a1
                "mov  rsi, rcx",         // arg1 = a2

                "mov  rsp, r8",
                "pop  rbx",
                "pop  rbp",
                "pop  r12",
                "pop  r13",
                "pop  r14",
                "pop  r15",
                "jmp  r9",               // func(a1, a2); ret -> destination
                in("rdi") to.0,
                in("rsi") func,
                in("rdx") a1,
                in("rcx") a2,
                options(noreturn),
            );
        }
    }

    #[inline(always)]
    unsafe fn make_context(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context {
        unsafe { native_make_context(stack_top, entry, arg) }
    }
}

// On AArch64 the C ABI makes v8–v15 (lower halves) callee-saved, but the
// switch code below saves only the general-purpose set — a task suspended
// with live floating-point state could otherwise resume with another task's
// register contents.  Saving v8–v15 unconditionally would cost 8 extra
// stores + 8 loads on every switch, even though integer code (the common
// case for a scheduler hot path) has nothing live there.
//
// So they are declared as clobbered instead, and the *compiler* spills
// exactly the live ones: free for integer code, correct for floating-point
// code.  `clobber_abi` covers the ordinary caller-saved set; x19–x28 are
// saved and restored by the blocks themselves.
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
