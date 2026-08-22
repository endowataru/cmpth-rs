//! [`NativeContext`] and [`LeanFrameContext`]: two [`ContextPolicy`]
//! implementations with the same observable behavior and different cost
//! shapes, both implemented with inline assembly.
//!
//! The switch primitives are inlined into their call sites with `asm!`
//! rather than called out of line: the return address stored in a saved
//! frame is a label inside the caller, so resuming a context lands straight
//! back in the caller's own code.
//!
//! # Which callee-saved registers the frame carries, and which are clobbers
//!
//! A switch destroys callee-saved registers — it loads the destination
//! context's values into them — which an ordinary call never does, so
//! `clobber_abi("C")` does not cover them: that macro expands to the
//! *caller*-saved set only.  Each callee-saved register therefore has to be
//! handled one of two ways:
//!
//! * **Declared as a clobber**, letting the compiler preserve whatever it
//!   actually has live there, in its *own* stack frame.  This is what
//!   [`LeanFrameContext`] does for `x21`–`x28` / `r14`–`r15` (and both
//!   policies do it for the `v8`–`v15` low halves, which AAPCS64 also makes
//!   callee-saved).
//! * **Saved into the context frame by the assembly**, unconditionally,
//!   every switch.  This is what [`NativeContext`] does for the same
//!   registers.
//!
//! Both are correct — a resumed context always continues at the very label
//! that saved it, so any value the compiler had live across the switch is
//! still in the suspended task's own stack frame, on the task's own stack,
//! which travels with it even when the task is stolen; nothing forces the
//! ctx frame itself to carry it.  What neither policy can avoid: registers
//! that cannot be named as an `asm!` operand at all (`x19`/`rbx` are
//! reserved by LLVM, `x29`/`rbp` are the frame pointer), the resume address,
//! and the two registers the [`make_context`](ContextPolicy::make_context)
//! entry protocol uses to carry `entry`/`arg` into the trampoline
//! (`x19`/`x20`, `r12`/`r13`) — those always live in the frame, because
//! `func` runs on the destination stack and tramples everything below its
//! top before the trampoline gets control, so they cannot live in memory
//! the caller owns instead.
//!
//! # Why two policies, and which one to pick
//!
//! Declaring a register as a clobber does not make its preservation cheaper
//! in any absolute sense — it moves the cost from "the switch, paid
//! unconditionally on every call, regardless of whether anything is live"
//! to "the enclosing function, paid once in its prologue/epilogue, but only
//! if that function is ever compiled to use the register for something
//! live across *any* clobbering call site in its body — including a switch
//! whose actual instructions may not even touch it, since AAPCS64/SysV's
//! callee-saved contract is whole-function, not per-program-point: once a
//! register is clobbered anywhere in a function, the compiler must save the
//! caller's value at entry and restore it before every return, whether or
//! not anything of the function's own is ever really live there."
//!
//! So which is faster is a genuine, measured tradeoff, not a strict
//! improvement in either direction — this is exactly the kind of
//! axis-of-variation the crate's trait-based design exists to make
//! swappable (see the crate-level docs and `ContextPolicy`) rather than
//! deciding once and hard-coding it:
//!
//! * [`NativeContext`] wins when a function calls the switch primitives
//!   *many times relative to how few registers it actually needs live
//!   across them* — e.g. a scheduler's idle loop, which yields repeatedly
//!   but carries almost nothing across a yield.  There the unconditional
//!   per-switch cost recurs every iteration, while the fixed frame pays for
//!   itself by carrying anything live for free.
//! * [`LeanFrameContext`] wins when a function calls the switch rarely
//!   (often once) but keeps several values live across the call, and loses
//!   when it declares registers clobbered that it happens to need nowhere
//!   at all — that case pays a save/restore pair for nothing.  Measured on
//!   this crate's own `fib`/`nqueens` recursion (spawn-once-per-level, a
//!   handful of live locals per level): a narrow but real regression
//!   (~1–2%) despite the switch site itself shrinking by 8 instructions,
//!   because the recursive frame picked up 3 extra callee-saved pairs it
//!   never uses, purely because they are declared clobbered somewhere in
//!   the function.
//!
//! Benchmark your own call sites before choosing; do not assume either
//! direction from instruction counts alone.
//!
//! `make_context` needs no assembly at all (it switches nothing, it only
//! writes a frame), and the only symbol left is the entry trampoline, which
//! has to be assembly because it is entered by `ret` rather than by a call
//! — and which both policies share, because both frame layouts place
//! `entry`/`arg` at the same first two words.

use crate::traits::stackful::{CondSwitchFnLike, Context, ContextPolicy, EntryFn, RestoreFnLike, SwapFnLike, SwitchFnLike, Transfer};

// ---------------------------------------------------------------------------
// Entry trampoline, shared by both policies
// ---------------------------------------------------------------------------
//
// Reached by `ret`/`br` rather than by a call: the switcher's `func` leaves
// its Transfer in the return-value register while `entry`/`arg` sit in the
// callee-saved registers the frame just restored, which is not a shape any
// Rust function signature can describe.  Bridging that to the ordinary C ABI
// takes three instructions.  Both frame layouts below place `entry`/`arg` at
// word 0/1 of the frame (`x19`/`x20`, `r12`/`r13`), so one trampoline serves
// both — only the *rest* of the frame differs between the two policies.
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

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
compile_error!("cmpth: no ContextPolicy implementation for this architecture");

// ---------------------------------------------------------------------------
// NativeContext: full frame, minimal clobbers
// ---------------------------------------------------------------------------

/// Default `ContextPolicy`: the ctx frame carries every callee-saved
/// register (`x19`–`x28`/`x29`/`x30` on AArch64, `rbx`/`rbp`/`r12`–`r15` +
/// return address on x86-64), saved and restored unconditionally by every
/// switch.  See the module docs for when to prefer this over
/// [`LeanFrameContext`].
pub struct NativeContext;

/// Build the [`NativeContext`] frame: 96 bytes on AArch64, 56 on x86-64.
///
/// # Safety
/// `stack_top` must be the top of an unused stack with at least one frame's
/// worth of space below it.
#[inline(always)]
unsafe fn make_context_native(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context {
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

// x86-64 (System V).  Same shape as the AArch64 implementation below; the
// differences worth calling out:
//
// * There is no `call` any more, so the return address a `call` used to push
//   is pushed explicitly: 7 pushes from a 16-byte-aligned rsp put ctx at
//   8 (mod 16), the alignment the rest of the protocol assumes.
// * Every callee-saved register is folded into the frame here (unlike
//   `LeanFrameContext`), so nothing needs declaring as a clobber beyond
//   `v8`-equivalent state — and System V has none: every xmm/ymm/zmm is
//   already caller-saved, so `clobber_abi("C")` covers them for free.
//
// Windows x64 is *not* covered by this: xmm6–xmm15 are callee-saved in its
// ABI, so it would need different handling.
#[cfg(target_arch = "x86_64")]
unsafe impl ContextPolicy for NativeContext {
    // Unverified on this (aarch64-only) machine; mirrors the aarch64
    // tail-land restructuring below.
    #[inline(always)]
    unsafe fn swap_context<F: SwapFnLike>(to: Context, a1: *mut (), a2: *mut ()) -> Transfer {
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
                "mov  r8,  rdi",         // r8 = to.0
                "mov  rcx, rdx",         // rcx = a2 (shift up)
                "mov  rdx, rsi",         // rdx = a1 (shift up)
                "mov  rsi, r8",          // rsi = to.0
                "mov  rdi, rsp",         // rdi = prev_ctx
                "mov  rsp, r8",          // onto `to`'s own stack
                "jmp  {f}",              // F::call(prev_ctx, to, a1, a2) --
                                         // never returns
                "3:",                    // reached only by an external land()
                f = sym <F as SwapFnLike>::call,
                inout("rdi") to.0 => _,
                inout("rsi") a1 => _,
                inout("rdx") a2 => _,
                lateout("rax") ret,
                clobber_abi("C"),
            );
        }
        Transfer(ret)
    }

    #[inline(always)]
    unsafe fn save_context<F: SwitchFnLike>(
        new_sp: *mut u8,
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
                "mov  r8,  rdi",         // r8  = new stack top
                "mov  r10, rsp",         // r10 = prev_ctx
                "mov  rdi, rsp",         // arg0 = prev_ctx (a1/a2 already sit
                                         // in rsi/rdx, untouched)
                "and  r8, -16",          // align the new stack top
                "mov  [r8 - 8], r10",    // prev_ctx, for the return path below
                "lea  r11, [rip + 4f]",
                "mov  [r8 - 24], r11",   // F::call's return address
                "lea  rsp, [r8 - 24]",   // == 8 (mod 16): F::call enters as
                                         // if called
                "jmp  {f}",              // F::call(prev_ctx, a1, a2) on the
                                         // new stack (returns iff no real
                                         // switch happened)

                "4:",                    // F::call returned: resume the saved context
                "mov  r9, [rsp + 8]",    // r9 = prev_ctx
                "mov  rsp, r9",
                "pop  rbx",
                "pop  rbp",
                "pop  r12",
                "pop  r13",
                "pop  r14",
                "pop  r15",
                "add  rsp, 8",
                "jmp  3f",               // -> `3:` (this instance's own)
                "3:",
                f = sym <F as SwitchFnLike>::call,
                inout("rdi") new_sp => _,
                inout("rsi") a1 => _,
                inout("rdx") a2 => _,
                lateout("rax") ret,
                clobber_abi("C"),
            );
        }
        Transfer(ret)
    }

    // Only the cancel path returns to this code -- see the aarch64
    // cond_swap_context above.
    #[inline(always)]
    unsafe fn cond_swap_context<F: CondSwitchFnLike>(
        to: Context,
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
                // rsp = prev_ctx.  r12/r13 are ours to use now: the previous
                // values are in the frame and cancel restores every
                // callee-saved register from it.
                "mov  r12, rdi",         // r12 = to.0
                "mov  r13, rsp",         // r13 = prev_ctx
                "mov  rcx, rdx",         // rcx = a2 (shift up)
                "mov  rdx, rsi",         // rdx = a1 (shift up)
                "mov  rsi, r12",         // rsi = to.0
                "mov  rdi, r13",         // rdi = prev_ctx

                // Run F::call on the destination stack, below its intact
                // frame. ctx == 8 (mod 16), so sub 8 aligns rsp for the call.
                "mov  rsp, r12",
                "sub  rsp, 8",
                "call {f}",              // rax = value; returns only on cancel
                "add  rsp, 8",

                "mov  rsp, r13",         // cancel: restore the previous context
                "pop  rbx",
                "pop  rbp",
                "pop  r12",
                "pop  r13",
                "pop  r14",
                "pop  r15",
                "pop  r11",
                "jmp  r11",              // -> `3:` (this instance's own;
                                         // cancel always resumes here)
                "3:",
                f = sym <F as CondSwitchFnLike>::call,
                inout("rdi") to.0 => _,
                inout("rsi") a1 => _,
                inout("rdx") a2 => _,
                lateout("rax") ret,
                clobber_abi("C"),
            );
        }
        Transfer(ret)
    }

    // Unverified on this (aarch64-only) machine; mirrors the aarch64
    // Strategy-B restructuring below (`F::call` performs the switch-out
    // itself via `land`). `jmp`, not `call`: `call` would push a return
    // address `F::call` (ending in `land`'s indirect jump, never a `ret`)
    // would never pop -- see the aarch64 `b`-vs-`bl` note. `to.0` is already
    // == 8 (mod 16) by construction (see `make_context_native`), the same
    // alignment a `call` would have produced, so `F::call` sees an
    // identical entry state either way.
    #[inline(always)]
    unsafe fn restore_context<F: RestoreFnLike>(to: Context, a1: *mut (), a2: *mut ()) -> ! {
        unsafe {
            core::arch::asm!(
                "mov  rsp, rdi",         // F::call runs on `to`'s own stack,
                                         // below its still-unread frame
                "jmp  {f}",              // F::call(to, a1, a2) -- never returns
                f = sym <F as RestoreFnLike>::call,
                in("rdi") to.0,
                in("rsi") a1,
                in("rdx") a2,
                options(noreturn),
            );
        }
    }

    #[inline(always)]
    unsafe fn land(ctx: Context, ret_value: *mut ()) -> ! {
        unsafe {
            core::arch::asm!(
                "mov  r8, rdi",          // r8 = ctx.0 (frame pointer)
                "mov  rsp, r8",
                "pop  rbx",
                "pop  rbp",
                "pop  r12",
                "pop  r13",
                "pop  r14",
                "pop  r15",
                "mov  rax, rsi",         // rax = ret_value, for the resumed side
                "pop  r11",
                "jmp  r11",
                in("rdi") ctx.0,
                in("rsi") ret_value,
                options(noreturn),
            );
        }
    }

    #[inline(always)]
    unsafe fn make_context(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context {
        unsafe { make_context_native(stack_top, entry, arg) }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe impl ContextPolicy for NativeContext {
    #[inline(always)]
    // `F::call` here always ends by calling `land` itself (see
    // `SwapFnLike`'s doc comment) -- so the `3:` label below is *never*
    // reached by this specific call's own `b {f}` returning; it exists
    // purely as the address some later, unrelated switch resumes into when
    // it names this saved frame as its own destination. `clobber_abi("C")`
    // still applies: `b` vs `bl` only changes whether a return address gets
    // pushed, not what `F::call`'s own execution clobbers.
    unsafe fn swap_context<F: SwapFnLike>(to: Context, a1: *mut (), a2: *mut ()) -> Transfer {
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

                "mov  x9,  x0",          // x9  = destination context (to.0)
                "mov  x11, sp",          // x11 = prev_ctx (our saved frame)
                "mov  x3,  x2",          // x3 = a2 (shift up before x0-x2 move)
                "mov  x2,  x1",          // x2 = a1
                "mov  x1,  x9",          // x1 = to.0
                "mov  x0,  x11",         // x0 = prev_ctx
                "mov  sp,  x9",          // onto `to`'s own stack, below its
                                         // still-unread frame
                "b    {f}",              // F::call(prev_ctx, to, a1, a2) --
                                         // never returns
                "3:",                    // reached only by an external land()
                                         // resuming this saved frame later
                f = sym <F as SwapFnLike>::call,
                inout("x0") to.0 => ret,
                inout("x1") a1 => _,
                inout("x2") a2 => _,
                lateout("v8") _, lateout("v9") _, lateout("v10") _, lateout("v11") _,
                lateout("v12") _, lateout("v13") _, lateout("v14") _, lateout("v15") _,
                clobber_abi("C"),
            );
        }
        Transfer(ret)
    }

    // `save_context` has no predetermined destination (see `SwitchFnLike`'s
    // doc comment), so `F::call` may return normally; the fallback path
    // (`F::call` returned without ever diverging into a real switch) lands
    // on our own just-saved frame directly -- `x19` is free to stash it
    // across the call (LLVM-reserved, never touched by `F::call`'s own
    // compiled body, so it survives an ordinary ABI-conforming call for
    // free; unlike `x20`/etc. this needs no help from the callee-saved
    // contract).
    #[inline(always)]
    unsafe fn save_context<F: SwitchFnLike>(
        new_sp: *mut u8,
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

                "mov  x9,  x0",          // x9 = new stack top
                "mov  x19, sp",          // x19 = prev_ctx (our saved frame)
                "bic  x9, x9, #15",      // align the new stack top
                "mov  x0,  x19",         // arg0 for F::call = prev_ctx
                "mov  sp,  x9",          // onto the new stack

                "bl   {f}",              // F::call(prev_ctx, a1, a2) -> Transfer
                                         // (returns iff no real switch happened)

                "mov  x1,  x0",          // x1 = F::call's returned value
                "mov  x9,  x19",         // x9 = our own frame (x19 still
                                         // valid: preserved across the call)
                "ldp  x19, x20, [x9,  #0]",
                "ldp  x21, x22, [x9, #16]",
                "ldp  x23, x24, [x9, #32]",
                "ldp  x25, x26, [x9, #48]",
                "ldp  x27, x28, [x9, #64]",
                "ldp  x29, x30, [x9, #80]",
                "add  sp,  x9, #96",
                "mov  x0,  x1",
                "br   x30",              // -> `3:` (this instance's own)
                "3:",
                f = sym <F as SwitchFnLike>::call,
                inout("x0") new_sp => ret,
                inout("x1") a1 => _,
                inout("x2") a2 => _,
                lateout("v8") _, lateout("v9") _, lateout("v10") _, lateout("v11") _,
                lateout("v12") _, lateout("v13") _, lateout("v14") _, lateout("v15") _,
                clobber_abi("C"),
            );
        }
        Transfer(ret)
    }

    // Only the cancel path returns to this code at all now: committing is a
    // tail call to `land` inside `F::call` itself (see `CondSwitchFnLike`'s
    // doc comment), so there's no flag left to branch on here -- any return
    // from `F::call` unconditionally means cancel, restoring `prev` (x20).
    #[inline(always)]
    unsafe fn cond_swap_context<F: CondSwitchFnLike>(
        to: Context,
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

                "mov  x19, x0",          // x19 = destination context
                "mov  x20, sp",          // x20 = prev_ctx (our own frame)
                "mov  x3,  x2",          // x3 = a2 (shift up)
                "mov  x2,  x1",          // x2 = a1
                "mov  x1,  x19",         // x1 = to.0
                "mov  x0,  x20",         // x0 = prev_ctx

                "mov  sp,  x19",         // run F::call on the destination
                                         // stack, below its intact frame
                "bl   {f}",              // F::call(prev_ctx, to, a1, a2) --
                                         // returns only on cancel

                "mov  x9,  x20",         // cancel: restore the previous
                                         // context (x20 preserved across the
                                         // call by the ordinary callee-saved
                                         // contract)
                "ldp  x19, x20, [x9,  #0]",
                "ldp  x21, x22, [x9, #16]",
                "ldp  x23, x24, [x9, #32]",
                "ldp  x25, x26, [x9, #48]",
                "ldp  x27, x28, [x9, #64]",
                "ldp  x29, x30, [x9, #80]",
                "add  sp,  x9, #96",
                "br   x30",              // -> `3:` (this instance's own;
                                         // cancel always resumes here)
                "3:",
                f = sym <F as CondSwitchFnLike>::call,
                inout("x0") to.0 => ret,
                inout("x1") a1 => _,
                inout("x2") a2 => _,
                lateout("v8") _, lateout("v9") _, lateout("v10") _, lateout("v11") _,
                lateout("v12") _, lateout("v13") _, lateout("v14") _, lateout("v15") _,
                clobber_abi("C"),
            );
        }
        Transfer(ret)
    }

    // `F::call` itself performs the switch-out (via `land`, inlined into its
    // own compiled body) once its Rust logic finishes, so this asm block is
    // nothing but "move onto `to`'s stack, then jump to `F::call`" -- no code
    // after it at all, since `F::call` never returns. Plain `b`, not `bl`:
    // `bl` would push a return address onto the RAS that `F::call` (ending
    // in `land`'s `br`, never a `ret`) would never pop, leaving a stale
    // entry that can misalign the RAS for unrelated `ret`s deeper in the
    // call stack later.
    #[inline(always)]
    unsafe fn restore_context<F: RestoreFnLike>(to: Context, a1: *mut (), a2: *mut ()) -> ! {
        unsafe {
            core::arch::asm!(
                "mov  sp,  x0",          // F::call runs on `to`'s own stack,
                                         // below its still-unread frame
                "b    {f}",              // F::call(to, a1, a2) -- never returns
                f = sym <F as RestoreFnLike>::call,
                in("x0") to.0,
                in("x1") a1,
                in("x2") a2,
                options(noreturn),
            );
        }
    }

    #[inline(always)]
    unsafe fn land(ctx: Context, ret_value: *mut ()) -> ! {
        unsafe {
            core::arch::asm!(
                "mov  x9,  x0",          // x9 = ctx.0 (frame pointer)
                "ldp  x19, x20, [x9,  #0]",
                "ldp  x21, x22, [x9, #16]",
                "ldp  x23, x24, [x9, #32]",
                "ldp  x25, x26, [x9, #48]",
                "ldp  x27, x28, [x9, #64]",
                "ldp  x29, x30, [x9, #80]",
                "add  sp,  x9, #96",
                "mov  x0,  x1",          // x0 = ret_value, for the resumed side
                "br   x30",
                in("x0") ctx.0,
                in("x1") ret_value,
                options(noreturn),
            );
        }
    }

    #[inline(always)]
    unsafe fn make_context(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context {
        unsafe { make_context_native(stack_top, entry, arg) }
    }
}

// ---------------------------------------------------------------------------
// LeanFrameContext: minimal frame, wide clobbers
// ---------------------------------------------------------------------------

/// Alternative `ContextPolicy`: the ctx frame carries only what cannot be
/// declared a clobber — `x19`/`x20`/`x29`/`x30` on AArch64,
/// `rbx`/`rbp`/`r12`/`r13` + return address on x86-64 — and declares
/// `x21`–`x28`/`r14`–`r15` as clobbered instead, so the compiler preserves
/// exactly what each caller has live there, in that caller's own frame.
/// See the module docs for when to prefer this over [`NativeContext`]; the
/// short version is: measure both on your actual call sites, because the
/// answer depends on how often you switch relative to how much you keep
/// live across the switch, not on either policy's instruction count alone.
///
/// # Safety
/// A context saved by one policy must never be resumed by the other: the
/// two frame layouts differ past the first two words, so `S::Ctx` must be
/// the same concrete type end-to-end for a given system — never mix
/// [`NativeContext`] and `LeanFrameContext` contexts within one scheduler.
pub struct LeanFrameContext;

/// Build the [`LeanFrameContext`] frame: 32 bytes on AArch64, 40 on x86-64.
///
/// # Safety
/// `stack_top` must be the top of an unused stack with at least one frame's
/// worth of space below it.
#[inline(always)]
unsafe fn make_context_lean(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context {
    let aligned = (stack_top as usize & !0xF) as *mut u8;
    let trampoline = cmpth_entry_trampoline as *const () as usize;

    #[cfg(target_arch = "aarch64")]
    {
        // 32-byte frame: x19, x20, x29 (fp), x30 (resume address).
        let ctx = aligned.wrapping_sub(32);
        let w = ctx as *mut usize;
        unsafe {
            w.add(0).write(entry as usize); // x19 = entry
            w.add(1).write(arg as usize); // x20 = arg
            w.add(2).write(0); // x29 = fp = 0 (terminates backtraces)
            w.add(3).write(trampoline); // x30 = trampoline
        }
        Context(ctx)
    }

    #[cfg(target_arch = "x86_64")]
    {
        // 40-byte frame: rbx, rbp, r12, r13, return address.
        let ctx = aligned.wrapping_sub(40);
        let w = ctx as *mut usize;
        unsafe {
            w.add(0).write(0); // rbx
            w.add(1).write(0); // rbp = 0 (terminates backtraces)
            w.add(2).write(entry as usize); // r12 = entry
            w.add(3).write(arg as usize); // r13 = arg
            w.add(4).write(trampoline); // resume address = trampoline
        }
        Context(ctx)
    }
}

#[cfg(target_arch = "x86_64")]
unsafe impl ContextPolicy for LeanFrameContext {
    // Unverified on this (aarch64-only) machine; mirrors the x86_64
    // NativeContext impl above.
    #[inline(always)]
    unsafe fn swap_context<F: SwapFnLike>(to: Context, a1: *mut (), a2: *mut ()) -> Transfer {
        let ret: *mut ();
        unsafe {
            core::arch::asm!(
                "lea  r11, [rip + 3f]",  // resume address = end of this block
                "push r11",              // ...where `call` would have pushed it
                "push r13",
                "push r12",
                "push rbp",
                "push rbx",
                // rsp = prev_ctx
                "mov  r8,  rdi",         // r8 = to.0
                "mov  rcx, rdx",         // rcx = a2 (shift up)
                "mov  rdx, rsi",         // rdx = a1 (shift up)
                "mov  rsi, r8",          // rsi = to.0
                "mov  rdi, rsp",         // rdi = prev_ctx
                "mov  rsp, r8",          // onto `to`'s own stack
                "jmp  {f}",              // F::call(prev_ctx, to, a1, a2) --
                                         // never returns
                "3:",
                f = sym <F as SwapFnLike>::call,
                inout("rdi") to.0 => _,
                inout("rsi") a1 => _,
                inout("rdx") a2 => _,
                lateout("rax") ret,
                lateout("r14") _, lateout("r15") _,
                clobber_abi("C"),
            );
        }
        Transfer(ret)
    }

    #[inline(always)]
    unsafe fn save_context<F: SwitchFnLike>(
        new_sp: *mut u8,
        a1: *mut (),
        a2: *mut (),
    ) -> Transfer {
        let ret: *mut ();
        unsafe {
            core::arch::asm!(
                "lea  r11, [rip + 3f]",
                "push r11",
                "push r13",
                "push r12",
                "push rbp",
                "push rbx",
                // rsp = prev_ctx
                "mov  r8,  rdi",         // r8  = new stack top
                "mov  r10, rsp",         // r10 = prev_ctx
                "mov  rdi, rsp",         // arg0 = prev_ctx (a1/a2 untouched)
                "and  r8, -16",          // align the new stack top
                "mov  [r8 - 8], r10",    // prev_ctx, for the return path below
                "lea  r11, [rip + 4f]",
                "mov  [r8 - 24], r11",   // F::call's return address
                "lea  rsp, [r8 - 24]",   // == 8 (mod 16): F::call enters as
                                         // if called
                "jmp  {f}",              // F::call(prev_ctx, a1, a2) on the
                                         // new stack

                "4:",                    // F::call returned: resume the saved context
                "mov  r9, [rsp + 8]",    // r9 = prev_ctx
                "mov  rsp, r9",
                "pop  rbx",
                "pop  rbp",
                "pop  r12",
                "pop  r13",
                "add  rsp, 8",
                "jmp  3f",               // -> `3:` (this instance's own)
                "3:",
                f = sym <F as SwitchFnLike>::call,
                inout("rdi") new_sp => _,
                inout("rsi") a1 => _,
                inout("rdx") a2 => _,
                lateout("rax") ret,
                lateout("r14") _, lateout("r15") _,
                clobber_abi("C"),
            );
        }
        Transfer(ret)
    }

    #[inline(always)]
    unsafe fn cond_swap_context<F: CondSwitchFnLike>(
        to: Context,
        a1: *mut (),
        a2: *mut (),
    ) -> Transfer {
        let ret: *mut ();
        unsafe {
            core::arch::asm!(
                "lea  r11, [rip + 3f]",
                "push r11",
                "push r13",
                "push r12",
                "push rbp",
                "push rbx",
                // rsp = prev_ctx.  r12/r13 are ours to use now: the previous
                // values are in the frame and cancel restores every
                // callee-saved register from it.
                "mov  r12, rdi",         // r12 = to.0
                "mov  r13, rsp",         // r13 = prev_ctx
                "mov  rcx, rdx",         // rcx = a2 (shift up)
                "mov  rdx, rsi",         // rdx = a1 (shift up)
                "mov  rsi, r12",         // rsi = to.0
                "mov  rdi, r13",         // rdi = prev_ctx

                // Run F::call on the destination stack, below its intact
                // frame. ctx == 8 (mod 16), so sub 8 aligns rsp for the call.
                "mov  rsp, r12",
                "sub  rsp, 8",
                "call {f}",              // rax = value; returns only on cancel
                "add  rsp, 8",

                "mov  rsp, r13",         // cancel: restore the previous context
                "pop  rbx",
                "pop  rbp",
                "pop  r12",
                "pop  r13",
                "pop  r11",
                "jmp  r11",              // -> `3:` (this instance's own)
                "3:",
                f = sym <F as CondSwitchFnLike>::call,
                inout("rdi") to.0 => _,
                inout("rsi") a1 => _,
                inout("rdx") a2 => _,
                lateout("rax") ret,
                lateout("r14") _, lateout("r15") _,
                clobber_abi("C"),
            );
        }
        Transfer(ret)
    }

    // Unverified on this (aarch64-only) machine. `jmp`, not `call` -- see
    // the NativeContext impl above for why.
    #[inline(always)]
    unsafe fn restore_context<F: RestoreFnLike>(to: Context, a1: *mut (), a2: *mut ()) -> ! {
        unsafe {
            core::arch::asm!(
                "mov  rsp, rdi",         // F::call runs on `to`'s own stack,
                                         // below its still-unread frame
                "jmp  {f}",              // F::call(to, a1, a2) -- never returns
                f = sym <F as RestoreFnLike>::call,
                in("rdi") to.0,
                in("rsi") a1,
                in("rdx") a2,
                options(noreturn),
            );
        }
    }

    #[inline(always)]
    unsafe fn land(ctx: Context, ret_value: *mut ()) -> ! {
        unsafe {
            core::arch::asm!(
                "mov  r8, rdi",          // r8 = ctx.0 (frame pointer)
                "mov  rsp, r8",
                "pop  rbx",
                "pop  rbp",
                "pop  r12",
                "pop  r13",
                "mov  rax, rsi",         // rax = ret_value, for the resumed side
                "pop  r11",
                "jmp  r11",
                in("rdi") ctx.0,
                in("rsi") ret_value,
                options(noreturn),
            );
        }
    }

    #[inline(always)]
    unsafe fn make_context(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context {
        unsafe { make_context_lean(stack_top, entry, arg) }
    }
}

// On AArch64, `x21`–`x28` and the low halves of `v8`–`v15` are callee-saved
// and are declared as clobbers rather than written to the frame: the
// compiler then preserves exactly what it has live, once per enclosing
// function rather than once per switch.  `x19`/`x20` stay in the frame
// (`x19` cannot be named as an operand at all, and both carry `entry`/`arg`
// for the `make_context` trampoline), as do `x29` (frame pointer) and `x30`
// (the resume address, which *is* the context's program counter).

#[cfg(target_arch = "aarch64")]
unsafe impl ContextPolicy for LeanFrameContext {
    // Same Strategy-B / tail-land shape as `NativeContext` above; only the
    // frame layout differs (32 bytes, x21-x28 declared as clobbers).
    #[inline(always)]
    unsafe fn swap_context<F: SwapFnLike>(to: Context, a1: *mut (), a2: *mut ()) -> Transfer {
        let ret: *mut ();
        unsafe {
            core::arch::asm!(
                "sub  sp, sp, #32",
                "stp  x19, x20, [sp,  #0]",
                "adr  x30, 3f",
                "stp  x29, x30, [sp, #16]",

                "mov  x9,  x0",          // x9  = destination context (to.0)
                "mov  x11, sp",          // x11 = prev_ctx
                "mov  x3,  x2",          // x3 = a2
                "mov  x2,  x1",          // x2 = a1
                "mov  x1,  x9",          // x1 = to.0
                "mov  x0,  x11",         // x0 = prev_ctx
                "mov  sp,  x9",
                "b    {f}",              // F::call(prev_ctx, to, a1, a2) --
                                         // never returns
                "3:",
                f = sym <F as SwapFnLike>::call,
                inout("x0") to.0 => ret,
                inout("x1") a1 => _,
                inout("x2") a2 => _,
                lateout("x21") _, lateout("x22") _, lateout("x23") _, lateout("x24") _,
                lateout("x25") _, lateout("x26") _, lateout("x27") _, lateout("x28") _,
                lateout("v8") _, lateout("v9") _, lateout("v10") _, lateout("v11") _,
                lateout("v12") _, lateout("v13") _, lateout("v14") _, lateout("v15") _,
                clobber_abi("C"),
            );
        }
        Transfer(ret)
    }

    #[inline(always)]
    unsafe fn save_context<F: SwitchFnLike>(
        new_sp: *mut u8,
        a1: *mut (),
        a2: *mut (),
    ) -> Transfer {
        let ret: *mut ();
        unsafe {
            core::arch::asm!(
                "sub  sp, sp, #32",
                "stp  x19, x20, [sp,  #0]",
                "adr  x30, 3f",
                "stp  x29, x30, [sp, #16]",

                "mov  x9,  x0",          // x9 = new stack top
                "mov  x19, sp",          // x19 = prev_ctx
                "bic  x9, x9, #15",
                "mov  x0,  x19",         // arg0 for F::call = prev_ctx
                "mov  sp,  x9",

                "bl   {f}",              // F::call(prev_ctx, a1, a2) -> Transfer

                "mov  x1,  x0",
                "mov  x9,  x19",
                "ldp  x19, x20, [x9,  #0]",
                "ldp  x29, x30, [x9, #16]",
                "add  sp,  x9, #32",
                "mov  x0,  x1",
                "br   x30",
                "3:",
                f = sym <F as SwitchFnLike>::call,
                inout("x0") new_sp => ret,
                inout("x1") a1 => _,
                inout("x2") a2 => _,
                lateout("x21") _, lateout("x22") _, lateout("x23") _, lateout("x24") _,
                lateout("x25") _, lateout("x26") _, lateout("x27") _, lateout("x28") _,
                lateout("v8") _, lateout("v9") _, lateout("v10") _, lateout("v11") _,
                lateout("v12") _, lateout("v13") _, lateout("v14") _, lateout("v15") _,
                clobber_abi("C"),
            );
        }
        Transfer(ret)
    }

    #[inline(always)]
    unsafe fn cond_swap_context<F: CondSwitchFnLike>(
        to: Context,
        a1: *mut (),
        a2: *mut (),
    ) -> Transfer {
        let ret: *mut ();
        unsafe {
            core::arch::asm!(
                "sub  sp, sp, #32",
                "stp  x19, x20, [sp,  #0]",
                "adr  x30, 3f",
                "stp  x29, x30, [sp, #16]",

                "mov  x19, x0",          // x19 = destination context
                "mov  x20, sp",          // x20 = prev_ctx
                "mov  x3,  x2",          // x3 = a2
                "mov  x2,  x1",          // x2 = a1
                "mov  x1,  x19",         // x1 = to.0
                "mov  x0,  x20",         // x0 = prev_ctx

                "mov  sp,  x19",
                "bl   {f}",              // F::call(prev_ctx, to, a1, a2) --
                                         // returns only on cancel

                "mov  x9,  x20",         // cancel: restore the previous context
                "ldp  x19, x20, [x9,  #0]",
                "ldp  x29, x30, [x9, #16]",
                "add  sp,  x9, #32",
                "br   x30",
                "3:",
                f = sym <F as CondSwitchFnLike>::call,
                inout("x0") to.0 => ret,
                inout("x1") a1 => _,
                inout("x2") a2 => _,
                lateout("x21") _, lateout("x22") _, lateout("x23") _, lateout("x24") _,
                lateout("x25") _, lateout("x26") _, lateout("x27") _, lateout("x28") _,
                lateout("v8") _, lateout("v9") _, lateout("v10") _, lateout("v11") _,
                lateout("v12") _, lateout("v13") _, lateout("v14") _, lateout("v15") _,
                clobber_abi("C"),
            );
        }
        Transfer(ret)
    }

    // `b`, not `bl` -- see the NativeContext impl above for why.
    #[inline(always)]
    unsafe fn restore_context<F: RestoreFnLike>(to: Context, a1: *mut (), a2: *mut ()) -> ! {
        unsafe {
            core::arch::asm!(
                "mov  sp,  x0",          // F::call runs on `to`'s own stack,
                                         // below its still-unread frame
                "b    {f}",              // F::call(to, a1, a2) -- never returns
                f = sym <F as RestoreFnLike>::call,
                in("x0") to.0,
                in("x1") a1,
                in("x2") a2,
                options(noreturn),
            );
        }
    }

    #[inline(always)]
    unsafe fn land(ctx: Context, ret_value: *mut ()) -> ! {
        unsafe {
            core::arch::asm!(
                "mov  x9,  x0",          // x9 = ctx.0 (frame pointer)
                "ldp  x19, x20, [x9,  #0]",
                "ldp  x29, x30, [x9, #16]",
                "add  sp,  x9, #32",
                "mov  x0,  x1",          // x0 = ret_value, for the resumed side
                "br   x30",
                in("x0") ctx.0,
                in("x1") ret_value,
                options(noreturn),
            );
        }
    }

    #[inline(always)]
    unsafe fn make_context(stack_top: *mut u8, entry: EntryFn, arg: *mut ()) -> Context {
        unsafe { make_context_lean(stack_top, entry, arg) }
    }
}
