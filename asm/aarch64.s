// AArch64 context-switch primitives for cmpth.
//
// Modeled on ComposableThreads' x86_64_context_policy: every switch function
// takes a plain function pointer `func` that is executed *after* the stack
// switch, on the stack of the destination context.  `func` returns a Transfer
// value in x0 and its `ret` lands directly on the destination's saved return
// address, so a complete switch costs one indirect branch into `func` plus its
// ordinary `ret` — no extra trampolines and no "post swap" bookkeeping on the
// suspended side.
//
// A saved context is a 96-byte frame pushed on the suspended thread's stack:
//   [ctx +  0] x19    [ctx +  8] x20
//   [ctx + 16] x21    [ctx + 24] x22
//   [ctx + 32] x23    [ctx + 40] x24
//   [ctx + 48] x25    [ctx + 56] x26
//   [ctx + 64] x27    [ctx + 72] x28
//   [ctx + 80] x29    [ctx + 88] x30 (return address of the suspended caller)
// The context value handed to Rust is the frame base (== sp at save time - 96,
// always 16-byte aligned).  Resuming a context = load the 12 registers, set
// sp = ctx + 96, and return to the saved x30 with x0 = the transfer value.
//
// Both Mach-O (_cmpth_*) and ELF (cmpth_*) symbol names are provided.

.text
.align 4

// ---------------------------------------------------------------------------
// swap_context(to: Context = x0, func = x1, a1 = x2, a2 = x3) -> Transfer
//
// Save the current context on the current stack, switch to `to`, then
// tail-call func(prev_ctx, a1, a2) on the destination stack.  func's `ret`
// resumes the destination with x0 = func's return value.  When the saved
// context is itself resumed later, swap_context appears to return with x0 =
// the transfer value supplied by the resumer.
// ---------------------------------------------------------------------------
.global _cmpth_swap_context
.global cmpth_swap_context
_cmpth_swap_context:
cmpth_swap_context:
    sub  sp, sp, #96
    stp  x19, x20, [sp,  #0]
    stp  x21, x22, [sp, #16]
    stp  x23, x24, [sp, #32]
    stp  x25, x26, [sp, #48]
    stp  x27, x28, [sp, #64]
    stp  x29, x30, [sp, #80]

    mov  x9,  x0            // x9  = destination context
    mov  x10, x1            // x10 = func
    mov  x0,  sp            // arg0 = prev_ctx (our freshly saved frame)
    mov  x1,  x2            // arg1 = a1
    mov  x2,  x3            // arg2 = a2

    ldp  x19, x20, [x9,  #0]
    ldp  x21, x22, [x9, #16]
    ldp  x23, x24, [x9, #32]
    ldp  x25, x26, [x9, #48]
    ldp  x27, x28, [x9, #64]
    ldp  x29, x30, [x9, #80]
    add  sp,  x9, #96       // pop the destination frame
    br   x10                // func(prev_ctx, a1, a2); ret -> destination

// ---------------------------------------------------------------------------
// save_context(new_sp = x0, func = x1, a1 = x2, a2 = x3) -> Transfer
//
// Save the current context, switch to a fresh stack (no saved frame there),
// then tail-call func(prev_ctx, a1, a2).  Used for child-first fork: func
// bootstraps the new task on its own stack.  If func returns, the previous
// context is resumed immediately with x0 = func's return value.
// ---------------------------------------------------------------------------
.global _cmpth_save_context
.global cmpth_save_context
_cmpth_save_context:
cmpth_save_context:
    sub  sp, sp, #96
    stp  x19, x20, [sp,  #0]
    stp  x21, x22, [sp, #16]
    stp  x23, x24, [sp, #32]
    stp  x25, x26, [sp, #48]
    stp  x27, x28, [sp, #64]
    stp  x29, x30, [sp, #80]

    mov  x9,  x0            // x9  = new stack top
    mov  x10, x1            // x10 = func
    mov  x0,  sp            // arg0 = prev_ctx
    mov  x1,  x2            // arg1 = a1
    mov  x2,  x3            // arg2 = a2

    and  x9, x9, #~15
    mov  x11, sp            // keep prev frame pointer for the return path
    mov  sp, x9
    str  x11, [sp, #-16]!   // push prev frame pointer on the new stack
    adr  x30, Lcmpth_save_context_ret
    br   x10                // func(prev_ctx, a1, a2) on the new stack

Lcmpth_save_context_ret:
    // func returned: resume the saved (previous) context.  x0 = Transfer.
    ldr  x9, [sp]
    ldp  x19, x20, [x9,  #0]
    ldp  x21, x22, [x9, #16]
    ldp  x23, x24, [x9, #32]
    ldp  x25, x26, [x9, #48]
    ldp  x27, x28, [x9, #64]
    ldp  x29, x30, [x9, #80]
    add  sp,  x9, #96
    ret

// ---------------------------------------------------------------------------
// cond_swap_context(to: Context = x0, func = x1, a1 = x2, a2 = x3) -> Transfer
//
// Save the current context, run func(prev_ctx, a1, a2) on the destination
// stack (below the destination's still-intact frame), then inspect the
// CondTransfer {x0 = value, x1 = flag} returned by func:
//   flag != 0 -> commit: resume the destination context.
//   flag == 0 -> cancel: resume the just-saved previous context; from the
//                caller's perspective cond_swap_context returns immediately.
// In both cases x0 carries func's transfer value.
// ---------------------------------------------------------------------------
.global _cmpth_cond_swap_context
.global cmpth_cond_swap_context
_cmpth_cond_swap_context:
cmpth_cond_swap_context:
    sub  sp, sp, #96
    stp  x19, x20, [sp,  #0]
    stp  x21, x22, [sp, #16]
    stp  x23, x24, [sp, #32]
    stp  x25, x26, [sp, #48]
    stp  x27, x28, [sp, #64]
    stp  x29, x30, [sp, #80]

    // x19/x20 are ours to use now: the previous values are in the frame and
    // both exit paths below restore every callee-saved register from a frame.
    mov  x19, x0            // x19 = destination context (survives the call)
    mov  x20, sp            // x20 = previous frame (survives the call)
    mov  x10, x1            // x10 = func
    mov  x0,  sp            // arg0 = prev_ctx
    mov  x1,  x2            // arg1 = a1
    mov  x2,  x3            // arg2 = a2

    mov  sp,  x19           // run func on the destination stack, below the
                            // destination frame at [x19, x19+96)
    blr  x10                // (x0, x1) = func(prev_ctx, a1, a2)

    cbnz x1, Lcmpth_cond_commit
    mov  x9, x20            // cancel: restore the previous context
    b    Lcmpth_cond_restore
Lcmpth_cond_commit:
    mov  x9, x19            // commit: restore the destination context
Lcmpth_cond_restore:
    ldp  x19, x20, [x9,  #0]
    ldp  x21, x22, [x9, #16]
    ldp  x23, x24, [x9, #32]
    ldp  x25, x26, [x9, #48]
    ldp  x27, x28, [x9, #64]
    ldp  x29, x30, [x9, #80]
    add  sp,  x9, #96
    ret                     // x0 = func's transfer value

// ---------------------------------------------------------------------------
// restore_context(to: Context = x0, func = x1, a1 = x2, a2 = x3) -> !
//
// Abandon the current stack, switch to `to`, and tail-call func(a1, a2) on
// the destination stack.  Used by exiting tasks: func performs the final
// bookkeeping (marking finished / freeing the dead task) on a live stack.
// ---------------------------------------------------------------------------
.global _cmpth_restore_context
.global cmpth_restore_context
_cmpth_restore_context:
cmpth_restore_context:
    mov  x9,  x0            // x9  = destination context
    mov  x10, x1            // x10 = func
    mov  x0,  x2            // arg0 = a1
    mov  x1,  x3            // arg1 = a2

    ldp  x19, x20, [x9,  #0]
    ldp  x21, x22, [x9, #16]
    ldp  x23, x24, [x9, #32]
    ldp  x25, x26, [x9, #48]
    ldp  x27, x28, [x9, #64]
    ldp  x29, x30, [x9, #80]
    add  sp,  x9, #96
    br   x10                // func(a1, a2); ret -> destination

// ---------------------------------------------------------------------------
// make_context(stack_top = x0, entry = x1, arg = x2) -> Context
//
// Build a context frame at the top of a fresh stack so that the first switch
// into it lands in `entry(transfer, arg)`.  The switcher's func runs first
// (as for any switch) and its return value becomes `transfer`.
// ---------------------------------------------------------------------------
.global _cmpth_make_context
.global cmpth_make_context
_cmpth_make_context:
cmpth_make_context:
    and  x0, x0, #~15
    sub  x0, x0, #96
    str  x1, [x0,  #0]      // x19 = entry
    str  x2, [x0,  #8]      // x20 = arg
    stp  xzr, xzr, [x0, #16]
    stp  xzr, xzr, [x0, #32]
    stp  xzr, xzr, [x0, #48]
    stp  xzr, xzr, [x0, #64]
    str  xzr, [x0, #80]     // fp = 0 (stack trace terminator)
    adr  x9, Lcmpth_entry_trampoline
    str  x9, [x0, #88]      // resume address = trampoline
    ret

Lcmpth_entry_trampoline:
    // Reached when the switcher's func returns after the first switch into a
    // made context.  x0 = Transfer (func's return), x19 = entry, x20 = arg.
    mov  x1, x20
    blr  x19                // entry(transfer, arg) -> !
    brk  #0                 // entry must never return
