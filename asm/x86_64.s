/* x86_64 context-switch primitives for cmpth (System V ABI).
 *
 * Same design as the AArch64 implementation: every switch function takes a
 * plain function pointer executed *after* the stack switch, on the
 * destination stack.  This eliminates all "saving in progress" bookkeeping
 * (flags, spin loops, post-swap states) because the continuation is
 * published only after the context is fully saved.
 *
 * Saved context frame -- 56 bytes pushed on the suspended thread's stack:
 *   [ctx +  0] rbx    [ctx +  8] rbp
 *   [ctx + 16] r12    [ctx + 24] r13
 *   [ctx + 32] r14    [ctx + 40] r15
 *   [ctx + 48] return address of the suspended caller
 *
 * ctx (= Context value) is the frame base, i.e. rsp after the 6 pushes.
 * Because at function entry rsp = 16n-8 (call pushed 8 bytes), after
 * 6 x 8 = 48 more bytes: ctx = 16n-56 == 8 (mod 16).
 * Resuming a context: pop 6 registers, then ret with rax = Transfer.
 *
 * Both Mach-O (_cmpth_*) and ELF (cmpth_*) symbol names are provided.
 */

.text
.align 4

/* ---------------------------------------------------------------------------
 * swap_context(to: Context = %rdi, func = %rsi, a1 = %rdx, a2 = %rcx)
 *              -> Transfer (%rax)
 *
 * Save current context, switch to `to`, tail-call func(prev_ctx, a1, a2) on
 * the destination stack.  func's ret resumes the destination with rax = its
 * return value.  When the saved context is itself resumed, swap_context
 * appears to return with rax = the resumer's Transfer value.
 * ---------------------------------------------------------------------------
 */
.global _cmpth_swap_context
.global cmpth_swap_context
_cmpth_swap_context:
cmpth_swap_context:
    pushq %r15
    pushq %r14
    pushq %r13
    pushq %r12
    pushq %rbp
    pushq %rbx
    /* %rsp = prev_ctx */

    movq  %rdi, %r8        /* r8  = to */
    movq  %rsi, %r9        /* r9  = func */
    movq  %rsp, %rdi       /* arg0 = prev_ctx */
    movq  %rdx, %rsi       /* arg1 = a1 */
    movq  %rcx, %rdx       /* arg2 = a2 */

    movq  %r8, %rsp        /* switch to destination context */
    popq  %rbx
    popq  %rbp
    popq  %r12
    popq  %r13
    popq  %r14
    popq  %r15
    /* [%rsp] = destination's saved return address */
    jmpq  *%r9             /* func(prev_ctx, a1, a2); ret -> destination */

/* ---------------------------------------------------------------------------
 * save_context(new_sp = %rdi, func = %rsi, a1 = %rdx, a2 = %rcx)
 *              -> Transfer (%rax)
 *
 * Save the current context, switch to a fresh stack (new_sp), tail-call
 * func(prev_ctx, a1, a2) there.  If func returns, the saved context resumes
 * immediately with rax = func's return value.
 *
 * Stack layout on new_sp (top = high address; nothing is written AT or
 * above aligned_new_sp -- the caller owns that region, e.g. the closure and
 * result slots that spawn places above exec_top):
 *   [aligned_new_sp -  8] = prev_ctx   (for the return trampoline)
 *   [aligned_new_sp - 24] = trampoline (func's "return address")
 * rsp at jmp = aligned_new_sp - 24 == 8 (mod 16): func enters as if called.
 * ---------------------------------------------------------------------------
 */
.global _cmpth_save_context
.global cmpth_save_context
_cmpth_save_context:
cmpth_save_context:
    pushq %r15
    pushq %r14
    pushq %r13
    pushq %r12
    pushq %rbp
    pushq %rbx
    /* %rsp = prev_ctx */

    movq  %rdi, %r8        /* r8  = new_sp (save before overwriting %rdi) */
    movq  %rsi, %r9        /* r9  = func */
    movq  %rsp, %r10       /* r10 = prev_ctx */
    movq  %rsp, %rdi       /* arg0 = prev_ctx */
    movq  %rdx, %rsi       /* arg1 = a1 */
    movq  %rcx, %rdx       /* arg2 = a2 */

    andq  $-16, %r8        /* align new_sp to 16 bytes: r8 = 16n */
    movq  %r10, -8(%r8)    /* [16n -  8] = prev_ctx (for trampoline after func) */
    leaq  Lcmpth_save_ctx_ret(%rip), %r11
    movq  %r11, -24(%r8)   /* [16n - 24] = trampoline (func's return address) */
    leaq  -24(%r8), %rsp   /* switch to new stack: rsp = 16n - 24 == 8 (mod 16) */

    jmpq  *%r9             /* func(prev_ctx, a1, a2); ret -> trampoline */

Lcmpth_save_ctx_ret:
    /* func returned: rax = Transfer, rsp = 16n - 16, [rsp + 8] = prev_ctx */
    movq  8(%rsp), %r9     /* r9 = prev_ctx */
    movq  %r9, %rsp        /* restore prev stack */
    popq  %rbx
    popq  %rbp
    popq  %r12
    popq  %r13
    popq  %r14
    popq  %r15
    ret                    /* rax = Transfer (preserved throughout) */

/* ---------------------------------------------------------------------------
 * cond_swap_context(to = %rdi, func = %rsi, a1 = %rdx, a2 = %rcx)
 *                   -> Transfer (%rax)
 *
 * Save the current context, run func(prev_ctx, a1, a2) on the destination
 * stack below its intact frame, then inspect the CondTransfer returned:
 *   flag (rdx) != 0 -> commit: resume the destination context.
 *   flag (rdx) == 0 -> cancel: resume the just-saved previous context.
 * In both cases rax carries func's value field as the Transfer return.
 *
 * ctx == 8 (mod 16), so sub 8 is needed to 16-byte-align rsp before call.
 * After call+ret rsp is restored to ctx via add 8.
 * ---------------------------------------------------------------------------
 */
.global _cmpth_cond_swap_context
.global cmpth_cond_swap_context
_cmpth_cond_swap_context:
cmpth_cond_swap_context:
    pushq %r15
    pushq %r14
    pushq %r13
    pushq %r12
    pushq %rbp
    pushq %rbx
    /* %rsp = prev_ctx */

    /* Preserve state across the call using callee-saved registers
     * (our saved values of r12-r15 are in the frame; we can use them freely).
     */
    movq  %rdi, %r12       /* r12 = to (destination context) */
    movq  %rsp, %r13       /* r13 = prev_ctx */
    movq  %rsi, %r14       /* r14 = func */

    movq  %rsp, %rdi       /* arg0 = prev_ctx */
    movq  %rdx, %rsi       /* arg1 = a1 */
    movq  %rcx, %rdx       /* arg2 = a2 */

    /* Switch to destination stack and call func *below* its intact frame.
     * ctx == 8 (mod 16) -> sub 8 gives 16-byte alignment required before call.
     */
    movq  %r12, %rsp
    subq  $8, %rsp
    callq *%r14            /* func(prev_ctx, a1, a2) -> rax=value, rdx=flag */
    addq  $8, %rsp         /* rsp = r12 = destination ctx */

    testq %rdx, %rdx
    jnz   Lcmpth_cond_commit
    movq  %r13, %rsp       /* cancel: restore prev_ctx */
    jmp   Lcmpth_cond_restore
Lcmpth_cond_commit:
    movq  %r12, %rsp       /* commit: restore destination ctx */
Lcmpth_cond_restore:
    popq  %rbx
    popq  %rbp
    popq  %r12
    popq  %r13
    popq  %r14
    popq  %r15
    ret                    /* rax = value (Transfer); rdx = flag preserved too */

/* ---------------------------------------------------------------------------
 * restore_context(to = %rdi, func = %rsi, a1 = %rdx, a2 = %rcx) -> !
 *
 * Abandon the current stack, switch to `to`, tail-call func(a1, a2) on the
 * destination stack.  Used by exiting tasks; no context is saved.
 * ---------------------------------------------------------------------------
 */
.global _cmpth_restore_context
.global cmpth_restore_context
_cmpth_restore_context:
cmpth_restore_context:
    movq  %rdi, %r8        /* r8 = to */
    movq  %rsi, %r9        /* r9 = func */
    movq  %rdx, %rdi       /* rdi = a1 */
    movq  %rcx, %rsi       /* rsi = a2 */

    movq  %r8, %rsp
    popq  %rbx
    popq  %rbp
    popq  %r12
    popq  %r13
    popq  %r14
    popq  %r15
    jmpq  *%r9             /* func(a1, a2); ret -> destination */

/* ---------------------------------------------------------------------------
 * make_context(stack_top = %rdi, entry = %rsi, arg = %rdx) -> Context (%rax)
 *
 * Build a context frame so that the first switch into it calls
 * entry(Transfer, arg).  entry and arg are stashed in the callee-saved slots
 * r12/r13; the switcher's func preserves them, and the trampoline reads them
 * after func's ret.
 *
 * Frame layout (ctx = aligned_stack_top - 56):
 *   [ctx +  0] rbx = 0   [ctx +  8] rbp = 0
 *   [ctx + 16] r12 = entry  [ctx + 24] r13 = arg
 *   [ctx + 32] r14 = 0   [ctx + 40] r15 = 0
 *   [ctx + 48] = trampoline address
 * ---------------------------------------------------------------------------
 */
.global _cmpth_make_context
.global cmpth_make_context
_cmpth_make_context:
cmpth_make_context:
    andq  $-16, %rdi       /* align stack_top */
    subq  $56, %rdi        /* allocate frame: ctx = stack_top - 56 */

    xorq  %rax, %rax
    movq  %rax,  0(%rdi)   /* rbx = 0 */
    movq  %rax,  8(%rdi)   /* rbp = 0 */
    movq  %rsi, 16(%rdi)   /* r12 = entry */
    movq  %rdx, 24(%rdi)   /* r13 = arg */
    movq  %rax, 32(%rdi)   /* r14 = 0 */
    movq  %rax, 40(%rdi)   /* r15 = 0 */
    leaq  Lcmpth_entry_trampoline(%rip), %rax
    movq  %rax, 48(%rdi)   /* [ctx+48] = trampoline (first resume lands here) */

    movq  %rdi, %rax       /* return Context = ctx */
    ret

Lcmpth_entry_trampoline:
    /* Reached when the switcher's func returns after the first switch into a
     * made context.  rax = Transfer (func's return), r12 = entry, r13 = arg.
     * rsp = stack_top (16-byte aligned), ready for call.
     */
    movq  %rax, %rdi       /* rdi = Transfer (arg0) */
    movq  %r13, %rsi       /* rsi = arg (arg1) */
    callq *%r12            /* entry(Transfer, arg) -> ! */
    ud2                    /* entry must never return */
