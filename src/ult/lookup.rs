//! Current-worker lookup policy.
//!
//! How code running on a ULT locates its own [`UltWorker`] (the basis of
//! `spawn`, `join`, `yield_now`, …).  Two implementations:
//!
//! * [`TlsCurrent`] — read the per-system OS-TLS slot (the classic way).
//! * [`SpCurrent`] — derive the task descriptor from the **stack pointer**:
//!   with [`ArenaStack`](crate::ult::stack::ArenaStack), stacks live in one
//!   reserved region at a power-of-two stride, so `sp` maps to its cell
//!   header (`base + (sp - base & !(stride-1))`) which holds the descriptor
//!   pointer, and the descriptor records the worker that resumed it.
//!   Falls back to TLS when `sp` is outside the arena (scheduler loop,
//!   external threads, heap-stack systems) or the descriptor belongs to a
//!   different system (nested schedulers).
//!
//! `SpCurrent` has a structural advantage over TLS beyond speed: the lookup
//! is safe to inline.  An inlined TLS read can be CSE'd by the compiler
//! across a context switch (see `OsTls`), but `sp` reads via `asm!` are
//! never merged, and the answer changes *with* the stack, not with the OS
//! thread.

use crate::traits::thread_system::TlsSlot;
use crate::ult::stack::slot_from_sp;
use crate::ult::system::UltSchedulerSystem;
use crate::ult::worker::UltWorker;

/// Policy for [`Worker::current`](crate::ult::worker::Worker::current).
/// Selected per system via
/// [`UltSchedulerSystem::Lookup`].
pub trait CurrentLookup<S: UltSchedulerSystem>: Send + Sync + 'static {
    fn current() -> Option<&'static UltWorker<S>>;
}

/// A unique identity for system `S`: the address of its `worker_tls` static.
#[inline]
pub(crate) fn system_id<S: UltSchedulerSystem>() -> *const () {
    S::worker_tls() as *const _ as *const ()
}

// ---------------------------------------------------------------------------
// TlsCurrent
// ---------------------------------------------------------------------------

/// Look the worker up in the per-system OS-TLS slot.
pub struct TlsCurrent;

impl<S: UltSchedulerSystem> CurrentLookup<S> for TlsCurrent {
    #[inline]
    fn current() -> Option<&'static UltWorker<S>> {
        let p = TlsSlot::get(S::worker_tls());
        if p.is_null() { None } else { Some(unsafe { &*p }) }
    }
}

// ---------------------------------------------------------------------------
// SpCurrent
// ---------------------------------------------------------------------------

/// Derive the worker from the stack pointer (requires
/// [`ArenaStack`](crate::ult::stack::ArenaStack); falls back to TLS
/// otherwise).
pub struct SpCurrent;

#[inline(always)]
fn current_sp() -> usize {
    let sp: usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags));
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!("mov {}, rsp", out(reg) sp, options(nomem, nostack, preserves_flags));
    }
    sp
}

impl<S: UltSchedulerSystem> CurrentLookup<S> for SpCurrent {
    #[inline]
    fn current() -> Option<&'static UltWorker<S>> {
        if let Some(slot) = slot_from_sp(current_sp()) {
            // The slot must belong to this system: on nested schedulers,
            // sp finds the *innermost* stack, whose worker is the wrong
            // type for an outer-layer call.
            if slot.system_id.get() == system_id::<S>() {
                let wk = slot.worker.get() as *const UltWorker<S>;
                debug_assert!(!wk.is_null());
                return Some(unsafe { &*wk });
            }
        }
        // Outside the arena (scheduler loop, external thread, heap stacks)
        // or a different layer's stack.
        <TlsCurrent as CurrentLookup<S>>::current()
    }
}
