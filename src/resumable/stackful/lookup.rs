//! Stack-pointer-derived worker lookup ([`SpCurrent`]) — see
//! [`crate::resumable::common::lookup`] for the base trait and the
//! TLS-based implementation every flavor can use.

use crate::resumable::common::lookup::{system_id, CurrentLookup, TlsCurrent};
use crate::resumable::stackful::stack::slot_from_sp;
use crate::resumable::stackful::system::UltSchedulerSystem;
use crate::resumable::common::worker::UltWorker;

/// Derive the worker from the stack pointer (requires
/// [`ArenaStack`](crate::resumable::stackful::stack::ArenaStack); falls back to TLS
/// otherwise). Inherently stackful: only implemented for
/// `S: UltSchedulerSystem`, since a stackless-only system has no per-task
/// stack pointer to derive a worker from.
///
/// Has a structural advantage over TLS beyond speed: the lookup is safe to
/// inline.  An inlined TLS read can be CSE'd by the compiler across a
/// context switch (see `OsTls`), but `sp` reads via `asm!` are never
/// merged, and the answer changes *with* the stack, not with the OS
/// thread.
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

impl<S: UltSchedulerSystem> CurrentLookup<S> for SpCurrent
where
    S::Desc: crate::resumable::stackful::desc::StackfulTaskDesc,
{
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
