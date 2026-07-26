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
use crate::ult::system::{SchedulerSystem, UltSchedulerSystem};
use crate::ult::worker::UltWorker;

/// Policy for [`Worker::current`](crate::ult::worker::Worker::current).
/// Selected per system via
/// [`SchedulerSystem::Lookup`]. Base-level (`S: SchedulerSystem`): a
/// stackless-only system still needs to locate its current worker, it just
/// can't use [`SpCurrent`] (see that impl) since there is no per-task stack
/// pointer to derive it from.
pub trait CurrentLookup<S: SchedulerSystem>: Send + Sync + 'static {
    fn current() -> Option<&'static UltWorker<S>>;
}

/// Like [`SpCurrent`], but for `spawn_async`/`recurse` storage instead of a
/// real stack: `S::AsyncPool` is arena-allocated (see
/// [`AsyncArenaStack`](crate::ult::stack::AsyncArenaStack)), and a
/// `spawn_async`'d future's fields (e.g. a `JoinHandle` awaited from within
/// it) live inside that same descriptor's memory. So `self`'s own address,
/// from within `JoinHandle::poll`/`RecursionFrame::poll`, maps to the
/// enclosing task's arena cell the same way a stack pointer maps to a ULT's
/// — no TLS needed. Falls back (returns `None`) when `addr` isn't in the
/// async-task arena at all (the root future, an oversized allocation, a
/// foreign executor) or belongs to a different system's cell (nested
/// schedulers).
#[inline]
pub(crate) fn worker_from_async_arena_addr<S: SchedulerSystem>(addr: usize) -> Option<&'static UltWorker<S>> {
    let slot = crate::ult::stack::slot_from_addr::<crate::ult::stack::AsyncTaskArenaKind>(addr)?;
    if slot.system_id.get() != system_id::<S>() {
        return None;
    }
    let wk = slot.worker.get() as *const UltWorker<S>;
    if wk.is_null() {
        return None;
    }
    Some(unsafe { &*wk })
}

/// A unique identity for system `S`: the address of its `worker_tls` static.
#[inline]
pub(crate) fn system_id<S: SchedulerSystem>() -> *const () {
    S::worker_tls() as *const _ as *const ()
}

// ---------------------------------------------------------------------------
// TlsCurrent
// ---------------------------------------------------------------------------

/// Look the worker up in the per-system OS-TLS slot.
pub struct TlsCurrent;

impl<S: SchedulerSystem> CurrentLookup<S> for TlsCurrent {
    #[inline]
    fn current() -> Option<&'static UltWorker<S>> {
        let p = TlsSlot::get(S::worker_tls());
        if p.is_null() { None } else { Some(unsafe { &*p }) }
    }
}

// ---------------------------------------------------------------------------
// InlineTlsCurrent
// ---------------------------------------------------------------------------

/// Like [`TlsCurrent`], but reads the slot via [`TlsSlot::get_inline`]
/// instead of [`TlsSlot::get`] — a single inlinable TLS access instead of
/// an opaque, non-inlinable function call per lookup.
///
/// Only sound for systems that can never migrate a task across OS threads
/// mid-poll, i.e. **stackless-only** systems (`ult_async_system!`'s output,
/// which never implements `UltSchedulerSystem` and so never does a real
/// context switch). `ult_async_system!` uses this as its default `Lookup`
/// for exactly that reason. A stackful or dual config must keep using
/// [`TlsCurrent`] — see `OsTls::get`'s doc comment for the CSE hazard this
/// would otherwise reintroduce.
pub struct InlineTlsCurrent;

impl<S: SchedulerSystem> CurrentLookup<S> for InlineTlsCurrent {
    #[inline]
    fn current() -> Option<&'static UltWorker<S>> {
        let p = TlsSlot::get_inline(S::worker_tls());
        if p.is_null() { None } else { Some(unsafe { &*p }) }
    }
}

// ---------------------------------------------------------------------------
// SpCurrent
// ---------------------------------------------------------------------------

/// Derive the worker from the stack pointer (requires
/// [`ArenaStack`](crate::ult::stack::ArenaStack); falls back to TLS
/// otherwise). Inherently stackful: only implemented for
/// `S: UltSchedulerSystem`, since a stackless-only system has no per-task
/// stack pointer to derive a worker from.
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
    S::Desc: crate::ult::desc::StackfulTaskDesc,
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
