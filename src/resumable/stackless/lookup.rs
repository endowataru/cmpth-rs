//! Poll-safe worker lookups for stackless code: [`InlineTlsCurrent`] (an
//! inlinable TLS read, sound only because a stackless-only system never
//! migrates a task across OS threads mid-poll) and
//! `worker_from_async_arena_addr` (derive the worker from an address
//! inside `spawn_async`/`recurse` arena storage — the async analogue of
//! [`stackful::lookup::SpCurrent`](crate::resumable::stackful::lookup::SpCurrent)'s
//! stack-pointer trick).

use crate::traits::common::TlsSlot;
use crate::resumable::common::lookup::{system_id, CurrentLookup};
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::worker::UltWorker;

// ---------------------------------------------------------------------------
// InlineTlsCurrent
// ---------------------------------------------------------------------------

/// Like [`TlsCurrent`](crate::resumable::common::lookup::TlsCurrent), but
/// reads the slot via [`TlsSlot::get_inline`] instead of [`TlsSlot::get`] —
/// a single inlinable TLS access instead of an opaque, non-inlinable
/// function call per lookup.
///
/// Only sound for systems that can never migrate a task across OS threads
/// mid-poll, i.e. **stackless-only** systems (a
/// [`UltAsyncIdentity`](crate::resumable::stackless::system::UltAsyncIdentity)
/// implementor, which never implements `StackfulSchedulerSystem` and so
/// never does a real context switch). The natural `Lookup` choice for
/// exactly that reason. A stackful or dual config must keep using
/// `TlsCurrent` — see `OsTls::get`'s doc comment for the CSE hazard this
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
// worker_from_async_arena_addr
// ---------------------------------------------------------------------------

/// Like `SpCurrent`, but for `spawn_async`/`recurse` storage instead of a
/// real stack: `S::AsyncPool` is arena-allocated (see
/// [`AsyncArenaStack`](crate::resumable::stackless::stack::AsyncArenaStack)), and a
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
    let slot = crate::resumable::common::stack::slot_from_addr::<crate::resumable::stackless::stack::AsyncTaskArenaKind>(addr)?;
    if slot.system_id.get() != system_id::<S>() {
        return None;
    }
    let wk = slot.worker.get() as *const UltWorker<S>;
    if wk.is_null() {
        return None;
    }
    Some(unsafe { &*wk })
}
