//! Poll-safe worker lookup for stackless code: [`InlineTlsCurrent`], an
//! inlinable TLS read, sound only because a stackless-only system never
//! migrates a task across OS threads mid-poll.

use crate::traits::common::TlsSlot;
use crate::resumable::common::lookup::CurrentLookup;
use crate::resumable::common::system::SchedulerSystem;

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
    fn current() -> Option<&'static S::Worker> {
        let p = TlsSlot::get_inline(S::worker_tls());
        if p.is_null() { None } else { Some(unsafe { &*p }) }
    }
}
