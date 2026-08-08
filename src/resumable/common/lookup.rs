//! Current-worker lookup: base trait + [`TlsCurrent`], the implementation
//! that works for every flavor. `stackless::lookup` also has a
//! flavor-specific implementation
//! ([`InlineTlsCurrent`](crate::resumable::stackless::lookup::InlineTlsCurrent)).

use crate::traits::common::TlsSlot;
use crate::resumable::common::system::WorkerSystem;

/// Policy for [`WorkerOps::current`](crate::resumable::common::worker::WorkerOps::current).
/// Selected per system via [`WorkerSystem::Lookup`]. Base-level
/// (`S: WorkerSystem`): every flavor, stackful or stackless, needs to
/// locate its current worker through this trait.
pub trait CurrentLookup<S: WorkerSystem>: Send + Sync + 'static {
    fn current() -> Option<&'static S::Worker>;
}

// ---------------------------------------------------------------------------
// TlsCurrent
// ---------------------------------------------------------------------------

/// Look the worker up in the per-system OS-TLS slot.
pub struct TlsCurrent;

impl<S: WorkerSystem> CurrentLookup<S> for TlsCurrent {
    #[inline]
    fn current() -> Option<&'static S::Worker> {
        let p = TlsSlot::get(S::worker_tls());
        if p.is_null() { None } else { Some(unsafe { &*p }) }
    }
}
