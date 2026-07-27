//! Current-worker lookup: base trait + [`TlsCurrent`], the implementation
//! that works for every flavor. `stackless::lookup` also has a
//! flavor-specific implementation
//! ([`InlineTlsCurrent`](crate::resumable::stackless::lookup::InlineTlsCurrent)).

use crate::traits::common::TlsSlot;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::worker::UltWorker;

/// Policy for [`Worker::current`](crate::resumable::common::worker::Worker::current).
/// Selected per system via [`SchedulerSystem::Lookup`]. Base-level
/// (`S: SchedulerSystem`): every flavor, stackful or stackless, needs to
/// locate its current worker through this trait.
pub trait CurrentLookup<S: SchedulerSystem>: Send + Sync + 'static {
    fn current() -> Option<&'static UltWorker<S>>;
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
