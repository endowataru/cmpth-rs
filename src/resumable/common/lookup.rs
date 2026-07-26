//! Current-worker lookup: base trait + [`TlsCurrent`], the implementation
//! that works for every flavor. Flavor-specific implementations live in
//! `stackful::lookup` ([`SpCurrent`](crate::resumable::stackful::lookup::SpCurrent))
//! and `stackless::lookup` ([`InlineTlsCurrent`](crate::resumable::stackless::lookup::InlineTlsCurrent)).

use crate::traits::thread_system::TlsSlot;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::worker::UltWorker;

/// Policy for [`Worker::current`](crate::resumable::common::worker::Worker::current).
/// Selected per system via
/// [`SchedulerSystem::Lookup`]. Base-level (`S: SchedulerSystem`): a
/// stackless-only system still needs to locate its current worker, it just
/// can't use `SpCurrent` (see `stackful::lookup`) since there is no
/// per-task stack pointer to derive it from.
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
