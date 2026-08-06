use crate::traits::component::tls::TlsSlot;
use crate::traits::system::TaskSystem;

/// A system that can host a nested scheduler on top of it — the capability
/// a nested `ThreadSystem`'s `worker_tls` needs from its `Base`. Split off
/// from [`ThreadSystem`](crate::traits::system::stackful::ThreadSystem) so a
/// system that is never used as a nesting base doesn't need to name a
/// `ThreadSpecific` slot type.
pub trait NestableSystem: TaskSystem {
    /// Thread-specific storage slot: one `*mut T` per thread (or per ULT) of
    /// this system.  A nested scheduler stores its per-worker pointer here,
    /// which is why a single slot per level is enough — everything else is
    /// reached through the worker pointer.
    type ThreadSpecific<T: 'static>: TlsSlot<T>;
}
