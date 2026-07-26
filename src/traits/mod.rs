//! Interface traits — no implementations live here.

pub mod barrier;
pub mod delegator;
pub mod lock;
pub mod scoped;
pub mod poller;
pub mod system;
pub mod thread_system;
pub mod wait;

pub use barrier::{BarrierWaitResult, DualBarrier, StackfulBarrier, StacklessBarrier};
pub use delegator::{Delegator, DelegatorConsumer};
pub use lock::{DualMutex, StackfulMutex, StacklessMutex};
pub use scoped::{ScopedStackfulTaskSystem, ScopedStacklessTaskSystem};
pub use poller::Poller;
pub use system::{StackfulSystem, StackfulTaskSystem, StacklessSystem, StacklessTaskSystem};
pub use thread_system::{JoinHandleLike, TaskSystem, TlsAnchor, TlsSlot, ThreadSystem};
pub use wait::{Resumable, StackfulResumable, StacklessResumable};

/// Bulk import for stackful (real-ULT, blocking-call) code:
/// `use cmpth::traits::stackful::*;`.
pub mod stackful {
    pub use crate::traits::{
        Delegator, DelegatorConsumer, JoinHandleLike, Resumable, ScopedStackfulTaskSystem, StackfulBarrier,
        StackfulMutex, StackfulResumable, StackfulSystem, StackfulTaskSystem, TaskSystem, ThreadSystem,
    };
}

/// Bulk import for stackless (`spawn_async`, `.await`-based) code:
/// `use cmpth::traits::stackless::*;`.
pub mod stackless {
    pub use crate::traits::{
        Resumable, ScopedStacklessTaskSystem, StacklessBarrier, StacklessMutex, StacklessResumable,
        StacklessSystem, StacklessTaskSystem, TaskSystem,
    };
}
