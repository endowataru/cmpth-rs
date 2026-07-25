//! Interface traits — no implementations live here.

pub mod barrier;
pub mod delegator;
pub mod lock;
pub mod scoped;
pub mod poller;
pub mod thread_system;
pub mod ult_system;
pub mod wait;

pub use barrier::{BarrierWaitResult, DualBarrier, StackfulBarrier, StacklessBarrier};
pub use delegator::{Delegator, DelegatorConsumer};
pub use lock::{DualMutex, StackfulMutex, StacklessMutex};
pub use scoped::{StackfulParallelInvoke, StacklessParallelInvoke};
pub use poller::Poller;
pub use thread_system::{JoinHandleLike, TlsAnchor, TlsSlot, ThreadSystem};
pub use ult_system::{AsyncWorkerSystem, UltSystem};
pub use wait::{Resumable, StackfulResumable, StacklessResumable};

/// Bulk import for stackful (real-ULT, blocking-call) code:
/// `use cmpth::traits::stackful::*;`.
pub mod stackful {
    pub use crate::traits::{
        Delegator, DelegatorConsumer, Resumable, StackfulBarrier, StackfulMutex, StackfulResumable,
        ThreadSystem, UltSystem,
    };
}

/// Bulk import for stackless (`spawn_async`, `.await`-based) code:
/// `use cmpth::traits::stackless::*;`.
pub mod stackless {
    pub use crate::traits::{AsyncWorkerSystem, Resumable, StacklessBarrier, StacklessMutex, StacklessResumable};
}
