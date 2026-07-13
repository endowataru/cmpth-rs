//! Interface traits — no implementations live here.

pub mod barrier;
pub mod delegator;
pub mod mutex;
pub mod poller;
pub mod suspended;
pub mod thread_system;

pub use barrier::{Barrier, BarrierWaitResult};
pub use delegator::{Delegator, DelegatorConsumer};
pub use mutex::{Condvar, Mutex};
pub use poller::Poller;
pub use suspended::SuspendedThread;
pub use thread_system::{JoinHandleLike, TlsAnchor, TlsSlot, ThreadSystem};
