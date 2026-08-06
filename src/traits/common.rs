//! Shared root types used by both the stackful and stackless flavors:
//! [`TaskSystem`], [`TaskDesc`], [`Resumable`], [`TlsAnchor`]/[`TlsSlot`].
//! See [`crate::traits::dual`] for [`DualMutex`](crate::traits::dual::DualMutex)/
//! [`DualBarrier`](crate::traits::dual::DualBarrier).

pub use crate::traits::component::task::{TaskDesc, TaskExitSink};
pub use crate::traits::component::tls::{TlsAnchor, TlsSlot};
pub(crate) use crate::traits::component::tls::TLS_ANCHOR_UNASSIGNED;
pub use crate::traits::component::wait::Resumable;
pub use crate::traits::system::TaskSystem;
pub use crate::traits::system::sync::BarrierWaitResult;
