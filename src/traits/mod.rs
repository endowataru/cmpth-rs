//! Interface traits — no implementations live here.
//!
//! Split by audience: [`system`] (what a library *user* calls — complete
//! system interfaces like `ThreadSystem`/`StacklessTaskSystem`) vs.
//! [`component`] (what an *implementer* plugs in to assemble one — task
//! descriptors, resumable wait-slots, context-switch policies, TLS slots).
//!
//! The former per-calling-convention modules ([`common`], [`stackful`],
//! [`stackless`], [`dual`], [`scoped`]) still exist as compatibility
//! facades over the [`system`]/[`component`] split above: [`stackful`]
//! (real-ULT, blocking-call) and [`stackless`] (`S::spawn`, `.await`-based)
//! each bulk-re-export everything a caller working in one flavor needs from
//! one `use`:
//! `use cmpth::traits::stackful::*;` or `use cmpth::traits::stackless::*;`.

pub mod component;
pub mod system;

pub mod common;
pub mod dual;
pub mod scoped;
pub mod stackful;
pub mod stackless;

pub use component::task::{TaskDesc, TaskExitSink};
pub use component::tls::{TlsAnchor, TlsSlot};
pub use component::wait::Resumable;
pub use system::TaskSystem;
pub use system::sync::{BarrierWaitResult, DualBarrier, DualMutex};
pub use system::scoped::{ScopedStackfulTaskSystem, ScopedStacklessTaskSystem};
pub use component::delegation::DelegatorConsumer;
pub use component::stackful::{
    CondTransfer, Context, ContextPolicy, HandoffTaskDesc, Poller, StackfulResumable, Transfer,
};
pub use system::block_on::BlockOnSystem;
pub use system::bundle::StackfulTaskSystem;
pub use system::delegation::{Delegator, DelegationSystem};
pub use system::nesting::NestableSystem;
pub use system::stackful::{JoinHandleLike, StackfulBuilder, StackfulInitSystem, ThreadSystem};
pub use system::suspend::SuspendableSystem;
pub use system::sync::{StackfulBarrier, StackfulMutex, StackfulSyncSystem};
pub use system::stackless::{StacklessBuilder, StacklessInitSystem, StacklessTaskSystem};
pub use system::sync::{StacklessBarrier, StacklessMutex, StacklessSyncSystem};
pub use component::stackless::{StacklessResumable, WakerTaskDesc};
