//! Interface traits — no implementations live here.
//!
//! Organized by calling convention, not by component: [`common`] (shared
//! by both flavors), [`stackful`] (real-ULT, blocking-call), [`stackless`]
//! (`S::spawn`, `.await`-based), [`dual`] (`DualBarrier`/`DualMutex`,
//! usable from either convention on one system), [`scoped`] (the
//! `parallel_call` family, which spans both flavors in one file since it's
//! historically its own independent unit — see that module's docs). A
//! caller working in one flavor gets everything they need from one bulk
//! import:
//! `use cmpth::traits::stackful::*;` or `use cmpth::traits::stackless::*;`.

pub mod common;
pub mod dual;
pub mod scoped;
pub mod stackful;
pub mod stackless;

pub use common::{BarrierWaitResult, JoinState, Resumable, TaskDesc, TaskSystem, TlsAnchor, TlsSlot, WakeOutcome};
pub use dual::{DualBarrier, DualMutex};
pub use scoped::{ScopedStackfulTaskSystem, ScopedStacklessTaskSystem};
pub use stackful::{
    CondTransfer, Context, ContextPolicy, Delegator, DelegatorConsumer, JoinHandleLike, Poller,
    StackfulBarrier, StackfulMutex, StackfulResumable, StackfulTaskSystem, ThreadSystem, Transfer,
};
pub use stackless::{StacklessBarrier, StacklessMutex, StacklessResumable, StacklessTaskSystem, WakerTaskDesc};
