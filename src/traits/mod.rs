//! Interface traits — no implementations live here.
//!
//! Organized by calling convention, not by component: [`common`] (shared
//! by both flavors), [`stackful`] (real-ULT, blocking-call), [`stackless`]
//! (`spawn_async`, `.await`-based), [`scoped`] (the `parallel_call` family,
//! which spans both flavors in one file since it's historically its own
//! independent unit — see that module's docs). A caller working in one
//! flavor gets everything they need from one bulk import:
//! `use cmpth::traits::stackful::*;` or `use cmpth::traits::stackless::*;`.

pub mod common;
pub mod scoped;
pub mod stackful;
pub mod stackless;

pub use common::{BarrierWaitResult, DualBarrier, DualMutex, JoinState, Resumable, TaskDesc, TaskSystem, TlsAnchor, TlsSlot, WakeOutcome};
pub use scoped::{ScopedStackfulTaskSystem, ScopedStacklessTaskSystem};
pub use stackful::{
    Delegator, DelegatorConsumer, JoinHandleLike, Poller, StackfulBarrier, StackfulMutex,
    StackfulResumable, StackfulTaskSystem, ThreadSystem,
};
pub use stackless::{StacklessBarrier, StacklessMutex, StacklessResumable, StacklessTaskSystem};
