//! Stackless (`spawn_async`, `.await`-based) interface: [`StacklessMutex`]/
//! [`StacklessBarrier`], [`StacklessResumable`], [`StacklessTaskSystem`].
//!
//! `use cmpth::traits::stackless::*;` also brings in the shared
//! [`TaskSystem`]/[`Resumable`] (re-exported from [`crate::traits::common`])
//! and [`ScopedStacklessTaskSystem`] (re-exported from
//! [`crate::traits::scoped`]) — everything a caller working purely in the
//! stackless flavor needs in one `use`.

pub use crate::traits::component::stackless::{StacklessResumable, WakerTaskDesc};
pub use crate::traits::component::wait::Resumable;
pub use crate::traits::system::TaskSystem;
pub use crate::traits::system::scoped::ScopedStacklessTaskSystem;
pub use crate::traits::system::stackless::StacklessTaskSystem;
pub use crate::traits::system::sync::{StacklessBarrier, StacklessMutex, StacklessSyncSystem};
