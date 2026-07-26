//! The scheduler modules for `spawn`/`spawn_async`-style schedulers: the
//! caller's own continuation is reified into something independently
//! resumable (a real context-switch continuation for stackful ULTs, a
//! pollable task for stackless ones) — hence the module name. See
//! [`crate::scoped`] for the sibling family where that never happens.

pub mod common;
pub mod dual;
pub mod stackful;
pub mod stackless;

pub use crate::traits::system::StackfulSystem;
pub use crate::resumable::common::thread::JoinHandle;
pub use crate::resumable::stackful::thread::spawn;
