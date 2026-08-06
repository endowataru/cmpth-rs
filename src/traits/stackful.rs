//! Stackful (real-ULT, blocking-call) interface: [`ThreadSystem`],
//! [`Delegator`], [`StackfulMutex`]/[`StackfulBarrier`],
//! [`StackfulResumable`], [`Poller`], [`StackfulTaskSystem`].
//!
//! `use cmpth::traits::stackful::*;` also brings in the shared
//! [`TaskSystem`]/[`Resumable`] (re-exported from [`crate::traits::common`])
//! and [`ScopedStackfulTaskSystem`] (re-exported from
//! [`crate::traits::scoped`]) — everything a caller working purely in the
//! stackful flavor needs in one `use`.

pub use crate::traits::component::stackful::{
    CondSwitchFn, CondTransfer, Context, ContextPolicy, EntryFn, HandoffTaskDesc, Poller,
    RestoreFn, StackfulResumable, SwitchFn, Transfer,
};
pub use crate::traits::component::delegation::DelegatorConsumer;
pub use crate::traits::component::wait::Resumable;
pub use crate::traits::system::TaskSystem;
pub use crate::traits::system::block_on::BlockOnSystem;
pub use crate::traits::system::bundle::StackfulTaskSystem;
pub use crate::traits::system::delegation::{Delegator, DelegationSystem};
pub use crate::traits::system::nesting::NestableSystem;
pub use crate::traits::system::scoped::ScopedStackfulTaskSystem;
pub use crate::traits::system::stackful::{JoinHandleLike, ThreadSystem};
pub use crate::traits::system::suspend::SuspendableSystem;
pub use crate::traits::system::sync::{StackfulBarrier, StackfulMutex, StackfulSyncSystem};
