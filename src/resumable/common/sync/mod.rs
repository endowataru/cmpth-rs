//! Sync primitives generic over the wait-slot flavor `N` (`StackfulResumable`/
//! `StacklessResumable`), rather than hardcoded to one calling convention —
//! see [`dual_mutex`]'s module doc. Common (not `stackful`-only or
//! `stackless`-only) because `DualMutex<S, T, N>`/`DualBarrier<S, N>`
//! conditionally implement `StackfulMutex`/`StacklessMutex` independently
//! depending on what `N` provides; today's only concrete user
//! (`SuspendedTask<S>`, satisfying both) happens to make them dual in
//! practice, but nothing here requires that.

pub mod dual_barrier;
pub mod dual_mutex;

pub use dual_barrier::DualBarrier;
pub use dual_mutex::{DualMutex, DualMutexGuard};
