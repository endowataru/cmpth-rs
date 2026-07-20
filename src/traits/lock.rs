//! [`StackfulMutex`]/[`StacklessMutex`]/[`DualMutex`] — same-named
//! stackful/stackless mutex traits.
//!
//! (Not named after the file: kept in `lock.rs` rather than `mutex.rs` to
//! avoid colliding with the older `traits::mutex` module, which still
//! holds the pre-existing `Mutex`/`Condvar` pair — see that module's docs.)
//!
//! Same disambiguation pattern as [`crate::traits::StackfulResumable`]/
//! [`crate::traits::StacklessResumable`]: both traits define a method
//! literally named `lock`; which one resolves at a call site depends on
//! which trait is `use`d there, not on a `_async` suffix.
//!
//! Each carries its own `new`, replacing the older `traits::Mutex`'s role
//! (which also paired a `Condvar` type — never used generically through
//! `S::Mutex`, only via concrete types like `McsCondvar`, so it's not
//! carried over here).
//!
//! The interface owns the name here, not the implementation: `DualMutex`
//! is the trait; the concrete generic-over-N type (`ult::sync::DualMutex`)
//! is re-exported under an alias (`UltDualMutex`) at the crate root to make
//! room, the same pattern already used for `Barrier`/`UltBarrier`.

use std::future::Future;
use std::ops::DerefMut;

pub trait StackfulMutex<T: Send>: Sized + Send + Sync {
    type Guard<'a>: DerefMut<Target = T> + 'a
    where
        Self: 'a,
        T: 'a;

    fn new(val: T) -> Self;

    fn lock(&self) -> Self::Guard<'_>;
}

pub trait StacklessMutex<T: Send>: Sized + Send + Sync {
    type Guard<'a>: DerefMut<Target = T> + 'a
    where
        Self: 'a,
        T: 'a;

    fn new(val: T) -> Self;

    fn lock<'a>(&'a self) -> impl Future<Output = Self::Guard<'a>> + Send
    where
        T: 'a;
}

/// A mutex usable from either calling convention. Blanket-derived: any type
/// implementing both flavors gets this for free, so it exists purely as a
/// convenience bound for generic code that wants "works either way" as one
/// name (`S::Mutex: DualMutex<T>`) instead of spelling out both traits.
pub trait DualMutex<T: Send>: StackfulMutex<T> + StacklessMutex<T> {}

impl<T: Send, M: StackfulMutex<T> + StacklessMutex<T>> DualMutex<T> for M {}
