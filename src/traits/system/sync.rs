use std::future::Future;
use std::ops::DerefMut;

/// Return value of [`StackfulBarrier::wait`], mirroring
/// `std::sync::BarrierWaitResult`.
pub struct BarrierWaitResult {
    pub is_leader: bool,
}

impl BarrierWaitResult {
    pub fn is_leader(&self) -> bool { self.is_leader }
}

/// [`StackfulMutex`]/[`StacklessMutex`] —
/// same-named stackful/stackless mutex traits.
///
/// Same disambiguation pattern as [`StackfulResumable`](crate::traits::component::stackful::StackfulResumable)/
/// [`StacklessResumable`](crate::traits::component::stackless::StacklessResumable): both
/// traits define a method literally named `lock`; which one resolves at a
/// call site depends on which trait is `use`d there, not on a `_async`
/// suffix.
///
/// Each carries its own `new`. There is no generic `Condvar` trait: it was
/// never used generically through `S::Mutex`, only via concrete types like
/// `McsCondvar`, so pairing types (`McsMutex`/`McsCondvar`,
/// `OsMutex`/`OsCondvar`, …) expose their condvar as an inherent type with
/// inherent methods instead.
///
/// The interface owns the name here, not the implementation:
/// [`DualMutex`] is the trait; the concrete
/// generic-over-N type (`resumable::common::sync::DualMutex`) is
/// re-exported under an alias (`UltDualMutex`) at the crate root to make
/// room, the same pattern already used for `Barrier`/`UltBarrier`.
pub trait StackfulMutex<T: Send>: Sized + Send + Sync {
    type Guard<'a>: DerefMut<Target = T> + 'a
    where
        Self: 'a,
        T: 'a;

    fn new(val: T) -> Self;

    fn lock(&self) -> Self::Guard<'_>;
}

/// Stackful/stackless-flavored barrier `wait`, same disambiguation pattern
/// as [`StackfulMutex`]/[`StacklessMutex`]:
/// both traits define a method literally named `wait`, resolved by which
/// trait is `use`d at the call site. Each carries its own `new`, same
/// reasoning as `StackfulMutex`/`StacklessMutex`.
pub trait StackfulBarrier: Sized + Send + Sync {
    fn new(count: usize) -> Self;
    fn wait(&self) -> BarrierWaitResult;
}

/// [`StackfulMutex`]/`StacklessMutex` —
/// same-named stackful/stackless mutex traits.
///
/// Same disambiguation pattern as
/// [`StackfulResumable`](crate::traits::component::stackful::StackfulResumable)/
/// [`StacklessResumable`](crate::traits::component::stackless::StacklessResumable): both traits define a method literally named
/// `lock`; which one resolves at a call site depends on which trait is
/// `use`d there, not on a `_async` suffix.
///
/// Each carries its own `new`. There is no generic `Condvar` trait: it was
/// never used generically through `S::Mutex`, only via concrete types like
/// `McsCondvar`, so pairing types (`McsMutex`/`McsCondvar`,
/// `OsMutex`/`OsCondvar`, …) expose their condvar as an inherent type with
/// inherent methods instead.
///
/// The interface owns the name here, not the implementation:
/// [`DualMutex`] is the trait; the concrete
/// generic-over-N type (`resumable::common::sync::DualMutex`) is
/// re-exported under an alias (`UltDualMutex`) at the crate root to make
/// room, the same pattern already used for `Barrier`/`UltBarrier`.
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

/// Stackful/stackless-flavored barrier `wait`, same disambiguation pattern
/// as [`StacklessMutex`]/[`StackfulMutex`]:
/// both traits define a method literally named `wait`, resolved by which
/// trait is `use`d at the call site. Each carries its own `new`, same
/// reasoning as `StackfulMutex`/`StacklessMutex`.
pub trait StacklessBarrier: Sized + Send + Sync {
    fn new(count: usize) -> Self;
    fn wait<'a>(&'a self) -> impl Future<Output = BarrierWaitResult> + Send + 'a;
}

/// A barrier usable from either calling convention — see [`DualMutex`] for
/// the same pattern applied to mutexes. The interface owns the name here
/// too: the concrete generic-over-N type
/// (`resumable::common::sync::DualBarrier`) is re-exported under an alias
/// (`UltDualBarrier`) at the crate root to make room.
pub trait DualBarrier: Sized + Send + Sync + StackfulBarrier + StacklessBarrier {}

impl<M: StackfulBarrier + StacklessBarrier> DualBarrier for M {}

/// A mutex usable from either calling convention. Blanket-derived: any type
/// implementing both flavors gets this for free, so it exists purely as a
/// convenience bound for generic code that wants "works either way" as one
/// name (`S::Mutex: DualMutex<T>`) instead of spelling out both traits.
pub trait DualMutex<T: Send>: StackfulMutex<T> + StacklessMutex<T> {}

impl<T: Send, M: StackfulMutex<T> + StacklessMutex<T>> DualMutex<T> for M {}
