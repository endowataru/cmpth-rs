//! [`Mutex`] and [`Condvar`] — thread-system-agnostic sync primitives.

use std::ops::DerefMut;

/// Mutex abstraction with a GAT guard tied to the borrow lifetime.
///
/// Each `Mutex` has an associated [`Condvar`] type so that the two are
/// always type-compatible (matching the ComposableThreads pairing model).
pub trait Mutex<T: Send>: Sized + Send + Sync {
    type Guard<'a>: DerefMut<Target = T> + 'a
    where
        Self: 'a,
        T: 'a;
    /// Condition variable that operates on guards of this mutex type.
    type Condvar: Condvar<Self, T> + Send + Sync;

    fn new(val: T) -> Self;
    fn lock(&self) -> Self::Guard<'_>;
}

/// Condition variable paired with mutex type `M`.
pub trait Condvar<M, T>: Sized + Send + Sync
where
    M: Mutex<T>,
    T: Send,
{
    fn new() -> Self;
    /// Release the guard, wait for a notification, re-acquire and return guard.
    fn wait<'a>(&self, guard: M::Guard<'a>) -> M::Guard<'a>
    where
        M: 'a,
        T: 'a;
    fn notify_one(&self);
    fn notify_all(&self);
}
