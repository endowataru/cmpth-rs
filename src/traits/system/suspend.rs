use crate::traits::component::wait::Resumable;
use crate::traits::system::TaskSystem;

/// A system with a parked-continuation ("suspended thread") type — the
/// capability [`crate::traits::system::sync`]'s mutex/barrier machinery and
/// [`crate::traits::system::delegation::DelegationSystem`] build on. Split
/// off from [`ThreadSystem`](crate::traits::system::stackful::ThreadSystem)
/// so plain `spawn`/`join` users never need to name a `SuspendedThread`
/// type.
///
/// `SuspendedThread: Resumable<Self>` is a real bound here (not just a
/// `Default + Send` marker): every consumer of `S::SuspendedThread` across
/// the delegator/mutex machinery needs `is_set`/`notify` at minimum, and
/// requiring it once at the declaration site lets every one of those call
/// sites drop its own copy-pasted `where <S as ThreadSystem>::SuspendedThread:
/// StackfulResumable<S>`-shaped clause.
pub trait SuspendableSystem: TaskSystem {
    /// Parked-continuation handle for this system.
    type SuspendedThread: Resumable<Self> + Send + Default;
}
