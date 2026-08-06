/// The durable capability every wait-slot has, regardless of what kind of
/// waiter (if any) is currently parked: a real ULT continuation, a
/// registered async [`Waker`](std::task::Waker), or nothing. Unlike
/// `is_set`'s answer, which changes per instance over time, this trait
/// itself is a fixed property of the type — same spirit as `Send`/`Sync`.
pub trait Resumable<S>: Default {
    /// True if a waiter is currently parked here.
    fn is_set(&self) -> bool;

    /// Wake whatever is parked here, if anything. Cheap and direct for a
    /// real ULT continuation; goes through `Waker::wake` only when the slot
    /// actually holds a registered async waiter.
    fn notify(&self);
}
