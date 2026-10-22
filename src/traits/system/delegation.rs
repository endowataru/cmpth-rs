use std::pin::Pin;

use crate::traits::component::delegation::{DelegatorConsumer, FnConsumer, PostConsumer, PostDelegator};
use crate::traits::system::stackful::SpawnableStackfulTaskSystem;
use crate::traits::system::suspend::SuspendableSystem;

/// Delegator: a queue-based lock that serialises access to a consumer and
/// runs the requests of callers that miss the lock on their behalf.
///
/// Dropping it stops whatever runs the queue (a consumer thread, if any);
/// no call may be in progress then.
pub trait Delegator<S: SuspendableSystem, C: DelegatorConsumer<S>>: Send + Sync + 'static {
    /// Start accepting calls.  The delegator stays at the returned address
    /// until dropped, so a consumer thread can point at it from the start.
    fn start(consumer: C) -> Pin<Box<Self>>;

    /// Run `req` with exclusive access to the consumer: inline if the lock
    /// is free, otherwise by whoever holds it while this thread sleeps.
    /// Returns once the request has finished, deferred or not.
    fn call(&self, req: C::Request<'_>) -> C::Response;

    /// Run `f` on the consumer as the lock holder, waiting for the lock if
    /// it is taken (for changing the consumer's own state, e.g. adding a
    /// file descriptor).
    fn with_lock<R>(&self, f: impl FnOnce(&mut C) -> R) -> R;
}

/// Running closures through a [`Delegator`] of [`FnConsumer`].
pub trait DelegatorRun<S: SuspendableSystem>: Delegator<S, FnConsumer> {
    /// Run `f` with exclusive access, inline or delegated.
    #[inline]
    fn run<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        let mut f = Some(f);
        let mut out = None;
        self.call(&mut || out = Some((f.take().unwrap())()));
        out.unwrap()
    }
}

impl<S: SuspendableSystem, D: Delegator<S, FnConsumer>> DelegatorRun<S> for D {}

/// Everything needed to host a [`Delegator`]: `spawn`/`join` (the consumer
/// runs as a real thread/ULT) plus a `SuspendedThread` type (what a
/// delegating caller parks on).
pub trait DelegationSystem: SpawnableStackfulTaskSystem + SuspendableSystem {
    /// Delegator type for this system.
    type Delegator<C: DelegatorConsumer<Self>>: Delegator<Self, C>;
}

// ---------------------------------------------------------------------------
// Choosing a PostDelegator from what the consumer needs
// ---------------------------------------------------------------------------

/// What a [`PostConsumer`] asks its delegator to keep cheap.  The axis is
/// deliberately the only one: posting never blocks and posts carry no
/// data, so what is left to choose is who pays.
pub trait PostFavor: 'static {
    /// The delegator `Ch` picks for this favour.
    type Pick<Ch: PostChoice, C: PostConsumer>: PostDelegator<C>;
}

/// Posting first: a post is a plain store; the server pays for finding
/// the posts (it looks at every producer, posted or not).  For posts on a
/// hot path from a few producers.
pub struct FavorPosting;

/// Serving first: an idle delegator costs nothing and a serve touches
/// only producers that posted; a post may take an atomic swap, and the
/// first one after a serve also wakes a server.  For many producers or
/// rare posts.
pub struct FavorServing;

impl PostFavor for FavorPosting {
    type Pick<Ch: PostChoice, C: PostConsumer> = Ch::Posting<C>;
}

impl PostFavor for FavorServing {
    type Pick<Ch: PostChoice, C: PostConsumer> = Ch::Serving<C>;
}

/// How a system maps each [`PostFavor`] to an implementation.  A system
/// picks its choice ([`PostingSystem::PostChoice`]), so the mapping is as
/// replaceable as the rest of the system.
pub trait PostChoice: 'static {
    type Posting<C: PostConsumer>: PostDelegator<C>;
    type Serving<C: PostConsumer>: PostDelegator<C>;
}

/// A system's choice of post delegators.
pub trait PostingSystem {
    type PostChoice: PostChoice;
}

/// The post delegator system `S` picks for consumer `C`.
pub type PostDelegatorOf<S, C> =
    <<C as PostConsumer>::Favor as PostFavor>::Pick<<S as PostingSystem>::PostChoice, C>;
