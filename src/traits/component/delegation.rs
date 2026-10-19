//! What a delegator runs, and the handles that tie a call to its caller.
//!
//! A caller makes a request; whoever holds the delegator runs it through
//! [`DelegatorConsumer::execute`].  The request either finishes at once
//! (`Poll::Ready`) or is kept by the consumer, which takes the caller's
//! [`Completion`] out of the [`CallCx`] and completes it later.  Completing
//! a call gives a [`Ready`]: the caller, with its response in place, which
//! the consumer hands back to the delegator from
//! [`DelegatorConsumer::progress`]; only the delegator wakes it, so it
//! knows when a caller (whose queue node it may still be reading) can
//! return.

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::task::Poll;

use crate::traits::component::stackful::StackfulResumable;
use crate::traits::component::wait::Resumable;
use crate::traits::system::suspend::SuspendableSystem;

/// The resource a delegator serialises access to (an RDMA queue, an epoll
/// set, a library that is not thread-safe, ...), and the requests it runs.
pub trait DelegatorConsumer<S: SuspendableSystem>: Send + 'static {
    /// One call's input.  It may borrow from the caller's frame: the call
    /// does not return before the request has run.
    type Request<'a>: Send;
    /// One call's output.
    type Response: Send;

    /// Run `req` (called by whoever holds the delegator).  Either finish
    /// it now with `Poll::Ready`, or take the caller's completion with
    /// [`CallCx::defer`] and return `Poll::Pending`; the caller then sleeps
    /// until that completion is completed.
    fn execute(&mut self, req: Self::Request<'_>, cx: &mut CallCx<'_, S, Self::Response>) -> Poll<Self::Response>;

    /// Hand back one completed deferred call, if any (polling e.g. a
    /// completion queue for more).  Called by the holder while
    /// [`is_active`](Self::is_active), repeatedly; keep further completions
    /// for the next calls.
    fn progress(&mut self) -> Option<Ready<S>> {
        None
    }

    /// True while some deferred call has not completed.
    fn is_active(&self) -> bool {
        false
    }
}

/// A call in progress: where its caller sleeps and where its response
/// lands.  Lives in the caller's frame, which stays put until the call
/// has been completed and the caller woken.  For delegator
/// implementations.
pub struct CallSlot<S: SuspendableSystem, R> {
    sth: S::SuspendedThread,
    out: UnsafeCell<MaybeUninit<R>>,
}

impl<S: SuspendableSystem, R> Default for CallSlot<S, R> {
    fn default() -> Self {
        CallSlot { sth: Default::default(), out: UnsafeCell::new(MaybeUninit::uninit()) }
    }
}

impl<S: SuspendableSystem, R> CallSlot<S, R> {
    /// Where the caller sleeps.
    pub fn sth(&self) -> &S::SuspendedThread {
        &self.sth
    }

    /// Where the caller sleeps, for moving the sleeping thread out.
    pub fn sth_mut(&mut self) -> &mut S::SuspendedThread {
        &mut self.sth
    }

    /// The context for running this call's request.
    pub fn cx(&self) -> CallCx<'_, S, R> {
        CallCx { slot: NonNull::from(self), deferred: false, _life: PhantomData }
    }

    /// Store the response of a request that finished at once and return
    /// the caller to wake.
    ///
    /// # Safety
    /// The caller is asleep on [`sth`](Self::sth) (or will not look at the
    /// response before it is woken), and nothing else completes this call.
    pub unsafe fn finish(&self, r: R) -> Ready<S> {
        unsafe { (*self.out.get()).write(r) };
        Ready(NonNull::from(&self.sth))
    }

    /// The response, once the call has completed.
    ///
    /// # Safety
    /// The call has completed, and this is called once.
    pub unsafe fn take(&self) -> R {
        unsafe { (*self.out.get()).assume_init_read() }
    }
}

/// Handed to [`DelegatorConsumer::execute`]: lets the consumer keep the
/// call instead of finishing it.
pub struct CallCx<'a, S: SuspendableSystem, R> {
    slot: NonNull<CallSlot<S, R>>,
    deferred: bool,
    _life: PhantomData<&'a CallSlot<S, R>>,
}

impl<S: SuspendableSystem, R> CallCx<'_, S, R> {
    /// Keep the call: `execute` must then return `Poll::Pending`, and the
    /// caller sleeps until the returned completion is completed.
    pub fn defer(&mut self) -> Completion<S, R> {
        debug_assert!(!self.deferred, "CallCx::defer called twice");
        self.deferred = true;
        Completion(self.slot)
    }

    /// Whether [`defer`](Self::defer) was called (for delegators, to check
    /// that it matches what `execute` returned).
    pub fn is_deferred(&self) -> bool {
        self.deferred
    }
}

/// The right, and the duty, to finish one deferred call.  Dropping it
/// leaves the caller asleep for good.
#[must_use = "dropping a Completion leaves its caller asleep"]
pub struct Completion<S: SuspendableSystem, R>(NonNull<CallSlot<S, R>>);

// SAFETY: it stands for the response (`R`) and for waking a thread, which
// `S::SuspendedThread` allows from any thread.
unsafe impl<S: SuspendableSystem, R: Send> Send for Completion<S, R> {}

impl<S: SuspendableSystem, R> Completion<S, R> {
    /// Store the response and return the caller to wake.
    pub fn complete(self, r: R) -> Ready<S> {
        let slot = self.0.as_ptr();
        unsafe {
            (*(*slot).out.get()).write(r);
            Ready(NonNull::from(&(*slot).sth))
        }
    }
}

/// A caller whose call has completed, not yet woken.
#[must_use = "dropping a Ready leaves its caller asleep"]
pub struct Ready<S: SuspendableSystem>(NonNull<S::SuspendedThread>);

// SAFETY: as for `Completion`.
unsafe impl<S: SuspendableSystem> Send for Ready<S> {}

impl<S: SuspendableSystem> Ready<S> {
    /// Make the caller runnable.
    ///
    /// # Safety
    /// For delegators: nothing reads the caller's queue node afterwards.
    pub unsafe fn notify(self) {
        // Move the parked thread out first: once notified, the caller may
        // return and free its slot.
        let sth = std::mem::take(unsafe { &mut *self.0.as_ptr() });
        sth.notify();
    }

    /// Switch to the caller now (the current thread stays runnable).
    ///
    /// # Safety
    /// As for [`notify`](Self::notify).
    pub unsafe fn enter(self)
    where
        S::SuspendedThread: StackfulResumable<S>,
    {
        let sth = std::mem::take(unsafe { &mut *self.0.as_ptr() });
        sth.enter();
    }

    /// Whether this is the caller of `slot` (for a delegator that is still
    /// reading that caller's node, or holds the lock as that caller).
    pub fn is_for<R>(&self, slot: &CallSlot<S, R>) -> bool {
        std::ptr::eq(self.0.as_ptr(), &slot.sth)
    }

    /// Drop without waking: for a holder whose own call completed.
    pub fn forget(self) {}
}

/// A consumer that runs the closures callers hand it: plain mutual
/// exclusion with delegation.  See [`DelegatorRun::run`](crate::traits::DelegatorRun::run).
pub struct FnConsumer;

impl<S: SuspendableSystem> DelegatorConsumer<S> for FnConsumer {
    type Request<'a> = &'a mut (dyn FnMut() + Send + 'a);
    type Response = ();

    #[inline]
    fn execute(&mut self, req: Self::Request<'_>, _cx: &mut CallCx<'_, S, ()>) -> Poll<()> {
        req();
        Poll::Ready(())
    }
}
