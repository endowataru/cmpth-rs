//! `CombiningDelegator` — a delegator without a consumer thread: whoever
//! holds the lock runs the requests of the callers queued behind it, then
//! wakes them.
//!
//! An MCS queue of nodes in the callers' frames.  A caller that finds the
//! queue empty runs its request itself; one that finds it busy links its
//! node behind its predecessor and sleeps.  The holder walks the queue,
//! running each successor's request and waking it, until it can swing the
//! tail back to empty.  A caller is woken only once its node's successor
//! is linked (or it was the last), so the holder never touches a frame that
//! has returned; that holds for deferred calls too, since completed ones
//! come back to the holder through `progress`.  A [`with_lock`](crate::traits::Delegator::with_lock) caller
//! is handed the lock instead, and drains the queue from its own node.
//!
//! Nothing runs when nobody calls: `progress` is polled by the holder once
//! per step while `is_active`, and the holder keeps the lock (polling)
//! until no deferred call is left.  Because no thread or ULT is
//! spawned, it works the same for ULTs and for OS threads (as far as
//! `S::SuspendedThread` parks both), and before any worker pool exists.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::task::Poll;

use crate::traits::{CallSlot, Delegator as DelegatorTrait, DelegatorConsumer, Ready, Resumable, StackfulResumable, SuspendableSystem};

/// A caller's queue node, with everything the holder touches for it in
/// one place: its request (written only when it is queued) and its slot.
///
/// `next` carries [`LOCK`] in its low bit when the successor is a
/// `with_lock` caller (asleep on its slot's thread, with no request): a
/// flag in the node would cost one more store before the tail swap.
struct Node<S: SuspendableSystem, C: DelegatorConsumer<S>> {
    next: AtomicPtr<Node<S, C>>,
    req: MaybeUninit<C::Request<'static>>,
    slot: CallSlot<S, C::Response>,
}

const LOCK: usize = 1;

impl<S: SuspendableSystem, C: DelegatorConsumer<S>> Node<S, C> {
    #[inline]
    fn new() -> Self {
        Node { next: AtomicPtr::new(null_mut()), req: MaybeUninit::uninit(), slot: CallSlot::default() }
    }
}

/// On its own cache line: every caller swaps `tail`.
#[repr(align(128))]
pub struct CombiningDelegator<S: SuspendableSystem, C: DelegatorConsumer<S>> {
    tail: AtomicPtr<Node<S, C>>,
    consumer: UnsafeCell<C>,
}

// SAFETY: the consumer is only touched by the lock holder.
unsafe impl<S: SuspendableSystem, C: DelegatorConsumer<S>> Send for CombiningDelegator<S, C> {}
unsafe impl<S: SuspendableSystem, C: DelegatorConsumer<S>> Sync for CombiningDelegator<S, C> {}

impl<S, C> CombiningDelegator<S, C>
where
    S: SuspendableSystem,
    S::SuspendedThread: StackfulResumable<S>,
    C: DelegatorConsumer<S>,
{
    pub fn new(consumer: C) -> Self {
        CombiningDelegator { tail: AtomicPtr::new(null_mut()), consumer: UnsafeCell::new(consumer) }
    }

    /// Holder only.
    #[allow(clippy::mut_from_ref)]
    fn consumer(&self) -> &mut C {
        unsafe { &mut *self.consumer.get() }
    }

    /// As the holder whose node is `me`: run the queued requests behind it
    /// until the queue is released (returns null) or handed to a
    /// `with_lock` caller (returns its node).  `own` is the
    /// holder's own deferred call, if any; the queue is not released before
    /// it completes, and the second result says whether it still has not.
    #[inline]
    fn drain(&self, me: *mut Node<S, C>, mut own: Option<&CallSlot<S, C::Response>>) -> (*mut Node<S, C>, bool) {
        let mut cur = me;
        // The caller to wake for `cur` (none for our own node).
        let mut cur_wake: Option<Ready<S>> = None;
        loop {
            let con = self.consumer();
            if con.is_active() {
                while let Some(r) = con.progress() {
                    if own.is_some_and(|o| r.is_for(o)) {
                        r.forget();
                        own = None;
                    } else if cur != me && r.is_for(unsafe { &(*cur).slot }) {
                        // Its caller may not return before we leave `cur`.
                        cur_wake = Some(r);
                    } else {
                        unsafe { r.notify() };
                    }
                }
            }
            let mut next = unsafe { (*cur).next.load(Ordering::Acquire) };
            if next.is_null() {
                if own.is_some() || con.is_active() {
                    // Deferred calls to poll before the queue can be released.
                    std::hint::spin_loop();
                    continue;
                }
                if self.tail.compare_exchange(cur, null_mut(), Ordering::AcqRel, Ordering::Acquire).is_ok() {
                    if let Some(w) = cur_wake {
                        unsafe { w.notify() };
                    }
                    return (null_mut(), false);
                }
                // A successor swapped the tail but has not linked yet: wait
                // for the link only (not retrying the CAS, which would take
                // the tail's line from the callers enqueueing).
                loop {
                    next = unsafe { (*cur).next.load(Ordering::Acquire) };
                    if !next.is_null() {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
            // `cur`'s successor is known: `cur` may return now.
            if let Some(w) = cur_wake.take() {
                unsafe { w.notify() };
            }
            if next.addr() & LOCK != 0 {
                return (next.map_addr(|a| a & !LOCK), own.is_some());
            }
            let n = unsafe { &*next };
            let req = unsafe { n.req.assume_init_read() };
            let slot = &n.slot;
            let mut cx = slot.cx();
            match self.consumer().execute(req, &mut cx) {
                Poll::Ready(v) => {
                    debug_assert!(!cx.is_deferred(), "CombiningDelegator: deferred and finished");
                    cur_wake = Some(unsafe { slot.finish(v) });
                }
                Poll::Pending => debug_assert!(cx.is_deferred(), "CombiningDelegator: pending without defer"),
            }
            cur = next;
        }
    }

    /// Hold the lock from node `me` until it is released or handed over;
    /// returns once `own` (if any) has completed.
    #[inline]
    fn hold(&self, me: *mut Node<S, C>, own: Option<&CallSlot<S, C::Response>>) {
        let (next, pending) = self.drain(me, own);
        if next.is_null() {
            return;
        }
        // Moved out first: once woken, its caller may return.
        let wake = move || std::mem::take(unsafe { &mut *(*next).slot.sth_mut() }).notify();
        match own {
            // Still waiting for our own call: sleep before the next holder
            // can complete it.
            Some(o) if pending => o.sth().wait_with(wake),
            _ => wake(),
        }
    }

    /// Enqueue `me`; true if the queue was empty (we hold the lock).
    /// Otherwise runs `fill` and sleeps until woken; `lock` marks a
    /// `with_lock` caller.
    #[inline]
    fn enqueue(&self, me: *mut Node<S, C>, lock: bool, fill: impl FnOnce()) -> bool {
        let prev = self.tail.swap(me, Ordering::AcqRel);
        if prev.is_null() {
            return true;
        }
        fill();
        // Link behind `prev` only once asleep: the holder may run our
        // request and wake us as soon as it sees the link.
        let link = if lock { me.map_addr(|a| a | LOCK) } else { me };
        unsafe { &(*me).slot }.sth().wait_with(move || unsafe { (*prev).next.store(link, Ordering::Release) });
        false
    }
}

impl<S, C> DelegatorTrait<S, C> for CombiningDelegator<S, C>
where
    S: SuspendableSystem,
    S::SuspendedThread: StackfulResumable<S>,
    C: DelegatorConsumer<S>,
{
    fn start(consumer: C) -> Pin<Box<Self>> {
        Box::pin(Self::new(consumer))
    }

    #[inline]
    fn call(&self, req: C::Request<'_>) -> C::Response {
        let mut node = Node::<S, C>::new();
        let me: *mut Node<S, C> = &mut node;
        let mut req = Some(req);
        let queued = !self.enqueue(me, false, || {
            // SAFETY: only the lifetime is erased; the holder moves the
            // request out before waking us.
            unsafe { (*me).req.as_mut_ptr().cast::<C::Request<'_>>().write(req.take().unwrap_unchecked()) };
        });
        if !queued {
            let req = unsafe { req.take().unwrap_unchecked() };
            let slot = unsafe { &(*me).slot };
            let mut cx = slot.cx();
            match self.consumer().execute(req, &mut cx) {
                Poll::Ready(v) => {
                    self.hold(me, None);
                    return v;
                }
                Poll::Pending => self.hold(me, Some(slot)),
            }
        }
        unsafe { (*me).slot.take() }
    }

    fn with_lock<R>(&self, f: impl FnOnce(&mut C) -> R) -> R {
        let mut node = Node::<S, C>::new();
        let me: *mut Node<S, C> = &mut node;
        // Either way we hold the lock afterwards.
        self.enqueue(me, true, || {});
        let r = f(self.consumer());
        self.hold(me, None);
        r
    }
}
