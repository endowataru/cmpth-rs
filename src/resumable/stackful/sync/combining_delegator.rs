//! `CombiningDelegator` — a delegator without a consumer thread: whoever
//! holds the lock runs the work of the callers queued behind it, then
//! wakes them.
//!
//! An MCS queue of nodes in the callers' frames.  A caller that finds the
//! queue empty runs `imm` itself; one that finds it busy writes its work
//! with `del`, links itself behind its predecessor and parks on the
//! `SuspendedThread` that `del` returned.  The holder walks the queue,
//! executing each successor's work through [`DelegatorConsumer::execute`]
//! and waking it, until it can swing the tail back to empty.  A node is
//! woken only once its own successor is linked (or it was the last), so
//! the holder never touches a frame that has returned.
//!
//! Nothing runs when nobody calls: `progress` is polled by the holder once
//! per executed item while `is_active`, not in the background.  Because no
//! thread or ULT is spawned, it works the same for ULTs and for OS threads
//! (as far as `S::SuspendedThread` parks both), and before any worker pool
//! exists.

use std::cell::UnsafeCell;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::traits::{Delegator as DelegatorTrait, DelegatorConsumer, Resumable, StackfulResumable, SuspendableSystem};

struct Node<S: SuspendableSystem, C: DelegatorConsumer<S>> {
    next: AtomicPtr<Node<S, C>>,
    work: UnsafeCell<C::Work>,
}

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
    /// Holder only.
    #[allow(clippy::mut_from_ref)]
    fn consumer(&self) -> &mut C {
        unsafe { &mut *self.consumer.get() }
    }

    fn poll_progress(&self) {
        let con = self.consumer();
        if con.is_active() {
            if let Some(w) = con.progress() {
                w.notify();
            }
        }
    }

    /// As the holder whose node is `me`: run the queued work behind it,
    /// then release the queue.
    fn drain(&self, me: *mut Node<S, C>) {
        let mut cur = me;
        // The thread to wake for `cur` (none for our own node).
        let mut cur_wake: Option<S::SuspendedThread> = None;
        loop {
            self.poll_progress();
            let mut next = unsafe { (*cur).next.load(Ordering::Acquire) };
            if next.is_null() {
                if self.tail.compare_exchange(cur, null_mut(), Ordering::AcqRel, Ordering::Acquire).is_ok() {
                    if let Some(w) = cur_wake {
                        w.notify();
                    }
                    return;
                }
                // A successor swapped the tail but has not linked yet.
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
                w.notify();
            }
            let (done, w) = self.consumer().execute(unsafe { &mut *(*next).work.get() });
            debug_assert!(done, "CombiningDelegator: work must finish in one execute");
            cur_wake = w;
            cur = next;
        }
    }
}

impl<S, C> DelegatorTrait<S, C> for CombiningDelegator<S, C>
where
    S: SuspendableSystem,
    S::SuspendedThread: StackfulResumable<S>,
    C: DelegatorConsumer<S>,
{
    fn start(consumer: C) -> Self {
        CombiningDelegator { tail: AtomicPtr::new(null_mut()), consumer: UnsafeCell::new(consumer) }
    }

    /// Nothing to stop: no consumer thread.
    fn stop(self) {}

    fn execute_or_delegate<Imm, Del>(&self, imm: Imm, del: Del)
    where
        Imm: FnOnce(&mut C) -> (bool, Option<S::SuspendedThread>),
        Del: FnOnce(&mut C::Work) -> &S::SuspendedThread,
    {
        let mut node = Node::<S, C> { next: AtomicPtr::new(null_mut()), work: UnsafeCell::new(Default::default()) };
        let me: *mut Node<S, C> = &mut node;
        let prev = self.tail.swap(me, Ordering::AcqRel);
        if prev.is_null() {
            let (done, w) = imm(self.consumer());
            debug_assert!(done, "CombiningDelegator: imm must finish");
            if let Some(w) = w {
                w.notify();
            }
            self.drain(me);
            return;
        }
        let sth: *const S::SuspendedThread = del(unsafe { &mut *(*me).work.get() });
        // Link behind `prev` only once parked: the holder may run our work
        // and wake us as soon as it sees the link.
        unsafe { &*sth }.wait_with(move || unsafe { (*prev).next.store(me, Ordering::Release) });
    }
}
