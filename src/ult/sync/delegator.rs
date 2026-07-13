use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::traits::{Delegator as DelegatorTrait, DelegatorConsumer, SuspendedThread};
use crate::ult::system::UltSystem;
use crate::ult::thread;

// ---------------------------------------------------------------------------
// DelegatorNode — content of each queue node
// ---------------------------------------------------------------------------

pub struct DelegatorNode<S: UltSystem, C: DelegatorConsumer<S>> {
    pub(super) sth:  S::SuspendedThread,
    pub(super) work: C::Work,
}

impl<S: UltSystem, C: DelegatorConsumer<S>> Default for DelegatorNode<S, C> {
    fn default() -> Self {
        DelegatorNode { sth: Default::default(), work: Default::default() }
    }
}

// ---------------------------------------------------------------------------
// SyncQueue — internal trait for MCS / ring-buffer backends
// ---------------------------------------------------------------------------

/// Queue backend for [`Delegator`].  Not part of the public API.
pub trait SyncQueue<S: UltSystem, C: DelegatorConsumer<S>>: Send + Sync {
    /// Try to acquire the lock or enqueue.
    /// Returns `(is_locked, prev_node, cur_node)`.
    /// `prev_node` is null when the queue was empty (i.e. is_locked == true).
    /// When `!is_locked`, `cur_node` is the newly enqueued node.
    fn start_lock(&self) -> (bool, *mut DelegatorNode<S, C>, *mut DelegatorNode<S, C>);

    /// Publish `cur` to its predecessor `prev` (called from the wait callback).
    fn set_next(&self, prev: *mut DelegatorNode<S, C>, cur: *mut DelegatorNode<S, C>);

    /// Return the current head node (the one holding the lock).
    fn get_head(&self) -> *mut DelegatorNode<S, C>;

    /// Try to unlock when the queue appears empty; returns true on success.
    fn try_unlock(&self, head: *mut DelegatorNode<S, C>) -> bool;

    /// Advance head to the next node if it has published itself.
    /// Returns `Some(next)` on success and frees/recycles the old head.
    fn try_follow_head(&self, head: *mut DelegatorNode<S, C>)
        -> Option<*mut DelegatorNode<S, C>>;
}

// ---------------------------------------------------------------------------
// Delegator<S, C, Q>
// ---------------------------------------------------------------------------

pub struct Delegator<S: UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> {
    queue:        Q,
    consumer:     std::cell::UnsafeCell<C>,
    consumer_sth: S::SuspendedThread,
    is_executed:  Cell<bool>,
    finished:     AtomicBool,
    // consumer ULT handle kept until stop()
    consumer_th:  std::cell::UnsafeCell<Option<thread::JoinHandle<S, ()>>>,
}

unsafe impl<S: UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Send
    for Delegator<S, C, Q> {}
unsafe impl<S: UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Sync
    for Delegator<S, C, Q> {}

impl<S: UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C> + Default>
    Delegator<S, C, Q>
{
    pub fn new(consumer: C) -> Self {
        Delegator {
            queue:        Q::default(),
            consumer:     std::cell::UnsafeCell::new(consumer),
            consumer_sth: Default::default(),
            is_executed:  Cell::new(true),
            finished:     AtomicBool::new(false),
            consumer_th:  std::cell::UnsafeCell::new(None),
        }
    }
}

// ---------------------------------------------------------------------------
// Core algorithm (shared between MCS and ring-buffer variants)
// ---------------------------------------------------------------------------

impl<S: UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Delegator<S, C, Q> {
    fn consumer(&self) -> &mut C {
        unsafe { &mut *self.consumer.get() }
    }

    // -- lock_or_delegate ----------------------------------------------------

    /// Returns `true` if the caller acquired the lock (should call `unlock` after).
    /// Returns `false` if the work was delegated; the caller is suspended and
    /// will be woken by the consumer.
    fn lock_or_delegate<Del>(&self, del: Del) -> bool
    where
        Del: FnOnce(&mut C::Work) -> &S::SuspendedThread,
    {
        let (is_locked, _prev, cur) = self.queue.start_lock();
        if is_locked {
            return true;
        }

        // We are enqueued at `cur`.  Fill in the work and get back the sth to
        // park on.  The actual linking (set_next) must happen INSIDE the wait
        // callback — after the continuation is saved — so the holder can only
        // call notify after our sth is ready.
        let work_ptr: *mut C::Work = unsafe { &mut (*cur).work };
        let sth_ref: &S::SuspendedThread = del(unsafe { &mut *work_ptr });

        // Park.  The callback links us into the predecessor's next pointer.
        let queue_ptr = &self.queue as *const Q;
        let prev_ptr = _prev;
        let cur_ptr = cur;
        sth_ref.wait_with(move || {
            if !prev_ptr.is_null() {
                unsafe { (*queue_ptr).set_next(prev_ptr, cur_ptr) };
            }
        });

        false
    }

    // -- unlock --------------------------------------------------------------

    fn unlock(&self) {
        self.is_executed.set(true);
        let head = self.queue.get_head();
        let is_active = self.consumer().is_active();

        if !is_active && self.queue.try_unlock(head) {
            return;
        }

        if let Some(next) = self.queue.try_follow_head(head) {
            self.is_executed.set(false);
            if unsafe { (*next).sth.is_set() } {
                // Next waiter is trying to lock — wake it directly.
                if !is_active {
                    self.consumer_sth.swap(unsafe { &(*next).sth });
                } else {
                    unsafe { (*next).sth.notify() };
                }
            } else {
                // Next waiter delegated work — consumer handles it.
                self.consumer_sth.notify();
            }
            return;
        }

        // No successor visible yet — wake the consumer to drain the queue.
        self.consumer_sth.notify();
    }

    fn unlock_and_wait(&self, wait_sth: &S::SuspendedThread) {
        self.is_executed.set(true);
        let head = self.queue.get_head();

        if let Some(next) = self.queue.try_follow_head(head) {
            self.is_executed.set(false);
            if unsafe { (*next).sth.is_set() } {
                wait_sth.swap(unsafe { &(*next).sth });
            } else {
                wait_sth.swap(&self.consumer_sth);
            }
            return;
        }

        // Queue empty or successor not yet visible: try to unlock atomically.
        let queue_ptr = &self.queue as *const Q;
        let head_ptr = head;
        wait_sth.wait_with(move || {
            // If the unlock fails, a successor appeared; consumer_sth is
            // notified in the unlock path, so nothing more to do here.
            unsafe { (*queue_ptr).try_unlock(head_ptr) };
        });
    }

    // -- consume -------------------------------------------------------------

    // Dedicated-consumer mode: `start` does not spawn a consumer thread yet
    // (`consumer_th` stays `None`), so these two methods are currently unused.
    #[allow(dead_code)]
    fn consume(&self) {
        let con = self.consumer();
        let mut is_executed = self.is_executed.get();
        let mut head = self.queue.get_head();

        if is_executed {
            if let Some(next) = self.queue.try_follow_head(head) {
                is_executed = false;
                head = next;
            }
        }

        let do_progress = con.is_active();
        let mut awake_sth: Option<S::SuspendedThread> = None;

        if !is_executed {
            if !unsafe { (*head).sth.is_set() } {
                // Delegated work — execute it.
                let (done, sth_opt) =
                    con.execute(unsafe { &mut (*head).work });
                is_executed = done;
                awake_sth = sth_opt;

                if is_executed {
                    self.is_executed.set(true);
                    if let Some(sth) = awake_sth {
                        sth.notify();
                    }
                    return;
                }
            } else {
                // Lock-holder path: a ULT wants to acquire the lock.
                self.consumer_sth.swap(unsafe { &(*head).sth });
                return;
            }
        }

        if do_progress {
            if let Some(sth) = con.progress() {
                if let Some(prev) = awake_sth {
                    prev.notify();
                }
                awake_sth = Some(sth);
            }
        }

        self.is_executed.set(is_executed);

        let is_active = con.is_active();
        if !is_active && is_executed {
            // Try to suspend consumer until next work arrives.
            let queue_ptr = &self.queue as *const Q;
            let head_ptr = head;
            self.consumer_sth.wait_with(move || {
                unsafe { (*queue_ptr).try_unlock(head_ptr) };
            });
        }

        if let Some(sth) = awake_sth {
            sth.notify();
        }
    }

    // -- consumer loop -------------------------------------------------------

    #[allow(dead_code)]
    fn consumer_loop(&self) {
        while !self.finished.load(Ordering::Acquire) {
            self.consume();
        }
    }
}

// ---------------------------------------------------------------------------
// Delegator impl
// ---------------------------------------------------------------------------

impl<S: UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C> + Default + 'static>
    DelegatorTrait<S, C> for Delegator<S, C, Q>
{
    fn start(consumer: C) -> Self {
        let del = Self::new(consumer);

        // Acquire the lock to initialise: the consumer ULT starts holding it.
        let (is_locked, _, _) = del.queue.start_lock();
        assert!(is_locked, "new queue must be empty");

        del
    }

    fn stop(self) {
        self.finished.store(true, Ordering::Release);
        let is_active = self.consumer().is_active();
        self.unlock();
        if !is_active {
            self.consumer_sth.notify();
        }
        if let Some(th) = unsafe { &mut *self.consumer_th.get() }.take() {
            th.join().ok();
        }
    }

    fn execute_or_delegate<Imm, Del>(&self, imm: Imm, del: Del)
    where
        Imm: FnOnce(&mut C) -> (bool, Option<S::SuspendedThread>),
        Del: FnOnce(&mut C::Work) -> &S::SuspendedThread,
    {
        let is_locked = self.lock_or_delegate(del);
        if is_locked {
            let (is_done, wait_sth) = imm(self.consumer());
            self.is_executed.set(is_done);
            match wait_sth {
                Some(ref sth) => self.unlock_and_wait(sth),
                None => self.unlock(),
            }
        }
    }
}
