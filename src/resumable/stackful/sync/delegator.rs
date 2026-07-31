use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::traits::{Delegator as DelegatorTrait, DelegatorConsumer, Resumable, StackfulResumable};
use crate::resumable::common::desc::WakerTaskDesc;
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::stackful::suspended::StackfulOnlyResumable;
use crate::traits::ThreadSystem;
use crate::resumable::common::thread;
use crate::resumable::stackful::thread::spawn;

// ---------------------------------------------------------------------------
// DelegatorNode — content of each queue node
// ---------------------------------------------------------------------------

pub struct DelegatorNode<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    pub(super) sth:  <S as ThreadSystem>::SuspendedThread,
    pub(super) work: C::Work,
}

impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>> Default for DelegatorNode<S, C> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    fn default() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        DelegatorNode { sth: Default::default(), work: Default::default() }
    }
}

// ---------------------------------------------------------------------------
// SyncQueue — internal trait for MCS / ring-buffer backends
// ---------------------------------------------------------------------------

/// Queue backend for [`Delegator`].  Not part of the public API.
pub trait SyncQueue<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>>: Send + Sync where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    /// Try to acquire the lock or enqueue.
    /// Returns `(is_locked, prev_node, cur_node)`.
    /// `prev_node` is null when the queue was empty (i.e. is_locked == true).
    /// When `!is_locked`, `cur_node` is the newly enqueued node.
    fn start_lock(&self) -> (bool, *mut DelegatorNode<S, C>, *mut DelegatorNode<S, C>) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable;

    /// Publish `cur` to its predecessor `prev` (called from the wait callback).
    fn set_next(&self, prev: *mut DelegatorNode<S, C>, cur: *mut DelegatorNode<S, C>) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable;

    /// Return the current head node (the one holding the lock).
    fn get_head(&self) -> *mut DelegatorNode<S, C> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable;

    /// Try to unlock when the queue appears empty; returns true on success.
    fn try_unlock(&self, head: *mut DelegatorNode<S, C>) -> bool where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable;

    /// Advance head to the next node if it has published itself.
    /// Returns `Some(next)` on success and frees/recycles the old head.
    fn try_follow_head(&self, head: *mut DelegatorNode<S, C>)
        -> Option<*mut DelegatorNode<S, C>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable;
}

// ---------------------------------------------------------------------------
// Delegator<S, C, Q>
// ---------------------------------------------------------------------------

pub struct Delegator<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    queue:        Q,
    consumer:     std::cell::UnsafeCell<C>,
    consumer_sth: <S as ThreadSystem>::SuspendedThread,
    is_executed:  Cell<bool>,
    finished:     AtomicBool,
    // consumer ULT handle kept until stop()
    consumer_th:  std::cell::UnsafeCell<Option<thread::JoinHandle<S, ()>>>,
    // Guards lazily spawning the consumer ULT — see `ensure_consumer_started`.
    consumer_started: AtomicBool,
}

unsafe impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Send
    for Delegator<S, C, Q> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {}
unsafe impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Sync
    for Delegator<S, C, Q> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {}

impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C> + Default>
    Delegator<S, C, Q> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable
{
    pub fn new(consumer: C) -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        Delegator {
            queue:        Q::default(),
            consumer:     std::cell::UnsafeCell::new(consumer),
            consumer_sth: Default::default(),
            is_executed:  Cell::new(true),
            finished:     AtomicBool::new(false),
            consumer_th:  std::cell::UnsafeCell::new(None),
            consumer_started: AtomicBool::new(false),
        }
    }
}

// ---------------------------------------------------------------------------
// Core algorithm (shared between MCS and ring-buffer variants)
// ---------------------------------------------------------------------------

impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Delegator<S, C, Q> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    fn consumer(&self) -> &mut C where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        unsafe { &mut *self.consumer.get() }
    }

    // -- consumer ULT startup -------------------------------------------------

    /// Spawn the dedicated consumer ULT on first use, not in `start()`.
    ///
    /// `start()` returns `Self` *by value*, so any address taken inside it
    /// (e.g. `&del`) is not guaranteed to be `self`'s final address — the
    /// caller is very likely to immediately move the returned value again
    /// (`Arc::new(Delegator::start(..))`, the common case, definitely moves
    /// it once more onto the heap). Capturing `&self` here instead, inside a
    /// `&self` method, is sound: by the time any caller can invoke
    /// `execute_or_delegate` concurrently from multiple ULTs at all, `self`
    /// must already be behind something that gives it a stable address
    /// (`Arc`, a `'static` reference, etc.) — that's already a precondition
    /// for sharing it, independent of this method.
    ///
    /// (An earlier version of this function spawned eagerly inside `start()`
    /// using a captured `&del as *const Self as usize`; that crashed with
    /// SIGBUS in ~1 out of 13 stress runs once the consumer ULT actually
    /// existed to dereference the stale address — exactly the failure mode
    /// this comment describes. Caught by repeated stress runs, not the first
    /// few passes.)
    fn ensure_consumer_started(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        if self.consumer_started.load(Ordering::Acquire) {
            return;
        }
        if self
            .consumer_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return; // another caller already won the race to start it
        }
        let self_ptr = self as *const Self as usize;
        let th = spawn::<S, (), _>(move || {
            let del = unsafe { &*(self_ptr as *const Self) };
            del.consumer_loop();
        });
        unsafe { *self.consumer_th.get() = Some(th) };
    }

    // -- lock_wait -------------------------------------------------------------

    /// Acquire the position, waiting if necessary — never delegating actual
    /// work, just parking on the queue node's own `sth` (which `unlock()`
    /// already recognizes as "next waiter wants the lock", via
    /// `sth.is_set()`). Used by `stop()`, mirroring the C++ reference's
    /// `stop_consumer()`, which does a full `lock(); ...; unlock();` cycle —
    /// not a bare `unlock()` — precisely so `unlock()`'s `get_head()` is
    /// always called by whoever currently, legitimately holds the position.
    fn lock_wait(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        let (is_locked, prev, cur) = self.queue.start_lock();
        if is_locked {
            return;
        }
        let queue_ptr = &self.queue as *const Q;
        let prev_ptr = prev;
        let cur_ptr = cur;
        let sth_ref: &<S as ThreadSystem>::SuspendedThread = unsafe { &(*cur).sth };
        sth_ref.wait_with(move || {
            if !prev_ptr.is_null() {
                unsafe { (*queue_ptr).set_next(prev_ptr, cur_ptr) };
            }
        });
    }

    // -- lock_or_delegate ----------------------------------------------------

    /// Returns `true` if the caller acquired the lock (should call `unlock` after).
    /// Returns `false` if the work was delegated; the caller is suspended and
    /// will be woken by the consumer.
    fn lock_or_delegate<Del>(&self, del: Del) -> bool
    where
        Del: FnOnce(&mut C::Work) -> &<S as ThreadSystem>::SuspendedThread, <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable
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
        let sth_ref: &<S as ThreadSystem>::SuspendedThread = del(unsafe { &mut *work_ptr });

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

    fn unlock(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
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

    fn unlock_and_wait(&self, wait_sth: &<S as ThreadSystem>::SuspendedThread) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
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

    // Dedicated-consumer mode: the consumer ULT spawned by `start()` runs
    // `consumer_loop`, which calls `consume` in a loop until `stop()` sets
    // `finished`.
    fn consume(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
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
        let mut awake_sth: Option<<S as ThreadSystem>::SuspendedThread> = None;

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
            // Try to suspend consumer until next work arrives — but only if
            // try_unlock actually succeeds (queue genuinely empty). If a
            // successor enqueued concurrently, cancel the suspend instead of
            // committing to it unconditionally: whoever enqueued does *not*
            // call unlock()/notify() on our behalf (they never won the lock),
            // so an unconditional park here would never be woken. Matches
            // the C++ reference's `try_unlock_and_wait`, built on a
            // conditional suspend for exactly this reason. On cancel, the
            // outer `consumer_loop`'s `while !finished { consume() }` simply
            // re-enters and re-derives state from scratch.
            let queue_ptr = &self.queue as *const Q;
            let head_ptr = head;
            self.consumer_sth.wait_with_cond(move || unsafe {
                (*queue_ptr).try_unlock(head_ptr)
            });
        }

        if let Some(sth) = awake_sth {
            sth.notify();
        }
    }

    // -- consumer loop -------------------------------------------------------

    fn consumer_loop(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        while !self.finished.load(Ordering::Acquire) {
            self.consume();
        }
    }
}

// ---------------------------------------------------------------------------
// Delegator impl
// ---------------------------------------------------------------------------

impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C> + Default + 'static>
    DelegatorTrait<S, C> for Delegator<S, C, Q> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable
{
    fn start(consumer: C) -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        let del = Self::new(consumer);

        // Acquire the lock to initialise: the consumer ULT starts holding it.
        let (is_locked, _, _) = del.queue.start_lock();
        assert!(is_locked, "new queue must be empty");

        // The consumer ULT itself is spawned lazily, on first
        // `execute_or_delegate` call — see `ensure_consumer_started` for why
        // spawning it here (before the caller has settled `del` at its final
        // address, e.g. inside an `Arc`) is unsound.
        del
    }

    fn stop(self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        // Become the position holder first (waiting our turn if someone
        // else currently holds it) — matching the C++ reference's
        // `stop_consumer()`, which does `lock(); ...; unlock();`, not a bare
        // `unlock()`. Without this, `unlock()`'s `get_head()` can read a
        // null `head` (whenever the queue is genuinely idle at the moment
        // `stop()` happens to be called), since nothing established this
        // caller as the current holder.
        self.lock_wait();
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
        Imm: FnOnce(&mut C) -> (bool, Option<<S as ThreadSystem>::SuspendedThread>),
        Del: FnOnce(&mut C::Work) -> &<S as ThreadSystem>::SuspendedThread, <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable
    {
        self.ensure_consumer_started();
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
