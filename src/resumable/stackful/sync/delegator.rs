use std::cell::Cell;
use std::mem::ManuallyDrop;
use std::pin::Pin;
use std::task::Poll;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::traits::{CallSlot, Delegator as DelegatorTrait, DelegatorConsumer, Resumable, StackfulResumable};
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::common::system::PoolSystem;
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::traits::{SpawnableStackfulTaskSystem, SuspendableSystem};
use crate::resumable::common::thread;
use crate::resumable::stackful::thread::spawn;

// ---------------------------------------------------------------------------
// DelegatorNode — content of each queue node
// ---------------------------------------------------------------------------

type Req<S, C> = ManuallyDrop<<C as DelegatorConsumer<S>>::Request<'static>>;

/// A queue node: a delegated call (its request and slot, both in the
/// caller's frame), or a caller waiting for the lock itself (asleep on
/// `sth`; `req` is null).
pub struct DelegatorNode<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>> where <S as PoolSystem>::Desc: StackfulTaskDesc {
    pub(super) sth:  <S as SuspendableSystem>::SuspendedThread,
    pub(super) req:  *mut Req<S, C>,
    pub(super) slot: *const CallSlot<S, C::Response>,
}

impl<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>> Default for DelegatorNode<S, C> where <S as PoolSystem>::Desc: StackfulTaskDesc {
    fn default() -> Self where <S as PoolSystem>::Desc: StackfulTaskDesc {
        DelegatorNode { sth: Default::default(), req: std::ptr::null_mut(), slot: std::ptr::null() }
    }
}

// ---------------------------------------------------------------------------
// SyncQueue — internal trait for MCS / ring-buffer backends
// ---------------------------------------------------------------------------

/// Queue backend for [`Delegator`].  Not part of the public API.
pub trait SyncQueue<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>>: Send + Sync where <S as PoolSystem>::Desc: StackfulTaskDesc {
    /// Try to acquire the lock or enqueue.
    /// Returns `(is_locked, prev_node, cur_node)`.
    /// `prev_node` is null when the queue was empty (i.e. is_locked == true).
    /// When `!is_locked`, `cur_node` is the newly enqueued node.
    fn start_lock(&self) -> (bool, *mut DelegatorNode<S, C>, *mut DelegatorNode<S, C>) where <S as PoolSystem>::Desc: StackfulTaskDesc;

    /// Publish `cur` to its predecessor `prev` (called from the wait callback).
    fn set_next(&self, prev: *mut DelegatorNode<S, C>, cur: *mut DelegatorNode<S, C>) where <S as PoolSystem>::Desc: StackfulTaskDesc;

    /// Return the current head node (the one holding the lock).
    fn get_head(&self) -> *mut DelegatorNode<S, C> where <S as PoolSystem>::Desc: StackfulTaskDesc;

    /// Try to unlock when the queue appears empty; returns true on success.
    fn try_unlock(&self, head: *mut DelegatorNode<S, C>) -> bool where <S as PoolSystem>::Desc: StackfulTaskDesc;

    /// Advance head to the next node if it has published itself.
    /// Returns `Some(next)` on success and frees/recycles the old head.
    fn try_follow_head(&self, head: *mut DelegatorNode<S, C>)
        -> Option<*mut DelegatorNode<S, C>> where <S as PoolSystem>::Desc: StackfulTaskDesc;
}

// ---------------------------------------------------------------------------
// Delegator<S, C, Q>
// ---------------------------------------------------------------------------

pub struct Delegator<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>>
where
    <S as PoolSystem>::Desc: StackfulTaskDesc,
    <S as SuspendableSystem>::SuspendedThread: StackfulResumable<S>,
{
    queue:        Q,
    consumer:     std::cell::UnsafeCell<C>,
    consumer_sth: <S as SuspendableSystem>::SuspendedThread,
    is_executed:  Cell<bool>,
    finished:     AtomicBool,
    // consumer ULT handle kept until stop()
    consumer_th:  std::cell::UnsafeCell<Option<thread::JoinHandle<S, ()>>>,
    // Guards lazily spawning the consumer ULT — see `ensure_consumer_started`.
    consumer_started: AtomicBool,
}

unsafe impl<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Send
    for Delegator<S, C, Q> where <S as PoolSystem>::Desc: StackfulTaskDesc, <S as SuspendableSystem>::SuspendedThread: StackfulResumable<S> {}
unsafe impl<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Sync
    for Delegator<S, C, Q> where <S as PoolSystem>::Desc: StackfulTaskDesc, <S as SuspendableSystem>::SuspendedThread: StackfulResumable<S> {}

impl<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C> + Default>
    Delegator<S, C, Q> where <S as PoolSystem>::Desc: StackfulTaskDesc, <S as SuspendableSystem>::SuspendedThread: StackfulResumable<S>
{
    pub fn new(consumer: C) -> Self where <S as PoolSystem>::Desc: StackfulTaskDesc {
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

impl<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Delegator<S, C, Q> where <S as PoolSystem>::Desc: StackfulTaskDesc, <S as SuspendableSystem>::SuspendedThread: StackfulResumable<S> {
    fn consumer(&self) -> &mut C where <S as PoolSystem>::Desc: StackfulTaskDesc {
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
    fn ensure_consumer_started(&self)
    where
        <S as PoolSystem>::Desc: StackfulTaskDesc,
        <S as SuspendableSystem>::SuspendedThread: StackfulResumable<S>,
        S::Worker: crate::resumable::stackful::worker::ContextSwitcher<S>
            + crate::resumable::common::worker::DescWorkerOps<S>,
    {
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
    fn lock_wait(&self)
    where
        <S as PoolSystem>::Desc: StackfulTaskDesc,
        <S as SuspendableSystem>::SuspendedThread: StackfulResumable<S>,
    {
        let (is_locked, prev, cur) = self.queue.start_lock();
        if is_locked {
            return;
        }
        // A null request marks a lock waiter; a ring slot may still hold
        // the request of its last user.
        unsafe {
            (*cur).req = std::ptr::null_mut();
            (*cur).slot = std::ptr::null();
        }
        let queue_ptr = &self.queue as *const Q;
        let prev_ptr = prev;
        let cur_ptr = cur;
        let sth_ref: &<S as SuspendableSystem>::SuspendedThread = unsafe { &(*cur).sth };
        sth_ref.wait_with(move || {
            if !prev_ptr.is_null() {
                unsafe { (*queue_ptr).set_next(prev_ptr, cur_ptr) };
            }
        });
    }

    // -- call --------------------------------------------------------------

    #[inline]
    fn call_impl(&self, req: C::Request<'_>) -> C::Response
    where
        <S as PoolSystem>::Desc: StackfulTaskDesc,
        <S as SuspendableSystem>::SuspendedThread: StackfulResumable<S>,
        S::Worker: crate::resumable::stackful::worker::ContextSwitcher<S>
            + crate::resumable::common::worker::DescWorkerOps<S>,
    {
        self.ensure_consumer_started();
        let slot = CallSlot::<S, C::Response>::default();
        let mut req = ManuallyDrop::new(req);
        let (is_locked, prev, cur) = self.queue.start_lock();
        if is_locked {
            let req = unsafe { ManuallyDrop::take(&mut req) };
            let mut cx = slot.cx();
            match self.consumer().execute(req, &mut cx) {
                Poll::Ready(v) => {
                    self.unlock();
                    return v;
                }
                Poll::Pending => {
                    debug_assert!(cx.is_deferred(), "Delegator: pending without defer");
                    self.unlock_and_wait(slot.sth());
                }
            }
        } else {
            // Enqueued at `cur`: point it at our request and slot, then
            // sleep.  The link (set_next) happens inside the wait callback,
            // once our continuation is saved, so the consumer can only wake
            // us after that.
            // SAFETY: only the lifetime is erased; the request is moved out
            // before we are woken.
            unsafe {
                (*cur).req = (&mut req as *mut ManuallyDrop<C::Request<'_>>).cast();
                (*cur).slot = &slot;
            }
            let queue_ptr = &self.queue as *const Q;
            slot.sth().wait_with(move || {
                if !prev.is_null() {
                    unsafe { (*queue_ptr).set_next(prev, cur) };
                }
            });
        }
        unsafe { slot.take() }
    }

    // -- unlock --------------------------------------------------------------

    fn unlock(&self)
    where
        <S as PoolSystem>::Desc: StackfulTaskDesc,
        <S as SuspendableSystem>::SuspendedThread: StackfulResumable<S>,
    {
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
                // (Not `consumer_sth.swap`: that would park this caller in
                // the consumer's slot.)
                if !is_active {
                    unsafe { (*next).sth.enter() };
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

    /// Release the lock and sleep on `wait_sth` (the caller's deferred
    /// call), as the C++ reference's `unlock_and_wait` does: while the
    /// consumer still has deferred calls to poll, hand the lock to it.
    fn unlock_and_wait(&self, wait_sth: &<S as SuspendableSystem>::SuspendedThread)
    where
        <S as PoolSystem>::Desc: StackfulTaskDesc,
        <S as SuspendableSystem>::SuspendedThread: StackfulResumable<S>,
    {
        self.is_executed.set(true);
        let is_active = self.consumer().is_active();
        let head = self.queue.get_head();

        if !is_active {
            // Sleep, then release; cancelled if a successor appeared.
            let unlocked = Cell::new(true);
            let queue_ptr = &self.queue as *const Q;
            wait_sth.wait_with_cond(|| {
                let ok = unsafe { (*queue_ptr).try_unlock(head) };
                unlocked.set(ok);
                ok
            });
            if unlocked.get() {
                return;
            }
        }

        if let Some(next) = self.queue.try_follow_head(head) {
            self.is_executed.set(false);
            if unsafe { (*next).sth.is_set() } {
                wait_sth.swap(unsafe { &(*next).sth });
                return;
            }
        }
        // Delegated work queued, or deferred calls to poll: the consumer.
        wait_sth.swap(&self.consumer_sth);
    }

    // -- consume -------------------------------------------------------------

    // Dedicated-consumer mode: the consumer ULT spawned by `start()` runs
    // `consumer_loop`, which calls `consume` in a loop until `stop()` sets
    // `finished`.
    fn consume(&self)
    where
        <S as PoolSystem>::Desc: StackfulTaskDesc,
        <S as SuspendableSystem>::SuspendedThread: StackfulResumable<S>,
    {
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
        let mut awake_sth: Option<crate::traits::Ready<S>> = None;

        if !is_executed {
            if !unsafe { (*head).req.is_null() } {
                // A delegated call — run it.  Deferred or not, it has been
                // executed: a deferred one is completed later.
                let req = unsafe { ManuallyDrop::take(&mut *(*head).req) };
                let slot = unsafe { &*(*head).slot };
                let mut cx = slot.cx();
                let r = con.execute(req, &mut cx);
                self.is_executed.set(true);
                if let Poll::Ready(v) = r {
                    // SAFETY: queue nodes are not in the callers' frames.
                    unsafe { slot.finish(v).notify() };
                }
                return;
            } else {
                // Lock-holder path: a ULT wants to acquire the lock.
                self.consumer_sth.swap(unsafe { &(*head).sth });
                return;
            }
        }

        if do_progress {
            if let Some(sth) = con.progress() {
                if let Some(prev) = awake_sth {
                    unsafe { prev.notify() };
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
            //
            // A caller completed by `progress` is woken first: once
            // suspended, nothing would wake it.  (The C++ reference
            // switches to it directly, `swap_with`.)
            if let Some(sth) = awake_sth.take() {
                unsafe { sth.notify() };
            }
            let queue_ptr = &self.queue as *const Q;
            let head_ptr = head;
            self.consumer_sth.wait_with_cond(move || unsafe {
                (*queue_ptr).try_unlock(head_ptr)
            });
        }

        if let Some(sth) = awake_sth {
            unsafe { sth.notify() };
        }
    }

    // -- consumer loop -------------------------------------------------------

    fn consumer_loop(&self)
    where
        <S as PoolSystem>::Desc: StackfulTaskDesc,
        <S as SuspendableSystem>::SuspendedThread: StackfulResumable<S>,
    {
        while !self.finished.load(Ordering::Acquire) {
            self.consume();
        }
    }
}

// ---------------------------------------------------------------------------
// Delegator impl
// ---------------------------------------------------------------------------

impl<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C> + Default + 'static>
    DelegatorTrait<S, C> for Delegator<S, C, Q>
where
    <S as PoolSystem>::Desc: StackfulTaskDesc,
    <S as SuspendableSystem>::SuspendedThread: StackfulResumable<S>,
    S::Worker: crate::resumable::stackful::worker::ContextSwitcher<S>
        + crate::resumable::common::worker::DescWorkerOps<S>,
{
    fn start(consumer: C) -> Pin<Box<Self>> {
        let del = Box::pin(Self::new(consumer));

        // Acquire the lock to initialise: the consumer ULT starts holding it.
        let (is_locked, _, _) = del.queue.start_lock();
        assert!(is_locked, "new queue must be empty");

        // The consumer ULT itself is spawned on the first call (see
        // `ensure_consumer_started`), so that a delegator can be made
        // outside the worker pool.
        del
    }

    #[inline]
    fn call(&self, req: C::Request<'_>) -> C::Response {
        self.call_impl(req)
    }

    fn with_lock<R>(&self, f: impl FnOnce(&mut C) -> R) -> R {
        self.ensure_consumer_started();
        self.lock_wait();
        let r = f(self.consumer());
        self.unlock();
        r
    }
}

impl<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Drop
    for Delegator<S, C, Q>
where
    <S as PoolSystem>::Desc: StackfulTaskDesc,
    <S as SuspendableSystem>::SuspendedThread: StackfulResumable<S>,
{
    fn drop(&mut self) {
        if !self.consumer_started.load(Ordering::Acquire) {
            return;
        }
        // Become the position holder first (waiting our turn if someone
        // else currently holds it) — matching the C++ reference's
        // `stop_consumer()`, which does `lock(); ...; unlock();`, not a bare
        // `unlock()`. Without this, `unlock()`'s `get_head()` can read a
        // null `head` (whenever the queue is genuinely idle), since nothing
        // established this caller as the current holder.
        self.lock_wait();
        self.finished.store(true, Ordering::Release);
        let is_active = self.consumer().is_active();
        self.unlock();
        if !is_active {
            self.consumer_sth.notify();
        }
        if let Some(th) = self.consumer_th.get_mut().take() {
            th.join().ok();
        }
    }
}
