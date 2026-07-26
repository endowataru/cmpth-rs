//! `delegator()` — the mpsc-style redesign of `Delegator`/`DelegatorTrait`
//! (docs/sync-async-unification.md's Delegator section covers the design
//! discussion). Kept alongside the existing `Delegator<S, C, Q>` rather than
//! replacing it, following this session's established pattern.
//!
//! Producer/consumer roles, matching `std::sync::mpsc::channel()`:
//! `Producer<S, C, Q>` is `Clone` (multi-producer); there is no separate
//! consumer-side handle at all — the consumer is a dedicated ULT spawned by
//! `delegator()`, and its lifecycle is tied entirely to `Arc` refcounting.
//! Once every `Producer` clone is dropped, `Inner`'s `Drop` impl signals
//! shutdown; there is no explicit `stop()`. A `Producer` clone leaked
//! somewhere leaks the consumer ULT along with it — the same failure mode
//! as any other `Arc` cycle/leak, just carrying a background execution
//! context instead of only memory.
//!
//! The core algorithm (`lock_or_delegate`/`unlock`/`unlock_and_wait`/
//! `consume`/`consumer_loop`) is unchanged from the fixed `Delegator` in
//! `delegator.rs` — copied, not reimplemented, specifically to keep the
//! already-stress-tested logic intact. What changes is everything around
//! it: the consumer ULT spawns eagerly in `delegator()` (sound now, since
//! `Inner` lives in an `Arc` from construction and is never moved again —
//! unlike the old `start()`, which returned `Self` by value), and shutdown
//! goes through `Drop` instead of a `stop(self)` that needed `Arc::try_unwrap`.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::traits::{DelegatorConsumer, Resumable, StackfulResumable};
use crate::ult::sync::delegator::SyncQueue;
use crate::ult::desc::{StackfulTaskDesc, WakerTaskDesc};
use crate::ult::system::{SchedulerSystem, UltSchedulerSystem};
use crate::traits::UltSystem;
use crate::ult::thread;

// ---------------------------------------------------------------------------
// Inner
// ---------------------------------------------------------------------------

struct Inner<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    queue: Q,
    consumer: std::cell::UnsafeCell<C>,
    consumer_sth: S::SuspendedThread,
    is_executed: Cell<bool>,
    finished: AtomicBool,
    consumer_th: std::cell::UnsafeCell<Option<thread::JoinHandle<S, ()>>>,
}

unsafe impl<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Send for Inner<S, C, Q> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {}
unsafe impl<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Sync for Inner<S, C, Q> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {}

impl<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C> + Default> Inner<S, C, Q> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    fn new(consumer: C) -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        Inner {
            queue: Q::default(),
            consumer: std::cell::UnsafeCell::new(consumer),
            consumer_sth: Default::default(),
            is_executed: Cell::new(true),
            finished: AtomicBool::new(false),
            consumer_th: std::cell::UnsafeCell::new(None),
        }
    }
}

// ---------------------------------------------------------------------------
// Core algorithm — copied unchanged from the fixed `Delegator` in
// delegator.rs (see that file's comments for the four-bugs-found history).
// ---------------------------------------------------------------------------

impl<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Inner<S, C, Q> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    fn consumer(&self) -> &mut C where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        unsafe { &mut *self.consumer.get() }
    }

    /// Acquire the position, waiting if necessary — never delegating actual
    /// work, just parking on the queue node's own `sth`. Used by `Drop`,
    /// mirroring the C++ reference's `stop_consumer()` (`lock(); ...;
    /// unlock();`, not a bare `unlock()`).
    fn lock_wait(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        let (is_locked, prev, cur) = self.queue.start_lock();
        if is_locked {
            return;
        }
        let queue_ptr = &self.queue as *const Q;
        let prev_ptr = prev;
        let cur_ptr = cur;
        let sth_ref: &S::SuspendedThread = unsafe { &(*cur).sth };
        sth_ref.wait_with(move || {
            if !prev_ptr.is_null() {
                unsafe { (*queue_ptr).set_next(prev_ptr, cur_ptr) };
            }
        });
    }

    fn lock_or_delegate<Del>(&self, del: Del) -> bool
    where
        Del: FnOnce(&mut C::Work) -> &S::SuspendedThread, <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc
    {
        let (is_locked, _prev, cur) = self.queue.start_lock();
        if is_locked {
            return true;
        }
        let work_ptr: *mut C::Work = unsafe { &mut (*cur).work };
        let sth_ref: &S::SuspendedThread = del(unsafe { &mut *work_ptr });
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

    fn unlock(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        self.is_executed.set(true);
        let head = self.queue.get_head();
        let is_active = self.consumer().is_active();

        if !is_active && self.queue.try_unlock(head) {
            return;
        }

        if let Some(next) = self.queue.try_follow_head(head) {
            self.is_executed.set(false);
            if unsafe { (*next).sth.is_set() } {
                if !is_active {
                    self.consumer_sth.swap(unsafe { &(*next).sth });
                } else {
                    unsafe { (*next).sth.notify() };
                }
            } else {
                self.consumer_sth.notify();
            }
            return;
        }

        self.consumer_sth.notify();
    }

    fn unlock_and_wait(&self, wait_sth: &S::SuspendedThread) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
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

        let queue_ptr = &self.queue as *const Q;
        let head_ptr = head;
        wait_sth.wait_with(move || {
            unsafe { (*queue_ptr).try_unlock(head_ptr) };
        });
    }

    fn consume(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
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
                let (done, sth_opt) = con.execute(unsafe { &mut (*head).work });
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

    fn consumer_loop(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        while !self.finished.load(Ordering::Acquire) {
            self.consume();
        }
    }

    fn execute_or_delegate<Imm, Del>(&self, imm: Imm, del: Del)
    where
        Imm: FnOnce(&mut C) -> (bool, Option<S::SuspendedThread>),
        Del: FnOnce(&mut C::Work) -> &S::SuspendedThread, <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc
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

// ---------------------------------------------------------------------------
// Drop: reuses the exact lock_wait -> finished -> unlock -> conditional
// notify sequence the old stop() used (see delegator.rs), rather than
// touching consumer_sth directly. Going through lock_wait() first means
// this is fully serialized by the same MCS chain every other participant
// (producers, the consumer's own swap-to-successor path) already goes
// through — a direct `if consumer_sth.is_set() { notify() }` here would
// race against the consumer's own in-flight wait_with_cond callback (which
// unconditionally stores its continuation before deciding whether to
// commit or cancel — see suspended.rs's wait_with_cond_impl), since Drop is
// not part of that serialization unless it explicitly joins it via
// lock_wait() first.
// ---------------------------------------------------------------------------

impl<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Drop for Inner<S, C, Q> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    fn drop(&mut self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        self.lock_wait();
        self.finished.store(true, Ordering::Release);
        let is_active = self.consumer().is_active();
        self.unlock();
        if !is_active {
            self.consumer_sth.notify();
        }
        // consumer_th's own Drop (JoinHandle) detaches safely without
        // blocking if the consumer ULT hasn't exited yet — no explicit
        // join, keeping this non-blocking beyond the bounded lock_wait()
        // above (bounded because no *new* work can arrive: every Producer
        // is already gone by the time this runs).
    }
}

// ---------------------------------------------------------------------------
// Producer — Clone, mpsc::Sender-like
// ---------------------------------------------------------------------------

pub struct Producer<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>>(Arc<Inner<S, C, Q>>) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc;

impl<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Clone for Producer<S, C, Q> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    fn clone(&self) -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        Producer(Arc::clone(&self.0))
    }
}

impl<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, Q: SyncQueue<S, C>> Producer<S, C, Q> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    /// Runs `imm` inline if uncontended, otherwise delegates via `del` and
    /// waits for the result. Blocks only on the caller's own work; any
    /// backlog left behind by other callers is handed to the consumer ULT
    /// via a non-blocking notify and does not extend this call's wait (see
    /// docs/sync-async-unification.md for the reasoning behind that as a
    /// design intent, not a hard guarantee).
    pub fn execute_or_delegate<Imm, Del>(&self, imm: Imm, del: Del)
    where
        Imm: FnOnce(&mut C) -> (bool, Option<S::SuspendedThread>),
        Del: FnOnce(&mut C::Work) -> &S::SuspendedThread, <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc
    {
        self.0.execute_or_delegate(imm, del)
    }
}

// ---------------------------------------------------------------------------
// delegator() — entry point
// ---------------------------------------------------------------------------

/// Start a consumer ULT running `consumer` and return a `Producer` for
/// submitting work to it. Cloning the `Producer` gives multiple producers;
/// there is no separate consumer-side handle. The consumer ULT keeps
/// running until every `Producer` clone (including this one) has been
/// dropped — no explicit stop call exists. A leaked `Producer` leaks the
/// consumer ULT along with it, exactly like any other `Arc` leak.
///
/// Must be called from within a worker (spawns the consumer ULT via
/// [`crate::ult::thread::spawn`]).
pub fn delegator<S, C, Q>(consumer: C) -> Producer<S, C, Q>
where
    S: UltSchedulerSystem + UltSystem,
    C: DelegatorConsumer<S>,
    Q: SyncQueue<S, C> + Default + 'static, <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc
{
    let inner = Arc::new(Inner::<S, C, Q>::new(consumer));

    let (is_locked, ..) = inner.queue.start_lock();
    assert!(is_locked, "fresh queue must be empty");

    // Sound to spawn eagerly here (unlike the old start()): `inner` is
    // already in its final, stable heap location (Arc), and will never be
    // moved again — every further use is through Arc clones of this same
    // allocation.
    let th = thread::spawn::<S, (), _>({
        let inner = Arc::clone(&inner);
        move || inner.consumer_loop()
    });
    unsafe { *inner.consumer_th.get() = Some(th) };

    Producer(inner)
}
