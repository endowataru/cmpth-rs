use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::spin::SpinLock;
use crate::traits::{Condvar as CondvarTrait, Mutex as MutexTrait, Resumable, StackfulMutex, StackfulResumable};
use crate::ult::system::UltSchedulerSystem;

// ---------------------------------------------------------------------------
// McsMutex
// ---------------------------------------------------------------------------

struct McsNode<S: UltSchedulerSystem> {
    next: AtomicPtr<McsNode<S>>,
    suspended: S::SuspendedThread,
}

pub struct McsMutex<S: UltSchedulerSystem, T: Send> {
    tail: AtomicPtr<McsNode<S>>,
    data: UnsafeCell<T>,
}

unsafe impl<S: UltSchedulerSystem, T: Send> Send for McsMutex<S, T> {}
unsafe impl<S: UltSchedulerSystem, T: Send> Sync for McsMutex<S, T> {}

pub struct McsMutexGuard<'a, S: UltSchedulerSystem, T: Send> {
    mutex: &'a McsMutex<S, T>,
    node: Box<McsNode<S>>,
}

impl<S: UltSchedulerSystem, T: Send> Deref for McsMutexGuard<'_, S, T> {
    type Target = T;
    fn deref(&self) -> &T { unsafe { &*self.mutex.data.get() } }
}

impl<S: UltSchedulerSystem, T: Send> DerefMut for McsMutexGuard<'_, S, T> {
    fn deref_mut(&mut self) -> &mut T { unsafe { &mut *self.mutex.data.get() } }
}

impl<S: UltSchedulerSystem, T: Send> Drop for McsMutexGuard<'_, S, T> {
    fn drop(&mut self) {
        let node_ptr: *mut McsNode<S> = &mut *self.node;
        if self.mutex.tail
            .compare_exchange(node_ptr, null_mut(), Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return;
        }
        // The successor has swapped itself into `tail` but publishes
        // `node.next` only from its post-switch callback.  Under cooperative
        // scheduling that callback may be queued behind *this* task (e.g. a
        // nested system whose workers are ULTs), so a pure spin can starve it
        // forever: yield periodically to let the enqueuer run.
        let mut next = self.node.next.load(Ordering::Acquire);
        let mut spins = 0u32;
        while next.is_null() {
            spins = spins.wrapping_add(1);
            if spins & 0x3F == 0 {
                use crate::ult::worker::Worker as _;
                if let Some(wk) = crate::ult::worker::UltWorker::<S>::current() {
                    wk.yield_now();
                }
            } else {
                std::hint::spin_loop();
            }
            next = self.node.next.load(Ordering::Acquire);
        }
        unsafe { (*next).suspended.notify() };
    }
}

impl<S: UltSchedulerSystem, T: Send> MutexTrait<T> for McsMutex<S, T> {
    type Guard<'a> = McsMutexGuard<'a, S, T> where Self: 'a, T: 'a;
    type Condvar = McsCondvar<S>;

    fn new(val: T) -> Self {
        McsMutex { tail: AtomicPtr::new(null_mut()), data: UnsafeCell::new(val) }
    }

    fn lock(&self) -> McsMutexGuard<'_, S, T> {
        let mut node = Box::new(McsNode {
            next: AtomicPtr::new(null_mut()),
            suspended: S::SuspendedThread::default(),
        });
        let node_ptr: *mut McsNode<S> = &mut *node;
        let prev = self.tail.swap(node_ptr, Ordering::AcqRel);
        if !prev.is_null() {
            node.suspended.wait_with(move || {
                unsafe { (*prev).next.store(node_ptr, Ordering::Release) };
            });
        }
        McsMutexGuard { mutex: self, node }
    }
}

impl<S: UltSchedulerSystem, T: Send> StackfulMutex<T> for McsMutex<S, T> {
    type Guard<'a> = McsMutexGuard<'a, S, T> where Self: 'a, T: 'a;

    fn new(val: T) -> Self {
        <Self as MutexTrait<T>>::new(val)
    }

    fn lock(&self) -> McsMutexGuard<'_, S, T> {
        MutexTrait::lock(self)
    }
}

// ---------------------------------------------------------------------------
// McsCondvar
// ---------------------------------------------------------------------------

pub struct McsCondvar<S: UltSchedulerSystem> {
    waiters: SpinLock<VecDeque<S::SuspendedThread>>,
}

impl<S: UltSchedulerSystem> McsCondvar<S> {
    pub fn new() -> Self {
        McsCondvar { waiters: SpinLock::new(VecDeque::new()) }
    }

    pub fn notify_one(&self) {
        if let Some(sth) = self.waiters.lock().pop_front() { sth.notify(); }
    }

    pub fn notify_all(&self) {
        let all: VecDeque<_> = std::mem::take(&mut *self.waiters.lock());
        for sth in all { sth.notify(); }
    }
}

impl<S: UltSchedulerSystem, T: Send> CondvarTrait<McsMutex<S, T>, T> for McsCondvar<S> {
    fn new() -> Self { McsCondvar::new() }

    fn wait<'a>(&self, guard: McsMutexGuard<'a, S, T>) -> McsMutexGuard<'a, S, T>
    where McsMutex<S, T>: 'a, T: 'a
    {
        let mutex = guard.mutex;
        let mut waiters = self.waiters.lock();
        waiters.push_back(S::SuspendedThread::default());
        let sth = waiters.back().unwrap() as *const S::SuspendedThread;
        unsafe { &*sth }.wait_with(move || {
            drop(waiters);
            drop(guard);
        });
        MutexTrait::lock(mutex)
    }

    fn notify_one(&self) { McsCondvar::notify_one(self); }
    fn notify_all(&self) { McsCondvar::notify_all(self); }
}
