use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::spin::SpinLock;
use crate::traits::{Resumable, StackfulMutex, StackfulResumable};
use crate::traits::StackfulParking;

// ---------------------------------------------------------------------------
// McsMutex
// ---------------------------------------------------------------------------

struct McsNode<S: StackfulParking> {
    next: AtomicPtr<McsNode<S>>,
    suspended: S::SuspendedThread,
}

pub struct McsMutex<S: StackfulParking, T: Send> {
    tail: AtomicPtr<McsNode<S>>,
    data: UnsafeCell<T>,
}

unsafe impl<S: StackfulParking, T: Send> Send for McsMutex<S, T> {}
unsafe impl<S: StackfulParking, T: Send> Sync for McsMutex<S, T> {}

pub struct McsMutexGuard<'a, S: StackfulParking, T: Send> {
    mutex: &'a McsMutex<S, T>,
    node: Box<McsNode<S>>,
}

impl<S: StackfulParking, T: Send> Deref for McsMutexGuard<'_, S, T> {
    type Target = T;
    fn deref(&self) -> &T { unsafe { &*self.mutex.data.get() } }
}

impl<S: StackfulParking, T: Send> DerefMut for McsMutexGuard<'_, S, T> {
    fn deref_mut(&mut self) -> &mut T { unsafe { &mut *self.mutex.data.get() } }
}

impl<S: StackfulParking, T: Send> Drop for McsMutexGuard<'_, S, T>
{
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
                <S as crate::traits::SpawnableStackfulTaskSystem>::yield_now();
            } else {
                std::hint::spin_loop();
            }
            next = self.node.next.load(Ordering::Acquire);
        }
        unsafe { (*next).suspended.notify() };
    }
}

impl<S: StackfulParking, T: Send> StackfulMutex<T> for McsMutex<S, T> {
    type Guard<'a> = McsMutexGuard<'a, S, T> where Self: 'a, T: 'a;

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

// ---------------------------------------------------------------------------
// McsCondvar
// ---------------------------------------------------------------------------

pub struct McsCondvar<S: StackfulParking> {
    waiters: SpinLock<VecDeque<S::SuspendedThread>>,
}

impl<S: StackfulParking> McsCondvar<S> {
    pub fn new() -> Self {
        McsCondvar { waiters: SpinLock::new(VecDeque::new()) }
    }

    /// Release `guard`, wait for a notification, then re-acquire and return
    /// a fresh guard.
    pub fn wait<'a, T: Send>(&self, guard: McsMutexGuard<'a, S, T>) -> McsMutexGuard<'a, S, T> {
        let mutex = guard.mutex;
        let mut waiters = self.waiters.lock();
        waiters.push_back(S::SuspendedThread::default());
        let sth = waiters.back().unwrap() as *const S::SuspendedThread;
        unsafe { &*sth }.wait_with(move || {
            drop(waiters);
            drop(guard);
        });
        StackfulMutex::lock(mutex)
    }

    pub fn notify_one(&self) {
        if let Some(sth) = self.waiters.lock().pop_front() { sth.notify(); }
    }

    pub fn notify_all(&self) {
        let all: VecDeque<_> = std::mem::take(&mut *self.waiters.lock());
        for sth in all { sth.notify(); }
    }
}

impl<S: StackfulParking> Default for McsCondvar<S> {
    fn default() -> Self {
        Self::new()
    }
}
