use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::spin::SpinLock;
use crate::traits::{Resumable, StackfulMutex, StackfulResumable};
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::stackful::worker::StackfulWorker;

// ---------------------------------------------------------------------------
// McsMutex
// ---------------------------------------------------------------------------

struct McsNode<S: StackfulSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    next: AtomicPtr<McsNode<S>>,
    suspended: S::SuspendedThread,
}

pub struct McsMutex<S: StackfulSchedulerSystem, T: Send> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    tail: AtomicPtr<McsNode<S>>,
    data: UnsafeCell<T>,
}

unsafe impl<S: StackfulSchedulerSystem, T: Send> Send for McsMutex<S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}
unsafe impl<S: StackfulSchedulerSystem, T: Send> Sync for McsMutex<S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}

pub struct McsMutexGuard<'a, S: StackfulSchedulerSystem, T: Send> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    mutex: &'a McsMutex<S, T>,
    node: Box<McsNode<S>>,
}

impl<S: StackfulSchedulerSystem, T: Send> Deref for McsMutexGuard<'_, S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    type Target = T;
    fn deref(&self) -> &T where <S as SchedulerSystem>::Desc: StackfulTaskDesc { unsafe { &*self.mutex.data.get() } }
}

impl<S: StackfulSchedulerSystem, T: Send> DerefMut for McsMutexGuard<'_, S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    fn deref_mut(&mut self) -> &mut T where <S as SchedulerSystem>::Desc: StackfulTaskDesc { unsafe { &mut *self.mutex.data.get() } }
}

impl<S: StackfulSchedulerSystem, T: Send> Drop for McsMutexGuard<'_, S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    fn drop(&mut self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
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
                use crate::resumable::common::worker::Worker as _;
                if let Some(wk) = crate::resumable::common::worker::UltWorker::<S>::current() {
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

impl<S: StackfulSchedulerSystem, T: Send> StackfulMutex<T> for McsMutex<S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    type Guard<'a> = McsMutexGuard<'a, S, T> where Self: 'a, T: 'a;

    fn new(val: T) -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        McsMutex { tail: AtomicPtr::new(null_mut()), data: UnsafeCell::new(val) }
    }

    fn lock(&self) -> McsMutexGuard<'_, S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
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

pub struct McsCondvar<S: StackfulSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    waiters: SpinLock<VecDeque<S::SuspendedThread>>,
}

impl<S: StackfulSchedulerSystem> McsCondvar<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    pub fn new() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        McsCondvar { waiters: SpinLock::new(VecDeque::new()) }
    }

    /// Release `guard`, wait for a notification, then re-acquire and return
    /// a fresh guard.
    pub fn wait<'a, T: Send>(&self, guard: McsMutexGuard<'a, S, T>) -> McsMutexGuard<'a, S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
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

    pub fn notify_one(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        if let Some(sth) = self.waiters.lock().pop_front() { sth.notify(); }
    }

    pub fn notify_all(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        let all: VecDeque<_> = std::mem::take(&mut *self.waiters.lock());
        for sth in all { sth.notify(); }
    }
}
