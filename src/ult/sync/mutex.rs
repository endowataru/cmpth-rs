use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};

use crate::spin::SpinLock;
use crate::traits::{Resumable, StackfulMutex, StackfulResumable};
use crate::ult::desc::StackfulTaskDesc;
use crate::ult::system::{SchedulerSystem, UltSchedulerSystem};

// ---------------------------------------------------------------------------
// MutexCore
// ---------------------------------------------------------------------------

pub struct MutexState<S: UltSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    pub(super) locked: bool,
    pub(super) waiters: VecDeque<S::SuspendedThread>,
}

pub trait MutexCore: Send + Sync + Sized where <<Self as MutexCore>::UltSchedulerSystem as SchedulerSystem>::Desc: StackfulTaskDesc {
    type UltSchedulerSystem: UltSchedulerSystem;
    type Data: Send;

    fn state(&self) -> &SpinLock<MutexState<Self::UltSchedulerSystem>>;
    fn data(&self) -> &UnsafeCell<Self::Data>;

    fn lock_impl(&self) -> MutexGuard<'_, Self> where <<Self as MutexCore>::UltSchedulerSystem as SchedulerSystem>::Desc: StackfulTaskDesc {
        let mut s = self.state().lock();
        if !s.locked {
            s.locked = true;
            return MutexGuard { mutex: self };
        }
        s.waiters.push_back(Default::default());
        let sth: *const <Self::UltSchedulerSystem as UltSchedulerSystem>::SuspendedThread = s.waiters.back().unwrap();
        unsafe { &*sth }.wait_with(move || drop(s));
        MutexGuard { mutex: self }
    }

    fn try_lock_impl(&self) -> Option<MutexGuard<'_, Self>> {
        let mut s = self.state().lock();
        if !s.locked {
            s.locked = true;
            Some(MutexGuard { mutex: self })
        } else {
            None
        }
    }

    fn unlock_impl(&self) where <<Self as MutexCore>::UltSchedulerSystem as SchedulerSystem>::Desc: StackfulTaskDesc {
        let next = {
            let mut s = self.state().lock();
            match s.waiters.pop_front() {
                Some(sth) => Some(sth),
                None => { s.locked = false; None }
            }
        };
        if let Some(sth) = next {
            sth.notify();
        }
    }
}

// ---------------------------------------------------------------------------
// MutexGuard
// ---------------------------------------------------------------------------

pub struct MutexGuard<'a, M: MutexCore> {
    mutex: &'a M,
}

impl<M: MutexCore> Deref for MutexGuard<'_, M> {
    type Target = M::Data;
    fn deref(&self) -> &M::Data { unsafe { &*self.mutex.data().get() } }
}

impl<M: MutexCore> DerefMut for MutexGuard<'_, M> {
    fn deref_mut(&mut self) -> &mut M::Data { unsafe { &mut *self.mutex.data().get() } }
}

impl<M: MutexCore> Drop for MutexGuard<'_, M> {
    fn drop(&mut self) { self.mutex.unlock_impl(); }
}

// ---------------------------------------------------------------------------
// Mutex
// ---------------------------------------------------------------------------

pub struct Mutex<S: UltSchedulerSystem, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    state: SpinLock<MutexState<S>>,
    data: UnsafeCell<T>,
}

unsafe impl<S: UltSchedulerSystem, T: Send> Send for Mutex<S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}
unsafe impl<S: UltSchedulerSystem, T: Send> Sync for Mutex<S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}

impl<S: UltSchedulerSystem, T: Send> MutexCore for Mutex<S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    type UltSchedulerSystem = S;
    type Data = T;
    fn state(&self) -> &SpinLock<MutexState<S>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc { &self.state }
    fn data(&self) -> &UnsafeCell<T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc { &self.data }
}

impl<S: UltSchedulerSystem, T: Send> StackfulMutex<T> for Mutex<S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    type Guard<'a> = MutexGuard<'a, Mutex<S, T>> where Self: 'a, T: 'a;

    fn new(val: T) -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        Mutex {
            state: SpinLock::new(MutexState { locked: false, waiters: VecDeque::new() }),
            data: UnsafeCell::new(val),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Self> where <S as SchedulerSystem>::Desc: StackfulTaskDesc { self.lock_impl() }
}

// ---------------------------------------------------------------------------
// Condvar
// ---------------------------------------------------------------------------

pub struct Condvar<S: UltSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    waiters: SpinLock<VecDeque<S::SuspendedThread>>,
}

unsafe impl<S: UltSchedulerSystem> Send for Condvar<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}
unsafe impl<S: UltSchedulerSystem> Sync for Condvar<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}

impl<S: UltSchedulerSystem> Condvar<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    pub fn new() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        Condvar { waiters: SpinLock::new(VecDeque::new()) }
    }

    pub fn wait<'a, T: Send>(
        &self,
        guard: MutexGuard<'a, Mutex<S, T>>,
    ) -> MutexGuard<'a, Mutex<S, T>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        let mutex = guard.mutex;
        std::mem::forget(guard);
        let mut w = self.waiters.lock();
        w.push_back(Default::default());
        let sth: *const S::SuspendedThread = w.back().unwrap();
        unsafe { &*sth }.wait_with(move || { drop(w); mutex.unlock_impl(); });
        mutex.lock_impl()
    }

    pub fn notify_one(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        if let Some(sth) = self.waiters.lock().pop_front() { sth.notify(); }
    }

    pub fn notify_all(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        let sths: Vec<_> = self.waiters.lock().drain(..).collect();
        for sth in sths { sth.notify(); }
    }
}

impl<S: UltSchedulerSystem> Default for Condvar<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    fn default() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc { Self::new() }
}
