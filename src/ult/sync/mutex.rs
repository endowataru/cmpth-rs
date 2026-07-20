use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};

use crate::spin::SpinLock;
use crate::traits::{Condvar as CondvarTrait, Mutex as MutexTrait, Resumable, StackfulResumable};
use crate::ult::system::UltSchedulerSystem;

// ---------------------------------------------------------------------------
// MutexCore
// ---------------------------------------------------------------------------

pub struct MutexState<S: UltSchedulerSystem> {
    pub(super) locked: bool,
    pub(super) waiters: VecDeque<S::SuspendedThread>,
}

pub trait MutexCore: Send + Sync + Sized {
    type UltSchedulerSystem: UltSchedulerSystem;
    type Data: Send;

    fn state(&self) -> &SpinLock<MutexState<Self::UltSchedulerSystem>>;
    fn data(&self) -> &UnsafeCell<Self::Data>;

    fn lock_impl(&self) -> MutexGuard<'_, Self> {
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

    fn unlock_impl(&self) {
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

pub struct Mutex<S: UltSchedulerSystem, T> {
    state: SpinLock<MutexState<S>>,
    data: UnsafeCell<T>,
}

unsafe impl<S: UltSchedulerSystem, T: Send> Send for Mutex<S, T> {}
unsafe impl<S: UltSchedulerSystem, T: Send> Sync for Mutex<S, T> {}

impl<S: UltSchedulerSystem, T: Send> MutexCore for Mutex<S, T> {
    type UltSchedulerSystem = S;
    type Data = T;
    fn state(&self) -> &SpinLock<MutexState<S>> { &self.state }
    fn data(&self) -> &UnsafeCell<T> { &self.data }
}

impl<S: UltSchedulerSystem, T: Send> MutexTrait<T> for Mutex<S, T> {
    type Guard<'a> = MutexGuard<'a, Mutex<S, T>> where Self: 'a, T: 'a;
    type Condvar = Condvar<S>;

    fn new(val: T) -> Self {
        Mutex {
            state: SpinLock::new(MutexState { locked: false, waiters: VecDeque::new() }),
            data: UnsafeCell::new(val),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Self> { self.lock_impl() }
}

// ---------------------------------------------------------------------------
// Condvar
// ---------------------------------------------------------------------------

pub struct Condvar<S: UltSchedulerSystem> {
    waiters: SpinLock<VecDeque<S::SuspendedThread>>,
}

unsafe impl<S: UltSchedulerSystem> Send for Condvar<S> {}
unsafe impl<S: UltSchedulerSystem> Sync for Condvar<S> {}

impl<S: UltSchedulerSystem> Condvar<S> {
    pub fn new() -> Self {
        Condvar { waiters: SpinLock::new(VecDeque::new()) }
    }

    pub fn wait<'a, T: Send>(
        &self,
        guard: MutexGuard<'a, Mutex<S, T>>,
    ) -> MutexGuard<'a, Mutex<S, T>> {
        let mutex = guard.mutex;
        std::mem::forget(guard);
        let mut w = self.waiters.lock();
        w.push_back(Default::default());
        let sth: *const S::SuspendedThread = w.back().unwrap();
        unsafe { &*sth }.wait_with(move || { drop(w); mutex.unlock_impl(); });
        mutex.lock_impl()
    }

    pub fn notify_one(&self) {
        if let Some(sth) = self.waiters.lock().pop_front() { sth.notify(); }
    }

    pub fn notify_all(&self) {
        let sths: Vec<_> = self.waiters.lock().drain(..).collect();
        for sth in sths { sth.notify(); }
    }
}

impl<S: UltSchedulerSystem> Default for Condvar<S> {
    fn default() -> Self { Self::new() }
}

impl<S: UltSchedulerSystem, T: Send> CondvarTrait<Mutex<S, T>, T> for Condvar<S> {
    fn new() -> Self { Condvar::new() }

    fn wait<'a>(&self, guard: MutexGuard<'a, Mutex<S, T>>) -> MutexGuard<'a, Mutex<S, T>>
    where Mutex<S, T>: 'a, T: 'a,
    {
        Condvar::wait(self, guard)
    }

    fn notify_one(&self) { Condvar::notify_one(self); }
    fn notify_all(&self) { Condvar::notify_all(self); }
}
