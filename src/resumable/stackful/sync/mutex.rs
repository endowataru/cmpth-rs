use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};

use crate::spin::SpinLock;
use crate::traits::{Resumable, StackfulMutex, StackfulResumable};
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackful::system::StackfulSchedulerSystem;

// ---------------------------------------------------------------------------
// MutexCore
// ---------------------------------------------------------------------------

pub struct MutexState<S: StackfulSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    pub(super) locked: bool,
    pub(super) waiters: VecDeque<S::SuspendedThread>,
}

/// Raw mutex storage: this crate's own `SpinLock<MutexState>` wait-queue
/// representation, plus the guarded payload. Implementing this opts a type
/// into [`StackfulMutex`] for free via the blanket impl below — the same
/// two-tier relationship as
/// [`TaskDescCore`](crate::resumable::common::desc::TaskDescCore)/[`TaskDesc`](crate::resumable::common::desc::TaskDesc).
pub trait MutexCore: Send + Sync + Sized where <<Self as MutexCore>::StackfulSchedulerSystem as SchedulerSystem>::Desc: StackfulTaskDesc {
    type StackfulSchedulerSystem: StackfulSchedulerSystem;
    type Data: Send;

    fn new_core(data: Self::Data) -> Self where <<Self as MutexCore>::StackfulSchedulerSystem as SchedulerSystem>::Desc: StackfulTaskDesc;
    fn state(&self) -> &SpinLock<MutexState<Self::StackfulSchedulerSystem>>;
    fn data(&self) -> &UnsafeCell<Self::Data>;
}

// --- shared lock/unlock algorithm, used by the blanket StackfulMutex impl
// below, by Drop for MutexGuard, and by Condvar (which locks/unlocks a
// Mutex directly around its wait queue, without going through a
// MutexGuard's Drop — see Condvar::wait). --------------------------------

fn mutex_lock<M: MutexCore>(m: &M) -> MutexGuard<'_, M> where <<M as MutexCore>::StackfulSchedulerSystem as SchedulerSystem>::Desc: StackfulTaskDesc {
    let mut s = m.state().lock();
    if !s.locked {
        s.locked = true;
        return MutexGuard { mutex: m };
    }
    s.waiters.push_back(Default::default());
    let sth: *const <M::StackfulSchedulerSystem as StackfulSchedulerSystem>::SuspendedThread = s.waiters.back().unwrap();
    unsafe { &*sth }.wait_with(move || drop(s));
    MutexGuard { mutex: m }
}

// Currently unused (no StackfulMutex::try_lock exists yet) — preserved
// as-is from the pre-split MutexCore::try_lock_impl, out of scope here.
#[allow(dead_code)]
fn mutex_try_lock<M: MutexCore>(m: &M) -> Option<MutexGuard<'_, M>> {
    let mut s = m.state().lock();
    if !s.locked {
        s.locked = true;
        Some(MutexGuard { mutex: m })
    } else {
        None
    }
}

fn mutex_unlock<M: MutexCore>(m: &M) where <<M as MutexCore>::StackfulSchedulerSystem as SchedulerSystem>::Desc: StackfulTaskDesc {
    let next = {
        let mut s = m.state().lock();
        match s.waiters.pop_front() {
            Some(sth) => Some(sth),
            None => { s.locked = false; None }
        }
    };
    if let Some(sth) = next {
        sth.notify();
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
    fn drop(&mut self) { mutex_unlock(self.mutex); }
}

// ---------------------------------------------------------------------------
// Mutex
// ---------------------------------------------------------------------------

pub struct Mutex<S: StackfulSchedulerSystem, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    state: SpinLock<MutexState<S>>,
    data: UnsafeCell<T>,
}

unsafe impl<S: StackfulSchedulerSystem, T: Send> Send for Mutex<S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}
unsafe impl<S: StackfulSchedulerSystem, T: Send> Sync for Mutex<S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}

impl<S: StackfulSchedulerSystem, T: Send> MutexCore for Mutex<S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    type StackfulSchedulerSystem = S;
    type Data = T;
    fn new_core(val: T) -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        Mutex {
            state: SpinLock::new(MutexState { locked: false, waiters: VecDeque::new() }),
            data: UnsafeCell::new(val),
        }
    }
    fn state(&self) -> &SpinLock<MutexState<S>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc { &self.state }
    fn data(&self) -> &UnsafeCell<T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc { &self.data }
}

/// Blanket [`StackfulMutex`] for any [`MutexCore`]: the lock/new algorithm
/// lives here (via the free functions above), not as trait defaults on
/// `MutexCore`, so that trait stays a pure accessor contract.
impl<M: MutexCore> StackfulMutex<M::Data> for M {
    type Guard<'a> = MutexGuard<'a, M> where Self: 'a, M::Data: 'a;

    fn new(val: M::Data) -> Self {
        M::new_core(val)
    }

    fn lock(&self) -> MutexGuard<'_, Self> {
        mutex_lock(self)
    }
}

// ---------------------------------------------------------------------------
// Condvar
// ---------------------------------------------------------------------------

pub struct Condvar<S: StackfulSchedulerSystem> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    waiters: SpinLock<VecDeque<S::SuspendedThread>>,
}

unsafe impl<S: StackfulSchedulerSystem> Send for Condvar<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}
unsafe impl<S: StackfulSchedulerSystem> Sync for Condvar<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {}

impl<S: StackfulSchedulerSystem> Condvar<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
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
        unsafe { &*sth }.wait_with(move || { drop(w); mutex_unlock(mutex); });
        mutex_lock(mutex)
    }

    pub fn notify_one(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        if let Some(sth) = self.waiters.lock().pop_front() { sth.notify(); }
    }

    pub fn notify_all(&self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        let sths: Vec<_> = self.waiters.lock().drain(..).collect();
        for sth in sths { sth.notify(); }
    }
}

impl<S: StackfulSchedulerSystem> Default for Condvar<S> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    fn default() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc { Self::new() }
}
