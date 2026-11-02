use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};

use crate::spin::SpinLock;
use crate::traits::{Resumable, StackfulMutex, StackfulResumable};
use crate::traits::{StackfulParking, SuspendableSystem};

// ---------------------------------------------------------------------------
// MutexCore
// ---------------------------------------------------------------------------

pub struct MutexState<S: StackfulParking> {
    pub(super) locked: bool,
    pub(super) waiters: VecDeque<S::SuspendedThread>,
}

/// Raw mutex storage: this crate's own `SpinLock<MutexState>` wait-queue
/// representation, plus the guarded payload. Implementing this opts a type
/// into [`StackfulMutex`] for free via the blanket impl below — the same
/// two-tier relationship as
/// [`TaskDescCore`](crate::resumable::common::desc::TaskDescCore)/[`TaskDesc`](crate::resumable::common::desc::TaskDesc).
pub trait MutexCore: Send + Sync + Sized {
    type System: StackfulParking;
    type Data: Send;

    fn new_core(data: Self::Data) -> Self;
    fn state(&self) -> &SpinLock<MutexState<Self::System>>;
    fn data(&self) -> &UnsafeCell<Self::Data>;
}

// --- shared lock/unlock algorithm, used by the blanket StackfulMutex impl
// below, by Drop for MutexGuard, and by Condvar (which locks/unlocks a
// Mutex directly around its wait queue, without going through a
// MutexGuard's Drop — see Condvar::wait). --------------------------------

fn mutex_lock<M: MutexCore>(m: &M) -> MutexGuard<'_, M> {
    let mut s = m.state().lock();
    if !s.locked {
        s.locked = true;
        return MutexGuard { mutex: m };
    }
    s.waiters.push_back(Default::default());
    let sth: *const <M::System as SuspendableSystem>::SuspendedThread = s.waiters.back().unwrap();
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

fn mutex_unlock<M: MutexCore>(m: &M) {
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

pub struct Mutex<S: StackfulParking, T> {
    state: SpinLock<MutexState<S>>,
    data: UnsafeCell<T>,
}

unsafe impl<S: StackfulParking, T: Send> Send for Mutex<S, T> {}
unsafe impl<S: StackfulParking, T: Send> Sync for Mutex<S, T> {}

impl<S: StackfulParking, T: Send> MutexCore for Mutex<S, T> {
    type System = S;
    type Data = T;
    fn new_core(val: T) -> Self {
        Mutex {
            state: SpinLock::new(MutexState { locked: false, waiters: VecDeque::new() }),
            data: UnsafeCell::new(val),
        }
    }
    fn state(&self) -> &SpinLock<MutexState<S>> { &self.state }
    fn data(&self) -> &UnsafeCell<T> { &self.data }
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

pub struct Condvar<S: StackfulParking> {
    waiters: SpinLock<VecDeque<S::SuspendedThread>>,
}

unsafe impl<S: StackfulParking> Send for Condvar<S> {}
unsafe impl<S: StackfulParking> Sync for Condvar<S> {}

impl<S: StackfulParking> Condvar<S> {
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
        unsafe { &*sth }.wait_with(move || { drop(w); mutex_unlock(mutex); });
        mutex_lock(mutex)
    }

    pub fn notify_one(&self) {
        if let Some(sth) = self.waiters.lock().pop_front() { sth.notify(); }
    }

    pub fn notify_all(&self) {
        let sths: Vec<_> = self.waiters.lock().drain(..).collect();
        for sth in sths { sth.notify(); }
    }
}

impl<S: StackfulParking> Default for Condvar<S> {
    fn default() -> Self { Self::new() }
}
