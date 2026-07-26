//! `DualMutex<S, T, N>` — a prototype MCS mutex generic over the wait-slot
//! flavor `N` (`BasicSuspendedThread<S>` / `SuspendedFuture<S>` /
//! `SuspendedTask<S>`), demonstrating the `docs/sync-async-unification.md`
//! design end to end. Kept separate from the existing, battle-tested
//! `McsMutex` rather than retrofitting it in place, to keep this exploratory
//! branch's blast radius small — see the design doc's "suggested
//! implementation order" for folding this back into `McsMutex` later.
//!
//! The "shared algorithm, parameterized wait step" idea from the design doc
//! is implemented here as a plain shared function (`start_lock`) rather than
//! a `macro_rules!`: declarative macros are hygienic, so a `$wait:expr`
//! fragment supplied at the call site can't see `let`-bindings (like `prev`/
//! `node_ptr`) introduced inside the macro body. A shared function sidesteps
//! that entirely and is arguably more idiomatic besides — the macro sketch
//! in the design doc should be read as illustrating the *shape* of the
//! sharing, not a literal recipe.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::traits::{Resumable, StackfulMutex, StackfulResumable, StacklessMutex, StacklessResumable};

struct UniNode<N> {
    next: AtomicPtr<UniNode<N>>,
    wait: N,
}

impl<N: Default> Default for UniNode<N> {
    fn default() -> Self {
        UniNode { next: AtomicPtr::new(null_mut()), wait: N::default() }
    }
}

pub struct DualMutex<S, T: Send, N> {
    tail: AtomicPtr<UniNode<N>>,
    data: std::cell::UnsafeCell<T>,
    _marker: PhantomData<S>,
}

unsafe impl<S, T: Send, N: Send> Send for DualMutex<S, T, N> {}
unsafe impl<S, T: Send, N: Send> Sync for DualMutex<S, T, N> {}

impl<S, T: Send, N: Default> DualMutex<S, T, N> {
    pub fn new(val: T) -> Self {
        DualMutex {
            tail: AtomicPtr::new(null_mut()),
            data: std::cell::UnsafeCell::new(val),
            _marker: PhantomData,
        }
    }
}

pub struct DualMutexGuard<'a, S, T: Send, N: Resumable<S>> {
    mutex: &'a DualMutex<S, T, N>,
    node: Box<UniNode<N>>,
    _marker: PhantomData<S>,
}

unsafe impl<S, T: Send, N: Resumable<S> + Send> Send for DualMutexGuard<'_, S, T, N> {}

impl<S, T: Send, N: Resumable<S>> Deref for DualMutexGuard<'_, S, T, N> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.mutex.data.get() }
    }
}

impl<S, T: Send, N: Resumable<S>> DerefMut for DualMutexGuard<'_, S, T, N> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.data.get() }
    }
}

impl<S, T: Send, N: Resumable<S>> Drop for DualMutexGuard<'_, S, T, N> {
    fn drop(&mut self) {
        let node_ptr: *mut UniNode<N> = &mut *self.node;
        if self
            .mutex
            .tail
            .compare_exchange(node_ptr, null_mut(), Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return;
        }
        // Simplified for the prototype: pure spin, no yield-on-stall handling
        // (the real McsMutex's Drop does this — see mcs_mutex.rs).
        let mut next = self.node.next.load(Ordering::Acquire);
        while next.is_null() {
            std::hint::spin_loop();
            next = self.node.next.load(Ordering::Acquire);
        }
        unsafe { (*next).wait.notify() };
    }
}

/// Shared prep: allocate this waiter's node, publish it as the new tail, and
/// return `(node, node_ptr, prev)` — everything both the sync and async
/// entry points need before diverging on "how do I wait if `prev` isn't
/// null". Raw pointers come back as `usize` so the async caller can freely
/// capture them in a `Send` closure across an `.await` point.
fn start_lock<S, T: Send, N: Default>(mutex: &DualMutex<S, T, N>) -> (Box<UniNode<N>>, usize, usize) {
    let node = Box::new(UniNode::default());
    let node_ptr = &*node as *const UniNode<N> as usize;
    let prev = mutex.tail.swap(node_ptr as *mut UniNode<N>, Ordering::AcqRel) as usize;
    (node, node_ptr, prev)
}

impl<S, T, N> StackfulMutex<T> for DualMutex<S, T, N>
where
    T: Send,
    S: Send + Sync + 'static,
    N: StackfulResumable<S> + Send + Sync + 'static,
{
    type Guard<'a>
        = DualMutexGuard<'a, S, T, N>
    where
        Self: 'a,
        T: 'a;

    fn new(val: T) -> Self {
        DualMutex::new(val)
    }

    fn lock(&self) -> Self::Guard<'_> {
        let (node, node_ptr, prev) = start_lock(self);
        if prev != 0 {
            node.wait.wait_with(move || {
                let prev = prev as *mut UniNode<N>;
                unsafe { (*prev).next.store(node_ptr as *mut UniNode<N>, Ordering::Release) };
            });
        }
        DualMutexGuard { mutex: self, node, _marker: PhantomData }
    }
}

impl<S, T, N> StacklessMutex<T> for DualMutex<S, T, N>
where
    T: Send,
    S: Send + Sync + 'static,
    N: StacklessResumable<S> + Send + Sync + 'static,
{
    type Guard<'a>
        = DualMutexGuard<'a, S, T, N>
    where
        Self: 'a,
        T: 'a;

    fn new(val: T) -> Self {
        DualMutex::new(val)
    }

    async fn lock<'a>(&'a self) -> Self::Guard<'a>
    where
        T: 'a,
    {
        let (node, node_ptr, prev) = start_lock(self);
        if prev != 0 {
            StacklessResumable::wait_with(&node.wait, move || {
                let prev = prev as *mut UniNode<N>;
                unsafe { (*prev).next.store(node_ptr as *mut UniNode<N>, Ordering::Release) };
            })
            .await;
        }
        DualMutexGuard { mutex: self, node, _marker: PhantomData }
    }
}
