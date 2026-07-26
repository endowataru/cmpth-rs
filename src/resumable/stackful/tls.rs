//! ULT-local storage: the `ThreadSpecific` implementation for systems nested
//! on top of a ULT scheduler.
//!
//! The value lives in the task descriptor of the current ULT, not in OS TLS,
//! so it migrates with the ULT when the outer scheduler moves it to another
//! OS thread.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::traits::thread_system::TlsSlot;
use crate::resumable::common::desc::TaskDesc;
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackful::system::UltSchedulerSystem;
use crate::resumable::common::worker::{UltWorker, Worker};

static NEXT_ULT_TLS_KEY: AtomicUsize = AtomicUsize::new(0);

#[repr(transparent)]
pub struct UltTls<S, T> {
    anchor: crate::traits::thread_system::TlsAnchor,
    _marker: PhantomData<fn() -> (S, T)>,
}

impl<S, T> UltTls<S, T> {
    pub const fn new() -> Self {
        UltTls { anchor: crate::traits::thread_system::TlsAnchor::new(), _marker: PhantomData }
    }

    fn key(&self) -> usize {
        use crate::traits::thread_system::TLS_ANCHOR_UNASSIGNED;
        let cur = self.anchor.index.load(Ordering::Relaxed);
        if cur != TLS_ANCHOR_UNASSIGNED {
            return cur;
        }
        // Race-safe one-time assignment (see `OsTls::assign_slot`'s doc
        // comment for why a CAS loop suffices in place of `OnceLock` here).
        loop {
            let cur = self.anchor.index.load(Ordering::Relaxed);
            if cur != TLS_ANCHOR_UNASSIGNED {
                return cur;
            }
            let candidate = NEXT_ULT_TLS_KEY.fetch_add(1, Ordering::Relaxed);
            if self
                .anchor
                .index
                .compare_exchange(TLS_ANCHOR_UNASSIGNED, candidate, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return candidate;
            }
        }
    }
}

impl<S, T> Default for UltTls<S, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: UltSchedulerSystem, T: 'static> TlsSlot<T> for UltTls<S, T> where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
    fn from_anchor(anchor: &'static crate::traits::thread_system::TlsAnchor) -> &'static Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        // Sound: repr(transparent) over TlsAnchor (PhantomData is a ZST).
        unsafe { &*(anchor as *const _ as *const Self) }
    }

    const INIT: Self = UltTls::new();

    fn get(&self) -> *mut T where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        let wk = UltWorker::<S>::current()
            .expect("cmpth: ULT-local storage accessed outside a worker");
        let desc = wk.cur_task.get();
        let map = unsafe { &*(*desc).tls().get() };
        map.as_ref()
            .and_then(|m| m.get(&self.key()).copied())
            .unwrap_or(std::ptr::null_mut())
            .cast()
    }

    fn set(&self, p: *mut T) where <S as SchedulerSystem>::Desc: StackfulTaskDesc {
        let wk = UltWorker::<S>::current()
            .expect("cmpth: ULT-local storage accessed outside a worker");
        let desc = wk.cur_task.get();
        let map = unsafe { &mut *(*desc).tls().get() };
        map.get_or_insert_with(HashMap::new).insert(self.key(), p.cast());
    }
}
