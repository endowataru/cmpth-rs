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
use crate::ult::system::UltSystem;
use crate::ult::worker::{UltWorker, Worker};

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
        *self.anchor.index.get_or_init(|| NEXT_ULT_TLS_KEY.fetch_add(1, Ordering::Relaxed))
    }
}

impl<S, T> Default for UltTls<S, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: UltSystem, T: 'static> TlsSlot<T> for UltTls<S, T> {
    fn from_anchor(anchor: &'static crate::traits::thread_system::TlsAnchor) -> &'static Self {
        // Sound: repr(transparent) over TlsAnchor (PhantomData is a ZST).
        unsafe { &*(anchor as *const _ as *const Self) }
    }

    const INIT: Self = UltTls::new();

    fn get(&self) -> *mut T {
        let wk = UltWorker::<S>::current()
            .expect("cmpth: ULT-local storage accessed outside a worker");
        let desc = wk.cur_task.get();
        let map = unsafe { &*(*desc).tls.get() };
        map.as_ref()
            .and_then(|m| m.get(&self.key()).copied())
            .unwrap_or(std::ptr::null_mut())
            .cast()
    }

    fn set(&self, p: *mut T) {
        let wk = UltWorker::<S>::current()
            .expect("cmpth: ULT-local storage accessed outside a worker");
        let desc = wk.cur_task.get();
        let map = unsafe { &mut *(*desc).tls.get() };
        map.get_or_insert_with(HashMap::new).insert(self.key(), p.cast());
    }
}
