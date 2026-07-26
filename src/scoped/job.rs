//! [`JobRef`]/[`StackJob`]/[`Latch`] — a stealable reference to a job living
//! on someone's native stack frame, plus the single-purpose completion flag
//! that guards it.
//!
//! Shared by [`sync_engine`](super::sync_engine): the whole point is that a
//! `parallel_call` branch is a plain value on the caller's own stack
//! frame, never separately allocated — unlike a `spawn`/`spawn_async` task,
//! which is a pooled or arena-backed [`crate::BasicTaskDesc`] with a general
//! join-protocol supporting sync joiners, async wakers, and async joiners
//! all at once. `join`'s two branches are always waited on by exactly one,
//! statically-known party (the call itself), never handed out as a reusable
//! handle, so a single `AtomicBool` is enough.
//!
//! [`super::async_engine`] reuses [`JobRef`] too (its trampoline just drives
//! a `Future` to completion instead of calling a `FnOnce` once), but needs
//! its own job/latch shape that can hold a registered [`Waker`](std::task::Waker) —
//! see that module's docs for why.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// Latch — single-purpose completion flag.
// ---------------------------------------------------------------------------

pub(super) struct Latch(AtomicBool);

impl Latch {
    pub(super) fn new() -> Self {
        Latch(AtomicBool::new(false))
    }
    #[inline]
    pub(super) fn set(&self) {
        self.0.store(true, Ordering::Release);
    }
    #[inline]
    pub(super) fn probe(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// JobRef / StackJob
// ---------------------------------------------------------------------------

/// Type-erased, two-word reference to a job living on someone's native
/// stack frame. Never separately allocated — the analogue of
/// `SuspendedUlt<D>`, but pointing at a stack value instead of a pooled
/// descriptor.
#[derive(Clone, Copy)]
pub(super) struct JobRef {
    pub(super) data: *const (),
    execute_fn: unsafe fn(*const ()),
}

unsafe impl Send for JobRef {}

impl JobRef {
    /// Build a `JobRef` from raw parts — used by [`super::async_engine`],
    /// whose jobs are `Arc`-owned rather than a `StackJob` this module
    /// controls end-to-end (see that module's docs for why).
    ///
    /// # Safety
    /// `execute_fn` must be a valid trampoline for whatever `data` actually
    /// points at.
    #[inline]
    pub(super) unsafe fn from_raw_parts(data: *const (), execute_fn: unsafe fn(*const ())) -> Self {
        JobRef { data, execute_fn }
    }

    #[inline]
    pub(super) unsafe fn execute(self) {
        unsafe { (self.execute_fn)(self.data) }
    }
}

/// A `parallel_call()` call's second branch. Lives on the caller's own
/// stack; never pooled, never boxed. `Sync` so a thief on another thread
/// can read `func`/`result` through `&StackJob` — synchronized entirely by
/// `latch` (the thief's pre-`set()` writes happen-before the pusher's
/// post-`probe()` reads, and only one side ever touches `func`/`result` at
/// a time: either the pusher, inline, if never stolen, or the thief,
/// exclusively, once `execute` has been dispatched to it).
pub(super) struct StackJob<F, R> {
    pub(super) latch: Latch,
    func: UnsafeCell<Option<F>>,
    result: UnsafeCell<Option<R>>,
}

unsafe impl<F: Send, R: Send> Sync for StackJob<F, R> {}

impl<F, R> StackJob<F, R>
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    pub(super) fn new(f: F) -> Self {
        StackJob { latch: Latch::new(), func: UnsafeCell::new(Some(f)), result: UnsafeCell::new(None) }
    }

    pub(super) fn as_job_ref(&self) -> JobRef {
        JobRef { data: self as *const Self as *const (), execute_fn: Self::execute_trampoline }
    }

    unsafe fn execute_trampoline(this: *const ()) {
        let this = unsafe { &*(this as *const Self) };
        let f = unsafe { (*this.func.get()).take() }.expect("cmpth: StackJob executed twice");
        let r = f();
        unsafe { *this.result.get() = Some(r) };
        // Publishes the result write above (Release) — a waiter's Acquire
        // probe()==true is guaranteed to see it.
        this.latch.set();
    }

    /// Run inline: we (the pusher) popped this job back off ourselves,
    /// unstolen. No latch involved — nobody else could have touched it.
    pub(super) fn run_inline(&self) -> R {
        let f = unsafe { (*self.func.get()).take() }.expect("cmpth: StackJob run twice");
        f()
    }

    pub(super) fn take_result(&self) -> R {
        unsafe { (*self.result.get()).take() }.expect("cmpth: StackJob latch set without a result")
    }
}
