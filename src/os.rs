//! `OsSystem`: the bottom-level implementation of [`ThreadSystem`], backed directly
//! by OS threads (`std::thread` + `std::sync`).  Every ULT scheduler is
//! parameterized by a base system; `OsSystem` is the base of the first level.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

use std::task::Context;

use crate::traits::{BarrierWaitResult, Delegator, DelegatorConsumer, StackfulBarrier, StackfulMutex, Poller};
use crate::traits::common::{TaskSystem, TlsSlot};
use crate::traits::stackful::{JoinHandleLike, ThreadSystem, noop_waker};

// ---------------------------------------------------------------------------
// OsPoller — busy-polling Poller for OsSystem
// ---------------------------------------------------------------------------

pub struct OsPoller {
    waker: std::task::Waker,
}

impl Poller for OsPoller {
    fn new() -> Self {
        OsPoller { waker: noop_waker() }
    }

    fn context<'a>(&'a self) -> Context<'a> {
        Context::from_waker(&self.waker)
    }

    fn wait(&self) {
        std::thread::yield_now();
    }
}

// ---------------------------------------------------------------------------
// OsSystem
// ---------------------------------------------------------------------------

pub struct OsSystem;

impl TaskSystem for OsSystem {
    // No managed worker pool -- any code can call `std::thread::spawn`
    // freely, so there's no stable per-thread index to report.
    fn worker_num() -> usize {
        0
    }

    fn num_workers() -> usize {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    }
}

impl ThreadSystem for OsSystem {
    type Poller = OsPoller;

    fn yield_now() {
        std::thread::yield_now();
    }

    type JoinHandle<T: Send + 'static> = std::thread::JoinHandle<T>;

    fn spawn<T, F>(f: F) -> std::thread::JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        std::thread::spawn(f)
    }

    type Mutex<T: Send> = OsMutex<T>;
    type Barrier = OsBarrier;
    type SuspendedThread = OsSuspendedThread;
    type Delegator<C: DelegatorConsumer<Self>> = OsDelegator<C>;
    type ThreadSpecific<T: 'static> = OsTls<T>;
}

impl<T: Send + 'static> JoinHandleLike<T> for std::thread::JoinHandle<T> {
    fn join(self) -> T {
        match self.join() {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        }
    }
}

// ---------------------------------------------------------------------------
// OsTls — per-OS-thread pointer slot
// ---------------------------------------------------------------------------

// All `OsTls` instances share one `thread_local!` array; each instance owns
// a process-wide slot index in it.  This keeps the number of real OS TLS
// variables at exactly one, no matter how many systems are stacked.
//
// The array is fixed-size and `Cell`-based (rather than `RefCell<Vec>`) so
// that `get` — on the spawn/join hot path via `UltWorker::current()` — is a
// TLS base load plus an indexed pointer load, with no borrow bookkeeping and
// no lazy-init branches beyond the slot lookup.  One slot is consumed per
// scheduler level; 16 covers any realistic nesting depth.
const OS_TLS_MAX_SLOTS: usize = 16;

thread_local! {
    static OS_TLS_SLOTS: [std::cell::Cell<*mut ()>; OS_TLS_MAX_SLOTS] =
        const { [const { std::cell::Cell::new(std::ptr::null_mut()) }; OS_TLS_MAX_SLOTS] };
}

static NEXT_OS_TLS_SLOT: AtomicUsize = AtomicUsize::new(0);

#[repr(transparent)]
pub struct OsTls<T> {
    anchor: crate::traits::common::TlsAnchor,
    _marker: PhantomData<fn(T) -> T>,
}

impl<T> Default for OsTls<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OsTls<T> {
    pub const fn new() -> Self {
        OsTls {
            anchor: crate::traits::common::TlsAnchor::new(),
            _marker: PhantomData,
        }
    }

    /// Fast path: a single `Relaxed` load against the (normally already
    /// eagerly-assigned, via [`warm_up`](Self::warm_up)) cached index — no
    /// `OnceLock`-style state check. Falls back to [`assign_slot`] only if
    /// nobody has assigned one yet.
    #[inline]
    fn slot(&self) -> usize {
        let s = self.anchor.index.load(Ordering::Relaxed);
        if s != crate::traits::common::TLS_ANCHOR_UNASSIGNED {
            s
        } else {
            self.assign_slot()
        }
    }

    /// Race-safe, one-time assignment: a plain `fetch_add` for a new index
    /// plus a CAS to publish it, not `OnceLock`. Sound because the
    /// "compute" step here is a wait-free, always-succeeds atomic increment
    /// (no arbitrary/blocking user closure like a general `OnceLock` has to
    /// support) — if two threads race here, the loser's fetched-but-unused
    /// index is simply wasted (never a correctness issue, and in practice
    /// this only happens if `warm_up` wasn't called before concurrent first
    /// use, which the scheduler's setup path avoids).
    #[cold]
    fn assign_slot(&self) -> usize {
        loop {
            let cur = self.anchor.index.load(Ordering::Relaxed);
            if cur != crate::traits::common::TLS_ANCHOR_UNASSIGNED {
                return cur;
            }
            let candidate = NEXT_OS_TLS_SLOT.fetch_add(1, Ordering::Relaxed);
            assert!(candidate < OS_TLS_MAX_SLOTS, "cmpth: too many OsTls slots (max {OS_TLS_MAX_SLOTS})");
            if self
                .anchor
                .index
                .compare_exchange(
                    crate::traits::common::TLS_ANCHOR_UNASSIGNED,
                    candidate,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return candidate;
            }
        }
    }
}

impl<T: 'static> TlsSlot<T> for OsTls<T> {
    const INIT: Self = OsTls::new();

    fn from_anchor(anchor: &'static crate::traits::common::TlsAnchor) -> &'static Self {
        // Sound: repr(transparent) over TlsAnchor (PhantomData is a ZST).
        unsafe { &*(anchor as *const _ as *const Self) }
    }

    // `inline(never)` is load-bearing, not a size tweak: if this body is
    // inlined, LLVM may CSE the thread-local base address across a context
    // switch (the switch is an opaque extern "C" call, so the compiler assumes
    // the OS thread cannot change underneath it).  A ULT that suspends on one
    // worker thread and resumes on another would then read the *old* thread's
    // slots.  An opaque call boundary forces a fresh TLS lookup per call.
    #[inline(never)]
    fn get(&self) -> *mut T {
        let slot = self.slot();
        OS_TLS_SLOTS.with(|v| v[slot].get()).cast()
    }

    #[inline(never)]
    fn set(&self, p: *mut T) {
        let slot = self.slot();
        OS_TLS_SLOTS.with(|v| v[slot].set(p.cast()));
    }

    // Safe to inline: callers of this method (see the trait doc comment)
    // have already proven the OS thread can't change underneath them, so
    // there is no context switch for the compiler to CSE the TLS base
    // address across in the first place.
    #[inline]
    fn get_inline(&self) -> *mut T {
        let slot = self.slot();
        OS_TLS_SLOTS.with(|v| v[slot].get()).cast()
    }

    // Resolve the array index now, single-threaded, before any worker OS
    // thread can race to assign it — see `Scheduler::new`'s callers, which
    // call this on the constructing thread before spawning workers.
    fn warm_up(&self) {
        self.slot();
    }
}

// ---------------------------------------------------------------------------
// std::sync newtypes (required for trait coherence)
// ---------------------------------------------------------------------------

pub struct OsMutex<T>(std::sync::Mutex<T>);

impl<T: Send> StackfulMutex<T> for OsMutex<T> {
    type Guard<'a>
        = std::sync::MutexGuard<'a, T>
    where
        Self: 'a,
        T: 'a;

    fn new(val: T) -> Self {
        OsMutex(std::sync::Mutex::new(val))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, T> {
        self.0.lock().unwrap()
    }
}

/// Condvar paired with [`OsMutex`]. Not part of any generic trait: never used
/// generically through `S::Mutex`, only via this concrete type, so its
/// interface is inherent methods rather than a `Condvar` trait.
pub struct OsCondvar(std::sync::Condvar);

impl OsCondvar {
    pub fn new() -> Self {
        OsCondvar(std::sync::Condvar::new())
    }

    pub fn wait<'a, T>(&self, guard: std::sync::MutexGuard<'a, T>) -> std::sync::MutexGuard<'a, T> {
        self.0.wait(guard).unwrap()
    }

    pub fn notify_one(&self) {
        self.0.notify_one();
    }

    pub fn notify_all(&self) {
        self.0.notify_all();
    }
}

impl Default for OsCondvar {
    fn default() -> Self {
        Self::new()
    }
}

pub struct OsBarrier(std::sync::Barrier);

impl StackfulBarrier for OsBarrier {
    fn new(count: usize) -> Self {
        OsBarrier(std::sync::Barrier::new(count))
    }

    fn wait(&self) -> BarrierWaitResult {
        BarrierWaitResult { is_leader: self.0.wait().is_leader() }
    }
}

// ---------------------------------------------------------------------------
// OsSuspendedThread — OS-level parker for use in OsDelegator
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct OsSuspendedThread {
    inner: Option<std::sync::Arc<OsParker>>,
}

struct OsParker {
    ready: std::sync::Mutex<bool>,
    cv:    std::sync::Condvar,
}

impl OsSuspendedThread {
    fn parker() -> std::sync::Arc<OsParker> {
        std::sync::Arc::new(OsParker {
            ready: std::sync::Mutex::new(false),
            cv:    std::sync::Condvar::new(),
        })
    }

    /// Park the current OS thread; `f` runs before blocking.
    pub fn wait_with<F: FnOnce()>(&mut self, f: F) {
        let p = Self::parker();
        self.inner = Some(p.clone());
        f();
        let mut ready = p.ready.lock().unwrap();
        while !*ready {
            ready = p.cv.wait(ready).unwrap();
        }
        self.inner = None;
    }

    pub fn notify(self) {
        if let Some(p) = self.inner {
            *p.ready.lock().unwrap() = true;
            p.cv.notify_one();
        }
    }
}

unsafe impl Send for OsSuspendedThread {}

// ---------------------------------------------------------------------------
// OsDelegator — delegator backed by OS mutex + condvar
// ---------------------------------------------------------------------------

struct OsDelegatorInner<C: DelegatorConsumer<OsSystem>> {
    consumer:  C,
    locked:    bool,
    queue:     std::collections::VecDeque<(C::Work, OsSuspendedThread)>,
    finished:  bool,
}

pub struct OsDelegator<C: DelegatorConsumer<OsSystem>> {
    inner: std::sync::Mutex<OsDelegatorInner<C>>,
    cv:    std::sync::Condvar,
    th:    std::cell::UnsafeCell<Option<std::thread::JoinHandle<()>>>,
}

unsafe impl<C: DelegatorConsumer<OsSystem>> Send for OsDelegator<C> {}
unsafe impl<C: DelegatorConsumer<OsSystem>> Sync for OsDelegator<C> {}

impl<C: DelegatorConsumer<OsSystem>> OsDelegator<C> {
    fn consumer_loop(
        inner: &std::sync::Mutex<OsDelegatorInner<C>>,
        cv:    &std::sync::Condvar,
    ) {
        loop {
            let mut guard = inner.lock().unwrap();
            // Wait until there is work or we are done.
            while guard.locked && guard.queue.is_empty() && !guard.finished {
                guard = cv.wait(guard).unwrap();
            }
            if guard.finished && guard.queue.is_empty() {
                break;
            }
            if let Some((mut work, sth)) = guard.queue.pop_front() {
                let con = &mut guard.consumer;
                let (done, wake_sth) = con.execute(&mut work);
                drop(guard);
                if done { sth.notify(); }
                if let Some(w) = wake_sth { w.notify(); }
            } else if guard.consumer.is_active() {
                let wake_sth = guard.consumer.progress();
                drop(guard);
                if let Some(w) = wake_sth { w.notify(); }
            } else {
                // No work; release lock and let callers in.
                guard.locked = false;
                cv.notify_all();
            }
        }
    }
}

impl<C: DelegatorConsumer<OsSystem>> Delegator<OsSystem, C> for OsDelegator<C> {
    fn start(consumer: C) -> Self {
        let del = OsDelegator {
            inner: std::sync::Mutex::new(OsDelegatorInner {
                consumer,
                locked:   true,
                queue:    std::collections::VecDeque::new(),
                finished: false,
            }),
            cv: std::sync::Condvar::new(),
            th: std::cell::UnsafeCell::new(None),
        };
        // Spawn consumer thread.
        let inner_ptr = &del.inner as *const _ as usize;
        let cv_ptr    = &del.cv    as *const _ as usize;
        let th = std::thread::spawn(move || {
            let inner = unsafe { &*(inner_ptr as *const std::sync::Mutex<OsDelegatorInner<C>>) };
            let cv    = unsafe { &*(cv_ptr    as *const std::sync::Condvar) };
            Self::consumer_loop(inner, cv);
        });
        unsafe { *del.th.get() = Some(th) };
        del
    }

    fn stop(self) {
        {
            let mut g = self.inner.lock().unwrap();
            g.finished = true;
        }
        self.cv.notify_all();
        if let Some(th) = unsafe { &mut *self.th.get() }.take() {
            th.join().ok();
        }
    }

    fn execute_or_delegate<Imm, Del>(&self, imm: Imm, del: Del)
    where
        Imm: FnOnce(&mut C) -> (bool, Option<OsSuspendedThread>),
        Del: FnOnce(&mut C::Work) -> &OsSuspendedThread,
    {
        let mut guard = self.inner.lock().unwrap();
        if !guard.locked {
            // Acquire lock inline.
            guard.locked = true;
            let (done, wake_sth) = imm(&mut guard.consumer);
            if let Some(w) = wake_sth { w.notify(); }
            if !done { guard.locked = false; }
            self.cv.notify_all();
        } else {
            // Delegate: push work onto queue and block.
            let mut work = C::Work::default();
            let mut sth = OsSuspendedThread::default();
            del(&mut work);
            sth.wait_with(|| {
                guard.queue.push_back((work, OsSuspendedThread::default()));
                self.cv.notify_all();
                drop(guard);
            });
        }
    }
}
