//! Idle policy: what a worker does when it has looked everywhere and found
//! nothing, and how a producer tells it that has changed.
//!
//! The two halves are one protocol, which is why they are one trait:
//! [`no_work_found`](IdlePolicy::no_work_found) (consumer side, called from
//! the shared idle loop) and [`new_work`](IdlePolicy::new_work) (producer
//! side, called after every publication of stealable work) must agree on how
//! a sleeping worker is woken without a lost wakeup. A policy that never
//! sleeps ([`SpinIdle`]) has nothing to agree on, and `new_work` compiles
//! away entirely.
//!
//! Swap the implementation via [`crate::WorkerSystem::Idle`].

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering, fence};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// Idle/wakeup protocol of one worker pool. One value is shared by all
/// workers of a pool (it lives in `PoolCore`).
///
/// # Contract
///
/// * `new_work` is called by *any* thread after it made an item visible to
///   thieves (local `push`/`defer`, external-queue push). It must be cheap
///   when nobody is idle.
/// * `no_work_found` is called by an idle worker after each search round
///   that found nothing. The caller re-searches after it returns, so a policy
///   may use consecutive calls as "announce intent to sleep", then "re-check",
///   then "sleep" (see [`ParkIdle`]).
/// * `work_found` is called when a search succeeds, ending the idle episode.
/// * `shutdown` is called once after `finished` has been set; every blocked
///   worker must return from `no_work_found` promptly.
pub trait IdlePolicy: Default + Send + Sync + 'static {
    /// Per-worker state for one idle episode (round counters, a sleep ticket).
    type Episode: Default;

    fn new_work(&self);

    fn work_found(&self, ep: &mut Self::Episode);

    /// `retry` is true when the search saw a victim that had work but could
    /// not be robbed right now ([`Steal::Retry`](crate::resumable::common::deque::Steal::Retry));
    /// such a round is not evidence of idleness.
    ///
    /// `yield_now` yields the *base* thread (`S::Base::yield_now`); policies
    /// must use it rather than `std::thread::yield_now`, since the base need
    /// not be an OS thread.
    fn no_work_found(&self, ep: &mut Self::Episode, retry: bool, finished: &AtomicBool, yield_now: impl Fn());

    fn shutdown(&self);

    /// Told the pool's worker count once, when the pool is built.
    fn set_workers(&self, _n: usize) {}
}

// ---------------------------------------------------------------------------
// SpinIdle
// ---------------------------------------------------------------------------

/// Never sleeps: spin, and yield to the OS every 64 consecutive empty rounds.
/// Lowest wake latency and zero producer-side cost, at the price of burning
/// a core per idle worker.
#[derive(Default)]
pub struct SpinIdle;

impl IdlePolicy for SpinIdle {
    type Episode = u32;

    #[inline(always)]
    fn new_work(&self) {}

    #[inline(always)]
    fn work_found(&self, ep: &mut u32) {
        *ep = 0;
    }

    #[inline(always)]
    fn no_work_found(&self, ep: &mut u32, retry: bool, _finished: &AtomicBool, yield_now: impl Fn()) {
        std::hint::spin_loop();
        if !retry {
            *ep += 1;
            if *ep & 0x3F == 0 {
                crate::duty::idle_hook();
                yield_now();
            }
        }
    }

    #[inline(always)]
    fn shutdown(&self) {}
}

// ---------------------------------------------------------------------------
// ParkIdle
// ---------------------------------------------------------------------------

const SPIN_ROUNDS: u32 = 64;
const YIELD_ROUNDS: u32 = 64;

/// Spins, then yields, then blocks the OS thread until a producer wakes it,
/// through `W` ([`FutexWait`] by default, [`CondvarWait`]).
///
/// Blocks the *OS thread*: the pool's `Base` must be a system whose threads
/// are OS threads (`OsSystem`); parking a worker that is itself a user-level
/// thread of another scheduler would stall that scheduler's worker.
///
/// Protocol (rayon's sleep protocol, reduced to one wait address):
///
/// 1. After enough empty rounds a worker *registers* (`sleepers += 1`),
///    snapshots `epoch`, and returns to the caller for one more full search.
/// 2. If that search finds nothing, the next call waits *only if `epoch`
///    is unchanged*.
/// 3. A producer, after publishing work, issues a `SeqCst` fence and reads
///    `sleepers`; if non-zero it bumps `epoch` and wakes one waiter.
///
/// Correctness is the classic store-buffering (Dekker) argument: the worker
/// does `sleepers += 1; fence; read queues`, the producer does `write queue;
/// fence; read sleepers`, so at least one of them sees the other. If the
/// producer sees the registration it bumps `epoch`, and the worker's wait
/// cannot miss it: the kernel compares `epoch` with the snapshot atomically
/// with queueing the waiter ([`FutexWait`]), or the check runs under the
/// lock the producer passes through ([`CondvarWait`]). If the worker sees the queue write, the extra
/// search finds the work. The fences are required on weak-memory hardware:
/// neither the deque push nor a `SeqCst` RMW alone orders a store before a
/// later load on another location.
#[derive(Default)]
pub struct ParkIdle<W: ParkWait = FutexWait> {
    sleepers: AtomicUsize,
    epoch: AtomicU32,
    wait: W,
}

/// How a parked worker blocks its OS thread until `epoch` changes: the
/// other half of [`ParkIdle`].
pub trait ParkWait: Default + Send + Sync + 'static {
    /// Block while `epoch == seen` and `finished` is false (spurious
    /// returns allowed: the caller re-checks).
    fn wait(&self, epoch: &AtomicU32, seen: u32, finished: &AtomicBool);

    /// [`wait`](Self::wait), returning after `timeout` at the latest.
    fn wait_timeout(&self, epoch: &AtomicU32, seen: u32, finished: &AtomicBool, timeout: Duration);

    /// Called after `epoch` was bumped: wake one waiter.
    fn wake_one(&self, epoch: &AtomicU32);

    /// Called after `epoch` was bumped: wake every waiter.
    fn wake_all(&self, epoch: &AtomicU32);
}

/// Waits on `epoch`'s address in the kernel (Linux `futex`, macOS
/// `os_sync_wait_on_address`, macOS 14.4+): a wake costs one system call
/// and no lock.  Elsewhere it falls back to yielding in a loop.
#[derive(Default)]
pub struct FutexWait;

impl ParkWait for FutexWait {
    fn wait(&self, epoch: &AtomicU32, seen: u32, finished: &AtomicBool) {
        if epoch.load(Ordering::SeqCst) == seen && !finished.load(Ordering::Acquire) {
            futex::wait(epoch, seen);
        }
    }

    fn wait_timeout(&self, epoch: &AtomicU32, seen: u32, finished: &AtomicBool, timeout: Duration) {
        if epoch.load(Ordering::SeqCst) == seen && !finished.load(Ordering::Acquire) {
            futex::wait_timeout(epoch, seen, timeout);
        }
    }

    #[inline(never)]
    fn wake_one(&self, epoch: &AtomicU32) {
        futex::wake(epoch, false);
    }

    fn wake_all(&self, epoch: &AtomicU32) {
        futex::wake(epoch, true);
    }
}

/// Waits on a condition variable (one mutex for the pool): portable, but a
/// wake takes the lock and the woken thread takes it again.
#[derive(Default)]
pub struct CondvarWait {
    lock: Mutex<()>,
    cv: Condvar,
}

impl ParkWait for CondvarWait {
    fn wait(&self, epoch: &AtomicU32, seen: u32, finished: &AtomicBool) {
        let mut g = self.lock.lock().unwrap();
        while epoch.load(Ordering::SeqCst) == seen && !finished.load(Ordering::Acquire) {
            g = self.cv.wait(g).unwrap();
        }
    }

    fn wait_timeout(&self, epoch: &AtomicU32, seen: u32, finished: &AtomicBool, timeout: Duration) {
        let g = self.lock.lock().unwrap();
        if epoch.load(Ordering::SeqCst) == seen && !finished.load(Ordering::Acquire) {
            drop(self.cv.wait_timeout(g, timeout).unwrap());
        }
    }

    #[inline(never)]
    fn wake_one(&self, _epoch: &AtomicU32) {
        drop(self.lock.lock().unwrap());
        self.cv.notify_one();
    }

    fn wake_all(&self, _epoch: &AtomicU32) {
        drop(self.lock.lock().unwrap());
        self.cv.notify_all();
    }
}

mod futex {
    use std::sync::atomic::AtomicU32;
    use std::time::Duration;

    #[cfg(target_os = "macos")]
    unsafe extern "C" {
        fn os_sync_wait_on_address(addr: *mut libc::c_void, value: u64, size: usize, flags: u32) -> i32;
        fn os_sync_wait_on_address_with_timeout(
            addr: *mut libc::c_void,
            value: u64,
            size: usize,
            flags: u32,
            clockid: u32,
            timeout_ns: u64,
        ) -> i32;
        fn os_sync_wake_by_address_any(addr: *mut libc::c_void, size: usize, flags: u32) -> i32;
        fn os_sync_wake_by_address_all(addr: *mut libc::c_void, size: usize, flags: u32) -> i32;
    }

    #[cfg(target_os = "macos")]
    pub(super) fn wait(a: &AtomicU32, seen: u32) {
        unsafe { os_sync_wait_on_address(a.as_ptr() as *mut _, seen as u64, 4, 0) };
    }

    #[cfg(target_os = "macos")]
    pub(super) fn wait_timeout(a: &AtomicU32, seen: u32, t: Duration) {
        // OS_CLOCK_MACH_ABSOLUTE_TIME
        let ns = (t.as_nanos() as u64).max(1);
        unsafe { os_sync_wait_on_address_with_timeout(a.as_ptr() as *mut _, seen as u64, 4, 0, 32, ns) };
    }

    #[cfg(target_os = "macos")]
    pub(super) fn wake(a: &AtomicU32, all: bool) {
        unsafe {
            if all {
                os_sync_wake_by_address_all(a.as_ptr() as *mut _, 4, 0);
            } else {
                os_sync_wake_by_address_any(a.as_ptr() as *mut _, 4, 0);
            }
        }
    }

    #[cfg(target_os = "linux")]
    pub(super) fn wait(a: &AtomicU32, seen: u32) {
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                a.as_ptr(),
                libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                seen,
                std::ptr::null::<libc::timespec>(),
            )
        };
    }

    #[cfg(target_os = "linux")]
    pub(super) fn wait_timeout(a: &AtomicU32, seen: u32, t: Duration) {
        let ts = libc::timespec { tv_sec: t.as_secs() as libc::time_t, tv_nsec: t.subsec_nanos() as libc::c_long };
        unsafe {
            libc::syscall(libc::SYS_futex, a.as_ptr(), libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG, seen, &ts as *const libc::timespec)
        };
    }

    #[cfg(target_os = "linux")]
    pub(super) fn wake(a: &AtomicU32, all: bool) {
        let n = if all { i32::MAX } else { 1 };
        unsafe { libc::syscall(libc::SYS_futex, a.as_ptr(), libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG, n) };
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    pub(super) fn wait(_a: &AtomicU32, _seen: u32) {
        std::thread::yield_now();
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    pub(super) fn wait_timeout(_a: &AtomicU32, _seen: u32, _t: Duration) {
        std::thread::yield_now();
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    pub(super) fn wake(_a: &AtomicU32, _all: bool) {}
}

#[derive(Default)]
pub struct ParkEpisode {
    rounds: u32,
    /// `Some(epoch snapshot)` iff this worker is registered in `sleepers`.
    ticket: Option<u32>,
    /// This worker keeps the duties' due times while parked
    /// (`duty::park_timeout`).
    timed: bool,
}

impl<W: ParkWait> ParkIdle<W> {
    #[inline(never)]
    fn wake_one(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        self.wait.wake_one(&self.epoch);
    }

    fn unregister(&self, ep: &mut ParkEpisode) {
        if ep.ticket.take().is_some() {
            self.sleepers.fetch_sub(1, Ordering::SeqCst);
        }
        ep.rounds = 0;
    }
}

impl<W: ParkWait> IdlePolicy for ParkIdle<W> {
    type Episode = ParkEpisode;

    #[inline]
    fn new_work(&self) {
        fence(Ordering::SeqCst);
        if self.sleepers.load(Ordering::Relaxed) != 0 {
            self.wake_one();
        }
    }

    #[inline]
    fn work_found(&self, ep: &mut ParkEpisode) {
        if ep.ticket.is_some() || ep.rounds != 0 || ep.timed {
            self.unregister(ep);
            if ep.timed {
                crate::duty::idle_done(&mut ep.timed);
            }
        }
    }

    fn no_work_found(&self, ep: &mut ParkEpisode, retry: bool, finished: &AtomicBool, yield_now: impl Fn()) {
        if retry {
            std::hint::spin_loop();
            return;
        }
        match ep.ticket {
            None => {
                ep.rounds += 1;
                if ep.rounds <= SPIN_ROUNDS {
                    std::hint::spin_loop();
                } else if ep.rounds <= SPIN_ROUNDS + YIELD_ROUNDS {
                    crate::duty::idle_hook();
                    yield_now();
                } else {
                    // Last chance before sleeping: once per park, which
                    // costs a system call anyway.
                    crate::duty::idle_hook();
                    self.sleepers.fetch_add(1, Ordering::SeqCst);
                    fence(Ordering::SeqCst);
                    ep.ticket = Some(self.epoch.load(Ordering::SeqCst));
                }
            }
            Some(seen) => {
                // The one worker that keeps the duties' deadlines while
                // asleep wakes at the earliest (see `duty::park_timeout`).
                match crate::duty::park_timeout(&mut ep.timed) {
                    Some(t) => self.wait.wait_timeout(&self.epoch, seen, finished, t),
                    None => self.wait.wait(&self.epoch, seen, finished),
                }
                self.unregister(ep);
            }
        }
    }

    fn shutdown(&self) {
        fence(Ordering::SeqCst);
        self.epoch.fetch_add(1, Ordering::SeqCst);
        self.wait.wake_all(&self.epoch);
        crate::duty::release_timed();
    }
}

// ---------------------------------------------------------------------------
// SearchIdle
// ---------------------------------------------------------------------------

/// How many idle workers may search (spin) at once, given the pool size,
/// how many are parked and how many search already.
pub trait SearchCap: Default + Send + Sync + 'static {
    fn cap(workers: usize, parked: usize, searching: usize) -> usize;
}

/// Half the busy workers, at least one (Go's scheduler): few searchers
/// when parallelism is low.  The default.
#[derive(Default)]
pub struct GoCap;

impl SearchCap for GoCap {
    #[inline]
    fn cap(workers: usize, parked: usize, searching: usize) -> usize {
        (workers.saturating_sub(parked + searching) / 2).max(1)
    }
}

/// Half the workers, at least one (Tokio's multi-threaded runtime): work
/// is picked up sooner when parallelism is low, at more energy.
#[derive(Default)]
pub struct HalfCap;

impl SearchCap for HalfCap {
    #[inline]
    fn cap(workers: usize, _parked: usize, _searching: usize) -> usize {
        (workers / 2).max(1)
    }
}

/// At most `N` searchers.
#[derive(Default)]
pub struct FixedCap<const N: usize>;

impl<const N: usize> SearchCap for FixedCap<N> {
    #[inline]
    fn cap(_workers: usize, _parked: usize, _searching: usize) -> usize {
        N
    }
}

/// Spin time limit of [`SearchIdle`]: a worker whose idle periods average
/// below this spins for twice its average, otherwise it parks at once.
const SEARCH_LIMIT_NS: u64 = 100_000;

/// The standard parking idle policy: a few idle workers search, the others
/// park ([`ParkIdle`]'s protocol, on a futex).
///
/// * At most `C::cap` workers search at once (Go and Tokio limit their
///   spinning threads the same way); a worker over the cap parks at once.
/// * A searcher spins for twice the average length of its own past idle
///   periods if that is below 100 us, else it parks at once.  The clock is
///   read on the idle worker itself, every 16 rounds; no OS timer.
/// * A producer wakes a sleeper only when nobody searches; a searcher that
///   finds work and was the last one wakes one sleeper, so that while work
///   is around someone keeps looking and parallelism ramps up one worker at
///   a time (Go's `wakep`, Tokio's `transition_worker_from_searching`).
/// * A worker kept awake by a poll hook searches as well and never times
///   out; it counts as a searcher when the cap allows.
///
/// Lost wakeups: a producer that skips the wake because a worker searches
/// published its work before its fence; a searcher that stops decrements
/// `searching` and registers as a sleeper before its fence, then searches
/// once more ([`ParkIdle`]'s protocol), so either the producer sees no
/// searcher (and then the registration, or the searcher sees the work) or
/// the searcher's last search sees the work.  A worker over the cap
/// increments `searching` only briefly and goes the same way.
pub struct SearchIdle<C: SearchCap = GoCap, W: ParkWait = FutexWait> {
    park: ParkIdle<W>,
    searching: AtomicUsize,
    workers: AtomicUsize,
    _cap: std::marker::PhantomData<C>,
}

impl<C: SearchCap, W: ParkWait> Default for SearchIdle<C, W> {
    fn default() -> Self {
        SearchIdle {
            park: ParkIdle::default(),
            searching: AtomicUsize::new(0),
            workers: AtomicUsize::new(1),
            _cap: std::marker::PhantomData,
        }
    }
}

#[derive(Default)]
pub struct SearchEpisode {
    park: ParkEpisode,
    /// `Some(start)` while this worker counts in `searching`.
    search_since: Option<Instant>,
    /// This episode decided to park.
    parking: bool,
    /// Start of this idle period, and the moving average (1/4 weight) of
    /// past periods.
    since: Option<Instant>,
    avg_ns: u64,
    spin_ns: u64,
}

impl<C: SearchCap, W: ParkWait> SearchIdle<C, W> {
    /// Become a searcher if the cap allows; `now` starts the spin.
    fn try_search(&self, ep: &mut SearchEpisode, now: Instant) -> bool {
        let n = self.searching.fetch_add(1, Ordering::SeqCst);
        let parked = self.park.sleepers.load(Ordering::Relaxed);
        if n < C::cap(self.workers.load(Ordering::Relaxed), parked, n) {
            ep.search_since = Some(now);
            ep.park.rounds = 0;
            true
        } else {
            self.searching.fetch_sub(1, Ordering::SeqCst);
            false
        }
    }

    /// Stop searching; whether this was the last searcher.
    fn stop_searching(&self, ep: &mut SearchEpisode) -> bool {
        if ep.search_since.take().is_some() {
            self.searching.fetch_sub(1, Ordering::SeqCst) == 1
        } else {
            false
        }
    }

    fn start_parking(&self, ep: &mut SearchEpisode) {
        ep.parking = true;
        // Straight to registration in `ParkIdle`'s protocol.
        ep.park.rounds = SPIN_ROUNDS + YIELD_ROUNDS;
    }
}

impl<C: SearchCap, W: ParkWait> IdlePolicy for SearchIdle<C, W> {
    type Episode = SearchEpisode;

    fn set_workers(&self, n: usize) {
        self.workers.store(n, Ordering::Relaxed);
    }

    #[inline]
    fn new_work(&self) {
        fence(Ordering::SeqCst);
        if self.searching.load(Ordering::Relaxed) == 0 && self.park.sleepers.load(Ordering::Relaxed) != 0 {
            self.park.wake_one();
        }
    }

    #[inline]
    fn work_found(&self, ep: &mut SearchEpisode) {
        if let Some(t) = ep.since.take() {
            let d = t.elapsed().as_nanos() as u64;
            ep.avg_ns = ep.avg_ns - ep.avg_ns / 4 + d / 4;
        }
        let last = self.stop_searching(ep);
        ep.parking = false;
        self.park.work_found(&mut ep.park);
        if last {
            fence(Ordering::SeqCst);
            if self.park.sleepers.load(Ordering::Relaxed) != 0 {
                self.park.wake_one();
            }
        }
    }

    fn no_work_found(&self, ep: &mut SearchEpisode, retry: bool, finished: &AtomicBool, yield_now: impl Fn()) {
        if retry {
            // Work nearby, or a poll hook waiting: stay awake, as a searcher
            // if the cap allows.
            if ep.search_since.is_none() && !ep.parking {
                self.try_search(ep, Instant::now());
            }
            std::hint::spin_loop();
            return;
        }
        if !ep.parking {
            match ep.search_since {
                None => {
                    let now = Instant::now();
                    if ep.since.is_none() {
                        ep.since = Some(now);
                    }
                    ep.spin_ns = if ep.avg_ns < SEARCH_LIMIT_NS { 2 * ep.avg_ns.max(1000) } else { 0 };
                    ep.spin_ns = ep.spin_ns.max(crate::duty::wait_spin_ns());
                    if self.try_search(ep, now) {
                        return;
                    }
                    self.start_parking(ep);
                }
                Some(since) => {
                    ep.park.rounds += 1;
                    if ep.park.rounds & 0xF != 0 || (since.elapsed().as_nanos() as u64) < ep.spin_ns {
                        if ep.park.rounds & 0x3 == 0 {
                            // A searcher runs the duties often: one may
                            // complete what a suspended task waits for.
                            crate::duty::idle_hook();
                        }
                        if ep.park.rounds & 0x3F == 0 {
                            yield_now();
                        } else {
                            std::hint::spin_loop();
                        }
                        return;
                    }
                    self.stop_searching(ep);
                    self.start_parking(ep);
                }
            }
        }
        self.park.no_work_found(&mut ep.park, false, finished, yield_now);
        if ep.park.ticket.is_none() && ep.park.rounds == 0 {
            // Woken from the park: start over if nothing turns up.
            ep.parking = false;
        }
    }

    fn shutdown(&self) {
        self.park.shutdown();
    }
}
