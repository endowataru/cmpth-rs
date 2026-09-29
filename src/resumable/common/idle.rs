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

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering, fence};
use std::sync::{Condvar, Mutex};

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

/// Spins, then yields, then blocks the OS thread on a condition variable.
///
/// Blocks the *OS thread*: the pool's `Base` must be a system whose threads
/// are OS threads (`OsSystem`); parking a worker that is itself a user-level
/// thread of another scheduler would stall that scheduler's worker.
///
/// Protocol (rayon's sleep protocol, reduced to one condvar):
///
/// 1. After enough empty rounds a worker *registers* (`sleepers += 1`),
///    snapshots `epoch`, and returns to the caller for one more full search.
/// 2. If that search finds nothing, the next call takes the lock and waits
///    *only if `epoch` is unchanged*.
/// 3. A producer, after publishing work, issues a `SeqCst` fence and reads
///    `sleepers`; if non-zero it bumps `epoch`, passes through the lock, and
///    notifies one waiter.
///
/// Correctness is the classic store-buffering (Dekker) argument: the worker
/// does `sleepers += 1; fence; read queues`, the producer does `write queue;
/// fence; read sleepers`, so at least one of them sees the other. If the
/// producer sees the registration it bumps `epoch`, and the worker's
/// check-under-lock cannot miss it (the producer's lock hand-off orders the
/// bump against the wait). If the worker sees the queue write, the extra
/// search finds the work. The fences are required on weak-memory hardware:
/// neither the deque push nor a `SeqCst` RMW alone orders a store before a
/// later load on another location.
#[derive(Default)]
pub struct ParkIdle {
    sleepers: AtomicUsize,
    epoch: AtomicU64,
    lock: Mutex<()>,
    cv: Condvar,
}

#[derive(Default)]
pub struct ParkEpisode {
    rounds: u32,
    /// `Some(epoch snapshot)` iff this worker is registered in `sleepers`.
    ticket: Option<u64>,
}

impl ParkIdle {
    #[inline(never)]
    fn wake_one(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        drop(self.lock.lock().unwrap());
        self.cv.notify_one();
    }

    fn unregister(&self, ep: &mut ParkEpisode) {
        if ep.ticket.take().is_some() {
            self.sleepers.fetch_sub(1, Ordering::SeqCst);
        }
        ep.rounds = 0;
    }
}

impl IdlePolicy for ParkIdle {
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
        if ep.ticket.is_some() || ep.rounds != 0 {
            self.unregister(ep);
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
                    yield_now();
                } else {
                    self.sleepers.fetch_add(1, Ordering::SeqCst);
                    fence(Ordering::SeqCst);
                    ep.ticket = Some(self.epoch.load(Ordering::SeqCst));
                }
            }
            Some(seen) => {
                let mut g = self.lock.lock().unwrap();
                while self.epoch.load(Ordering::SeqCst) == seen && !finished.load(Ordering::Acquire) {
                    g = self.cv.wait(g).unwrap();
                }
                drop(g);
                self.unregister(ep);
            }
        }
    }

    fn shutdown(&self) {
        fence(Ordering::SeqCst);
        self.epoch.fetch_add(1, Ordering::SeqCst);
        drop(self.lock.lock().unwrap());
        self.cv.notify_all();
    }
}
