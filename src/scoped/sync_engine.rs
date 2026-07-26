//! OS-thread-pool engine backing [`ScopedStackfulTaskSystem`](crate::traits::ScopedStackfulTaskSystem).
//!
//! Mirrors `rayon::join`: push the second branch as a stealable [`JobRef`],
//! run the first branch as an ordinary nested call, then either pop our own
//! job back off (not stolen — finish it with one more ordinary call) or
//! help execute other stealable work while waiting on the latch (stolen).
//! The un-stolen path never touches the latch, the deque's steal side, or
//! any heap allocation at all.
//!
//! # Why this is ~6-7x faster than `spawn`/`spawn_async` on `fib`
//!
//! Measured directly (original `fork_join` prototype, `docs/stackless-perf-investigation.md`):
//! for `fib(34)` (~9.2M `parallel_call()` calls), only 12-61 of them (2-4
//! workers) ever actually got stolen — under 0.001%. `spawn`'s child-first
//! design does a real context switch *unconditionally*, on every call,
//! whether or not the pushed continuation is ever stolen; `spawn_async`
//! similarly registers every call as a fully-fledged pollable task.
//! `parallel_call` only pays for the deque/latch/help-first machinery on
//! the handful of calls a steal actually happens to — the other 99.999%+
//! degrade to two ordinary nested function calls plus one uncontended local
//! deque push/pop.

use crossbeam_deque::{Injector, Steal, Stealer, Worker as Deque};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::job::{JobRef, StackJob};

// ---------------------------------------------------------------------------
// Registry / worker context
// ---------------------------------------------------------------------------

struct Registry {
    stealers: Vec<Stealer<JobRef>>,
    injector: Injector<JobRef>,
    shutdown: AtomicBool,
}

struct WorkerContext {
    index: usize,
    deque: Deque<JobRef>,
    registry: Arc<Registry>,
}

thread_local! {
    static CURRENT: Cell<*const WorkerContext> = const { Cell::new(std::ptr::null()) };
}

fn current_context() -> &'static WorkerContext {
    let p = CURRENT.with(|c| c.get());
    assert!(!p.is_null(), "cmpth: scoped::parallel_call called outside scoped::run");
    unsafe { &*p }
}

/// Non-panicking counterpart of [`current_context`], for `TaskSystem`'s
/// `worker_num`/`num_workers` (which must report *something* even when
/// called from outside a worker, unlike `parallel_call`/`run`).
fn try_current_context() -> Option<&'static WorkerContext> {
    let p = CURRENT.with(|c| c.get());
    if p.is_null() { None } else { Some(unsafe { &*p }) }
}

pub(crate) fn current_worker_num() -> Option<usize> {
    try_current_context().map(|wk| wk.index)
}

pub(crate) fn current_num_workers() -> Option<usize> {
    try_current_context().map(|wk| wk.registry.stealers.len())
}

/// Try to make progress once: pop our own local job, else steal from
/// another worker, else check the global injector. Returns `false` if
/// nothing was found anywhere right now. Shared by the idle worker loop
/// and `parallel_call`'s help-while-waiting loop — the same "what do I do
/// when I have nothing of my own to run" logic either way.
fn try_execute_one(wk: &WorkerContext) -> bool {
    if let Some(job) = wk.deque.pop() {
        unsafe { job.execute() };
        return true;
    }
    let n = wk.registry.stealers.len();
    for off in 1..n {
        let i = (wk.index + off) % n;
        loop {
            match wk.registry.stealers[i].steal() {
                Steal::Success(job) => {
                    unsafe { job.execute() };
                    return true;
                }
                Steal::Empty => break,
                Steal::Retry => continue,
            }
        }
    }
    loop {
        match wk.registry.injector.steal() {
            Steal::Success(job) => {
                unsafe { job.execute() };
                return true;
            }
            Steal::Empty => return false,
            Steal::Retry => continue,
        }
    }
}

// ---------------------------------------------------------------------------
// parallel_call — the public primitive
// ---------------------------------------------------------------------------

/// Work-first fork-join, mirroring `rayon::join`. `a` and `b` are borrowed
/// only for the duration of this call (both are guaranteed complete before
/// this returns), so — unlike `spawn`/`spawn_async` — neither needs
/// `'static`.
///
/// Must be called from within [`run`] (on one of its worker threads,
/// possibly nested inside another call's `a`/`b`).
pub(crate) fn parallel_call<Fa, Fb, Ra, Rb>(a: Fa, b: Fb) -> (Ra, Rb)
where
    Fa: FnOnce() -> Ra + Send,
    Fb: FnOnce() -> Rb + Send,
    Ra: Send,
    Rb: Send,
{
    let wk = current_context();
    let job_b = StackJob::new(b);
    let job_ref = job_b.as_job_ref();
    wk.deque.push(job_ref);

    let ra = a();

    let rb = match wk.deque.pop() {
        Some(popped) if std::ptr::eq(popped.data, job_ref.data) => {
            // Not stolen: finish it ourselves, one plain call — the whole
            // point. No latch, no steal-side traffic at all.
            job_b.run_inline()
        }
        popped => {
            // `popped` should only ever be `None` here (properly nested
            // calls always leave the deque exactly as they found it, aside
            // from `job_b` itself) — but if something else somehow came
            // back, put it back rather than dropping work.
            if let Some(other) = popped {
                wk.deque.push(other);
            }
            // Stolen: help execute other stealable work while waiting.
            while !job_b.latch.probe() {
                if !try_execute_one(wk) {
                    std::hint::spin_loop();
                }
            }
            job_b.take_result()
        }
    };
    (ra, rb)
}

// ---------------------------------------------------------------------------
// run — bring up the worker pool, run the root closure, tear down
// ---------------------------------------------------------------------------

/// Start `num_workers` OS threads (the calling thread becomes worker 0),
/// run `f` as the root job, and block until it (and everything it
/// transitively `parallel_call`s) completes.
pub(crate) fn run<F, R>(num_workers: usize, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    assert!(num_workers >= 1, "need at least one worker");
    let deques: Vec<Deque<JobRef>> = (0..num_workers).map(|_| Deque::new_lifo()).collect();
    let stealers: Vec<Stealer<JobRef>> = deques.iter().map(|d| d.stealer()).collect();
    let registry = Arc::new(Registry { stealers, injector: Injector::new(), shutdown: AtomicBool::new(false) });

    let mut deques = deques.into_iter();
    let worker0_deque = deques.next().unwrap();

    let handles: Vec<_> = deques
        .enumerate()
        .map(|(i, deque)| {
            let idx = i + 1;
            let registry = Arc::clone(&registry);
            std::thread::spawn(move || {
                let ctx = WorkerContext { index: idx, deque, registry };
                CURRENT.with(|c| c.set(&ctx as *const _));
                loop {
                    if try_execute_one(&ctx) {
                        continue;
                    }
                    if ctx.registry.shutdown.load(Ordering::Acquire) {
                        break;
                    }
                    std::hint::spin_loop();
                }
                // Drain anything left so a straggler steal doesn't miss
                // work pushed just before shutdown was observed.
                while try_execute_one(&ctx) {}
            })
        })
        .collect();

    let root = StackJob::new(f);
    let root_ref = root.as_job_ref();
    let ctx0 = WorkerContext { index: 0, deque: worker0_deque, registry: Arc::clone(&registry) };
    CURRENT.with(|c| c.set(&ctx0 as *const _));
    // Run the root job directly — no steal-check needed for the very first
    // one, nobody else has had a chance to touch it yet.
    unsafe { root_ref.execute() };

    registry.shutdown.store(true, Ordering::Release);
    for h in handles {
        h.join().expect("cmpth: parallel_call worker thread panicked");
    }

    root.take_result()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fib(n: u64) -> u64 {
        if n <= 1 {
            return n;
        }
        let (a, b) = parallel_call(|| fib(n - 1), || fib(n - 2));
        a + b
    }

    #[test]
    fn fib_matches_sequential() {
        for workers in [1, 2, 4] {
            let r = run(workers, || fib(20));
            assert_eq!(r, 6765, "workers={workers}");
        }
    }

    #[test]
    fn nested_join_many_levels() {
        let r = run(2, || fib(24));
        assert_eq!(r, 46368);
    }

    #[test]
    fn borrows_non_static_data() {
        let data = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        let sum = run(4, || {
            fn rec(s: &[u64]) -> u64 {
                if s.len() <= 1 {
                    return s.first().copied().unwrap_or(0);
                }
                let mid = s.len() / 2;
                let (a, b) = parallel_call(|| rec(&s[..mid]), || rec(&s[mid..]));
                a + b
            }
            rec(&data)
        });
        assert_eq!(sum, 36);
    }
}
