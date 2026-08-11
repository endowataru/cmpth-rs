//! OS-thread-pool engine backing [`ScopedStackfulTaskSystem`](crate::traits::ScopedStackfulTaskSystem).
//!
//! Mirrors `rayon::join`: push the second branch as a stealable [`TaskRef`],
//! run the first branch as an ordinary nested call, then either pop our own
//! task back off (not stolen — finish it with one more ordinary call) or
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
//!
//! # Worker-pool machinery
//!
//! Bring-up/teardown/idle-loop reuse [`ScopedWorker`]/[`ScopedRegistry`]
//! (`super::worker`) — the same `WorkerRunQueue`/`WorkerOps`/`LocalQueue`
//! machinery `resumable`'s `Scheduler<S>`/`worker_idle_loop` use, not a
//! bespoke deque/thread-local pair. Only `parallel_call` itself (the hot
//! path this module exists for) stays engine-specific — dispatch elsewhere
//! is structurally identical to `resumable::common::scheduler::worker_idle_loop`,
//! just without a task-pool/external-queue axis to check (`scoped` has
//! none).

use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::resumable::common::deque::Steal;
use crate::resumable::common::system::WorkerSystem;
use crate::resumable::common::worker::{LocalQueue, WorkerOps};
use crate::traits::stackful::SpawnableStackfulTaskSystem;

use super::system::ScopedTaskSystem;
use super::task::StackTask;
use super::worker::{ScopedRegistry, ScopedWorker};

// ---------------------------------------------------------------------------
// Worker lookup / idle dispatch
// ---------------------------------------------------------------------------

/// Try to make progress once: pop our own local task, else steal from
/// another worker. Returns `false` if nothing was found anywhere right now.
/// Shared by the idle worker loop and `parallel_call`'s help-while-waiting
/// loop — the same "what do I do when I have nothing of my own to run"
/// logic either way. Mirrors
/// [`worker_idle_loop`](crate::resumable::common::scheduler::worker_idle_loop)'s
/// body, minus the task-pool/external-queue checks `scoped` has none of.
fn try_execute_one(wk: &ScopedWorker) -> bool {
    if let Some(task) = wk.try_pop() {
        unsafe { task.execute() };
        return true;
    }
    if let Steal::Success(task) = wk.try_steal() {
        unsafe { task.execute() };
        return true;
    }
    false
}

/// The idle/dispatch loop every worker OS thread (other than the one
/// driving `run`'s root task inline) runs until shutdown.
fn worker_idle_loop(wk: &ScopedWorker, registry: &ScopedRegistry) {
    let mut idle_rounds = 0u32;
    while !registry.finished.load(Ordering::Acquire) {
        if try_execute_one(wk) {
            idle_rounds = 0;
            continue;
        }
        std::hint::spin_loop();
        idle_rounds += 1;
        if idle_rounds & 0x3F == 0 {
            <ScopedTaskSystem as WorkerSystem>::Base::yield_now();
        }
    }
    // Drain anything left so a straggler steal doesn't miss work pushed
    // just before shutdown was observed.
    while try_execute_one(wk) {}
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
    let wk = ScopedWorker::current().expect("cmpth: scoped::parallel_call called outside scoped::run");
    let task_b = StackTask::new(b);
    let task_ref = task_b.as_task_ref();
    wk.push(task_ref);

    let ra = a();

    let rb = match wk.try_pop() {
        Some(popped) if std::ptr::eq(popped.data, task_ref.data) => {
            // Not stolen: finish it ourselves, one plain call — the whole
            // point. No latch, no steal-side traffic at all.
            task_b.run_inline()
        }
        popped => {
            // `popped` should only ever be `None` here (properly nested
            // calls always leave the deque exactly as they found it, aside
            // from `task_b` itself) — but if something else somehow came
            // back, put it back rather than dropping work.
            if let Some(other) = popped {
                wk.push(other);
            }
            // Stolen: help execute other stealable work while waiting.
            while !task_b.latch.probe() {
                if !try_execute_one(wk) {
                    std::hint::spin_loop();
                }
            }
            task_b.take_result()
        }
    };
    (ra, rb)
}

// ---------------------------------------------------------------------------
// run — bring up the worker pool, run the root closure, tear down
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// init — standalone counterpart to `run`, backing
// `StackfulInitSystem`/`StackfulBuilder` for `ScopedTaskSystem`.
//
// This engine has no ULT/context-switch concept at all (see this module's
// own doc comment: a `parallel_call` branch is a stack-resident closure, run
// inline or executed directly by whichever thread steals its `TaskRef` — no
// continuation is ever reified the way a stackful ULT's stack is), so there
// is no "the caller's own continuation becomes stealable" property to
// mirror here the way `resumable::stackful::init` mirrors it for real ULTs.
// What standalone init *does* mean for this engine: bring the pool up, then
// return — the calling OS thread permanently stays this pool's worker 0 (it
// never migrates, same as it never did under the old `run`), and ordinary
// code after `init()` can call `parallel_call` immediately. Teardown
// (`Drop`) mirrors `run`'s own teardown: signal shutdown, join the other
// worker threads.
// ---------------------------------------------------------------------------

/// RAII guard returned by `init`. Dropping it drains any work left on
/// worker 0's own deque, signals shutdown, and joins the other worker OS
/// threads.
///
/// `pub`, not `pub(crate)`: this is [`ScopedTaskSystem`](super::ScopedTaskSystem)'s
/// [`StackfulInitSystem::Init`](crate::traits::stackful::StackfulInitSystem::Init)
/// — a public associated type needs an at-least-as-public backing type,
/// even though every field here (and the `init` function itself) stays
/// crate-private; same opaque-struct shape as
/// [`resumable::stackful::init::StackfulInit`](crate::resumable::stackful::init::StackfulInit).
pub struct SyncInit {
    registry: Arc<ScopedRegistry>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

pub(crate) fn init(num_workers: usize) -> SyncInit {
    assert!(num_workers >= 1, "need at least one worker");
    assert!(
        ScopedWorker::current().is_none(),
        "cmpth: nested scoped::init() of the same engine on one thread"
    );

    let workers: Vec<ScopedWorker> = (0..num_workers).map(ScopedWorker::new).collect();
    let stealers = workers.iter().map(ScopedWorker::deque_stealer).collect();
    let registry = Arc::new(ScopedRegistry {
        workers: workers.into_boxed_slice(),
        stealers,
        finished: std::sync::atomic::AtomicBool::new(false),
    });
    for w in registry.workers.iter() {
        w.bind_registry(Arc::as_ptr(&registry));
    }

    let handles: Vec<_> = (1..num_workers)
        .map(|idx| {
            let registry = Arc::clone(&registry);
            std::thread::spawn(move || {
                let wk = &registry.workers[idx];
                super::worker::set_current(wk as *const ScopedWorker);
                worker_idle_loop(wk, &registry);
                super::worker::set_current(std::ptr::null());
            })
        })
        .collect();

    super::worker::set_current(&registry.workers[0] as *const ScopedWorker);

    SyncInit { registry, handles }
}

impl Drop for SyncInit {
    fn drop(&mut self) {
        // No context switch happens anywhere in this engine (every
        // `parallel_call` branch either runs as an ordinary nested call or
        // gets executed directly by whichever thread steals it — there is
        // no suspended continuation to resume), so unlike the stackful ULT
        // initializer this has no panic-across-switch hazard: an ordinary
        // `Drop` while unwinding is perfectly sound here.
        let wk0 = &self.registry.workers[0];
        while try_execute_one(wk0) {}
        self.registry.finished.store(true, Ordering::Release);
        for h in self.handles.drain(..) {
            h.join().expect("cmpth: parallel_call worker thread panicked");
        }
        super::worker::set_current(std::ptr::null());
    }
}

/// Start `num_workers` OS threads (the calling thread becomes worker 0),
/// run `f` as the root task, and block until it (and everything it
/// transitively `parallel_call`s) completes.
pub(crate) fn run<F, R>(num_workers: usize, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    assert!(num_workers >= 1, "need at least one worker");
    let workers: Vec<ScopedWorker> = (0..num_workers).map(ScopedWorker::new).collect();
    let stealers = workers.iter().map(ScopedWorker::deque_stealer).collect();
    let registry = Arc::new(ScopedRegistry {
        workers: workers.into_boxed_slice(),
        stealers,
        finished: std::sync::atomic::AtomicBool::new(false),
    });
    for w in registry.workers.iter() {
        w.bind_registry(Arc::as_ptr(&registry));
    }

    let handles: Vec<_> = (1..num_workers)
        .map(|idx| {
            let registry = Arc::clone(&registry);
            std::thread::spawn(move || {
                let wk = &registry.workers[idx];
                super::worker::set_current(wk as *const ScopedWorker);
                worker_idle_loop(wk, &registry);
                super::worker::set_current(std::ptr::null());
            })
        })
        .collect();

    let root = StackTask::new(f);
    let root_ref = root.as_task_ref();
    let wk0 = &registry.workers[0];
    super::worker::set_current(wk0 as *const ScopedWorker);
    // Run the root task directly — no steal-check needed for the very first
    // one, nobody else has had a chance to touch it yet.
    unsafe { root_ref.execute() };

    registry.finished.store(true, Ordering::Release);
    for h in handles {
        h.join().expect("cmpth: parallel_call worker thread panicked");
    }
    super::worker::set_current(std::ptr::null());

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
