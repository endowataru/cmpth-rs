//! Poll-based counterpart to [`sync_engine`](super::sync_engine) backing
//! [`ScopedStacklessTaskSystem`](crate::traits::ScopedStacklessTaskSystem): same
//! worker-pool/steal shape (reusing [`super::task::TaskRef`]'s stack-resident,
//! type-erased task representation, and [`super::worker`]'s shared
//! `WorkerRunQueue`/`WorkerOps`/`LocalQueue` machinery), but bodies are
//! [`Future`]s driven via polling instead of plain closures called once.
//!
//! The future returned by [`parallel_call`] never blocks the OS thread
//! polling *it* while `a` is still running — `a` is polled transparently
//! (`Pending` propagates straight through, exactly like an ordinary nested
//! `.await`). Only once `a` completes do we check on `b`: if it's still
//! sitting unstolen in our own local deque we reclaim it and drive it
//! inline (nobody else could be touching it — same "not stolen, no
//! steal-side traffic" fast path [`sync_engine`](super::sync_engine) has);
//! if it was genuinely stolen we register a [`Waker`] on its
//! [`AsyncTask::latch`] and return `Pending` instead of busy-waiting.
//!
//! `b`'s storage is an `Arc<AsyncTask<Rb>>`, not a borrowed stack frame like
//! [`super::task::StackTask`]: the future returned by [`parallel_call`] can
//! be dropped (cancelled) at any poll boundary, including while `b` is
//! still being driven by a thief on another worker thread, so its storage
//! must be able to outlive the caller's own frame — see
//! [`ScopedStacklessTaskSystem`](crate::traits::ScopedStacklessTaskSystem)'s
//! doc comment for why this is the one place this engine accepts a small
//! heap allocation per call. A thief that actually steals a branch commits
//! its own dedicated worker OS thread to driving it to completion via
//! [`drive`] — a small busy loop that re-polls on wake and helps execute
//! other stealable async tasks while idle, mirroring
//! [`sync_engine`](super::sync_engine)'s "help while waiting" loop. That's
//! the one place this engine still blocks an OS thread synchronously —
//! deliberately: a dedicated pool worker has nothing better to do while a
//! task it grabbed isn't ready, same as the sync engine.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

use crate::resumable::common::deque::Steal;
use crate::resumable::common::system::WorkerSystem;
use crate::resumable::common::worker::{LocalQueue, WorkerOps};
use crate::traits::stackful::SpawnableStackfulTaskSystem;

use super::system::ScopedTaskSystem;
use super::task::TaskRef;
use super::worker::{ScopedRegistry, ScopedWorker};

// ---------------------------------------------------------------------------
// AsyncLatch — like `task::Latch`, but can hold a registered Waker: a stolen
// branch may still be running when the pusher wants to wait on it (unlike
// the sync engine, which only ever busy-polls a bool), so late registration
// must be race-free against a concurrent `set()`. Same CAS discipline as
// `resumable::common::desc::WakerTaskDesc::try_register_waker` — check-already-done and
// install-the-waiter are one atomic step, so a `set()` that races a
// `register()` can never be missed.
// ---------------------------------------------------------------------------

const PENDING: usize = 0;
const DONE: usize = 1;

struct AsyncLatch(AtomicUsize);

impl AsyncLatch {
    fn new() -> Self {
        AsyncLatch(AtomicUsize::new(PENDING))
    }

    /// Publish completion (thief side) and wake whoever registered, if
    /// anyone did.
    fn set(&self) {
        let old = self.0.swap(DONE, Ordering::AcqRel);
        if old != PENDING {
            let w = unsafe { Box::from_raw(old as *mut Waker) };
            w.wake();
        }
    }

    /// Try to install `waker` (pusher side). Returns `false` if the task was
    /// already finished by the time this ran — caller should take the
    /// result immediately instead of waiting.
    fn register(&self, waker: &Waker) -> bool {
        let mut cur = self.0.load(Ordering::Acquire);
        loop {
            if cur == DONE {
                return false;
            }
            let boxed = Box::into_raw(Box::new(waker.clone())) as usize;
            match self.0.compare_exchange_weak(cur, boxed, Ordering::Release, Ordering::Acquire) {
                Ok(_) => {
                    if cur != PENDING {
                        // Superseded a previous registration (re-poll after
                        // a spurious wake): drop it, it's stale.
                        drop(unsafe { Box::from_raw(cur as *mut Waker) });
                    }
                    return true;
                }
                Err(c) => cur = c,
            }
        }
    }
}

unsafe impl Send for AsyncLatch {}
unsafe impl Sync for AsyncLatch {}

// ---------------------------------------------------------------------------
// AsyncTask — `b`'s storage. `Arc`-owned (see module docs) rather than
// stack-resident: whichever of {pusher gets it back unstolen, thief steals
// it} runs first takes `body` out under the mutex: `Taken` on the loser's
// side is unreachable, not a possible outcome, since the deque only ever
// hands the task to one of them.
// ---------------------------------------------------------------------------

enum Body<Fut> {
    Pending(Pin<Box<Fut>>),
    Taken,
}

struct AsyncTask<Fut: Future> {
    body: Mutex<Body<Fut>>,
    result: Mutex<Option<Fut::Output>>,
    latch: AsyncLatch,
}

impl<Fut> AsyncTask<Fut>
where
    Fut: Future + Send + 'static,
    Fut::Output: Send + 'static,
{
    fn new(fut: Fut) -> Self {
        AsyncTask {
            body: Mutex::new(Body::Pending(Box::pin(fut))),
            result: Mutex::new(None),
            latch: AsyncLatch::new(),
        }
    }

    fn take_body(&self) -> Pin<Box<Fut>> {
        let mut guard = self.body.lock().unwrap();
        match std::mem::replace(&mut *guard, Body::Taken) {
            Body::Pending(fut) => fut,
            Body::Taken => unreachable!("cmpth: AsyncTask driven twice"),
        }
    }

    /// Drive to completion (blocking busy+help loop), store the result,
    /// then publish + wake. Used both by a thief (via the type-erased
    /// [`TaskRef`] trampoline) and by the pusher's own inline fast path when
    /// it gets `b` back unstolen.
    fn drive_to_completion(&self) {
        let mut fut = self.take_body();
        let out = drive(fut.as_mut());
        *self.result.lock().unwrap() = Some(out);
        self.latch.set();
    }

    fn take_result(&self) -> Fut::Output {
        self.result.lock().unwrap().take().expect("cmpth: AsyncTask latch set without a result")
    }

    unsafe fn execute_trampoline(data: *const ()) {
        let task = unsafe { Arc::from_raw(data as *const Self) };
        task.drive_to_completion();
    }

    /// A plain associated fn, not a `self: &Arc<Self>` method — that
    /// receiver form isn't a blessed arbitrary self type on stable Rust
    /// (only `Arc<Self>` by value is), so the `Arc` is just an ordinary
    /// parameter here.
    fn as_task_ref(task: &Arc<Self>) -> TaskRef {
        // Leaks one strong ref into the raw pointer; reclaimed either here
        // (unstolen: `Arc::from_raw` below, no trampoline call) or by
        // `execute_trampoline` (stolen: reconstructed there instead).
        let data = Arc::into_raw(Arc::clone(task)) as *const ();
        // Safety: `TaskRef` is only ever constructed for tasks whose type
        // `Fut` matches `execute_trampoline`'s own monomorphization here.
        unsafe { TaskRef::from_raw_parts(data, Self::execute_trampoline) }
    }
}

// ---------------------------------------------------------------------------
// drive — poll-on-wake, help-while-idle busy loop. The one place this
// engine blocks an OS thread: driving a future (the pool's root, or a task a
// thief just grabbed) to completion without a dedicated stack for it to
// suspend onto.
// ---------------------------------------------------------------------------

struct WakeFlag(AtomicBool);

impl Wake for WakeFlag {
    fn wake(self: Arc<Self>) {
        self.0.store(true, Ordering::Release);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.store(true, Ordering::Release);
    }
}

fn drive<Fut: Future + ?Sized>(mut fut: Pin<&mut Fut>) -> Fut::Output {
    let wk = ScopedWorker::current().expect("cmpth: scoped::parallel_call (async) called outside run_async");
    let woken = Arc::new(WakeFlag(AtomicBool::new(true)));
    let waker = Waker::from(Arc::clone(&woken));
    let mut cx = Context::from_waker(&waker);
    loop {
        if woken.0.swap(false, Ordering::AcqRel) {
            if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                return v;
            }
        }
        if !try_execute_one(wk) {
            std::hint::spin_loop();
        }
    }
}

// ---------------------------------------------------------------------------
// Worker idle dispatch — same shape as sync_engine's, kept separate (own
// registry/worker set per `run_async` call) since the two engines never
// share workers, only the `ScopedWorker`/`WorkerRunQueue` machinery itself.
// ---------------------------------------------------------------------------

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
    while try_execute_one(wk) {}
}

// ---------------------------------------------------------------------------
// parallel_call — the public primitive
// ---------------------------------------------------------------------------

enum State<Fa: Future> {
    RunningA(Pin<Box<Fa>>),
    WaitingB(Fa::Output),
    Done,
}

/// Returned by [`parallel_call`]. See the module docs for the state
/// machine this drives.
pub(crate) struct ParallelInvoke<Fa: Future, Fb: Future> {
    state: State<Fa>,
    task: Arc<AsyncTask<Fb>>,
    /// `task`'s `TaskRef::data`, as a plain integer once pushed — lets
    /// `poll` tell "got our own task back unstolen" apart from "someone
    /// else's task came back" the same way `sync_engine::parallel_call`
    /// does, without a raw pointer field (which would otherwise make this
    /// struct not automatically `Send`).
    pushed: Option<usize>,
}

impl<Fa, Fb> Future for ParallelInvoke<Fa, Fb>
where
    Fa: Future + Send + 'static,
    Fb: Future + Send + 'static,
    Fa::Output: Send + 'static,
    Fb::Output: Send + 'static,
{
    type Output = (Fa::Output, Fb::Output);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Safety: `state`/`task`/`pushed` are all moved wholesale, never
        // individually pinned to a self-referential address; only the
        // boxed future inside `State::RunningA` needs pin-projecting, and
        // it's already behind its own independent `Pin<Box<_>>`.
        let this = unsafe { self.get_unchecked_mut() };

        if this.pushed.is_none() {
            let wk = ScopedWorker::current().expect("cmpth: scoped::parallel_call (async) called outside run_async");
            let task_ref = AsyncTask::as_task_ref(&this.task);
            this.pushed = Some(task_ref.data as usize);
            wk.push(task_ref);
        }

        if let State::RunningA(a) = &mut this.state {
            match a.as_mut().poll(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(ra) => this.state = State::WaitingB(ra),
            }
        }

        if matches!(this.state, State::Done) {
            panic!("cmpth: ParallelInvoke polled after completion");
        }

        let wk = ScopedWorker::current().expect("cmpth: scoped::parallel_call (async) called outside run_async");
        let pushed = this.pushed.expect("cmpth: task_b not pushed before WaitingB");
        match wk.try_pop() {
            Some(popped) if popped.data as usize == pushed => {
                // Not stolen: reclaim the leaked ref (we still hold our own
                // `this.task` handle) and drive it inline.
                drop(unsafe { Arc::from_raw(popped.data as *const AsyncTask<Fb>) });
                this.task.drive_to_completion();
            }
            popped => {
                if let Some(other) = popped {
                    wk.push(other);
                }
                if this.task.latch.register(cx.waker()) {
                    return Poll::Pending;
                }
                // Else: already finished by the time we tried to register
                // — fall through and take the result now.
            }
        }

        let rb = this.task.take_result();
        let State::WaitingB(ra) = std::mem::replace(&mut this.state, State::Done) else {
            unreachable!("cmpth: state was checked to be WaitingB above")
        };
        Poll::Ready((ra, rb))
    }
}

/// See [`ScopedStacklessTaskSystem::parallel_call`](crate::traits::ScopedStacklessTaskSystem::parallel_call)
/// for why this takes thunks rather than already-built futures. Both are
/// called eagerly, right here — plain, ordinary evaluation, no `.await`
/// involved on this side.
pub(crate) fn parallel_call<Fa, Fb, MkA, MkB>(mk_a: MkA, mk_b: MkB) -> ParallelInvoke<Fa, Fb>
where
    MkA: FnOnce() -> Fa,
    MkB: FnOnce() -> Fb,
    Fa: Future + Send + 'static,
    Fb: Future + Send + 'static,
    Fa::Output: Send + 'static,
    Fb::Output: Send + 'static,
{
    let task = Arc::new(AsyncTask::new(mk_b()));
    ParallelInvoke { state: State::RunningA(Box::pin(mk_a())), task, pushed: None }
}

// ---------------------------------------------------------------------------
// run_async — bring up the worker pool, drive the root future, tear down
// ---------------------------------------------------------------------------

// No longer reachable from any public API path: `ScopedTaskSystem` only
// implements `StackfulInitSystem` (via `sync_engine`'s own `init`/`run`),
// not `StacklessInitSystem` — this engine's `run_async` never got a
// standalone-init counterpart added alongside it (unlike `sync_engine`,
// which did), so it currently exists purely for this module's own tests
// below. Kept `pub(crate)` (not deleted) since it's real, working
// infrastructure a future `StacklessInitSystem` impl for `ScopedTaskSystem`
// could reuse directly.
#[allow(dead_code)]
pub(crate) fn run_async<F>(num_workers: usize, root: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    assert!(num_workers >= 1, "need at least one worker");
    let workers: Vec<ScopedWorker> = (0..num_workers).map(ScopedWorker::new).collect();
    let stealers = workers.iter().map(ScopedWorker::deque_stealer).collect();
    let registry = Arc::new(ScopedRegistry {
        workers: workers.into_boxed_slice(),
        stealers,
        finished: AtomicBool::new(false),
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

    let wk0 = &registry.workers[0];
    super::worker::set_current(wk0 as *const ScopedWorker);
    let mut root = Box::pin(root);
    drive(root.as_mut());

    registry.finished.store(true, Ordering::Release);
    for h in handles {
        h.join().expect("cmpth: parallel_call worker thread panicked");
    }
    super::worker::set_current(std::ptr::null());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    // Recursive `async fn`s can't pass their own opaque return type as a
    // bare generic argument to anything (E0733) — this is the actual proof
    // that taking thunks (`parallel_call(mk_a, mk_b)`, not
    // `parallel_call(a, b)`) really does dodge it, not just a claim in a
    // doc comment.
    fn fib(n: u64) -> impl Future<Output = u64> + Send {
        async move {
            if n <= 1 {
                return n;
            }
            let (a, b) = parallel_call(move || fib(n - 1), move || fib(n - 2)).await;
            a + b
        }
    }

    #[test]
    fn fib_matches_sequential() {
        for workers in [1, 2, 4] {
            let result = Arc::new(AtomicU64::new(0));
            let result2 = Arc::clone(&result);
            run_async(workers, async move {
                result2.store(fib(20).await, Ordering::Release);
            });
            assert_eq!(result.load(Ordering::Acquire), 6765, "workers={workers}");
        }
    }

    #[test]
    fn nested_join_many_levels() {
        // Deep enough, with few enough workers, that real steals happen —
        // exercises `AsyncLatch::register`/`set`'s wake path, not just the
        // unstolen inline fast path.
        let result = Arc::new(AtomicU64::new(0));
        let result2 = Arc::clone(&result);
        run_async(2, async move {
            result2.store(fib(24).await, Ordering::Release);
        });
        assert_eq!(result.load(Ordering::Acquire), 46368);
    }

    #[test]
    fn many_independent_parallel_invokes() {
        // Several independent parallel_call trees live on the pool at
        // once, none of them the root future itself — checks that workers
        // correctly multiplex unrelated work via stealing, not just a
        // single tree.
        let counter = Arc::new(AtomicU64::new(0));
        run_async(4, {
            let counter = Arc::clone(&counter);
            async move {
                let mut sum = 0u64;
                for i in 0..50u64 {
                    let counter = Arc::clone(&counter);
                    let (a, b) = parallel_call(
                        move || async move {
                            counter.fetch_add(1, Ordering::Relaxed);
                            fib(15).await
                        },
                        move || fib(16),
                    )
                    .await;
                    sum += a + b + i;
                }
                assert_eq!(sum, 50 * (610 + 987) + (0..50u64).sum::<u64>());
            }
        });
        assert_eq!(counter.load(Ordering::Acquire), 50);
    }
}
