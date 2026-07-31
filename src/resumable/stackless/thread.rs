//! Stackless thread functions: `spawn_async`, `recurse`, and
//! `.await`-ing a [`JoinHandle`]
//! (shared with [`stackful::thread`](crate::resumable::stackful::thread) —
//! see `common::thread` for the handle type itself).

use std::alloc::Layout;
use std::future::Future;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::thread::{align_down, drop_stack_result, JoinHandle, StackResult};
use crate::resumable::common::desc::{HasBaseOwned, JoinState, SuspendedTaskToken, TaskDesc, TaskDescAlloc, WakeOutcome, WakerTaskDesc};
use crate::resumable::stackless::desc::{AsyncTaskDesc, HasPollFn, TaskPollResult};
use crate::resumable::common::pool::{DescPool, DynamicPool};
use crate::resumable::common::worker::{LocalQueue, UltWorker, Worker};

// ---------------------------------------------------------------------------
// .await-ing a JoinHandle
// ---------------------------------------------------------------------------

impl<S: SchedulerSystem, T: Send + 'static> Future for JoinHandle<S, T>
where
    S::Desc: AsyncTaskDesc,
{
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        // `self`'s own address, captured before `get_mut()`: `self` (this
        // JoinHandle) is a field of the enclosing `spawn_async`'d future,
        // stored inline inside that future's own arena-allocated descriptor
        // (see `worker_from_async_arena_addr`) — so this address doubles as
        // a way to find the enclosing task's own worker, no TLS needed.
        let self_addr = &*self as *const Self as usize;
        let this = self.get_mut(); // JoinHandle: Unpin
        let desc = this.desc;

        // Fast path: `wk.polling_async` is non-null exactly while
        // `run_async_poll` is synchronously driving `joiner`'s own future on
        // this worker (see `run_async_poll`) — and the same `&mut Context`
        // it built for `joiner` propagates, by construction, through every
        // `.await` reached from within that future's body (the desugaring
        // never substitutes a different one). So whenever `poll` is invoked
        // synchronously through that chain, `cx.waker()` *is* `joiner`'s own
        // waker; registering `joiner`'s descriptor directly needs no
        // `Box<Waker>` allocation. A hand-rolled `Future` that manually
        // swaps in a foreign `Context` inside that span would violate this —
        // not something any code in this crate does.
        //
        // `current_worker_from_cx` finds that worker via, in order: the
        // arena cell `self_addr` lands in (see
        // `worker_from_async_arena_addr`), or `UltWorker::<S>::current()`
        // (the TLS fallback).
        let current_wk = current_worker_from_cx::<S>(self_addr);

        // Reclaim fast path: if `desc` is still sitting untouched on our
        // own local deque (nobody has started or stolen it), pop it back
        // and run it directly, right here — no deque round trip through
        // the outer worker_loop, no separate dispatch cycle. Mirrors
        // `fork_join::join()`'s "not stolen -> plain nested call" fast
        // path, translated to spawn_async/await. If this runs `desc` to
        // completion, the registration below sees FINISHED immediately
        // (its own existing check) and falls straight to the Ready path;
        // if `desc` goes Pending instead (it has its own un-reclaimable
        // child), registration proceeds exactly as before.
        if let Some(wk) = current_wk {
            try_reclaim_and_run::<S>(wk, desc);
        }

        let registered = match current_wk {
            Some(wk) => {
                let joiner = wk.polling_async.get();
                if !joiner.is_null() {
                    unsafe { (*desc).try_register_async_joiner(joiner) }
                } else {
                    unsafe { (*desc).try_register_waker(cx.waker().clone()) }
                }
            }
            None => unsafe { (*desc).try_register_waker(cx.waker().clone()) },
        };
        if registered {
            return Poll::Pending;
        }
        // FINISHED: consume the handle (null desc so Drop becomes a no-op).
        let handle = unsafe { std::ptr::read(this) };
        this.desc = std::ptr::null_mut();
        let result = match current_wk {
            Some(wk) => handle.take_result(wk),
            None => handle.take_result_no_worker(),
        };
        Poll::Ready(match result {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        })
    }
}

/// Find the current worker for a `JoinHandle::poll` call: first via
/// `self_addr`'s arena cell (no TLS — see `worker_from_async_arena_addr`),
/// otherwise `UltWorker::<S>::current()` (TLS).
fn current_worker_from_cx<S: SchedulerSystem>(self_addr: usize) -> Option<&'static UltWorker<S>> {
    if let Some(wk) = crate::resumable::stackless::lookup::worker_from_async_arena_addr::<S>(self_addr) {
        return Some(wk);
    }
    UltWorker::<S>::current()
}

/// See [`JoinHandle::poll`]'s reclaim fast path. Pops `wk`'s own local
/// deque; if what comes back is `desc` itself (nobody else got to it —
/// crossbeam-deque's push/pop-vs-steal synchronization makes this a
/// reliable check, not a race) *and* `desc` is a `spawn_async` task (has a
/// `poll_fn` — `JoinHandle` is also the `Future` impl for plain stackful
/// `spawn()` handles, which have no `poll_fn` and need a real context
/// switch, not a direct poll, to run), drives it directly via
/// `run_async_poll` instead of leaving it for some worker to pick up
/// later. Anything else popped (a different task, or `desc` itself but
/// stackful) goes right back — not something this fast path can help with.
fn try_reclaim_and_run<S>(wk: &UltWorker<S>, desc: *mut S::Desc)
where
    S: SchedulerSystem,
    S::Desc: AsyncTaskDesc,
{
    match wk.pop_local() {
        Some(popped) if std::ptr::eq(popped.desc(), desc) => {
            if popped.is_poll_fn_dispatch() {
                let poll_fn = popped.poll_fn()
                    .expect("cmpth: descriptor committed to poll_fn dispatch but poll_fn unset");
                crate::resumable::stackless::worker::run_async_poll(wk, desc, poll_fn);
            } else {
                wk.push_local_top(popped);
            }
        }
        Some(other) => wk.push_local_top(other),
        None => {}
    }
}

// ---------------------------------------------------------------------------
// spawn_async — async Future as a lightweight task
// ---------------------------------------------------------------------------

/// Spawn a `Future` as a task.  The future is stored directly in a small
/// buffer (no 64 KB stack); the ULT executor polls it without any context
/// switch.
///
/// Returns a [`SpawnAction`]: a `Future` whose *only* `.await` performs the
/// actual registration (finding the calling worker, pool allocation,
/// calling `mk()` and writing its result into place, pushing to the deque)
/// and resolves to a [`JoinHandle`], which is then `.await`ed (or
/// `.join()`ed) a second time to get the result — i.e.
/// `spawn_async(mk).await.await`, or more commonly
/// `let h = spawn_async(mk).await; /* ... other work ... */ h.await`.
///
/// This shape is deliberate, not incidental, and callers must not "flatten"
/// it by keeping the pre-registration value around unawaited: the point of
/// requiring the first `.await` immediately is that the *user's own code*
/// then unambiguously marks the spot where the task becomes real, letting a
/// future scheduler change (e.g. a genuinely child-first/work-first
/// implementation, which needs to know exactly what "the rest of this
/// function from here" means) without an API break. It is not there to
/// dodge a thread-local lookup — `poll` below uses one directly, same as
/// any other call into this module — and must not be removed or reordered
/// for a particular implementation's convenience; see [`recurse`] for the
/// same rule stated the other way around (an already-immediate-`Poll::Ready`
/// implementation is exactly how a help-first strategy is expressed here,
/// with no fewer `.await`s in caller code).
///
/// Takes a **thunk** (`mk`), not an already-constructed future: an
/// already-built `F` would have to be held by value inside `SpawnAction`
/// until it can be moved into the task's storage — exactly the
/// infinitely-sized-embedding problem [`recurse`] exists to avoid (E0733)
/// for a directly self-recursive `F`, just relocated one call outward.
/// `mk()` runs exactly once, inside `spawn_now`, once the descriptor
/// it writes into already exists.
///
/// Storage comes from `S::AsyncPool` (see [`SchedulerSystem::AsyncPool`]):
/// futures that fit its configured slot size are served from its free list
/// like any pooled ULT stack; larger ones fall back to a one-off
/// allocation, freed directly rather than returned to the pool.
///
/// The actual registration work happens *eagerly*, right here, not deferred
/// into `SpawnAction::poll` — see `spawn_now`'s docs for why keeping that
/// work as a plain, ordinarily-called function (rather than embedded in a
/// `Future::poll` body) matters for how well the compiler can optimize the
/// enclosing `async fn`'s generated state machine. `SpawnAction` itself
/// stays a real, crate-owned `Future` type (not `std::future::ready`)
/// purely so a future work-first rewrite has a `poll` body it can still
/// change — see `SpawnAction`'s own docs.
pub fn spawn_async<S, T, F, Mk>(mk: Mk) -> SpawnAction<S, T>
where
    S: SchedulerSystem,
    F: Future<Output = T> + Send + 'static,
    Mk: FnOnce() -> F + Send + 'static,
    T: Send + 'static,
    S::Desc: AsyncTaskDesc,
{
    SpawnAction { handle: Some(spawn_now::<S, T, F, Mk>(mk)) }
}

/// Does the actual work of registering a task: finds the calling worker,
/// allocates from `S::AsyncPool`, calls `mk()` and writes its result into
/// place, and pushes the new task to the deque. Returns the completed
/// [`JoinHandle`] directly — an ordinary, eagerly-called function, not a
/// `Future`/`poll` body.
///
/// Deliberately factored out of [`SpawnAction::poll`] (which used to do all
/// of this inline): a plain function call like this one is easy for the
/// compiler to reason about and inline into the caller's own generated
/// state machine, same as any other non-generator code path. Burying this
/// same logic inside a `poll` implementation forces the compiler to treat
/// it as part of a generator's resumable body, which is a harder shape to
/// optimize. A future work-first rewrite that needs to run this (or a
/// child's body) conditionally from inside `poll` can still call this
/// function from there — the separation doesn't remove that option, it
/// just keeps today's help-first path (call eagerly, wrap the already-done
/// result) on the easy-to-optimize side of that boundary.
fn spawn_now<S, T, F, Mk>(mk: Mk) -> JoinHandle<S, T>
where
    S: SchedulerSystem,
    F: Future<Output = T> + Send + 'static,
    Mk: FnOnce() -> F,
    T: Send + 'static,
    S::Desc: AsyncTaskDesc,
{
    let wk = UltWorker::<S>::current().expect("cmpth: spawn_async called outside a worker");

    // Stack layout (same scheme as spawn, but no execution stack below F):
    //
    //   stack_top (high)
    //   ┌─────────────────┐
    //   │ StackResult<T>  │  ← result_addr
    //   ├─────────────────┤
    //   │ Future F        │  ← f_addr
    //   └─────────────────┘ ← base
    let result_layout = Layout::new::<StackResult<T>>();
    let f_layout = Layout::new::<F>();
    // Enough capacity to place both with worst-case alignment padding.
    let stack_size =
        result_layout.size() + result_layout.align() + f_layout.size() + f_layout.align() + 16;

    let desc = wk.shared().async_task_pool.alloc(wk.num(), true, stack_size);
    let mut token = SuspendedTaskToken(desc);
    token.commit_as_poll_fn();
    token.base_mut().scheduler = wk.shared.get() as *const ();
    // Arena-backed AsyncPool systems get a cell slot here; tag it with this
    // system's identity once (mirrors `spawn`'s own slot setup) so
    // `worker_from_async_arena_addr` can guard against a nested scheduler's
    // descriptor landing in the same arena.
    if let Some(slot) = token.base().slot {
        unsafe { (*slot).system_id.set(crate::resumable::common::lookup::system_id::<S>()) };
    }

    let stack_top = unsafe { (*desc).stack_top() } as usize;
    let result_addr = align_down(stack_top - result_layout.size(), result_layout.align());
    let f_addr = align_down(result_addr.wrapping_sub(f_layout.size()), f_layout.align().max(1));

    let result_ptr = result_addr as *mut StackResult<T>;
    let f_ptr = f_addr as *mut F;

    unsafe { f_ptr.write(mk()) };
    token.set_poll_fn(Some(poll_spawned_task::<S, T, F>));

    // Push to the deque as a ready-to-poll task.
    wk.push_local_top(token);

    JoinHandle { desc, result_ptr, result_drop: drop_stack_result::<T>, _marker: PhantomData }
}

/// Returned by [`spawn_async`]; see its docs for why the one `.await` is
/// mandatory rather than an implementation convenience.
///
/// Not generic over `F`/`Mk`: by the time this is constructed, `spawn_now`
/// has already consumed both and produced the finished [`JoinHandle`] — see
/// [`spawn_async`]'s docs for why that work happens eagerly rather than
/// inside `poll`. Kept as a crate-owned type (not `std::future::ready`, whose
/// `poll` is fixed and can never be changed to return `Poll::Pending`) so a
/// future work-first rewrite still has a `poll` body of its own to modify.
pub struct SpawnAction<S: SchedulerSystem, T> {
    handle: Option<JoinHandle<S, T>>,
}

impl<S: SchedulerSystem, T> Unpin for SpawnAction<S, T> {}

impl<S: SchedulerSystem, T> Future for SpawnAction<S, T> {
    type Output = JoinHandle<S, T>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<JoinHandle<S, T>> {
        let this = self.get_mut(); // SpawnAction: Unpin
        Poll::Ready(this.handle.take().expect("cmpth: SpawnAction polled after completion"))
    }
}

/// Type-erased poll function stored in `BasicTaskDesc::poll_fn` for async tasks.
///
/// Polls `F` once and reports what the caller's poll loop
/// ([`crate::resumable::stackless::worker::run_async_poll`]) should do next — see
/// [`TaskPollResult`]. When this completion claims a waiting
/// `AsyncJoiner` outright (`try_wake_state` returns `ClaimedParked`),
/// reports `ReadyAndContinue` with that descriptor instead of pushing it
/// to a deque: the caller's loop polls it directly next (symmetric
/// transfer), skipping a push/pop round trip for the common
/// parent-was-waiting-on-us case.
unsafe fn poll_spawned_task<S, T, F>(
    desc: *mut S::Desc,
    cx: &mut Context<'_>,
) -> TaskPollResult<S::Desc>
where
    S: SchedulerSystem,
    T: Send + 'static,
    F: Future<Output = T> + Send + 'static,
    S::Desc: AsyncTaskDesc,
{
    let stack_top = unsafe { (*desc).stack_top() } as usize;
    let result_layout = Layout::new::<StackResult<T>>();
    let f_layout = Layout::new::<F>();
    let result_addr = align_down(stack_top - result_layout.size(), result_layout.align());
    let f_addr = align_down(result_addr.wrapping_sub(f_layout.size()), f_layout.align().max(1));

    let f_ptr = f_addr as *mut F;
    let result_ptr = result_addr as *mut StackResult<T>;

    let poll_result = catch_unwind(AssertUnwindSafe(|| unsafe {
        Pin::new_unchecked(&mut *f_ptr).poll(cx)
    }));

    let sr = match poll_result {
        Ok(Poll::Pending) => return TaskPollResult::Pending,
        Ok(Poll::Ready(val)) => {
            unsafe { std::ptr::drop_in_place(f_ptr) };
            StackResult::Ok(val)
        }
        Err(e) => {
            unsafe { std::ptr::drop_in_place(f_ptr) };
            StackResult::Err(e)
        }
    };

    // Task done.  Invalidate the waker before signalling the joiner so that
    // any concurrent wake() becomes a no-op (IDLE state).
    unsafe { (*desc).mark_idle() };

    unsafe { result_ptr.write(sr) };

    // Publish FINISHED and settle whoever the old state names.  Runs on the
    // scheduler stack (no context-switch-target decision needed).
    match unsafe { (*desc).publish_finished() } {
        JoinState::SyncJoiner(j_desc) => {
            // Push the waiting ULT back to the deque.  This is always called
            // from within a worker (execute → run_async_poll → poll_fn).
            let wk = UltWorker::<S>::current()
                .expect("cmpth: poll_spawned_task called outside a worker");
            wk.push_local_top(SuspendedTaskToken(j_desc));
        }
        JoinState::AsyncWaker(w) => unsafe { Box::from_raw(w) }.wake(),
        JoinState::AsyncJoiner(j) => {
            // Claim j's next poll directly if nobody else can be driving it
            // (it was genuinely PARKED, not concurrently POLLING/NOTIFIED
            // elsewhere) — see TaskPollResult::ReadyAndContinue.
            match unsafe { (*j).try_wake_state() } {
                WakeOutcome::ClaimedParked => return TaskPollResult::ReadyAndContinue(j),
                WakeOutcome::SetNotified | WakeOutcome::NoOp => {}
            }
        }
        // JoinHandle still exists; it will read the result and free desc.
        JoinState::Running => {}
        JoinState::Detached => {
            // Detached task: drop result and return desc to the async pool
            // now. Always called from within a worker (execute →
            // run_async_poll → poll_fn), so a pool-relative wk_num is
            // available.
            unsafe { std::ptr::drop_in_place(result_ptr) };
            let wk = UltWorker::<S>::current()
                .expect("cmpth: poll_spawned_task called outside a worker");
            unsafe { wk.shared().async_task_pool.dealloc(wk.num(), desc) };
        }
        JoinState::Finished => unreachable!("cmpth: double async task completion"),
    }

    TaskPollResult::Ready
}

// ---------------------------------------------------------------------------
// recurse — pooled Box::pin replacement for self-recursive async fn bodies
// ---------------------------------------------------------------------------

/// Wrap a recursive async call's future, avoiding a `Box::pin` heap
/// allocation. Storage comes from a per-worker free list keyed by size
/// (see `Scheduler::recursion_pool`, reached via `wk.shared()`) instead of
/// the global allocator, falling back to a raw allocation when called
/// outside a worker.
///
/// An `async fn` cannot directly recurse — the call `f(n - 1).await`
/// inside `f`'s own body would need `f`'s state machine to embed another
/// instance of itself, an infinitely-sized type (E0733). `Box::pin` is
/// Rust's standard workaround; this is a cheaper one for the common case
/// where the recursive call is only ever awaited by its immediate caller:
/// unlike [`spawn_async`], the returned [`RecursionFrame`] is never a
/// schedulable task — no `TaskDesc`/`join_state`, no `Waker` construction,
/// never pushed to a deque or stealable. Its `poll` just forwards to the
/// wrapped future using the caller's own `Context`, exactly like an
/// ordinary (non-recursive) nested `.await` would.
///
/// Takes a **thunk** (`mk`), not an already-constructed `F` — for API
/// consistency with [`spawn_async`] (both accept "how to build the future"
/// rather than the future itself), not because `recurse` itself would risk
/// E0733 either way: `recurse` is a plain, eager, synchronous function, and
/// `mk()` is called and its result written into pool storage immediately,
/// inside this call, before `RecursionFrame` (a fixed-size `NonNull<F>`
/// handle, independent of `F`'s size) is ever constructed or held across
/// any `.await`. Nothing here is deferred to a `poll` call, and `poll`
/// below still uses a thread-local lookup directly — see [`spawn_async`]'s
/// docs for why that first `.await` exists for a different reason than
/// dodging one, and is not something to add or remove per implementation.
///
/// ```
/// # use cmpth::resumable::stackless::thread::{recurse, spawn_async};
/// # use cmpth::DefaultNestedDualTaskSystem as S;
/// fn fib(n: u64) -> impl std::future::Future<Output = u64> + Send {
///     async move {
///         if n <= 1 { return n; }
///         let h1 = spawn_async::<S, _, _, _>(move || fib(n - 1)).await;
///         let r2 = recurse::<S, _, _>(move || fib(n - 2)).await;
///         h1.await + r2
///     }
/// }
/// ```
pub fn recurse<S, F, Mk>(mk: Mk) -> RecursionFrame<S, F>
where
    S: SchedulerSystem,
    F: Future,
    Mk: FnOnce() -> F,
{
    let layout = Layout::new::<F>();
    let raw = match UltWorker::<S>::current() {
        Some(wk) => wk.shared().recursion_pool.alloc(wk.num(), layout),
        None => unsafe { std::alloc::alloc(layout) },
    };
    if raw.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    let typed = raw as *mut F;
    unsafe { typed.write(mk()) };
    RecursionFrame { ptr: unsafe { std::ptr::NonNull::new_unchecked(typed) }, _marker: PhantomData }
}

/// See [`recurse`]. Holds a pool-backed `F`, polled in place; never a
/// schedulable task.
pub struct RecursionFrame<S: SchedulerSystem, F> {
    ptr: std::ptr::NonNull<F>,
    _marker: PhantomData<S>,
}

unsafe impl<S: SchedulerSystem, F: Send> Send for RecursionFrame<S, F> {}
// The pointee is never moved (only ever touched through the stable
// pointer, exactly like `Pin<Box<F>>`), so the wrapper itself is Unpin
// regardless of whether `F` is.
impl<S: SchedulerSystem, F> Unpin for RecursionFrame<S, F> {}

impl<S: SchedulerSystem, F: Future> Future for RecursionFrame<S, F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        let this = self.get_mut();
        unsafe { Pin::new_unchecked(&mut *this.ptr.as_ptr()).poll(cx) }
    }
}

impl<S: SchedulerSystem, F> Drop for RecursionFrame<S, F> {
    fn drop(&mut self) {
        unsafe { std::ptr::drop_in_place(self.ptr.as_ptr()) };
        let layout = Layout::new::<F>();
        match UltWorker::<S>::current() {
            Some(wk) => unsafe {
                wk.shared().recursion_pool.dealloc(wk.num(), self.ptr.as_ptr() as *mut u8, layout)
            },
            None => unsafe { std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout) },
        }
    }
}

// ---------------------------------------------------------------------------
// fork_async_parent_first — the stackless-only counterpart to
// fork_parent_first, used by `run_async`'s root task
// ---------------------------------------------------------------------------

/// Parent-first async fork: like
/// [`stackful::thread::fork_parent_first`](crate::resumable::stackful::thread::fork_parent_first),
/// but for a stackless root task
/// ([`crate::resumable::stackless::scheduler::run_async`]'s entry point).
///
/// No current worker is required — there isn't one yet at that point in
/// `run_async`; the caller pushes the returned continuation directly into
/// `workers[0]`'s deque, exactly like `fork_parent_first` does for the
/// stackful root. `has_handle = false` (no `JoinHandle` is produced), so
/// completion runs the same `JoinState::Detached` path as the stackful
/// root's `exit()` — reuses [`poll_spawned_task`] directly with `T = ()`.
pub(crate) fn fork_async_parent_first<S, F>(f: F, scheduler: *const ()) -> SuspendedTaskToken<S::Desc>
where
    S: SchedulerSystem,
    S::Desc: AsyncTaskDesc,
    F: Future<Output = ()> + Send + 'static,
{
    let result_layout = Layout::new::<StackResult<()>>();
    let f_layout = Layout::new::<F>();
    let stack_size = result_layout.size()
        + result_layout.align()
        + f_layout.size()
        + f_layout.align()
        + 16;

    // Allocated directly (not through S::AsyncPool): there's no current
    // worker yet to own a pool slot, and this runs exactly once per
    // `run_async` call anyway, so there's nothing to gain from pooling it.
    // Still wrapped via `Node::wrap_fresh` (not a bare `Box::new`), and
    // marked `oversized` unconditionally, so its eventual dealloc (through
    // the pool, like any other completed async task) can recover the node
    // via `Node::node_of` and always raw-frees it, rather than risking it
    // being pushed onto a free list sized for `S::ASYNC_POOL_SIZE`, which
    // this allocation doesn't necessarily match.
    let payload = S::Desc::alloc(stack_size, false);
    let desc = crate::resumable::common::pool::Node::wrap_fresh(0, true, payload);
    let mut token = SuspendedTaskToken(desc);
    token.commit_as_poll_fn();
    token.base_mut().scheduler = scheduler;

    let stack_top = unsafe { (*desc).stack_top() } as usize;
    let result_addr = align_down(stack_top - result_layout.size(), result_layout.align());
    let f_addr = align_down(result_addr.wrapping_sub(f_layout.size()), f_layout.align().max(1));
    let f_ptr = f_addr as *mut F;

    unsafe { f_ptr.write(f) };
    token.set_poll_fn(Some(poll_spawned_task::<S, (), F>));

    token
}
