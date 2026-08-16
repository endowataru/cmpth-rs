//! Stackless thread functions: `spawn_async`, `recurse`, and
//! `.await`-ing a [`JoinHandle`]
//! (shared with [`stackful::thread`](crate::resumable::stackful::thread) —
//! see `common::thread` for the handle type itself).

use std::alloc::Layout;
use std::cell::Cell;
use std::future::Future;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::resumable::common::system::{SchedulerSystem, WorkerSystem};
use crate::resumable::stackless::system::StacklessSchedulerSystem;
use crate::resumable::common::thread::{align_down, drop_stack_result, JoinHandle, StackResult};
use crate::resumable::common::desc::{HasExternalQueue, SuspendedTaskToken, TaskDesc, TaskDescAlloc, TaskDescCore, TaskExitSink};
use crate::resumable::stackless::desc::WakerTaskDesc;
use crate::resumable::stackless::desc::{AsyncTaskDesc, HasPollFn, TaskPollResult};
use crate::resumable::common::worker::{AsyncTaskPool, DescWorkerOps, LocalQueue, RecursionAlloc, UltWorker, WorkerOps};

// ---------------------------------------------------------------------------
// .await-ing a JoinHandle
// ---------------------------------------------------------------------------

impl<S: StacklessSchedulerSystem, T: Send + 'static> Future for JoinHandle<S, T>
where
    S::Worker: DescWorkerOps<S>,
{
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
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
        let current_wk = S::Worker::current();

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
                let joiner = wk.polling_async();
                if !joiner.is_null() {
                    // SAFETY: `joiner` is the descriptor this worker is
                    // currently, synchronously, driving via
                    // `run_async_poll` — exclusively ours to hand off
                    // through this registration for as long as that poll
                    // is in progress (same contract the old raw-pointer
                    // signature spelled out at this call site).
                    let token = unsafe { SuspendedTaskToken::from_raw(joiner) };
                    this.desc_ref().try_register_async_joiner(token).is_ok()
                } else {
                    this.desc_ref().try_register_waker(cx.waker().clone())
                }
            }
            None => this.desc_ref().try_register_waker(cx.waker().clone()),
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
fn try_reclaim_and_run<S>(wk: &S::Worker, desc: *mut S::Desc)
where
    S: StacklessSchedulerSystem,
    S::Worker: DescWorkerOps<S>,
{
    match wk.try_pop() {
        Some(popped) => {
            let popped: crate::resumable::common::desc::SuspendedTaskToken<S::Desc> = popped.into();
            if std::ptr::eq(popped.desc(), desc) {
                if popped.is_poll_fn_dispatch() {
                    let poll_fn = popped.poll_fn()
                        .expect("cmpth: descriptor committed to poll_fn dispatch but poll_fn unset");
                    crate::resumable::stackless::worker::run_async_poll::<S>(wk, desc, poll_fn);
                } else {
                    wk.push(popped.into());
                }
            } else {
                wk.push(popped.into());
            }
        }
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
/// Storage comes from `S::AsyncPool` (see [`PoolSystem::AsyncPool`](crate::resumable::common::system::PoolSystem::AsyncPool)):
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
    S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>,
    F: Future<Output = T> + Send + 'static,
    Mk: FnOnce() -> F + Send + 'static,
    T: Send + 'static,
{
    SpawnAction { handle: Some(spawn_now::<S, T, F, Mk>(mk)) }
}

// Stack layout (same scheme as spawn, but no execution stack below F):
//
//   stack_top (high)
//   ┌─────────────────┐
//   │ StackResult<T>  │  ← result_addr
//   ├─────────────────┤
//   │ Future F        │  ← f_addr
//   └─────────────────┘ ← base
//
/// Flat storage size needed for `F` + `StackResult<T>` — shared by
/// [`build_async_task`]'s allocation and `parallel_call`'s warm-cache
/// fits-check: any pool-path descriptor's buffer is always exactly
/// `S::ASYNC_POOL_SIZE` bytes regardless of the smaller size a particular
/// call actually requested (`ReturnPool::alloc`,
/// `resumable/common/pool.rs`), so anything whose own `stack_size` is at
/// or below that threshold can safely reuse one.
pub(crate) fn async_branch_stack_size<T, F>() -> usize {
    let result_layout = Layout::new::<StackResult<T>>();
    let f_layout = Layout::new::<F>();
    // Enough capacity to place both with worst-case alignment padding.
    result_layout.size() + result_layout.align() + f_layout.size() + f_layout.align() + 16
}

/// Write `mk()`'s future into an already-allocated descriptor's flat
/// storage and commit it to poll_fn dispatch. Shared by
/// [`build_async_task`]'s cold-alloc path (right after `alloc_async_task`)
/// and `parallel_call`'s warm-reuse path, which skips allocation,
/// `commit_as_poll_fn`, and `set_external_queue` entirely — worker-stable/
/// one-time-commit fields a never-dispatched descriptor never had a chance
/// to invalidate, so only this half needs redoing on reuse.
///
/// # Safety
/// `desc` must be exclusively owned by the caller for the duration of this
/// call, with storage for at least `async_branch_stack_size::<T, F>()`
/// bytes below its `stack_top()`.
pub(crate) unsafe fn write_async_branch<S, T, F, Mk>(desc: *mut S::Desc, mk: Mk) -> (*mut F, *mut StackResult<T>)
where
    S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>,
    F: Future<Output = T> + Send + 'static,
    Mk: FnOnce() -> F,
    T: Send + 'static,
{
    let stack_top = unsafe { (*desc).stack_top() } as usize;
    let result_layout = Layout::new::<StackResult<T>>();
    let f_layout = Layout::new::<F>();
    let result_addr = align_down(stack_top - result_layout.size(), result_layout.align());
    let f_addr = align_down(result_addr.wrapping_sub(f_layout.size()), f_layout.align().max(1));

    let result_ptr = result_addr as *mut StackResult<T>;
    let f_ptr = f_addr as *mut F;

    unsafe { f_ptr.write(mk()) };
    // SAFETY: forwarded from this function's own contract.
    let mut token = unsafe { SuspendedTaskToken::from_raw(desc) };
    token.set_poll_fn(Some(poll_spawned_task::<S, T, F>));

    (f_ptr, result_ptr)
}

/// Allocate an async task descriptor from `S::AsyncPool`, write `mk()`'s
/// future into place, and commit it to poll_fn dispatch — everything
/// [`spawn_now`] needs before pushing and wrapping a [`JoinHandle`], and
/// everything `parallel_call`'s pushed branch
/// (`resumable::stackless::system::ScopedStacklessTaskSystem::parallel_call`)
/// needs before deciding *when* to push and what to build around it, on the
/// cold path (no warm-cached descriptor available — see
/// `UltWorker::branch_warm_async_pop`). Deliberately doesn't push or wrap
/// anything itself — see [`spawn_now`]'s own doc comment for why this stays
/// a plain, eagerly-called function rather than a `poll` body.
pub(crate) fn build_async_task<S, T, F, Mk>(wk: &UltWorker<S>, mk: Mk) -> BuiltAsyncTask<S::Desc, F, T>
where
    S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>,
    F: Future<Output = T> + Send + 'static,
    Mk: FnOnce() -> F,
    T: Send + 'static,
{
    let stack_size = async_branch_stack_size::<T, F>();

    let mut token = wk.alloc_async_task(true, stack_size);
    let desc = token.as_desc() as *const S::Desc as *mut S::Desc;
    token.commit_as_poll_fn();
    token.set_external_queue(wk.external_queue() as *const _);
    drop(token); // no Drop side effect; write_async_branch re-tokenizes `desc`

    // SAFETY: `desc` was freshly allocated above with `stack_size` bytes of
    // storage (>= what `write_async_branch` needs for this exact `T`/`F`),
    // and has never been wrapped in a token since `alloc_async_task`
    // returned it here — trivially exclusive.
    let (f_ptr, result_ptr) = unsafe { write_async_branch::<S, T, F, Mk>(desc, mk) };

    BuiltAsyncTask { desc, f_ptr, result_ptr }
}

/// Send-safe bundle of what [`build_async_task`] produces: raw pointers
/// aren't `Send` by default, but these are exclusively owned (nothing else
/// can reach `desc`/`f_ptr`/`result_ptr` until the holder explicitly acts
/// on them) — same reasoning [`JoinHandle`]'s own `unsafe impl Send`
/// already rests on (`common/thread.rs`), same bound shape (only the
/// result type `T` needs to be `Send`; `F`'s `Send + 'static` bound is
/// already required wherever this is constructed).
pub(crate) struct BuiltAsyncTask<D, F, T> {
    pub(crate) desc: *mut D,
    pub(crate) f_ptr: *mut F,
    pub(crate) result_ptr: *mut StackResult<T>,
}

unsafe impl<D, F, T: Send> Send for BuiltAsyncTask<D, F, T> {}

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
    S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>,
    F: Future<Output = T> + Send + 'static,
    Mk: FnOnce() -> F,
    T: Send + 'static,
{
    let wk = UltWorker::<S>::current().expect("cmpth: spawn_async called outside a worker");
    let built = build_async_task::<S, T, F, Mk>(wk, mk);

    // SAFETY: `desc` was freshly built above and has never been wrapped in
    // a token before — trivially exclusive.
    let token = unsafe { SuspendedTaskToken::from_raw(built.desc) };
    // Push to the run queue as a ready-to-poll task.
    wk.push(token.into());

    JoinHandle { desc: built.desc, result_ptr: built.result_ptr, result_drop: drop_stack_result::<T>, _marker: PhantomData }
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
// Bound carried on the struct itself (not just the impls) because it holds
// a `JoinHandle<S, T>` field -- `JoinHandle` only needs plain
// `SchedulerSystem` now (see that struct's own comment), so this does too.
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

/// [`TaskExitSink`] for [`poll_spawned_task`]'s completion: capture the
/// continuation to resume (if any) rather than acting on it immediately, so
/// the caller can still take the symmetric-transfer fast path
/// ([`TaskPollResult::ReadyAndContinue`]) for a same-system async joiner
/// instead of an unconditional deque round trip. Dispatched purely on
/// `cont.is_poll_fn_dispatch()`, not on which join-state arm produced
/// `cont` (that distinction is exactly what [`TaskExitSink`] hides): a real
/// ULT continuation (a stackful sync joiner racing to register on the same
/// dual-flavor task) is never poll_fn-dispatchable, so it always falls back
/// to a plain deque push; a same-system async joiner always is, by
/// construction (see [`WakerTaskDesc::try_register_async_joiner`]'s docs),
/// so it always takes the symmetric-transfer path.
struct PollSpawnedSink<S: SchedulerSystem, T> {
    desc_ptr: *mut S::Desc,
    result_ptr: *mut StackResult<T>,
    continue_with: Cell<Option<<S::Desc as TaskDesc>::Suspended>>,
}

impl<S, T: Send + 'static> TaskExitSink<S::Desc> for PollSpawnedSink<S, T>
where
    S: SchedulerSystem + WorkerSystem<Worker = UltWorker<S>>,
    S::Desc: AsyncTaskDesc,
{
    fn resume(&self, cont: <S::Desc as TaskDesc>::Suspended) {
        if cont.is_poll_fn_dispatch() {
            self.continue_with.set(Some(cont));
        } else {
            // A real ULT continuation — push it back to the run queue like
            // any other requeued task. Always called from within a worker
            // (execute → run_async_poll → poll_fn).
            let wk = UltWorker::<S>::current()
                .expect("cmpth: poll_spawned_task called outside a worker");
            wk.push(cont.into());
        }
    }

    fn reclaim(&self) {
        // Detached task: drop result and return desc to the async pool
        // now. Always called from within a worker (execute →
        // run_async_poll → poll_fn), so a pool-relative wk_num is
        // available.
        unsafe { std::ptr::drop_in_place(self.result_ptr) };
        let wk = UltWorker::<S>::current()
            .expect("cmpth: poll_spawned_task called outside a worker");
        unsafe { wk.free_async_task(self.desc_ptr) };
    }
}

/// Type-erased poll function stored in `DualTaskDesc::poll_fn` for async tasks.
///
/// Polls `F` once and reports what the caller's poll loop
/// ([`crate::resumable::stackless::worker::run_async_poll`]) should do next — see
/// [`TaskPollResult`]. When this completion claims a waiting same-system
/// async joiner outright, reports `ReadyAndContinue` with that descriptor
/// instead of pushing it to a deque: the caller's loop polls it directly
/// next (symmetric transfer), skipping a push/pop round trip for the
/// common parent-was-waiting-on-us case — see [`PollSpawnedSink`].
unsafe fn poll_spawned_task<S, T, F>(
    desc: *mut S::Desc,
    cx: &mut Context<'_>,
) -> TaskPollResult<S::Desc>
where
    S: SchedulerSystem + WorkerSystem<Worker = UltWorker<S>>,
    T: Send + 'static,
    F: Future<Output = T> + Send + 'static,
    S::Desc: AsyncTaskDesc,
{
    // One relay point for the whole function: `desc` is a live descriptor
    // for as long as `poll_spawned_task` is running (guaranteed by
    // `run_async_poll`'s caller contract), so everything below reaches it
    // through this single `&S::Desc` instead of repeated raw derefs.
    let desc_ref: &S::Desc = unsafe { &*desc };

    let stack_top = desc_ref.stack_top() as usize;
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
    desc_ref.mark_idle();

    unsafe { result_ptr.write(sr) };

    // Publish FINISHED and settle whoever the old state names.  Runs on the
    // scheduler stack (no context-switch-target decision needed).
    let sink = PollSpawnedSink::<S, T> {
        desc_ptr: desc,
        result_ptr,
        continue_with: Cell::new(None),
    };
    desc_ref.finish_and_settle(&sink);

    if let Some(cont) = sink.continue_with.take() {
        // Claimed a same-system async joiner directly — continue polling
        // it next instead of a deque round trip.
        return TaskPollResult::ReadyAndContinue(cont.into_raw());
    }

    TaskPollResult::Ready
}

// ---------------------------------------------------------------------------
// recurse — pooled Box::pin replacement for self-recursive async fn bodies
// ---------------------------------------------------------------------------

/// Wrap a recursive async call's future, avoiding a `Box::pin` heap
/// allocation. Storage comes from a per-worker free list keyed by size
/// (see `Scheduler::recursion_pool`, reached via
/// [`RecursionAlloc::alloc_recursion_frame`]) instead of the global
/// allocator, falling back to a raw allocation when called outside a
/// worker.
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
    S: WorkerSystem,
    S::Worker: RecursionAlloc,
    F: Future,
    Mk: FnOnce() -> F,
{
    let layout = Layout::new::<F>();
    let raw = match S::Worker::current() {
        Some(wk) => wk.alloc_recursion_frame(layout),
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
// Bounded on plain `WorkerSystem` + `S::Worker: RecursionAlloc` (not
// `SchedulerSystem`): nothing here ever names `SuspendedTaskToken<S::Desc>`
// or dispatches an item -- `alloc_recursion_frame`/`free_recursion_frame`
// are `RecursionAlloc` methods, `UltWorker<S>` implements that trait
// unconditionally for any `S: WorkerSystem` (see `common::worker`), so this
// needs neither `SuspendedToken` nor `Worker` pinned to anything concrete.
// `Drop` impls must restate exactly the bounds the type definition has, so
// the bound has to live here regardless.
pub struct RecursionFrame<S: WorkerSystem, F>
where
    S::Worker: RecursionAlloc,
{
    ptr: std::ptr::NonNull<F>,
    _marker: PhantomData<S>,
}

unsafe impl<S: WorkerSystem, F: Send> Send for RecursionFrame<S, F> where S::Worker: RecursionAlloc {}
// The pointee is never moved (only ever touched through the stable
// pointer, exactly like `Pin<Box<F>>`), so the wrapper itself is Unpin
// regardless of whether `F` is.
impl<S: WorkerSystem, F> Unpin for RecursionFrame<S, F> where S::Worker: RecursionAlloc {}

impl<S: WorkerSystem, F: Future> Future for RecursionFrame<S, F>
where
    S::Worker: RecursionAlloc,
{
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        let this = self.get_mut();
        unsafe { Pin::new_unchecked(&mut *this.ptr.as_ptr()).poll(cx) }
    }
}

impl<S: WorkerSystem, F> Drop for RecursionFrame<S, F>
where
    S::Worker: RecursionAlloc,
{
    fn drop(&mut self) {
        unsafe { std::ptr::drop_in_place(self.ptr.as_ptr()) };
        let layout = Layout::new::<F>();
        match S::Worker::current() {
            Some(wk) => unsafe {
                wk.free_recursion_frame(self.ptr.as_ptr() as *mut u8, layout)
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
/// completion runs the same abandoned/`reclaim` path as the stackful
/// root's `exit()` — reuses [`poll_spawned_task`] directly with `T = ()`.
pub(crate) fn fork_async_parent_first<S, F>(f: F, external_queue: *const S::ExternalQueue) -> SuspendedTaskToken<S::Desc>
where
    S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>,
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
    // SAFETY: `desc` was just freshly allocated above and has never been
    // wrapped in a token before — trivially exclusive.
    let mut token = unsafe { SuspendedTaskToken::from_raw(desc) };
    token.commit_as_poll_fn();
    token.set_external_queue(external_queue);

    let stack_top = token.as_desc().stack_top() as usize;
    let result_addr = align_down(stack_top - result_layout.size(), result_layout.align());
    let f_addr = align_down(result_addr.wrapping_sub(f_layout.size()), f_layout.align().max(1));
    let f_ptr = f_addr as *mut F;

    unsafe { f_ptr.write(f) };
    token.set_poll_fn(Some(poll_spawned_task::<S, (), F>));

    token
}

// ---------------------------------------------------------------------------
// BranchPoll — the un-stolen fast path for parallel_call's pushed branch
// ---------------------------------------------------------------------------

/// The un-stolen fast path for `parallel_call`'s pushed branch `b`
/// (`resumable::stackless::system::ScopedStacklessTaskSystem::parallel_call`):
/// once popped back by identity (nobody stole it), `b`'s `Fb` value is
/// polled directly with `parallel_call`'s own ambient `Context` — no
/// `poll_fn` indirection, no per-poll `RawWaker` construction, none of
/// `run_async_poll`'s bookkeeping (`mark_polling`/`set_polling_async`/the
/// `TaskPollResult` match). Modeled directly on [`RecursionFrame`], but
/// reclaiming an `S::AsyncPool` slot (`free_async_task`) instead of a
/// `RecursionAlloc` one, since `b` was built as a real (if never actually
/// dispatched) task via [`build_async_task`], not a `recurse()` frame.
pub(crate) struct BranchPoll<S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>, Fb> {
    /// Null once handled (`poll`'s `Ready` arm, or after `Drop` has run) —
    /// same idiom as [`JoinHandle`]'s own `Drop` (`common/thread.rs`).
    pub(crate) desc: *mut S::Desc,
    pub(crate) f_ptr: *mut Fb,
    /// Whether `desc` came from the warm-cache-eligible size class (see
    /// `async_branch_stack_size`/`ASYNC_POOL_SIZE`) — decides whether
    /// reclaiming it (in `poll`'s `Ready` arm, or `Drop`) pushes it back to
    /// `UltWorker::branch_warm_async` instead of returning it to the
    /// general pool. Sound in *both* places, not just `poll`'s `Ready` arm:
    /// `BranchPoll` never gives `Fb` a reference to `desc` itself (only to
    /// `f_ptr`, the flat byte storage, polled with the caller's own ambient
    /// `Context`), so `desc`'s own fields (`join_state`/`poll_fn`) are
    /// provably untouched regardless of how many `Pending`s `Fb` returns
    /// before completing, or whether it's dropped mid-flight instead —
    /// exclusivity was already established by the `try_pop`+identity check
    /// before this was ever constructed, not by "never suspended".
    pub(crate) fits: bool,
}

/// Shared by [`BranchPoll`]'s `poll` (`Ready` arm) and `Drop`: return
/// `desc` to the warm cache if it's eligible, otherwise the general pool.
fn reclaim_branch<S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>>(
    wk: &UltWorker<S>,
    desc: *mut S::Desc,
    fits: bool,
) {
    if fits {
        wk.branch_warm_async_push(desc);
    } else {
        unsafe { wk.free_async_task(desc) };
    }
}

impl<S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>, Fb> Unpin for BranchPoll<S, Fb> {}

// SAFETY: `desc`/`f_ptr` are raw pointers into pool-owned memory this value
// exclusively holds for as long as it's alive (see the struct's own doc
// comment) — same reasoning `JoinHandle`'s own `unsafe impl Send` rests on.
// `Fb: Send` is already required wherever `BranchPoll<S, Fb>` is
// constructed (`parallel_call`'s own `Fb: Future<Output = Rb> + Send +
// 'static` bound), so nothing weaker slips through this impl.
unsafe impl<S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>, Fb: Send> Send for BranchPoll<S, Fb> {}

impl<S, Fb> Future for BranchPoll<S, Fb>
where
    S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>,
    Fb: Future,
{
    type Output = Fb::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Fb::Output> {
        let this = self.get_mut();
        // No catch_unwind here — matches the un-stolen path elsewhere in
        // this crate (stackful's `f_ptr.read()()`): a panic just unwinds
        // through this call normally, and Drop below still reclaims the
        // pool slot if that happens before Ready.
        match unsafe { Pin::new_unchecked(&mut *this.f_ptr) }.poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(val) => {
                unsafe { std::ptr::drop_in_place(this.f_ptr) };
                let wk = UltWorker::<S>::current().expect("cmpth: worker vanished");
                reclaim_branch::<S>(wk, this.desc, this.fits);
                this.desc = std::ptr::null_mut();
                Poll::Ready(val)
            }
        }
    }
}

impl<S, Fb> Drop for BranchPoll<S, Fb>
where
    S: StacklessSchedulerSystem + WorkerSystem<Worker = UltWorker<S>>,
{
    fn drop(&mut self) {
        if self.desc.is_null() {
            return; // already handled in poll's Ready branch
        }
        unsafe { std::ptr::drop_in_place(self.f_ptr) };
        match UltWorker::<S>::current() {
            Some(wk) => reclaim_branch::<S>(wk, self.desc, self.fits),
            // No worker context (e.g. dropped during scheduler teardown):
            // same fallback JoinHandle::drop uses for the same situation.
            None => unsafe { crate::resumable::common::pool::free_desc(self.desc) },
        }
    }
}
