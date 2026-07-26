//! Thread functions: fork (child-first and parent-first), exit, join.

use std::alloc::Layout;
use std::any::Any;
use std::future::Future;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::task::{Context, Poll};

use crate::context::{ContextPolicy, Transfer};
use crate::traits::thread_system::JoinHandleLike;
use crate::ult::system::{SchedulerSystem, UltSchedulerSystem};
use crate::ult::desc::{
    AsyncTaskDesc, JoinState, StackfulTaskDesc, SuspendedUlt, TaskDesc, TaskDescAlloc, TaskPollResult,
    WakeOutcome, WakerTaskDesc, JS_FINISHED,
};
use crate::ult::pool::{DescPool, DynamicPool};
use crate::ult::worker::{ContextSwitcher, LocalQueue, StackfulWorker, TaskPool, UltWorker, Worker};

// Still needed for fork_parent_first (root task entry).
pub(crate) type ErasedBody = Box<dyn FnOnce() -> Box<dyn Any + Send> + Send>;

// Result stored directly on the child's stack, avoiding a Box for the success
// case.  The Err variant still boxes because that is what catch_unwind produces.
enum StackResult<T> {
    Ok(T),
    Err(Box<dyn Any + Send>),
}

// ---------------------------------------------------------------------------
// spawn (child-first fork)
// ---------------------------------------------------------------------------

/// Spawn a ULT.  Child-first: the child starts immediately on this worker and
/// the parent's continuation is pushed to the deque for stealing.
///
/// The closure `F` and the result slot `StackResult<T>` are placed directly on
/// the child's stack, avoiding two heap allocations that the old Box-erasure
/// approach required.
pub fn spawn<S, T, F>(f: F) -> JoinHandle<S, T>
where
    S: UltSchedulerSystem,
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
    <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc,
{
    let wk = UltWorker::<S>::current().expect("cmpth: spawn called outside a worker");
    let desc = wk.alloc_task(true, S::STACK_SIZE);
    unsafe { (*desc).scheduler().set(wk.shared.get() as *const ()) };
    if let Some(slot) = unsafe { (*desc).slot().get() } {
        unsafe { (*slot).system_id.set(crate::ult::lookup::system_id::<S>()) };
    }
    let stack_top = unsafe { (*desc).stack_top() } as usize;

    // Reserve space at the top of the child's stack (high addresses) for the
    // closure and the result slot.  The execution stack gets the rest below.
    //
    //   stack_top (high)
    //   ┌─────────────────┐
    //   │ StackResult<T>  │  ← result_addr
    //   ├─────────────────┤
    //   │ F               │  ← f_addr
    //   ├─────────────────┤
    //   │ (exec stack)    │  ← exec_top and below
    //   └─────────────────┘ ← stack base

    let result_layout = Layout::new::<StackResult<T>>();
    let f_layout = Layout::new::<F>();

    let result_addr = align_down(stack_top - result_layout.size(), result_layout.align());
    let f_addr = align_down(result_addr.wrapping_sub(f_layout.size()), f_layout.align().max(1));
    let exec_top = align_down(f_addr, 16) as *mut u8;

    let result_ptr = result_addr as *mut StackResult<T>;
    let f_ptr = f_addr as *mut F;

    // Write the closure onto the child's stack before switching.
    unsafe { f_ptr.write(f) };

    wk.suspend_to_new(exec_top, desc, move |wk, prev| {
        // Running on the child's stack.  Publish the parent for stealing, run
        // the closure, then exit via exit_with_result.
        wk.push_local_top(prev);
        let val = catch_unwind(AssertUnwindSafe(|| unsafe { f_ptr.read() }()));
        // The closure may have suspended and resumed on a different worker,
        // but every resume records the worker in the descriptor — cheaper
        // than a TLS lookup.
        let wk = unsafe { &*((*desc).worker().get() as *const UltWorker<S>) };
        debug_assert!(std::ptr::eq(wk, UltWorker::<S>::current().expect("cmpth: worker vanished")));
        debug_assert!(std::ptr::eq(wk.cur_task.get(), desc));
        exit_with_result(wk, desc, result_ptr, val)
    });

    JoinHandle { desc, result_ptr, result_drop: drop_stack_result::<T>, _marker: PhantomData }
}

#[inline]
fn align_down(addr: usize, align: usize) -> usize {
    addr & !(align - 1)
}

/// Parent-first fork: package `body` as a ready continuation without running
/// it.  Used for the root task of `run` and by [`PollerUltQueue::on_start`].
///
/// `scheduler` is a type-erased `*const Scheduler<S>` stored on the
/// descriptor for external-thread wake support.
pub(crate) fn fork_parent_first<S: UltSchedulerSystem>(body: ErasedBody, scheduler: *const ()) -> SuspendedUlt<S::Desc> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    use crate::ult::stack::StackAlloc as _;
    let desc = S::Desc::alloc_with(S::StackAlloc::alloc_stack(S::STACK_SIZE).into(), false);
    unsafe { (*desc).scheduler().set(scheduler) };
    if let Some(slot) = unsafe { (*desc).slot().get() } {
        unsafe { (*slot).system_id.set(crate::ult::lookup::system_id::<S>()) };
    }
    let arg = Box::into_raw(Box::new(body));
    let ctx = unsafe {
        S::Ctx::make_context((*desc).stack_top(), task_entry::<S>, arg as *mut ())
    };
    unsafe { (*desc).init_saved_context(ctx.0) };
    SuspendedUlt(desc)
}

unsafe extern "C" fn task_entry<S: UltSchedulerSystem>(transfer: Transfer, arg: *mut ()) -> ! where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    let wk = unsafe { &*(transfer.0 as *const UltWorker<S>) };
    let desc = wk.cur_task.get();
    let body = *unsafe { Box::from_raw(arg as *mut ErasedBody) };
    let result = catch_unwind(AssertUnwindSafe(body));
    // See spawn: the descriptor tracks the current worker across migrations.
    let wk = unsafe { &*((*desc).worker().get() as *const UltWorker<S>) };
    debug_assert!(std::ptr::eq(wk, UltWorker::<S>::current().expect("cmpth: worker vanished")));
    debug_assert!(std::ptr::eq(wk.cur_task.get(), desc));
    unsafe { *(*desc).result().get() = Some(result) };
    exit(wk, desc)
}

// ---------------------------------------------------------------------------
// exit helpers
// ---------------------------------------------------------------------------

/// Exit a spawned task.
///
/// One atomic decides everything.  A parked sync joiner and the detached
/// state are both *stable* (the joiner cannot act until resumed; a dropped
/// handle never comes back), so a plain Acquire read selects those paths.
/// Anything else (`RUNNING` or a registered async waker) can still change
/// concurrently — late joiner registration, waker replacement, detach — so
/// the exit callback publishes `FINISHED` with a `swap` *after* the context
/// switch and settles whichever party it finds in the old value.
fn exit_with_result<S: UltSchedulerSystem, T: Send + 'static>(
    wk: &UltWorker<S>,
    desc: *mut S::Desc,
    result_ptr: *mut StackResult<T>,
    val: Result<T, Box<dyn Any + Send>>,
) -> ! where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    match unsafe { (*desc).read_join_state() } {
        JoinState::SyncJoiner(j_desc) => {
            // Direct handoff: switch straight to the parked joiner.
            let sr = match val { Ok(v) => StackResult::Ok(v), Err(e) => StackResult::Err(e) };
            unsafe { result_ptr.write(sr) };
            wk.exit_to_cont(SuspendedUlt(j_desc), move |_wk| unsafe {
                (*desc).commit_finished();
            })
        }
        JoinState::Detached => {
            // No handle: drop val on the task's own stack before the context
            // switch so destructors run correctly.
            drop(val);
            wk.exit_to_sched(move |wk| unsafe { wk.free_task(desc) })
        }
        _ => {
            let sr = match val { Ok(v) => StackResult::Ok(v), Err(e) => StackResult::Err(e) };
            unsafe { result_ptr.write(sr) };
            wk.exit_to_sched(move |wk| {
                match unsafe { (*desc).publish_finished() } {
                    // No joiner appeared: the JoinHandle collects the result.
                    JoinState::Running => {}
                    // A joiner registered while we were exiting.
                    JoinState::SyncJoiner(j) => wk.push_local_top(SuspendedUlt(j)),
                    JoinState::AsyncWaker(w) => unsafe { Box::from_raw(w) }.wake(),
                    JoinState::AsyncJoiner(j) => unsafe { crate::ult::waker::try_wake_async::<S>(j) },
                    // The handle was dropped while we were exiting: the
                    // result already sits on our (still-allocated) stack.
                    JoinState::Detached => unsafe {
                        result_ptr.drop_in_place();
                        wk.free_task(desc);
                    },
                    JoinState::Finished => unreachable!("cmpth: double task exit"),
                }
            })
        }
    }
}

/// Exit for parent-first tasks (`fork_parent_first`): the result, if kept,
/// is already in `desc.result`.  Same state machine as `exit_with_result`.
fn exit<S: UltSchedulerSystem>(wk: &UltWorker<S>, desc: *mut S::Desc) -> ! where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    match unsafe { (*desc).read_join_state() } {
        JoinState::SyncJoiner(j_desc) => {
            wk.exit_to_cont(SuspendedUlt(j_desc), move |_wk| unsafe {
                (*desc).commit_finished();
            })
        }
        // Root tasks start in this state; desc.result drops with the desc.
        JoinState::Detached => wk.exit_to_sched(move |wk| unsafe { wk.free_task(desc) }),
        _ => wk.exit_to_sched(move |wk| {
            match unsafe { (*desc).publish_finished() } {
                JoinState::Running => {}
                JoinState::SyncJoiner(j) => wk.push_local_top(SuspendedUlt(j)),
                JoinState::AsyncWaker(w) => unsafe { Box::from_raw(w) }.wake(),
                JoinState::AsyncJoiner(j) => unsafe { crate::ult::waker::try_wake_async::<S>(j) },
                JoinState::Detached => unsafe { wk.free_task(desc) },
                JoinState::Finished => unreachable!("cmpth: double task exit"),
            }
        }),
    }
}

// ---------------------------------------------------------------------------
// JoinHandle
// ---------------------------------------------------------------------------

pub struct JoinHandle<S: SchedulerSystem, T> {
    desc: *mut S::Desc,
    result_ptr: *mut StackResult<T>,
    // Type-erased drop for the result slot; avoids a T: Send + 'static bound
    // on the Drop impl (Rust disallows extra bounds there).
    result_drop: unsafe fn(*mut ()),
    _marker: PhantomData<(S, T)>,
}

unsafe fn drop_stack_result<T>(ptr: *mut ()) {
    unsafe { std::ptr::drop_in_place(ptr as *mut StackResult<T>) };
}

unsafe impl<S: SchedulerSystem, T: Send> Send for JoinHandle<S, T> {}
// JoinHandle holds only raw pointers; it is safe to move at any time.
impl<S: SchedulerSystem, T> Unpin for JoinHandle<S, T> {}

impl<S: SchedulerSystem, T: Send + 'static> JoinHandle<S, T> {
    fn take_result(self, wk: &UltWorker<S>) -> Result<T, Box<dyn Any + Send>> {
        let desc = self.desc;
        let result_ptr = self.result_ptr;
        std::mem::forget(self);
        let sr = unsafe { result_ptr.read() };
        S::free_finished_desc(wk, desc);
        match sr {
            StackResult::Ok(val) => Ok(val),
            StackResult::Err(e) => Err(e),
        }
    }

    fn take_result_no_worker(self) -> Result<T, Box<dyn Any + Send>> {
        let desc = self.desc;
        let result_ptr = self.result_ptr;
        std::mem::forget(self);
        let sr = unsafe { result_ptr.read() };
        unsafe { S::Desc::free(desc) };
        match sr {
            StackResult::Ok(val) => Ok(val),
            StackResult::Err(e) => Err(e),
        }
    }
}

// Blocking `.join()`: inherently stackful (parks the calling ULT via
// `cond_suspend_to_sched`), so this is a separate impl block bounded on
// `UltSchedulerSystem` rather than widening the base block above — a
// stackless-only `JoinHandle` (from `spawn_async`) only ever gets `.await`ed
// (see `Future for JoinHandle` below), never `.join()`ed.
impl<S: UltSchedulerSystem, T: Send + 'static> JoinHandle<S, T>
where
    S::Desc: StackfulTaskDesc,
{
    pub fn join(self) -> Result<T, Box<dyn Any + Send>> {
        let wk = UltWorker::<S>::current().expect("cmpth: join called outside a worker");
        let desc = self.desc;

        // Fast path: the child already exited.  Child-first spawn guarantees
        // this whenever the parent continuation was not stolen, so the whole
        // fork-join hot path lands here.  FINISHED is published with Release
        // after the result write; the Acquire read makes the result visible.
        if unsafe { (*desc).is_finished() } {
            return self.take_result(wk);
        }

        // Slow path: register this task as the sync joiner with one CAS.
        // cond_suspend cancels the suspension when the child finished in the
        // meantime (the CAS loses to the exit path's swap).
        // The returned worker is the one we resumed on — no TLS re-read.
        let wk = wk.cond_suspend_to_sched(move |_wk, prev| {
            let joiner = prev.as_ref().expect("cond_suspend contract").desc();
            if unsafe { (*desc).try_register_sync_joiner(joiner) } {
                let _ = prev.take().expect("cond_suspend contract").into_raw();
            }
            // else: leave `prev` in place -> cancel, resume at once
        });

        debug_assert!(unsafe { (*desc).join_state().load(Ordering::Relaxed) } == JS_FINISHED);
        self.take_result(wk)
    }
}

impl<S: SchedulerSystem, T> Drop for JoinHandle<S, T> {
    // The common case (already consumed by `Future::poll`, `desc` null) is a
    // single branch; without this hint the compiler was leaving the whole
    // function (including the cold detach path) as a real call at every
    // drop-glue site (e.g. the `.await` desugaring's temporary), paying
    // call/return overhead for what should fold into a no-op check.
    #[inline]
    fn drop(&mut self) {
        if self.desc.is_null() {
            return; // consumed by Future::poll
        }
        let desc = self.desc;
        let result_ptr = self.result_ptr as *mut ();
        let result_drop = self.result_drop;

        // RUNNING or an async waker (a parked sync joiner is impossible: join
        // consumes the handle) -> detach, the exit path cleans up. Already
        // finished -> this handle owns the result and the descriptor.
        if unsafe { (*desc).try_mark_detached() } {
            unsafe { result_drop(result_ptr) };
            match UltWorker::<S>::current() {
                Some(wk) => S::free_finished_desc(wk, desc),
                None => unsafe { S::Desc::free(desc) },
            }
        }
    }
}

impl<S: UltSchedulerSystem, T: Send + 'static> JoinHandleLike<T> for JoinHandle<S, T>
where
    S::Desc: StackfulTaskDesc,
{
    fn join(self) -> T {
        match JoinHandle::join(self) {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        }
    }
}

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
    if let Some(wk) = crate::ult::lookup::worker_from_async_arena_addr::<S>(self_addr) {
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
            match unsafe { (*desc).poll_fn().get() } {
                Some(poll_fn) => crate::ult::worker::run_async_poll(wk, desc, poll_fn),
                None => wk.push_local_top(popped),
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
/// `mk()` runs exactly once, inside [`spawn_now`], once the descriptor
/// it writes into already exists.
///
/// Storage comes from `S::AsyncPool` (see [`SchedulerSystem::AsyncPool`]):
/// futures that fit its configured slot size are served from its free list
/// like any pooled ULT stack; larger ones fall back to a one-off
/// allocation, freed directly rather than returned to the pool.
///
/// The actual registration work happens *eagerly*, right here, not deferred
/// into `SpawnAction::poll` — see [`spawn_now`]'s docs for why keeping that
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
    unsafe { (*desc).scheduler().set(wk.shared.get() as *const ()) };
    // Arena-backed AsyncPool systems get a cell slot here; tag it with this
    // system's identity once (mirrors `spawn`'s own slot setup) so
    // `worker_from_async_arena_addr` can guard against a nested scheduler's
    // descriptor landing in the same arena.
    if let Some(slot) = unsafe { (*desc).slot().get() } {
        unsafe { (*slot).system_id.set(crate::ult::lookup::system_id::<S>()) };
    }

    let stack_top = unsafe { (*desc).stack_top() } as usize;
    let result_addr = align_down(stack_top - result_layout.size(), result_layout.align());
    let f_addr = align_down(result_addr.wrapping_sub(f_layout.size()), f_layout.align().max(1));

    let result_ptr = result_addr as *mut StackResult<T>;
    let f_ptr = f_addr as *mut F;

    unsafe { f_ptr.write(mk()) };
    unsafe { (*desc).poll_fn().set(Some(poll_spawned_task::<S, T, F>)) };

    // Push to the deque as a ready-to-poll task.
    wk.push_local_top(SuspendedUlt(desc));

    JoinHandle { desc, result_ptr, result_drop: drop_stack_result::<T>, _marker: PhantomData }
}

/// Returned by [`spawn_async`]; see its docs for why the one `.await` is
/// mandatory rather than an implementation convenience.
///
/// Not generic over `F`/`Mk`: by the time this is constructed, [`spawn_now`]
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
/// ([`crate::ult::worker::run_async_poll`]) should do next — see
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
            wk.push_local_top(SuspendedUlt(j_desc));
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
/// (see [`UltWorker::recursion_pool_take`]) instead of the global
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
/// # use cmpth::ult::thread::{recurse, spawn_async};
/// # use cmpth::DefaultUltUltSystem as S;
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

/// Parent-first async fork: like [`fork_parent_first`], but for a stackless
/// root task ([`crate::ult::scheduler::run_async`]'s entry point).
///
/// No current worker is required — there isn't one yet at that point in
/// `run_async`; the caller pushes the returned continuation directly into
/// `workers[0]`'s deque, exactly like `fork_parent_first` does for the
/// stackful root. `has_handle = false` (no `JoinHandle` is produced), so
/// completion runs the same `JoinState::Detached` path as the stackful
/// root's `exit()` — reuses [`poll_spawned_task`] directly with `T = ()`.
pub(crate) fn fork_async_parent_first<S, F>(f: F, scheduler: *const ()) -> SuspendedUlt<S::Desc>
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
    // Marked `oversized` unconditionally so its eventual dealloc (through
    // the pool, like any other completed async task) always raw-frees it
    // rather than risking it being pushed onto a free list sized for
    // `S::ASYNC_POOL_SIZE`, which this allocation doesn't necessarily match.
    let desc = S::Desc::alloc(stack_size, false);
    unsafe { (*desc).oversized().set(true) };
    unsafe { (*desc).scheduler().set(scheduler) };

    let stack_top = unsafe { (*desc).stack_top() } as usize;
    let result_addr = align_down(stack_top - result_layout.size(), result_layout.align());
    let f_addr = align_down(result_addr.wrapping_sub(f_layout.size()), f_layout.align().max(1));
    let f_ptr = f_addr as *mut F;

    unsafe { f_ptr.write(f) };
    unsafe { (*desc).poll_fn().set(Some(poll_spawned_task::<S, (), F>)) };

    SuspendedUlt(desc)
}
