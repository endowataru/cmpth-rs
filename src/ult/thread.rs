//! Thread functions: fork (child-first and parent-first), exit, join.

use std::alloc::Layout;
use std::any::Any;
use std::future::Future;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::task::{Context, Poll, Waker};

use crate::context::{ContextPolicy, Transfer};
use crate::traits::thread_system::JoinHandleLike;
use crate::ult::system::UltSystem;
use crate::ult::desc::{
    decode_join_state, JoinState, SuspendedUlt, UltDesc, JS_ASYNC_TAG, JS_FINISHED,
};
use crate::ult::worker::{ContextSwitcher, LocalQueue, TaskPool, UltWorker, Worker};

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
    S: UltSystem,
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let wk = UltWorker::<S>::current().expect("cmpth: spawn called outside a worker");
    let desc = wk.alloc_task(true);
    unsafe { (*desc).scheduler = wk.shared.get() as *const () };
    if let Some(slot) = unsafe { (*desc).slot } {
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
        let wk = unsafe { &*((*desc).worker.get() as *const UltWorker<S>) };
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
pub(crate) fn fork_parent_first<S: UltSystem>(body: ErasedBody, scheduler: *const ()) -> SuspendedUlt {
    use crate::ult::stack::StackAlloc as _;
    let desc = UltDesc::alloc_with(S::StackAlloc::alloc_stack(S::STACK_SIZE).into(), false);
    unsafe { (*desc).scheduler = scheduler };
    if let Some(slot) = unsafe { (*desc).slot } {
        unsafe { (*slot).system_id.set(crate::ult::lookup::system_id::<S>()) };
    }
    let arg = Box::into_raw(Box::new(body));
    let ctx = unsafe {
        S::Ctx::make_context((*desc).stack_top(), task_entry::<S>, arg as *mut ())
    };
    unsafe { (*desc).ctx.store(ctx.0, Ordering::Release) };
    SuspendedUlt(desc)
}

unsafe extern "C" fn task_entry<S: UltSystem>(transfer: Transfer, arg: *mut ()) -> ! {
    let wk = unsafe { &*(transfer.0 as *const UltWorker<S>) };
    let desc = wk.cur_task.get();
    let body = *unsafe { Box::from_raw(arg as *mut ErasedBody) };
    let result = catch_unwind(AssertUnwindSafe(body));
    // See spawn: the descriptor tracks the current worker across migrations.
    let wk = unsafe { &*((*desc).worker.get() as *const UltWorker<S>) };
    debug_assert!(std::ptr::eq(wk, UltWorker::<S>::current().expect("cmpth: worker vanished")));
    debug_assert!(std::ptr::eq(wk.cur_task.get(), desc));
    unsafe { *(*desc).result.get() = Some(result) };
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
fn exit_with_result<S: UltSystem, T: Send + 'static>(
    wk: &UltWorker<S>,
    desc: *mut UltDesc,
    result_ptr: *mut StackResult<T>,
    val: Result<T, Box<dyn Any + Send>>,
) -> ! {
    match decode_join_state(unsafe { (*desc).join_state.load(Ordering::Acquire) }) {
        JoinState::SyncJoiner(j_desc) => {
            // Direct handoff: switch straight to the parked joiner.
            let sr = match val { Ok(v) => StackResult::Ok(v), Err(e) => StackResult::Err(e) };
            unsafe { result_ptr.write(sr) };
            wk.exit_to_cont(SuspendedUlt(j_desc), move |_wk| unsafe {
                (*desc).join_state.store(JS_FINISHED, Ordering::Release);
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
                let old = unsafe { (*desc).join_state.swap(JS_FINISHED, Ordering::AcqRel) };
                match decode_join_state(old) {
                    // No joiner appeared: the JoinHandle collects the result.
                    JoinState::Running => {}
                    // A joiner registered while we were exiting.
                    JoinState::SyncJoiner(j) => wk.push_local_top(SuspendedUlt(j)),
                    JoinState::AsyncWaker(w) => unsafe { Box::from_raw(w) }.wake(),
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
fn exit<S: UltSystem>(wk: &UltWorker<S>, desc: *mut UltDesc) -> ! {
    match decode_join_state(unsafe { (*desc).join_state.load(Ordering::Acquire) }) {
        JoinState::SyncJoiner(j_desc) => {
            wk.exit_to_cont(SuspendedUlt(j_desc), move |_wk| unsafe {
                (*desc).join_state.store(JS_FINISHED, Ordering::Release);
            })
        }
        // Root tasks start in this state; desc.result drops with the desc.
        JoinState::Detached => wk.exit_to_sched(move |wk| unsafe { wk.free_task(desc) }),
        _ => wk.exit_to_sched(move |wk| {
            let old = unsafe { (*desc).join_state.swap(JS_FINISHED, Ordering::AcqRel) };
            match decode_join_state(old) {
                JoinState::Running => {}
                JoinState::SyncJoiner(j) => wk.push_local_top(SuspendedUlt(j)),
                JoinState::AsyncWaker(w) => unsafe { Box::from_raw(w) }.wake(),
                JoinState::Detached => unsafe { wk.free_task(desc) },
                JoinState::Finished => unreachable!("cmpth: double task exit"),
            }
        }),
    }
}

// ---------------------------------------------------------------------------
// JoinHandle
// ---------------------------------------------------------------------------

pub struct JoinHandle<S: UltSystem, T> {
    desc: *mut UltDesc,
    result_ptr: *mut StackResult<T>,
    // Type-erased drop for the result slot; avoids a T: Send + 'static bound
    // on the Drop impl (Rust disallows extra bounds there).
    result_drop: unsafe fn(*mut ()),
    _marker: PhantomData<(S, T)>,
}

unsafe fn drop_stack_result<T>(ptr: *mut ()) {
    unsafe { std::ptr::drop_in_place(ptr as *mut StackResult<T>) };
}

unsafe impl<S: UltSystem, T: Send> Send for JoinHandle<S, T> {}
// JoinHandle holds only raw pointers; it is safe to move at any time.
impl<S: UltSystem, T> Unpin for JoinHandle<S, T> {}

impl<S: UltSystem, T: Send + 'static> JoinHandle<S, T> {
    pub fn join(self) -> Result<T, Box<dyn Any + Send>> {
        let wk = UltWorker::<S>::current().expect("cmpth: join called outside a worker");
        let desc = self.desc;

        // Fast path: the child already exited.  Child-first spawn guarantees
        // this whenever the parent continuation was not stolen, so the whole
        // fork-join hot path lands here.  FINISHED is published with Release
        // after the result write; the Acquire read makes the result visible.
        if unsafe { (*desc).join_state.load(Ordering::Acquire) } == JS_FINISHED {
            return self.take_result(wk);
        }

        // Slow path: register this task as the sync joiner with one CAS.
        // cond_suspend cancels the suspension when the child finished in the
        // meantime (the CAS loses to the exit path's swap).
        // The returned worker is the one we resumed on — no TLS re-read.
        let wk = wk.cond_suspend_to_sched(move |_wk, prev| {
            let j = prev.as_ref().expect("cond_suspend contract").desc() as usize;
            let mut cur = unsafe { (*desc).join_state.load(Ordering::Relaxed) };
            loop {
                if cur == JS_FINISHED {
                    return; // leave `prev` in place -> cancel, resume at once
                }
                match unsafe {
                    (*desc).join_state.compare_exchange_weak(
                        cur, j, Ordering::Release, Ordering::Acquire,
                    )
                } {
                    Ok(_) => {
                        // A sync join supersedes any registered async waker.
                        if let JoinState::AsyncWaker(w) = decode_join_state(cur) {
                            drop(unsafe { Box::from_raw(w) });
                        }
                        let _ = prev.take().expect("cond_suspend contract").into_raw();
                        return; // committed: we stay parked
                    }
                    Err(c) => cur = c,
                }
            }
        });

        debug_assert_eq!(
            unsafe { (*desc).join_state.load(Ordering::Relaxed) },
            JS_FINISHED
        );
        self.take_result(wk)
    }

    fn take_result(self, wk: &UltWorker<S>) -> Result<T, Box<dyn Any + Send>> {
        let desc = self.desc;
        let result_ptr = self.result_ptr;
        std::mem::forget(self);
        let sr = unsafe { result_ptr.read() };
        // Async task descs bypass the pool (variable size allocation).
        if unsafe { (*desc).poll_fn.is_some() } {
            unsafe { UltDesc::free(desc) };
        } else {
            unsafe { wk.free_task(desc) };
        }
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
        unsafe { UltDesc::free(desc) };
        match sr {
            StackResult::Ok(val) => Ok(val),
            StackResult::Err(e) => Err(e),
        }
    }
}

impl<S: UltSystem, T> Drop for JoinHandle<S, T> {
    fn drop(&mut self) {
        if self.desc.is_null() {
            return; // consumed by Future::poll
        }
        let desc = self.desc;
        let result_ptr = self.result_ptr as *mut ();
        let result_drop = self.result_drop;

        let mut cur = unsafe { (*desc).join_state.load(Ordering::Acquire) };
        loop {
            if cur == JS_FINISHED {
                // Task done: this handle owns the result and the descriptor.
                unsafe { result_drop(result_ptr) };
                // Async task descs bypass the pool (variable size).
                if unsafe { (*desc).poll_fn.is_some() } {
                    unsafe { UltDesc::free(desc) };
                } else {
                    match UltWorker::<S>::current() {
                        Some(wk) => unsafe { wk.free_task(desc) },
                        None => unsafe { UltDesc::free(desc) },
                    }
                }
                return;
            }
            // RUNNING or an async waker (a parked sync joiner is impossible:
            // join consumes the handle).  Detach; the exit path cleans up.
            match unsafe {
                (*desc).join_state.compare_exchange_weak(
                    cur,
                    crate::ult::desc::JS_DETACHED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
            } {
                Ok(_) => {
                    if let JoinState::AsyncWaker(w) = decode_join_state(cur) {
                        drop(unsafe { Box::from_raw(w) });
                    }
                    return;
                }
                Err(c) => cur = c,
            }
        }
    }
}

impl<S: UltSystem, T: Send + 'static> JoinHandleLike<T> for JoinHandle<S, T> {
    fn join(self) -> T {
        match JoinHandle::join(self) {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        }
    }
}

impl<S: UltSystem, T: Send + 'static> Future for JoinHandle<S, T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        let this = self.get_mut(); // JoinHandle: Unpin
        let desc = this.desc;

        let mut cur = unsafe { (*desc).join_state.load(Ordering::Acquire) };
        if cur != JS_FINISHED {
            // Register (or replace) the async waker, boxed so that ownership
            // transfers atomically with the state word.
            let new = Box::into_raw(Box::new(cx.waker().clone())) as usize | JS_ASYNC_TAG;
            loop {
                if cur == JS_FINISHED {
                    // Finished while we were registering: discard ours.
                    drop(unsafe { Box::from_raw((new & !JS_ASYNC_TAG) as *mut Waker) });
                    break;
                }
                match unsafe {
                    (*desc).join_state.compare_exchange_weak(
                        cur, new, Ordering::Release, Ordering::Acquire,
                    )
                } {
                    Ok(_) => {
                        if let JoinState::AsyncWaker(w) = decode_join_state(cur) {
                            drop(unsafe { Box::from_raw(w) });
                        }
                        return Poll::Pending;
                    }
                    Err(c) => cur = c,
                }
            }
        }
        // FINISHED: consume the handle (null desc so Drop becomes a no-op).
        let handle = unsafe { std::ptr::read(this) };
        this.desc = std::ptr::null_mut();
        let result = match UltWorker::<S>::current() {
            Some(wk) => handle.take_result(wk),
            None => handle.take_result_no_worker(),
        };
        Poll::Ready(match result {
            Ok(v) => v,
            Err(e) => std::panic::resume_unwind(e),
        })
    }
}

// ---------------------------------------------------------------------------
// spawn_async — async Future as a lightweight task
// ---------------------------------------------------------------------------

/// Spawn a `Future` as a task.  The future is stored directly in a small
/// heap allocation (no 64 KB stack); the ULT executor polls it without any
/// context switch.
///
/// Returns a [`JoinHandle`] that can be `await`ed or `.join()`ed like a
/// spawned ULT.  Pool bypass: async task descs are always freed with
/// `UltDesc::free`, never returned to the fixed-size pool.
pub fn spawn_async<S, T, F>(f: F) -> JoinHandle<S, T>
where
    S: UltSystem,
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
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
    let stack_size = result_layout.size()
        + result_layout.align()
        + f_layout.size()
        + f_layout.align()
        + 16;

    // Allocate desc outside the pool (variable size).
    let desc = UltDesc::alloc(stack_size, true);
    unsafe { (*desc).scheduler = wk.shared.get() as *const () };

    let stack_top = unsafe { (*desc).stack_top() } as usize;
    let result_addr = align_down(stack_top - result_layout.size(), result_layout.align());
    let f_addr = align_down(result_addr.wrapping_sub(f_layout.size()), f_layout.align().max(1));

    let result_ptr = result_addr as *mut StackResult<T>;
    let f_ptr = f_addr as *mut F;

    unsafe { f_ptr.write(f) };
    unsafe { (*desc).poll_fn = Some(async_poll_fn::<S, T, F>) };

    // Push to the deque as a ready-to-poll task.
    wk.push_local_top(SuspendedUlt(desc));

    JoinHandle { desc, result_ptr, result_drop: drop_stack_result::<T>, _marker: PhantomData }
}

/// Type-erased poll function stored in `UltDesc::poll_fn` for async tasks.
///
/// Polls `F` once.  Returns `true` when `Poll::Ready` (the task is done and
/// the caller must not access `desc` afterwards for the detached case).
/// Returns `false` when `Poll::Pending`.
unsafe fn async_poll_fn<S, T, F>(
    desc: *mut UltDesc,
    cx: &mut Context<'_>,
) -> bool
where
    S: UltSystem,
    T: Send + 'static,
    F: Future<Output = T> + Send + 'static,
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
        Ok(Poll::Pending) => return false,
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
    unsafe { (*desc).waker_refs.store(crate::ult::desc::IDLE, std::sync::atomic::Ordering::Release) };

    unsafe { result_ptr.write(sr) };

    // Publish FINISHED and settle whoever the old state names.  Runs on the
    // scheduler stack (no context-switch-target decision needed).
    let old = unsafe { (*desc).join_state.swap(JS_FINISHED, Ordering::AcqRel) };
    match decode_join_state(old) {
        JoinState::SyncJoiner(j_desc) => {
            // Push the waiting ULT back to the deque.  This is always called
            // from within a worker (execute → run_async_poll → poll_fn).
            let wk = UltWorker::<S>::current()
                .expect("cmpth: async_poll_fn called outside a worker");
            wk.push_local_top(SuspendedUlt(j_desc));
        }
        JoinState::AsyncWaker(w) => unsafe { Box::from_raw(w) }.wake(),
        // JoinHandle still exists; it will read the result and free desc.
        JoinState::Running => {}
        JoinState::Detached => {
            // Detached task: drop result and free desc now.
            unsafe { std::ptr::drop_in_place(result_ptr) };
            unsafe { UltDesc::free(desc) };
        }
        JoinState::Finished => unreachable!("cmpth: double async task completion"),
    }

    true
}
