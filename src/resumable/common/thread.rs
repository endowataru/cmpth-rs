//! [`JoinHandle`] — shared by both the stackful `spawn` path
//! ([`stackful::thread`](crate::resumable::stackful::thread)) and the
//! stackless `spawn_async` path
//! ([`stackless::thread`](crate::resumable::stackless::thread)): both
//! produce the same handle type, differing only in which flavor-specific
//! `impl` blocks their `S::Desc` bound lets them use (blocking `.join()`
//! needs `StackfulTaskDesc`; `.await` needs `AsyncTaskDesc`).

use std::any::Any;
use std::marker::PhantomData;

use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::common::desc::TaskDesc;
use crate::resumable::common::pool::free_desc;
use crate::resumable::common::worker::{UltWorker, Worker};

// Result stored directly on the child's stack, avoiding a Box for the success
// case.  The Err variant still boxes because that is what catch_unwind produces.
pub(crate) enum StackResult<T> {
    Ok(T),
    Err(Box<dyn Any + Send>),
}

#[inline]
pub(crate) fn align_down(addr: usize, align: usize) -> usize {
    addr & !(align - 1)
}

pub struct JoinHandle<S: SchedulerSystem, T> {
    pub(crate) desc: *mut S::Desc,
    pub(crate) result_ptr: *mut StackResult<T>,
    // Type-erased drop for the result slot; avoids a T: Send + 'static bound
    // on the Drop impl (Rust disallows extra bounds there).
    pub(crate) result_drop: unsafe fn(*mut ()),
    pub(crate) _marker: PhantomData<(S, T)>,
}

pub(crate) unsafe fn drop_stack_result<T>(ptr: *mut ()) {
    unsafe { std::ptr::drop_in_place(ptr as *mut StackResult<T>) };
}

unsafe impl<S: SchedulerSystem, T: Send> Send for JoinHandle<S, T> {}
// JoinHandle holds only raw pointers; it is safe to move at any time.
impl<S: SchedulerSystem, T> Unpin for JoinHandle<S, T> {}

impl<S: SchedulerSystem, T: Send + 'static> JoinHandle<S, T> {
    pub(crate) fn take_result(self, wk: &UltWorker<S>) -> Result<T, Box<dyn Any + Send>> {
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

    pub(crate) fn take_result_no_worker(self) -> Result<T, Box<dyn Any + Send>> {
        let desc = self.desc;
        let result_ptr = self.result_ptr;
        std::mem::forget(self);
        let sr = unsafe { result_ptr.read() };
        unsafe { free_desc(desc) };
        match sr {
            StackResult::Ok(val) => Ok(val),
            StackResult::Err(e) => Err(e),
        }
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
                None => unsafe { free_desc(desc) },
            }
        }
    }
}
