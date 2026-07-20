//! Pluggable task-descriptor pool.
//!
//! The [`DescPool`] trait makes the pooling strategy a per-`UltSystem`
//! configuration axis.  Two implementations are provided:
//!
//! * [`SimplePool`] — per-worker free list with no cross-worker return.  The
//!   lowest-overhead option, but pool lists can become imbalanced when one
//!   worker creates tasks and another finishes them.
//! * [`ReturnPool`] — returns each descriptor to the worker that allocated
//!   it, batching cross-worker returns to amortise the spinlock cost.  Based
//!   on the `basic_return_pool` design in the C++ reference implementation.

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ult::desc::{BasicTaskDesc, TaskDesc};
use crate::ult::stack::{HeapStack, StackAlloc};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Pluggable pool for [`BasicTaskDesc`] allocation / deallocation.
///
/// The pool lives in the
/// [`Scheduler`](crate::ult::scheduler::Scheduler) and is shared across all
/// workers, hence the `Send + Sync` bounds.  Each worker identifies itself
/// via `wk_num` so implementations can maintain per-worker free lists without
/// locking the common case.
pub trait DescPool: Send + Sync + 'static {
    /// Create a pool for a scheduler with `num_workers` workers whose tasks
    /// need stacks of `stack_size` bytes.
    fn new_pool(num_workers: usize, stack_size: usize) -> Self;

    /// Allocate a task descriptor for worker `wk_num`.
    fn alloc(&self, wk_num: usize, has_handle: bool) -> *mut BasicTaskDesc;

    /// Return a finished descriptor from worker `wk_num` back to the pool.
    ///
    /// # Safety
    /// No other references to `desc` may exist after this call.
    unsafe fn dealloc(&self, wk_num: usize, desc: *mut BasicTaskDesc);
}

// ---------------------------------------------------------------------------
// SimplePool
// ---------------------------------------------------------------------------

/// Per-worker free list with no cross-worker return.
///
/// [`dealloc`](DescPool::dealloc) always pushes to the **current** worker's
/// list regardless of which worker originally allocated the descriptor.  This
/// is the simplest and lowest-overhead implementation, but it can cause list
/// imbalance under fork-heavy / join-heavy workloads where one worker spawns
/// tasks and others finish them.  Use [`ReturnPool`] to eliminate that
/// imbalance.
///
/// `CAP` is the maximum number of descriptors retained per worker; excess
/// descriptors are freed immediately.
pub struct SimplePool<A: StackAlloc = HeapStack, const CAP: usize = 256> {
    stack_size: usize,
    lists: Box<[UnsafeCell<Vec<*mut BasicTaskDesc>>]>,
    _alloc: PhantomData<A>,
}

// Each worker accesses only its own slot; Sync is sound by construction.
unsafe impl<A: StackAlloc, const CAP: usize> Send for SimplePool<A, CAP> {}
unsafe impl<A: StackAlloc, const CAP: usize> Sync for SimplePool<A, CAP> {}

impl<A: StackAlloc, const CAP: usize> DescPool for SimplePool<A, CAP> {
    fn new_pool(num_workers: usize, stack_size: usize) -> Self {
        let lists = (0..num_workers)
            .map(|_| UnsafeCell::new(Vec::new()))
            .collect();
        SimplePool { stack_size, lists, _alloc: PhantomData }
    }

    fn alloc(&self, wk_num: usize, has_handle: bool) -> *mut BasicTaskDesc {
        let list = unsafe { &mut *self.lists[wk_num].get() };
        match list.pop() {
            Some(desc) => {
                unsafe { (*desc).reinit(has_handle) };
                desc
            }
            None => {
                let desc = BasicTaskDesc::alloc_with(A::alloc_stack(self.stack_size).into(), has_handle);
                unsafe { (*desc).alloc_wk().set(wk_num) };
                desc
            }
        }
    }

    unsafe fn dealloc(&self, wk_num: usize, desc: *mut BasicTaskDesc) {
        let list = unsafe { &mut *self.lists[wk_num].get() };
        if list.len() < CAP {
            list.push(desc);
        } else {
            unsafe { BasicTaskDesc::free(desc) };
        }
    }
}

impl<A: StackAlloc, const CAP: usize> Drop for SimplePool<A, CAP> {
    fn drop(&mut self) {
        for cell in self.lists.iter() {
            for &desc in unsafe { &*cell.get() }.iter() {
                unsafe { BasicTaskDesc::free(desc) };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ReturnPool internals
// ---------------------------------------------------------------------------

/// Per-producer staging list: descriptors headed back to a specific home
/// worker, accumulated before a batch flush to avoid per-item lock overhead.
struct ProList {
    first: *mut BasicTaskDesc,
    last: *mut BasicTaskDesc,
    num: usize,
}

impl ProList {
    const fn empty() -> Self {
        ProList { first: null_mut(), last: null_mut(), num: 0 }
    }
}

/// "Local" half of a worker pool entry: only touched by the owning worker.
///
/// Placed on its own cache line to avoid false sharing with the remote half.
#[repr(C, align(64))]
struct LocalHalf {
    /// Head of the local free list.  Lock-free: only the owning worker reads
    /// or writes this.
    con_local: UnsafeCell<*mut BasicTaskDesc>,
}

/// "Remote" half of a worker pool entry: written by other workers.
///
/// Protected by `lock` and placed on its own cache line.
#[repr(C, align(64))]
struct RemoteHalf {
    lock: AtomicBool,
    /// Head of the remote free list.  Other workers prepend full batches here
    /// under the spinlock; the owning worker drains it into `con_local` in a
    /// single bulk move.
    con_remote: UnsafeCell<*mut BasicTaskDesc>,
}

struct WorkerEntry {
    local: LocalHalf,
    remote: RemoteHalf,
}

// Safety: LocalHalf is only touched by the owning worker; RemoteHalf is
// spinlock-protected.
unsafe impl Send for WorkerEntry {}
unsafe impl Sync for WorkerEntry {}

// ---------------------------------------------------------------------------
// ReturnPool
// ---------------------------------------------------------------------------

/// Pool that returns each descriptor to the worker that originally allocated
/// it, based on the `basic_return_pool` design from the C++ reference
/// implementation.
///
/// # Memory layout per worker
///
/// ```text
/// WorkerEntry
/// ├── local (cache line 0) — owned by this worker, no sync
/// │   └── con_local: *mut BasicTaskDesc   — local free list
/// └── remote (cache line 1) — shared, spinlock-protected
///     ├── lock: AtomicBool
///     └── con_remote: *mut BasicTaskDesc  — remote mailbox
///
/// pro_arrays[cur_wk][alloc_wk]      — staging lists, owned by cur_wk
/// ```
///
/// # Deallocation
///
/// * Same worker (`alloc_wk == cur_wk`): push to `con_local` directly.
/// * Different worker: stage in `pro_arrays[cur_wk][alloc_wk]`.  When the
///   staging list reaches `THRESHOLD` items, prepend the whole batch to
///   `alloc_wk`'s `con_remote` under one spinlock acquisition.
///
/// # Allocation
///
/// 1. Pop from `con_local` (no lock — fast path).
/// 2. If empty, drain `con_remote` into `con_local` under one spinlock, then
///    pop from the newly-filled local list.
/// 3. If still empty, allocate fresh with `BasicTaskDesc::alloc`.
pub struct ReturnPool<A: StackAlloc = HeapStack, const THRESHOLD: usize = 16> {
    stack_size: usize,
    workers: Box<[WorkerEntry]>,
    /// `pro_arrays[cur_wk]` is a `Vec<ProList>` of length `num_workers`.
    /// `pro_arrays[cur_wk][alloc_wk]` holds staged descriptors to be returned
    /// to `alloc_wk`.  Only accessed by worker `cur_wk`.
    pro_arrays: Box<[UnsafeCell<Vec<ProList>>]>,
    _alloc: PhantomData<A>,
}

// Safety: each worker accesses only its own slots in pro_arrays and
// local halves; remote halves are spinlock-protected.
unsafe impl<A: StackAlloc, const THRESHOLD: usize> Send for ReturnPool<A, THRESHOLD> {}
unsafe impl<A: StackAlloc, const THRESHOLD: usize> Sync for ReturnPool<A, THRESHOLD> {}

#[inline]
fn spin_lock(b: &AtomicBool) {
    loop {
        if b.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
            return;
        }
        while b.load(Ordering::Relaxed) {
            std::hint::spin_loop();
        }
    }
}

#[inline]
fn spin_unlock(b: &AtomicBool) {
    b.store(false, Ordering::Release);
}

impl<A: StackAlloc, const THRESHOLD: usize> DescPool for ReturnPool<A, THRESHOLD> {
    fn new_pool(num_workers: usize, stack_size: usize) -> Self {
        assert!(THRESHOLD >= 1, "ReturnPool THRESHOLD must be >= 1");
        let workers = (0..num_workers)
            .map(|_| WorkerEntry {
                local: LocalHalf { con_local: UnsafeCell::new(null_mut()) },
                remote: RemoteHalf {
                    lock: AtomicBool::new(false),
                    con_remote: UnsafeCell::new(null_mut()),
                },
            })
            .collect();
        let pro_arrays = (0..num_workers)
            .map(|_| UnsafeCell::new(
                (0..num_workers).map(|_| ProList::empty()).collect::<Vec<_>>()
            ))
            .collect();
        ReturnPool { stack_size, workers, pro_arrays, _alloc: PhantomData }
    }

    fn alloc(&self, wk_num: usize, has_handle: bool) -> *mut BasicTaskDesc {
        let we = &self.workers[wk_num];
        let con_local = unsafe { &mut *we.local.con_local.get() };

        // Fast path: take from lock-free local list.
        if !con_local.is_null() {
            let desc = *con_local;
            *con_local = unsafe { (*desc).pool_next().get() };
            unsafe { (*desc).reinit(has_handle) };
            return desc;
        }

        // Slow path: drain remote mailbox into local list (one lock acquisition).
        let head = {
            spin_lock(&we.remote.lock);
            let cr = unsafe { &mut *we.remote.con_remote.get() };
            let h = *cr;
            *cr = null_mut();
            spin_unlock(&we.remote.lock);
            h
        };
        if !head.is_null() {
            // Bulk move: return head, the rest become the new local list.
            *con_local = unsafe { (*head).pool_next().get() };
            unsafe { (*head).reinit(has_handle) };
            return head;
        }

        // Miss: allocate fresh and record the home worker.
        let desc = BasicTaskDesc::alloc_with(A::alloc_stack(self.stack_size).into(), has_handle);
        unsafe { (*desc).alloc_wk().set(wk_num) };
        desc
    }

    unsafe fn dealloc(&self, cur_wk: usize, desc: *mut BasicTaskDesc) {
        let alloc_wk = unsafe { (*desc).alloc_wk().get() };

        if alloc_wk == cur_wk {
            // Home worker: push directly to the lock-free local list.
            let we = &self.workers[cur_wk];
            let con_local = unsafe { &mut *we.local.con_local.get() };
            unsafe { (*desc).pool_next().set(*con_local) };
            *con_local = desc;
            return;
        }

        // Non-home worker: stage in pro_arrays[cur_wk][alloc_wk].
        let pro_arr = unsafe { &mut *self.pro_arrays[cur_wk].get() };
        let pro = &mut pro_arr[alloc_wk];
        let old_num = pro.num;

        if old_num < THRESHOLD {
            // Accumulate: prepend desc to the staging list.
            unsafe { (*desc).pool_next().set(pro.first) };
            if old_num == 0 {
                pro.last = desc; // first item also becomes the tail
            }
            pro.first = desc;
            pro.num = old_num + 1;
        } else {
            // Batch full: flush the old batch to the home worker's mailbox,
            // then start a fresh batch containing only the current descriptor.
            let alloc_we = &self.workers[alloc_wk];
            let old_first = pro.first;
            let old_last = pro.last;

            spin_lock(&alloc_we.remote.lock);
            let cr = unsafe { &mut *alloc_we.remote.con_remote.get() };
            // Prepend batch: batch_tail → old head of con_remote.
            unsafe { (*old_last).pool_next().set(*cr) };
            *cr = old_first;
            spin_unlock(&alloc_we.remote.lock);

            // Start fresh batch with only desc.
            unsafe { (*desc).pool_next().set(null_mut()) };
            pro.first = desc;
            pro.last = desc;
            pro.num = 1;
        }
    }
}

/// Free every descriptor in a `pool_next`-threaded linked list.
unsafe fn free_list(mut p: *mut BasicTaskDesc) {
    while !p.is_null() {
        let next = unsafe { (*p).pool_next().get() };
        unsafe { BasicTaskDesc::free(p) };
        p = next;
    }
}

impl<A: StackAlloc, const THRESHOLD: usize> Drop for ReturnPool<A, THRESHOLD> {
    fn drop(&mut self) {
        for (wk_num, we) in self.workers.iter().enumerate() {
            unsafe { free_list(*we.local.con_local.get()) };
            unsafe { free_list(*we.remote.con_remote.get()) };
            for pro in unsafe { &*self.pro_arrays[wk_num].get() } {
                unsafe { free_list(pro.first) };
            }
        }
    }
}
