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

use std::alloc::Layout;
use std::cell::{Cell, UnsafeCell};
use std::marker::PhantomData;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ult::desc::TaskDescAlloc;
#[allow(unused_imports)]
use crate::ult::desc::TaskDesc; // supertrait methods (pool_next/alloc_wk) need this in scope
use crate::ult::stack::{HeapStack, StackAlloc};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Pluggable pool for descriptor allocation / deallocation.
///
/// The pool lives in the
/// [`Scheduler`](crate::ult::scheduler::Scheduler) and is shared across all
/// workers, hence the `Send + Sync` bounds.  Each worker identifies itself
/// via `wk_num` so implementations can maintain per-worker free lists without
/// locking the common case.
///
/// Generic over the descriptor type `D` (see
/// [`crate::SchedulerSystem::Desc`]); every concrete system today sets
/// `D = BasicTaskDesc`.
///
/// `alloc`'s `size` lets one pool serve requests of varying size (needed for
/// `spawn_async`, where the required storage depends on the concrete
/// `Future` type) without forcing every caller to pre-allocate the worst
/// case: a request that fits the pool's configured slot size is served from
/// the free list exactly as before; an oversized request bypasses the free
/// list entirely with a one-off allocation (see
/// [`TaskDesc::oversized`](crate::ult::desc::TaskDesc::oversized)), and
/// `dealloc` frees it directly instead of returning it to the pool. Fixed
/// stack-size ULT callers (`spawn`) simply pass the same size every time, so
/// this never affects them today — but it also means a future per-task
/// custom stack size for `spawn` needs no further interface change here.
pub trait DescPool<D: TaskDescAlloc>: Send + Sync + 'static {
    /// Create a pool for a scheduler with `num_workers` workers whose tasks
    /// need up to `stack_size` bytes in the common case.
    fn new_pool(num_workers: usize, stack_size: usize) -> Self;

    /// Allocate a task descriptor for worker `wk_num` with storage for at
    /// least `size` bytes.
    fn alloc(&self, wk_num: usize, has_handle: bool, size: usize) -> *mut D;

    /// Return a finished descriptor from worker `wk_num` back to the pool
    /// (or free it directly, if it was an oversized one-off allocation).
    ///
    /// # Safety
    /// No other references to `desc` may exist after this call.
    unsafe fn dealloc(&self, wk_num: usize, desc: *mut D);
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
pub struct SimplePool<D: TaskDescAlloc, A: StackAlloc = HeapStack, const CAP: usize = 256> {
    stack_size: usize,
    lists: Box<[UnsafeCell<Vec<*mut D>>]>,
    _alloc: PhantomData<A>,
}

// Each worker accesses only its own slot; Sync is sound by construction.
unsafe impl<D: TaskDescAlloc, A: StackAlloc, const CAP: usize> Send for SimplePool<D, A, CAP> {}
unsafe impl<D: TaskDescAlloc, A: StackAlloc, const CAP: usize> Sync for SimplePool<D, A, CAP> {}

impl<D: TaskDescAlloc, A: StackAlloc, const CAP: usize> DescPool<D> for SimplePool<D, A, CAP> {
    fn new_pool(num_workers: usize, stack_size: usize) -> Self {
        let lists = (0..num_workers)
            .map(|_| UnsafeCell::new(Vec::new()))
            .collect();
        SimplePool { stack_size, lists, _alloc: PhantomData }
    }

    fn alloc(&self, wk_num: usize, has_handle: bool, size: usize) -> *mut D {
        if size > self.stack_size {
            // Oversized: one-off allocation, bypasses the free list entirely.
            let desc = D::alloc(size, has_handle);
            unsafe { (*desc).oversized().set(true) };
            return desc;
        }

        let list = unsafe { &mut *self.lists[wk_num].get() };
        match list.pop() {
            Some(desc) => {
                unsafe { (*desc).reinit(has_handle) };
                desc
            }
            None => {
                let desc = D::alloc_with(A::alloc_stack(self.stack_size).into(), has_handle);
                unsafe { (*desc).alloc_wk().set(wk_num) };
                desc
            }
        }
    }

    unsafe fn dealloc(&self, wk_num: usize, desc: *mut D) {
        if unsafe { (*desc).oversized().get() } {
            unsafe { D::free(desc) };
            return;
        }

        let list = unsafe { &mut *self.lists[wk_num].get() };
        if list.len() < CAP {
            list.push(desc);
        } else {
            unsafe { D::free(desc) };
        }
    }
}

impl<D: TaskDescAlloc, A: StackAlloc, const CAP: usize> Drop for SimplePool<D, A, CAP> {
    fn drop(&mut self) {
        for cell in self.lists.iter() {
            for &desc in unsafe { &*cell.get() }.iter() {
                unsafe { D::free(desc) };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ReturnPool internals
// ---------------------------------------------------------------------------

/// Per-producer staging list: descriptors headed back to a specific home
/// worker, accumulated before a batch flush to avoid per-item lock overhead.
struct ProList<D> {
    first: *mut D,
    last: *mut D,
    num: usize,
}

impl<D> ProList<D> {
    const fn empty() -> Self {
        ProList { first: null_mut(), last: null_mut(), num: 0 }
    }
}

/// "Local" half of a worker pool entry: only touched by the owning worker.
///
/// Placed on its own cache line to avoid false sharing with the remote half.
#[repr(C, align(64))]
struct LocalHalf<D> {
    /// Head of the local free list.  Lock-free: only the owning worker reads
    /// or writes this.
    con_local: UnsafeCell<*mut D>,
}

/// "Remote" half of a worker pool entry: written by other workers.
///
/// Protected by `lock` and placed on its own cache line.
#[repr(C, align(64))]
struct RemoteHalf<D> {
    lock: AtomicBool,
    /// Head of the remote free list.  Other workers prepend full batches here
    /// under the spinlock; the owning worker drains it into `con_local` in a
    /// single bulk move.
    con_remote: UnsafeCell<*mut D>,
}

struct WorkerEntry<D> {
    local: LocalHalf<D>,
    remote: RemoteHalf<D>,
}

// Safety: LocalHalf is only touched by the owning worker; RemoteHalf is
// spinlock-protected.
unsafe impl<D> Send for WorkerEntry<D> {}
unsafe impl<D> Sync for WorkerEntry<D> {}

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
/// │   └── con_local: *mut D   — local free list
/// └── remote (cache line 1) — shared, spinlock-protected
///     ├── lock: AtomicBool
///     └── con_remote: *mut D  — remote mailbox
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
/// 3. If still empty, allocate fresh with `D::alloc_with`.
pub struct ReturnPool<D: TaskDescAlloc, A: StackAlloc = HeapStack, const THRESHOLD: usize = 16> {
    stack_size: usize,
    workers: Box<[WorkerEntry<D>]>,
    /// `pro_arrays[cur_wk]` is a `Vec<ProList<D>>` of length `num_workers`.
    /// `pro_arrays[cur_wk][alloc_wk]` holds staged descriptors to be returned
    /// to `alloc_wk`.  Only accessed by worker `cur_wk`.
    pro_arrays: Box<[UnsafeCell<Vec<ProList<D>>>]>,
    _alloc: PhantomData<A>,
}

// Safety: each worker accesses only its own slots in pro_arrays and
// local halves; remote halves are spinlock-protected.
unsafe impl<D: TaskDescAlloc, A: StackAlloc, const THRESHOLD: usize> Send for ReturnPool<D, A, THRESHOLD> {}
unsafe impl<D: TaskDescAlloc, A: StackAlloc, const THRESHOLD: usize> Sync for ReturnPool<D, A, THRESHOLD> {}

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

impl<D: TaskDescAlloc, A: StackAlloc, const THRESHOLD: usize> DescPool<D> for ReturnPool<D, A, THRESHOLD> {
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

    fn alloc(&self, wk_num: usize, has_handle: bool, size: usize) -> *mut D {
        if size > self.stack_size {
            // Oversized: one-off allocation, bypasses the free list entirely.
            let desc = D::alloc(size, has_handle);
            unsafe { (*desc).oversized().set(true) };
            return desc;
        }

        // Safety: `wk_num` is always a worker index for this same pool's
        // `Scheduler` (constructed with the same `num_workers`), so it is
        // always in range for `self.workers` — no need to pay a bounds
        // check on every alloc.
        let we = unsafe { self.workers.get_unchecked(wk_num) };
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
        let desc = D::alloc_with(A::alloc_stack(self.stack_size).into(), has_handle);
        unsafe { (*desc).alloc_wk().set(wk_num) };
        desc
    }

    unsafe fn dealloc(&self, cur_wk: usize, desc: *mut D) {
        if unsafe { (*desc).oversized().get() } {
            unsafe { D::free(desc) };
            return;
        }

        let alloc_wk = unsafe { (*desc).alloc_wk().get() };

        if alloc_wk == cur_wk {
            // Home worker: push directly to the lock-free local list.
            // Safety: same invariant as `alloc` above — `cur_wk` is always
            // in range for `self.workers`.
            let we = unsafe { self.workers.get_unchecked(cur_wk) };
            let con_local = unsafe { &mut *we.local.con_local.get() };
            unsafe { (*desc).pool_next().set(*con_local) };
            *con_local = desc;
            return;
        }

        // Non-home worker: stage in pro_arrays[cur_wk][alloc_wk].
        // Safety: `cur_wk` indexes `self.pro_arrays` (one entry per worker);
        // `alloc_wk` was recorded from a valid `wk_num` at allocation time
        // (see `alloc`'s miss path), so it indexes `pro_arr` (also
        // one entry per worker) — both always in range.
        let pro_arr = unsafe { &mut *self.pro_arrays.get_unchecked(cur_wk).get() };
        let pro = unsafe { pro_arr.get_unchecked_mut(alloc_wk) };
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
            // Safety: same invariant as above — `alloc_wk` is always in
            // range for `self.workers`.
            let alloc_we = unsafe { self.workers.get_unchecked(alloc_wk) };
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
unsafe fn free_list<D: TaskDescAlloc>(mut p: *mut D) {
    while !p.is_null() {
        let next = unsafe { (*p).pool_next().get() };
        unsafe { D::free(p) };
        p = next;
    }
}

impl<D: TaskDescAlloc, A: StackAlloc, const THRESHOLD: usize> Drop for ReturnPool<D, A, THRESHOLD> {
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

// ---------------------------------------------------------------------------
// StaticPool / DynamicPool
// ---------------------------------------------------------------------------

/// A pool where every allocation is the same, fixed size — decided once at
/// construction. No per-call size: `alloc`/`dealloc` only need `wk_num`.
///
/// This is the same fixed-slot free-list mechanism [`ReturnPool`] already
/// has for descriptors, generalized to raw bytes: home-worker tracking and
/// cross-worker return batching, with none of [`TaskDescAlloc`]'s
/// construction/reinit machinery. [`DescPool`]-based pools (`spawn`'s
/// fixed-`STACK_SIZE` case in particular) are conceptually `StaticPool`
/// users layered with task-specific construction; that layering is not
/// implemented yet — see [`DynamicPool`]'s doc comment for the piece that
/// is.
pub trait StaticPool: Sync + 'static {
    /// Create a pool for `num_workers` workers, each allocation sized
    /// (and aligned) per `layout`.
    fn new(num_workers: usize, layout: Layout) -> Self;

    /// Allocate (or reuse a freed block) for worker `wk_num`.
    fn alloc(&self, wk_num: usize) -> *mut u8;

    /// Return a block from worker `wk_num` to the pool.
    ///
    /// # Safety
    /// `ptr` must have come from [`alloc`](Self::alloc) on this same pool,
    /// and no other references to it may exist after this call.
    unsafe fn dealloc(&self, wk_num: usize, ptr: *mut u8);
}

/// A pool where allocation size varies per call, served from a
/// [`StaticPool`]-backed free list sized at construction (the `threshold`)
/// with a one-off allocation fallback for anything bigger — the same
/// oversized-request handling [`ReturnPool`]/[`DescPool::alloc`] already
/// have for `AsyncPool`, pulled out one layer so [`recurse`](crate::ult::thread::recurse)
/// can use it directly instead of duplicating it.
pub trait DynamicPool: Sync + 'static {
    /// Create a pool for `num_workers` workers whose common-case requests
    /// fit within `threshold` (size *and* align).
    fn new(num_workers: usize, threshold: Layout) -> Self;

    /// Allocate (or reuse a freed block for) `layout`, for worker `wk_num`.
    fn alloc(&self, wk_num: usize, layout: Layout) -> *mut u8;

    /// Return a block to the pool.
    ///
    /// # Safety
    /// `ptr` must have come from [`alloc`](Self::alloc) on this same pool
    /// with this same `layout`, and no other references to it may exist
    /// after this call.
    unsafe fn dealloc(&self, wk_num: usize, ptr: *mut u8, layout: Layout);
}

// ---------------------------------------------------------------------------
// BlockPool — the StaticPool implementation
// ---------------------------------------------------------------------------

/// Free-list node prepended (hidden) before every block [`BlockPool`] hands
/// out — the generic equivalent of `BasicTaskDesc`'s `pool_next`/`alloc_wk`
/// fields, for callers with no descriptor of their own to carry them on.
struct BlockHeader {
    next: Cell<*mut BlockHeader>,
    alloc_wk: Cell<usize>,
}

/// [`StaticPool`] backed by the exact free-list mechanism [`ReturnPool`]
/// uses for descriptors (`con_local`/`con_remote`/`pro_arrays`, batched
/// cross-worker returns under a spinlock) — reusing [`WorkerEntry`]/
/// [`ProList`] directly, just instantiated with [`BlockHeader`] instead of
/// a `D: TaskDescAlloc`. Each returned block is `header_layout.extend(payload_layout)`
/// bytes; the header stays hidden before the pointer callers see.
pub struct BlockPool<const THRESHOLD: usize = 16> {
    /// Combined `(BlockHeader, payload)` layout — what's actually allocated
    /// for a fresh block.
    block_layout: Layout,
    /// Byte offset from the block's start to the payload (>= `size_of::<BlockHeader>()`,
    /// rounded up for `payload_layout`'s own alignment).
    payload_offset: usize,
    workers: Box<[WorkerEntry<BlockHeader>]>,
    pro_arrays: Box<[UnsafeCell<Vec<ProList<BlockHeader>>>]>,
}

unsafe impl<const THRESHOLD: usize> Send for BlockPool<THRESHOLD> {}
unsafe impl<const THRESHOLD: usize> Sync for BlockPool<THRESHOLD> {}

impl<const THRESHOLD: usize> BlockPool<THRESHOLD> {
    #[inline]
    fn header_of(&self, payload: *mut u8) -> *mut BlockHeader {
        unsafe { payload.sub(self.payload_offset) as *mut BlockHeader }
    }

    #[inline]
    fn payload_of(&self, header: *mut BlockHeader) -> *mut u8 {
        unsafe { (header as *mut u8).add(self.payload_offset) }
    }

    fn alloc_fresh(&self, wk_num: usize) -> *mut BlockHeader {
        let raw = unsafe { std::alloc::alloc(self.block_layout) } as *mut BlockHeader;
        if raw.is_null() {
            std::alloc::handle_alloc_error(self.block_layout);
        }
        unsafe {
            (*raw).next.set(null_mut());
            (*raw).alloc_wk.set(wk_num);
        }
        raw
    }
}

impl<const THRESHOLD: usize> StaticPool for BlockPool<THRESHOLD> {
    fn new(num_workers: usize, payload_layout: Layout) -> Self {
        assert!(THRESHOLD >= 1, "BlockPool THRESHOLD must be >= 1");
        let header_layout = Layout::new::<BlockHeader>();
        let (block_layout, payload_offset) = header_layout
            .extend(payload_layout)
            .expect("cmpth: BlockPool payload layout overflow");
        let block_layout = block_layout.pad_to_align();

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
            .map(|_| UnsafeCell::new((0..num_workers).map(|_| ProList::empty()).collect::<Vec<_>>()))
            .collect();

        BlockPool { block_layout, payload_offset, workers, pro_arrays }
    }

    #[inline]
    fn alloc(&self, wk_num: usize) -> *mut u8 {
        let we = unsafe { self.workers.get_unchecked(wk_num) };
        let con_local = unsafe { &mut *we.local.con_local.get() };

        if !con_local.is_null() {
            let header = *con_local;
            *con_local = unsafe { (*header).next.get() };
            return self.payload_of(header);
        }

        let head = {
            spin_lock(&we.remote.lock);
            let cr = unsafe { &mut *we.remote.con_remote.get() };
            let h = *cr;
            *cr = null_mut();
            spin_unlock(&we.remote.lock);
            h
        };
        if !head.is_null() {
            *con_local = unsafe { (*head).next.get() };
            return self.payload_of(head);
        }

        self.payload_of(self.alloc_fresh(wk_num))
    }

    #[inline]
    unsafe fn dealloc(&self, cur_wk: usize, ptr: *mut u8) {
        let header = self.header_of(ptr);
        let alloc_wk = unsafe { (*header).alloc_wk.get() };

        if alloc_wk == cur_wk {
            let we = unsafe { self.workers.get_unchecked(cur_wk) };
            let con_local = unsafe { &mut *we.local.con_local.get() };
            unsafe { (*header).next.set(*con_local) };
            *con_local = header;
            return;
        }

        let pro_arr = unsafe { &mut *self.pro_arrays.get_unchecked(cur_wk).get() };
        let pro = unsafe { pro_arr.get_unchecked_mut(alloc_wk) };
        let old_num = pro.num;

        if old_num < THRESHOLD {
            unsafe { (*header).next.set(pro.first) };
            if old_num == 0 {
                pro.last = header;
            }
            pro.first = header;
            pro.num = old_num + 1;
        } else {
            let alloc_we = unsafe { self.workers.get_unchecked(alloc_wk) };
            let old_first = pro.first;
            let old_last = pro.last;

            spin_lock(&alloc_we.remote.lock);
            let cr = unsafe { &mut *alloc_we.remote.con_remote.get() };
            unsafe { (*old_last).next.set(*cr) };
            *cr = old_first;
            spin_unlock(&alloc_we.remote.lock);

            unsafe { (*header).next.set(null_mut()) };
            pro.first = header;
            pro.last = header;
            pro.num = 1;
        }
    }
}

impl<const THRESHOLD: usize> Drop for BlockPool<THRESHOLD> {
    fn drop(&mut self) {
        let free_chain = |mut p: *mut BlockHeader| {
            while !p.is_null() {
                let next = unsafe { (*p).next.get() };
                unsafe { std::alloc::dealloc(p as *mut u8, self.block_layout) };
                p = next;
            }
        };
        for (wk_num, we) in self.workers.iter().enumerate() {
            free_chain(unsafe { *we.local.con_local.get() });
            free_chain(unsafe { *we.remote.con_remote.get() });
            for pro in unsafe { &*self.pro_arrays[wk_num].get() } {
                free_chain(pro.first);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ThresholdPool — the DynamicPool implementation
// ---------------------------------------------------------------------------

/// [`DynamicPool`] adapter: wraps any [`StaticPool`], serving requests that
/// fit `threshold` from it and falling back to a one-off allocation for
/// anything bigger — the same split [`ReturnPool::alloc`] already makes
/// for oversized `spawn_async` futures, generalized to any `P`.
pub struct ThresholdPool<P: StaticPool> {
    threshold: Layout,
    inner: P,
}

impl<P: StaticPool> DynamicPool for ThresholdPool<P> {
    fn new(num_workers: usize, threshold: Layout) -> Self {
        ThresholdPool { threshold, inner: P::new(num_workers, threshold) }
    }

    #[inline]
    fn alloc(&self, wk_num: usize, layout: Layout) -> *mut u8 {
        if layout.size() <= self.threshold.size() && layout.align() <= self.threshold.align() {
            self.inner.alloc(wk_num)
        } else {
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            ptr
        }
    }

    #[inline]
    unsafe fn dealloc(&self, wk_num: usize, ptr: *mut u8, layout: Layout) {
        if layout.size() <= self.threshold.size() && layout.align() <= self.threshold.align() {
            unsafe { self.inner.dealloc(wk_num, ptr) };
        } else {
            unsafe { std::alloc::dealloc(ptr, layout) };
        }
    }
}
