//! Pluggable task-descriptor pool.
//!
//! The [`DescPool`] trait makes the pooling strategy a per-`ThreadSystem`
//! configuration axis.  Two implementations are provided:
//!
//! * [`SimplePool`] — per-worker free list with no cross-worker return.  The
//!   lowest-overhead option, but pool lists can become imbalanced when one
//!   worker creates tasks and another finishes them.
//! * [`ReturnPool`] — returns each descriptor to the worker that allocated
//!   it, batching cross-worker returns to amortise the spinlock cost.  Based
//!   on the `basic_return_pool` design in the C++ reference implementation.
//!
//! Both are thin typed wrappers around the same free-list core as
//! [`BlockPool`] (see `PoolNode`/`node_take`/`node_give`): the pool
//! linkage (`next`/`alloc_wk`/`oversized`) is not a `TaskDesc` field —
//! it lives in a `Node<D>` the pool prepends around the descriptor, the
//! same way `BlockPool` prepends a `BlockHeader` around its type-erased
//! payloads. Descriptor pools know `D` at compile time, so the node<->payload
//! conversion (`Node::node_of`/`Node::payload_of`) is an ordinary struct
//! field access, resolved via `std::mem::offset_of!` — no runtime offset
//! like `BlockPool`'s `payload_offset` needed, since that only exists to
//! support genuinely type-erased payloads (see [`DynamicPool`]'s doc
//! comment for why `recurse` alone needs that).

use std::alloc::Layout;
use std::cell::{Cell, UnsafeCell};
use std::marker::PhantomData;
use std::mem::offset_of;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::resumable::common::desc::TaskDescAlloc;
use crate::resumable::common::stack::{HeapStack, StackAlloc};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Pluggable pool for descriptor allocation / deallocation.
///
/// The pool lives in the
/// [`Scheduler`](crate::resumable::common::scheduler::Scheduler) and is shared across all
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
/// list entirely with a one-off allocation (tracked by the pool's own
/// `Node` wrapper, not by the descriptor), and `dealloc` frees it directly
/// instead of returning it to the pool. Fixed stack-size ULT callers
/// (`spawn`) simply pass the same size every time, so this never affects
/// them today — but it also means a future per-task custom stack size for
/// `spawn` needs no further interface change here.
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
// PoolNode / Node<D> — shared free-list node interface
// ---------------------------------------------------------------------------

/// A type usable as a node in the free-list machinery shared by every pool
/// in this module: an intrusive `next` link plus a home-worker index. The
/// bookkeeping functions below (`node_take`/`node_give`) only ever touch
/// these two fields — they never look at whatever payload sits behind the
/// node — so the same implementation serves both `Node<D>` (a real,
/// compile-time-known `D` behind it) and `BlockHeader` (opaque
/// type-erased bytes behind it).
trait PoolNode: Sized {
    fn next(&self) -> &Cell<*mut Self>;
    fn alloc_wk(&self) -> &Cell<usize>;
}

/// Free-list node wrapping a compile-time-known payload `D`. The pool-only
/// counterpart to `TaskDesc`'s old `pool_next`/`alloc_wk`/`oversized`
/// fields: descriptors no longer carry pool bookkeeping themselves, pools
/// prepend it via this wrapper instead — the same idea as `BlockHeader`,
/// just with a real field instead of a runtime-offset payload, since `D` is
/// known here.
pub(crate) struct Node<D> {
    next: Cell<*mut Node<D>>,
    alloc_wk: Cell<usize>,
    oversized: Cell<bool>,
    payload: D,
}

impl<D> PoolNode for Node<D> {
    fn next(&self) -> &Cell<*mut Self> { &self.next }
    fn alloc_wk(&self) -> &Cell<usize> { &self.alloc_wk }
}

impl<D> Node<D> {
    /// Recover the owning node from a payload pointer handed back by
    /// `Node::payload_of`. Compile-time-constant offset (`D` is known),
    /// unlike `BlockPool::header_of`'s runtime `payload_offset`.
    #[inline]
    fn node_of(payload: *mut D) -> *mut Node<D> {
        unsafe { (payload as *mut u8).sub(offset_of!(Node<D>, payload)) as *mut Node<D> }
    }

    #[inline]
    fn payload_of(node: *mut Node<D>) -> *mut D {
        unsafe { &raw mut (*node).payload }
    }

    /// Box a fresh node around an already-constructed `payload` (from
    /// `D::alloc_with`/`D::alloc`), not yet reachable from any pool's free
    /// list, and return the payload pointer callers see. `pub(crate)`: also
    /// used directly by [`crate::resumable::stackless::thread::fork_async_parent_first`]
    /// for the one-off root async descriptor, allocated before any worker
    /// (hence any pool) exists, but still dealloc'd through the pool later
    /// like any other completed async task — see that function's own doc
    /// comment for why it needs `oversized = true` from birth.
    pub(crate) fn wrap_fresh(alloc_wk: usize, oversized: bool, payload: D) -> *mut D {
        let node = Box::into_raw(Box::new(Node {
            next: Cell::new(null_mut()),
            alloc_wk: Cell::new(alloc_wk),
            oversized: Cell::new(oversized),
            payload,
        }));
        Self::payload_of(node)
    }
}

/// Free a descriptor pointer that was handed out by some pool's `alloc`
/// (i.e. wrapped via `Node::wrap_fresh` at some point) — drops the whole
/// `Node<D>`, not just the payload. `pub(crate)`: also used directly by
/// [`crate::resumable::common::thread::JoinHandle`]'s no-worker fallback
/// paths, which need to free a descriptor without going through any
/// specific pool instance.
///
/// # Safety
/// `payload` must have come from a `Node<D>`-wrapping pool, and no other
/// references to it may exist after this call.
pub(crate) unsafe fn free_desc<D>(payload: *mut D) {
    unsafe { drop(Box::from_raw(Node::node_of(payload))) };
}

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

/// Try to take a node from worker `wk_num`'s free list: local fast path
/// (lock-free), else drain the remote mailbox under one lock acquisition.
/// `None` means the free list is empty — the caller must allocate fresh.
#[inline]
fn node_take<N: PoolNode>(workers: &[WorkerEntry<N>], wk_num: usize) -> Option<*mut N> {
    // Safety: `wk_num` is always a worker index for this same pool's
    // `Scheduler` (constructed with the same `num_workers`), so it is
    // always in range — no need to pay a bounds check on every call.
    let we = unsafe { workers.get_unchecked(wk_num) };
    let con_local = unsafe { &mut *we.local.con_local.get() };

    if !con_local.is_null() {
        let node = *con_local;
        *con_local = unsafe { (*node).next().get() };
        return Some(node);
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
        *con_local = unsafe { (*head).next().get() };
        return Some(head);
    }

    None
}

/// Return `node` (originally allocated by worker `(*node).alloc_wk()`) to
/// the pool from `cur_wk`: push directly to `cur_wk`'s own local list if
/// it's the home worker, otherwise stage in `pro_arrays[cur_wk][alloc_wk]`,
/// batch-flushing to the home worker's remote mailbox under one lock
/// acquisition once the staging list reaches `threshold`.
///
/// # Safety
/// `node` must not be referenced again after this call.
#[inline]
unsafe fn node_give<N: PoolNode>(
    workers: &[WorkerEntry<N>],
    pro_arrays: &[UnsafeCell<Vec<ProList<N>>>],
    cur_wk: usize,
    node: *mut N,
    threshold: usize,
) {
    let alloc_wk = unsafe { (*node).alloc_wk().get() };

    if alloc_wk == cur_wk {
        // Home worker: push directly to the lock-free local list.
        let we = unsafe { workers.get_unchecked(cur_wk) };
        let con_local = unsafe { &mut *we.local.con_local.get() };
        unsafe { (*node).next().set(*con_local) };
        *con_local = node;
        return;
    }

    // Non-home worker: stage in pro_arrays[cur_wk][alloc_wk].
    let pro_arr = unsafe { &mut *pro_arrays.get_unchecked(cur_wk).get() };
    let pro = unsafe { pro_arr.get_unchecked_mut(alloc_wk) };
    let old_num = pro.num;

    if old_num < threshold {
        // Accumulate: prepend node to the staging list.
        unsafe { (*node).next().set(pro.first) };
        if old_num == 0 {
            pro.last = node; // first item also becomes the tail
        }
        pro.first = node;
        pro.num = old_num + 1;
    } else {
        // Batch full: flush the old batch to the home worker's mailbox,
        // then start a fresh batch containing only the current node.
        let alloc_we = unsafe { workers.get_unchecked(alloc_wk) };
        let old_first = pro.first;
        let old_last = pro.last;

        spin_lock(&alloc_we.remote.lock);
        let cr = unsafe { &mut *alloc_we.remote.con_remote.get() };
        unsafe { (*old_last).next().set(*cr) };
        *cr = old_first;
        spin_unlock(&alloc_we.remote.lock);

        unsafe { (*node).next().set(null_mut()) };
        pro.first = node;
        pro.last = node;
        pro.num = 1;
    }
}

/// Construct the `workers`/`pro_arrays` pair every `WorkerEntry`-based pool
/// needs, shared by [`ReturnPool`]/[`SimplePool`]/[`BlockPool`].
fn new_workers_and_pro_arrays<N>(
    num_workers: usize,
) -> (Box<[WorkerEntry<N>]>, Box<[UnsafeCell<Vec<ProList<N>>>]>) {
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
    (workers, pro_arrays)
}

/// Walk every node reachable from a pool's lists at drop time, calling
/// `free_one` on each.
fn drop_all_nodes<N: PoolNode>(
    workers: &[WorkerEntry<N>],
    pro_arrays: &[UnsafeCell<Vec<ProList<N>>>],
    mut free_one: impl FnMut(*mut N),
) {
    fn free_chain<N: PoolNode>(mut p: *mut N, free_one: &mut impl FnMut(*mut N)) {
        while !p.is_null() {
            let next = unsafe { (*p).next().get() };
            free_one(p);
            p = next;
        }
    }
    for (wk_num, we) in workers.iter().enumerate() {
        free_chain(unsafe { *we.local.con_local.get() }, &mut free_one);
        free_chain(unsafe { *we.remote.con_remote.get() }, &mut free_one);
        for pro in unsafe { &*pro_arrays[wk_num].get() } {
            free_chain(pro.first, &mut free_one);
        }
    }
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
    lists: Box<[UnsafeCell<Vec<*mut Node<D>>>]>,
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
            let payload = D::alloc(size, has_handle);
            return Node::wrap_fresh(wk_num, true, payload);
        }

        let list = unsafe { &mut *self.lists[wk_num].get() };
        match list.pop() {
            Some(node) => {
                unsafe { (*node).payload.reinit(has_handle) };
                Node::payload_of(node)
            }
            None => {
                let payload = D::alloc_with(A::alloc_stack(self.stack_size).into(), has_handle);
                Node::wrap_fresh(wk_num, false, payload)
            }
        }
    }

    unsafe fn dealloc(&self, wk_num: usize, desc: *mut D) {
        let node = Node::node_of(desc);
        if unsafe { (*node).oversized.get() } {
            unsafe { free_desc(desc) };
            return;
        }

        let list = unsafe { &mut *self.lists[wk_num].get() };
        if list.len() < CAP {
            list.push(node);
        } else {
            unsafe { drop(Box::from_raw(node)) };
        }
    }
}

impl<D: TaskDescAlloc, A: StackAlloc, const CAP: usize> Drop for SimplePool<D, A, CAP> {
    fn drop(&mut self) {
        for cell in self.lists.iter() {
            for &node in unsafe { &*cell.get() }.iter() {
                unsafe { drop(Box::from_raw(node)) };
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
/// │   └── con_local: *mut Node<D>   — local free list
/// └── remote (cache line 1) — shared, spinlock-protected
///     ├── lock: AtomicBool
///     └── con_remote: *mut Node<D>  — remote mailbox
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
    workers: Box<[WorkerEntry<Node<D>>]>,
    /// `pro_arrays[cur_wk]` is a `Vec<ProList<Node<D>>>` of length
    /// `num_workers`. `pro_arrays[cur_wk][alloc_wk]` holds staged nodes to
    /// be returned to `alloc_wk`.  Only accessed by worker `cur_wk`.
    pro_arrays: Box<[UnsafeCell<Vec<ProList<Node<D>>>>]>,
    _alloc: PhantomData<A>,
}

// Safety: each worker accesses only its own slots in pro_arrays and
// local halves; remote halves are spinlock-protected.
unsafe impl<D: TaskDescAlloc, A: StackAlloc, const THRESHOLD: usize> Send for ReturnPool<D, A, THRESHOLD> {}
unsafe impl<D: TaskDescAlloc, A: StackAlloc, const THRESHOLD: usize> Sync for ReturnPool<D, A, THRESHOLD> {}

impl<D: TaskDescAlloc, A: StackAlloc, const THRESHOLD: usize> DescPool<D> for ReturnPool<D, A, THRESHOLD> {
    fn new_pool(num_workers: usize, stack_size: usize) -> Self {
        assert!(THRESHOLD >= 1, "ReturnPool THRESHOLD must be >= 1");
        let (workers, pro_arrays) = new_workers_and_pro_arrays(num_workers);
        ReturnPool { stack_size, workers, pro_arrays, _alloc: PhantomData }
    }

    fn alloc(&self, wk_num: usize, has_handle: bool, size: usize) -> *mut D {
        if size > self.stack_size {
            // Oversized: one-off allocation, bypasses the free list entirely.
            let payload = D::alloc(size, has_handle);
            return Node::wrap_fresh(wk_num, true, payload);
        }

        match node_take(&self.workers, wk_num) {
            Some(node) => {
                unsafe { (*node).payload.reinit(has_handle) };
                Node::payload_of(node)
            }
            None => {
                // Miss: allocate fresh and record the home worker.
                let payload = D::alloc_with(A::alloc_stack(self.stack_size).into(), has_handle);
                Node::wrap_fresh(wk_num, false, payload)
            }
        }
    }

    unsafe fn dealloc(&self, cur_wk: usize, desc: *mut D) {
        let node = Node::node_of(desc);
        if unsafe { (*node).oversized.get() } {
            unsafe { free_desc(desc) };
            return;
        }
        unsafe { node_give(&self.workers, &self.pro_arrays, cur_wk, node, THRESHOLD) };
    }
}

impl<D: TaskDescAlloc, A: StackAlloc, const THRESHOLD: usize> Drop for ReturnPool<D, A, THRESHOLD> {
    fn drop(&mut self) {
        drop_all_nodes(&self.workers, &self.pro_arrays, |node| unsafe {
            drop(Box::from_raw(node));
        });
    }
}

// ---------------------------------------------------------------------------
// StaticPool / DynamicPool
// ---------------------------------------------------------------------------

/// A pool where every allocation is the same, fixed size — decided once at
/// construction. No per-call size: `alloc`/`dealloc` only need `wk_num`.
///
/// This is the same fixed-slot free-list mechanism [`ReturnPool`] uses for
/// descriptors (both are thin wrappers around `node_take`/`node_give`),
/// generalized to raw, type-erased bytes for callers with no concrete `D`
/// known at compile time — see [`DynamicPool`]'s doc comment for why
/// `recurse` specifically needs that erasure.
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
/// have for `AsyncPool`, pulled out one layer so [`recurse`](crate::resumable::stackless::thread::recurse)
/// can use it directly instead of duplicating it.
///
/// This erasure is not optional here the way it was for `ReturnPool`:
/// `RecursionPool` is a single associated type on `SchedulerSystem`, chosen
/// once per `S`, long before every `F` any `recurse::<S, F, _>` call
/// anywhere in the program might ever use is known. One shared pool
/// instance has to serve arbitrarily many distinct `F` types through the
/// program's lifetime, so its payload can only be described by a `Layout`,
/// not a concrete type — unlike `ReturnPool<D>`, where `D` is fixed once as
/// `SchedulerSystem::Desc`.
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
/// out — the type-erased counterpart to `Node<D>`, for payloads with no
/// compile-time-known type to carry `next`/`alloc_wk` as real fields.
struct BlockHeader {
    next: Cell<*mut BlockHeader>,
    alloc_wk: Cell<usize>,
}

impl PoolNode for BlockHeader {
    fn next(&self) -> &Cell<*mut Self> { &self.next }
    fn alloc_wk(&self) -> &Cell<usize> { &self.alloc_wk }
}

/// [`StaticPool`] backed by the same `node_take`/`node_give` free-list
/// core [`ReturnPool`] uses, instantiated with `BlockHeader` instead of a
/// `Node<D>`. Each returned block is `header_layout.extend(payload_layout)`
/// bytes; the header stays hidden before the pointer callers see.
pub struct BlockPool<const THRESHOLD: usize = 16> {
    /// Combined `(BlockHeader, payload)` layout — what's actually allocated
    /// for a fresh block.
    block_layout: Layout,
    /// Byte offset from the block's start to the payload (>= `size_of::<BlockHeader>()`,
    /// rounded up for `payload_layout`'s own alignment). Unlike
    /// `Node::<D>::node_of`'s compile-time `offset_of!`, this has to be a
    /// runtime field: `payload_layout` itself is only known at construction
    /// time (see [`DynamicPool`]'s doc comment for why).
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

        let (workers, pro_arrays) = new_workers_and_pro_arrays(num_workers);

        BlockPool { block_layout, payload_offset, workers, pro_arrays }
    }

    #[inline]
    fn alloc(&self, wk_num: usize) -> *mut u8 {
        match node_take(&self.workers, wk_num) {
            Some(header) => self.payload_of(header),
            None => self.payload_of(self.alloc_fresh(wk_num)),
        }
    }

    #[inline]
    unsafe fn dealloc(&self, cur_wk: usize, ptr: *mut u8) {
        let header = self.header_of(ptr);
        unsafe { node_give(&self.workers, &self.pro_arrays, cur_wk, header, THRESHOLD) };
    }
}

impl<const THRESHOLD: usize> Drop for BlockPool<THRESHOLD> {
    fn drop(&mut self) {
        drop_all_nodes(&self.workers, &self.pro_arrays, |header| unsafe {
            std::alloc::dealloc(header as *mut u8, self.block_layout);
        });
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
