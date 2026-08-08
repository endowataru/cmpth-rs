//! Shared task-stack allocation machinery: [`StackAlloc`] (the pluggable
//! policy trait), [`StackMem`]/[`UltStackMemory`] (storage), and
//! [`HeapStack`] (the one allocator this crate provides).
//!
//! # Guard pages
//!
//! Every stack is a single `mmap` region: a leading `PROT_NONE` guard page
//! (stacks grow DOWN into it, so a real overflow faults instead of
//! silently corrupting whatever memory happened to be adjacent), followed
//! by the usable, page-rounded stack region committed
//! `PROT_READ | PROT_WRITE` via `mprotect`. One `mmap`/`munmap` pair per
//! stack — no pooling, coloring, or shared reserved address range: this
//! crate previously had a pooled arena allocator for `spawn_async`/
//! `recurse` storage (with an address-masking "which worker owns this
//! cell" lookup keyed off the pool's own stride), but measurement showed
//! the lookup lost to a plain TLS read (~3ns/call slower on macOS) and the
//! pooling itself was never used for real context-switched ULT stacks in
//! the first place — only guard pages carried their weight, so that's all
//! that's left.

use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// UltStackMemory trait
// ---------------------------------------------------------------------------

/// Type-level representation of an allocated task stack.
///
/// The only implementation is [`HeapStackMem`], produced by [`HeapStack`]
/// and converted into [`StackMem`] for storage inside
/// [`DualTaskDesc`](crate::resumable::dual::desc::DualTaskDesc).
pub trait UltStackMemory: Send + 'static {
    /// Pointer one byte past the top of the usable stack region.
    fn stack_top(&self) -> *mut u8;
}

/// Guard-paged stack memory: `ptr`/`size` are the usable, already
/// page-rounded region; the `PROT_NONE` guard page sits immediately below
/// `ptr` (derived as `ptr.sub(page_size())` when munmapping).
pub struct HeapStackMem {
    pub(crate) ptr: *mut u8,
    pub(crate) size: usize,
}

unsafe impl Send for HeapStackMem {}

impl UltStackMemory for HeapStackMem {
    fn stack_top(&self) -> *mut u8 { unsafe { self.ptr.add(self.size) } }
}

impl Drop for HeapStackMem {
    fn drop(&mut self) {
        unsafe { dealloc_heap_stack(self.ptr, self.size) };
    }
}

// ---------------------------------------------------------------------------
// StackMem — internal type-erased stack storage inside UltDesc
// ---------------------------------------------------------------------------

/// An allocated task stack stored inside [`DualTaskDesc`](crate::resumable::dual::desc::DualTaskDesc).
///
/// Produced by converting a typed [`UltStackMemory`] value (via `From`); freed
/// when the owning descriptor is dropped.  Root pseudo-descriptors use the
/// `None` variant.
pub enum StackMem {
    /// Root pseudo-descriptors have no stack.
    None,
    /// Guard-paged heap allocation — `ptr`/`size` are the usable region,
    /// same layout as [`HeapStackMem`].
    Heap { ptr: *mut u8, size: usize },
}

unsafe impl Send for StackMem {}
unsafe impl Sync for StackMem {}

impl StackMem {
    pub(crate) fn top(&self) -> *mut u8 {
        match *self {
            StackMem::None => std::ptr::null_mut(),
            StackMem::Heap { ptr, size } => unsafe { ptr.add(size) },
        }
    }
}

impl Drop for StackMem {
    fn drop(&mut self) {
        match *self {
            StackMem::None => {}
            StackMem::Heap { ptr, size } => unsafe { dealloc_heap_stack(ptr, size) },
        }
    }
}

impl From<HeapStackMem> for StackMem {
    fn from(m: HeapStackMem) -> Self {
        let s = StackMem::Heap { ptr: m.ptr, size: m.size };
        std::mem::forget(m); // ownership transferred to StackMem
        s
    }
}

/// Shared by [`HeapStackMem::drop`] and [`StackMem`]'s `Heap` arm (the same
/// region, reached through two different owning types depending on whether
/// a descriptor has claimed it yet — see [`HeapStack::alloc_stack`] for the
/// matching allocation.
///
/// # Safety
/// `ptr`/`size` must be a still-live `(ptr, size)` pair previously returned
/// by `HeapStack::alloc_stack`, not yet freed.
unsafe fn dealloc_heap_stack(ptr: *mut u8, size: usize) {
    let page = page_size();
    let base = unsafe { ptr.sub(page) };
    let ret = unsafe { libc::munmap(base as *mut libc::c_void, page + size) };
    debug_assert_eq!(ret, 0, "cmpth: munmap failed");
}

// ---------------------------------------------------------------------------
// StackAlloc policy
// ---------------------------------------------------------------------------

/// Stack allocation policy.  Selected per system via
/// [`StackfulWorkerSystem::StackAlloc`](crate::StackfulWorkerSystem::StackAlloc)
/// (for real ULT stacks) or as the `A` parameter of
/// [`ReturnPool`](crate::resumable::common::pool::ReturnPool)/
/// [`SimplePool`](crate::resumable::common::pool::SimplePool) (for
/// `spawn_async`/`recurse` storage) — generic, not stackful-specific,
/// despite the name.
pub trait StackAlloc: Send + Sync + 'static {
    /// The concrete stack-memory type produced by this allocator.
    type Mem: UltStackMemory + Into<StackMem>;
    #[doc(hidden)]
    fn alloc_stack(size: usize) -> Self::Mem;
}

/// The one stack allocator this crate provides: a guard-paged `mmap`
/// region per stack (see the module doc). Used for real ULT stacks (via
/// `S::StackAlloc`), `spawn_async`/`recurse` pool storage, and
/// `DualTaskDesc::alloc`'s oversized-request fallback alike.
pub struct HeapStack;

impl StackAlloc for HeapStack {
    type Mem = HeapStackMem;
    fn alloc_stack(size: usize) -> HeapStackMem {
        let page = page_size();
        let usable = round_up(size.max(16), page);
        let mmap_len = page + usable;
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mmap_len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert!(base != libc::MAP_FAILED, "cmpth: failed to reserve stack guard region");
        let usable_ptr = unsafe { (base as *mut u8).add(page) };
        let ret = unsafe {
            libc::mprotect(
                usable_ptr as *mut libc::c_void,
                usable,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        };
        assert_eq!(ret, 0, "cmpth: mprotect(commit) failed");
        HeapStackMem { ptr: usable_ptr, size: usable }
    }
}

#[inline]
fn round_up(v: usize, to: usize) -> usize {
    (v + to - 1) & !(to - 1)
}

pub(crate) fn page_size() -> usize {
    static PAGE: AtomicUsize = AtomicUsize::new(0);
    let p = PAGE.load(Ordering::Relaxed);
    if p != 0 {
        return p;
    }
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    PAGE.store(p, Ordering::Relaxed);
    p
}
