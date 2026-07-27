//! Shared task-stack allocation machinery: [`StackAlloc`] (the pluggable
//! policy trait), [`StackMem`]/[`UltStackMemory`] (storage), [`HeapStack`]
//! (the plain-heap implementation), and the generic arena mechanism
//! (`Arena`/`ArenaMeta`/`ArenaKind`/`arena_init`/`alloc_arena_cell`/
//! `slot_from_addr`) parameterized over an `ArenaKind` so it can back
//! multiple independent arenas without collision — see
//! [`stackless::stack`](crate::resumable::stackless::stack) for the
//! `spawn_async`/`recurse` storage kind (`AsyncArenaStack`/
//! `AsyncTaskArenaKind`), currently its only user.
//!
//! # Arena cell layout
//!
//! ```text
//! cell (stride-aligned, stride = power of two)
//! ├── +0 .. +page              PROT_NONE guard (stacks grow DOWN into it)
//! ├── +page .. +stride-16      the stack (top = cell+stride-16, minus
//! │                            a per-cell coloring offset)
//! └── +stride-16 .. +stride    slot: [worker, system_id]
//! ```
//!
//! The slot holds the **worker pointer directly** (maintained by the switch
//! shims), not the descriptor: the lookup is then
//! `sp → cell slot → done` — two dependent loads — instead of chasing
//! `sp → slot → desc → worker`.  It also sits just above the stack top,
//! the same neighborhood the spawn path uses for the closure/result area,
//! so it reads hot memory.  (Placing it at the cell base was measurably
//! slower: that line is 64 KiB from the active stack region and always
//! cold.)
//!
//! The region is reserved PROT_NONE and committed per cell with `mprotect`,
//! so reserving gigabytes of address space costs no physical memory.
//!
//! # Multiple independent arenas (`ArenaKind`)
//!
//! The arena machinery (`Arena`/`ArenaMeta`/`arena_init`/`slot_from_addr`) is
//! parameterized by an `ArenaKind` — *not* a plain generic function with a
//! function-local `static`: a `static` item nested inside a generic function
//! is **not** duplicated per monomorphization when its own type/initializer
//! doesn't mention the generic parameter (verified empirically; it is one
//! shared instance across every instantiation). `ArenaKind` instead
//! dispatches to a genuinely separate top-level `static` pair per concrete
//! kind (the same pattern `SchedulerSystem::worker_tls()` already uses for
//! per-system TLS anchors), which macro-expansion or a manual `impl` can
//! provide once per kind.

use std::alloc::Layout;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Bytes reserved above the stack top for the `[worker, system_id]` slot
/// (16 also keeps the stack top 16-aligned).
pub(crate) const CELL_SLOT: usize = 16;

/// The per-cell lookup slot, just above the stack top.
///
/// `worker` is rewritten by the switch shims every time a task on this
/// stack is resumed; `system_id` is written once when the cell is paired
/// with its descriptor.  Heap and root descriptors point their
/// `UltDesc::slot` at an inline dummy instead, so the shims can store
/// unconditionally.
#[repr(C)]
pub struct CellSlot {
    pub(crate) worker: std::cell::Cell<*const ()>,
    pub(crate) system_id: std::cell::Cell<*const ()>,
}

/// Total address space reserved for the arena.
pub(crate) const ARENA_RESERVE: usize = 1 << 34; // 16 GiB of address space

// ---------------------------------------------------------------------------
// UltStackMemory trait
// ---------------------------------------------------------------------------

/// Type-level representation of an allocated task stack.
///
/// Concrete implementations: `HeapStackMem` (plain heap) and
/// [`ArenaStackMem`] (arena cell).  Both are produced by their corresponding
/// [`StackAlloc`] implementation and converted into [`StackMem`] for storage
/// inside [`BasicTaskDesc`](crate::resumable::common::desc::BasicTaskDesc).
pub trait UltStackMemory: Send + 'static {
    /// Pointer one byte past the top of the usable stack region.
    fn stack_top(&self) -> *mut u8;
    /// The arena lookup slot at the top of this cell, or `None` for heap stacks.
    fn cell_slot(&self) -> Option<*mut CellSlot>;
}

/// Heap-allocated stack memory (16-byte aligned).
pub struct HeapStackMem {
    pub(crate) ptr: *mut u8,
    pub(crate) size: usize,
}

unsafe impl Send for HeapStackMem {}

impl UltStackMemory for HeapStackMem {
    fn stack_top(&self) -> *mut u8 { unsafe { self.ptr.add(self.size) } }
    fn cell_slot(&self) -> Option<*mut CellSlot> { None }
}

impl Drop for HeapStackMem {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, heap_layout(self.size)) };
    }
}

/// Arena-cell stack memory.  `ptr` is the usable base (`cell + page`);
/// `ptr + size` is the stack top (below the fixed cell slot). `stride` is
/// stored directly (not re-derived from a global) so freeing/slot lookups
/// never need to guess which kind's arena this cell came from.
pub struct ArenaStackMem {
    pub(crate) ptr: *mut u8,
    pub(crate) size: usize,
    pub(crate) stride: usize,
}

unsafe impl Send for ArenaStackMem {}

impl UltStackMemory for ArenaStackMem {
    fn stack_top(&self) -> *mut u8 { unsafe { self.ptr.add(self.size) } }
    fn cell_slot(&self) -> Option<*mut CellSlot> {
        let cell = (self.ptr as usize) & !(self.stride - 1);
        Some((cell + self.stride - CELL_SLOT) as *mut CellSlot)
    }
}

impl Drop for ArenaStackMem {
    fn drop(&mut self) {
        let cell = (self.ptr as usize) & !(self.stride - 1);
        push_free_cell(cell);
    }
}

// A cell being freed doesn't know its own `ArenaKind` (only its address and
// stride); every kind's free list lives in its own `Arena`, keyed by the
// same `(base, stride)` pair the cell was carved from, so this just walks
// the process-wide registry of initialized arenas.  In practice there is
// only ever one or two (`AsyncTaskArenaKind`, plus whatever future kinds
// register), so a linear scan under a lock is fine.
static ARENA_REGISTRY: Mutex<Vec<&'static Arena>> = Mutex::new(Vec::new());

fn push_free_cell(cell: usize) {
    for ar in ARENA_REGISTRY.lock().unwrap().iter() {
        if cell >= ar.base && cell < ar.base + ARENA_RESERVE {
            ar.free.lock().unwrap().push(cell);
            return;
        }
    }
    unreachable!("cmpth: freed arena cell {cell:#x} matches no registered arena");
}

// ---------------------------------------------------------------------------
// StackMem — internal type-erased stack storage inside UltDesc
// ---------------------------------------------------------------------------

/// An allocated task stack stored inside [`BasicTaskDesc`](crate::resumable::common::desc::BasicTaskDesc).
///
/// Produced by converting a typed [`UltStackMemory`] value (via `From`); freed
/// when the owning descriptor is dropped.  Root pseudo-descriptors use the
/// `None` variant.
pub enum StackMem {
    /// Root pseudo-descriptors have no stack.
    None,
    /// Heap allocation (16-byte aligned).
    Heap { ptr: *mut u8, size: usize },
    /// Arena cell; `ptr` is the usable base (`cell + page`), and
    /// `ptr + size == cell + stride - CELL_SLOT` is the stack top.
    Arena { ptr: *mut u8, size: usize, stride: usize },
}

unsafe impl Send for StackMem {}
unsafe impl Sync for StackMem {}

impl StackMem {
    pub(crate) fn top(&self) -> *mut u8 {
        match *self {
            StackMem::None => std::ptr::null_mut(),
            StackMem::Heap { ptr, size } | StackMem::Arena { ptr, size, .. } => unsafe {
                ptr.add(size)
            },
        }
    }

    /// The arena cell slot for an arena-backed stack (at the fixed offset
    /// `cell + stride - CELL_SLOT`, independent of the coloring applied to
    /// the stack top), or `None` for other stack kinds. Uses the stride
    /// stored directly on this value — no ambiguity about which
    /// [`ArenaKind`] it came from.
    pub(crate) fn cell_slot(&self) -> Option<*mut CellSlot> {
        match *self {
            StackMem::Arena { ptr, stride, .. } => {
                let cell = (ptr as usize) & !(stride - 1);
                Some((cell + stride - CELL_SLOT) as *mut CellSlot)
            }
            _ => None,
        }
    }
}

impl Drop for StackMem {
    fn drop(&mut self) {
        match *self {
            StackMem::None => {}
            StackMem::Heap { ptr, size } => unsafe {
                std::alloc::dealloc(ptr, heap_layout(size));
            },
            StackMem::Arena { ptr, stride, .. } => {
                let cell = (ptr as usize) & !(stride - 1);
                push_free_cell(cell);
            }
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

impl From<ArenaStackMem> for StackMem {
    fn from(m: ArenaStackMem) -> Self {
        let s = StackMem::Arena { ptr: m.ptr, size: m.size, stride: m.stride };
        std::mem::forget(m); // ownership transferred to StackMem
        s
    }
}

pub(crate) fn heap_layout(size: usize) -> Layout {
    Layout::from_size_align(size.max(16), 16).expect("cmpth: bad stack layout")
}

// ---------------------------------------------------------------------------
// StackAlloc policy
// ---------------------------------------------------------------------------

/// Stack allocation policy.  Selected per system via
/// [`StackfulSchedulerSystem::StackAlloc`](crate::StackfulSchedulerSystem::StackAlloc)
/// (for real ULT stacks) or as the `A` parameter of
/// [`ReturnPool`](crate::resumable::common::pool::ReturnPool)/
/// [`SimplePool`](crate::resumable::common::pool::SimplePool) (for
/// `spawn_async`/`recurse` storage) — generic, not stackful-specific, despite
/// the name: [`AsyncArenaStack`](crate::resumable::stackless::stack::AsyncArenaStack)
/// implements it too.
pub trait StackAlloc: Send + Sync + 'static {
    /// The concrete stack-memory type produced by this allocator.
    type Mem: UltStackMemory + Into<StackMem>;
    #[doc(hidden)]
    fn alloc_stack(size: usize) -> Self::Mem;
}

/// Plain heap-allocated stacks (16-byte aligned). Despite the name, not
/// stackful-specific in practice: real ULT stacks default to it (via
/// `S::StackAlloc`), but `BasicTaskDesc::alloc`'s `spawn_async`
/// oversized-request fallback also uses it directly (that "stack" only
/// stores a `Future` — no code runs on it, so it never needs the arena).
pub struct HeapStack;

impl StackAlloc for HeapStack {
    type Mem = HeapStackMem;
    fn alloc_stack(size: usize) -> HeapStackMem {
        let ptr = unsafe { std::alloc::alloc(heap_layout(size.max(16))) };
        assert!(!ptr.is_null(), "cmpth: stack allocation failed");
        HeapStackMem { ptr, size: size.max(16) }
    }
}

/// Shared allocation logic for any [`ArenaKind`] (see the module docs for
/// why this can't just be a generic function with function-local statics).
pub(crate) fn alloc_arena_cell<K: ArenaKind>(size: usize) -> ArenaStackMem {
    let size = (size.max(16) + 15) & !15;
    let ar = arena_init::<K>(size);
    let page = page_size();

    // Every cell has the same usable size, fixed by the stride.
    let usable = ar.stride - page - CELL_SLOT;
    assert!(
        size <= usable,
        "cmpth: arena stack request {size} exceeds cell capacity {usable} \
         (stride fixed at {} by the first allocation of this arena kind)",
        ar.stride
    );

    let cell = ar
        .free
        .lock()
        .unwrap()
        .pop()
        .unwrap_or_else(|| {
            let cell = ar.next.fetch_add(ar.stride, Ordering::Relaxed);
            assert!(
                cell + ar.stride <= ar.base + ARENA_RESERVE,
                "cmpth: stack arena exhausted"
            );
            // Commit everything except the guard page at the cell base.
            let ret = unsafe {
                libc::mprotect(
                    (cell + page) as *mut libc::c_void,
                    ar.stride - page,
                    libc::PROT_READ | libc::PROT_WRITE,
                )
            };
            assert_eq!(ret, 0, "cmpth: mprotect(commit) failed");
            cell
        });

    // Cache coloring: stride is a power of two, so without an offset
    // every cell's stack-top lines map to the same L1 set and deep
    // fork chains (one live stack per recursion level) exceed the
    // associativity.  Shift each cell's top by a cell-dependent amount
    // (multiples of 3 cache lines on 128-byte-line machines, staying
    // within one L1 set-index window); the lookup slot stays at the
    // fixed cell offset.
    //
    // Number of distinct offsets is capped by how many actually fit in
    // `usable`, not a bare 31: for a kind with a small `usable` (e.g. the
    // async-task arena's tiny per-cell payload on a 4 KiB-page system —
    // `usable` there is far smaller than a stackful arena's), 31 slots of
    // 384 bytes each (up to 11520 total) can exceed `usable` itself, which
    // would underflow `usable - color` below and silently wrap to a huge
    // `size`. `.max(1)` keeps the modulus valid (color 0) when even one
    // slot doesn't fit.
    let color_slots = (usable / 384).clamp(1, 31);
    let color = ((cell / ar.stride) % color_slots) * 384;

    ArenaStackMem { ptr: (cell + page) as *mut u8, size: usable - color, stride: ar.stride }
}

// ---------------------------------------------------------------------------
// Arena singleton(s) — see `ArenaKind` in the module docs
// ---------------------------------------------------------------------------

pub(crate) struct Arena {
    pub(crate) base: usize,
    pub(crate) stride: usize,
    next: AtomicUsize,
    free: Mutex<Vec<usize>>,
}

// Published copies for the lock-free lookup fast path, packed into one
// cache line.  `base` is written last (Release); a reader that observes it
// non-zero (Acquire) also sees `len` / `stride`.
#[repr(C, align(64))]
pub(crate) struct ArenaMeta {
    base: AtomicUsize,
    len: AtomicUsize,
    stride: AtomicUsize,
}

pub(crate) const ARENA_META_INIT: ArenaMeta = ArenaMeta {
    base: AtomicUsize::new(0),
    len: AtomicUsize::new(0),
    stride: AtomicUsize::new(0),
};

/// A distinct family of arena-allocated cells, each with its own reserved
/// mmap region and stride (fixed independently by its own first
/// allocation). Implementors provide a genuinely separate top-level
/// `static` pair — see the module docs for why a generic function with a
/// function-local `static` does *not* achieve this.
pub(crate) trait ArenaKind: 'static {
    #[doc(hidden)]
    fn cell() -> &'static std::sync::OnceLock<Arena>;
    #[doc(hidden)]
    fn meta() -> &'static ArenaMeta;
}

pub(crate) fn arena_init<K: ArenaKind>(first_size: usize) -> &'static Arena {
    K::cell().get_or_init(|| {
        let meta = K::meta();
        // Stride: smallest power of two fitting guard page + stack + slot.
        let stride = (page_size() + first_size + CELL_SLOT).next_power_of_two();

        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                ARENA_RESERVE + stride, // slack so `base` can be aligned up
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert!(
            base != libc::MAP_FAILED,
            "cmpth: failed to reserve stack arena address space"
        );
        let base = round_up(base as usize, stride);

        meta.len.store(ARENA_RESERVE, Ordering::Relaxed);
        meta.stride.store(stride, Ordering::Relaxed);
        meta.base.store(base, Ordering::Release);

        let ar = Arena { base, stride, next: AtomicUsize::new(base), free: Mutex::new(Vec::new()) };
        ar
    });
    let ar = K::cell().get().unwrap();
    // Register once (idempotent-ish: only ever happens on the get_or_init
    // above in practice, but harmless if called again).
    {
        let mut reg = ARENA_REGISTRY.lock().unwrap();
        if !reg.iter().any(|r| std::ptr::eq(*r, ar)) {
            reg.push(ar);
        }
    }
    ar
}

/// Map an arbitrary address to the lookup slot of the arena cell it lies
/// on, for the given [`ArenaKind`]. Returns `None` when `addr` is outside
/// that specific arena (a different kind's cells, OS-thread stacks,
/// heap-allocated stacks, external threads).
#[inline]
pub(crate) fn slot_from_addr<K: ArenaKind>(addr: usize) -> Option<&'static CellSlot> {
    let meta = K::meta();
    let base = meta.base.load(Ordering::Acquire);
    if base == 0 {
        return None;
    }
    let off = addr.wrapping_sub(base);
    if off >= meta.len.load(Ordering::Relaxed) {
        return None;
    }
    let stride = meta.stride.load(Ordering::Relaxed);
    // Slot just above the stack top of the cell containing addr.
    let slot = base + (off & !(stride - 1)) + stride - CELL_SLOT;
    Some(unsafe { &*(slot as *const CellSlot) })
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
