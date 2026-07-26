//! Task-stack allocation policy.
//!
//! Two implementations of [`StackAlloc`]:
//!
//! * [`HeapStack`] — plain heap allocation (the classic behavior).
//! * [`ArenaStack`] — stacks carved out of one reserved, stride-aligned mmap
//!   region.  Enables [`SpCurrent`](crate::ult::lookup::SpCurrent): any code
//!   running on an arena stack can find its own task descriptor from the
//!   stack pointer alone, with no TLS access.  Each cell also begins with a
//!   PROT_NONE guard page, so stack overflow faults immediately instead of
//!   silently corrupting a neighbor.
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
//! # Multiple independent arenas ([`ArenaKind`])
//!
//! The arena machinery (`Arena`/`ArenaMeta`/`arena_init`/`slot_from_addr`) is
//! parameterized by an [`ArenaKind`] — *not* a plain generic function with a
//! function-local `static`: a `static` item nested inside a generic function
//! is **not** duplicated per monomorphization when its own type/initializer
//! doesn't mention the generic parameter (verified empirically; it is one
//! shared instance across every instantiation). `ArenaKind` instead
//! dispatches to a genuinely separate top-level `static` pair per concrete
//! kind (the same pattern `SchedulerSystem::worker_tls()` already uses for
//! per-system TLS anchors), which macro-expansion or a manual `impl` can
//! provide once per kind.
//!
//! Two arenas share this module's logic today: [`UltStackArenaKind`] (the
//! original ULT-stack arena, unchanged behavior) and
//! [`AsyncTaskArenaKind`] (`spawn_async`/`recurse` storage — see
//! `crate::ult::pool`). Each kind's stride is fixed independently by *its
//! own* first allocation, so a small async-task request can never collide
//! with a large ULT stack's stride requirement.

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
const ARENA_RESERVE: usize = 1 << 34; // 16 GiB of address space

// ---------------------------------------------------------------------------
// UltStackMemory trait
// ---------------------------------------------------------------------------

/// Type-level representation of an allocated task stack.
///
/// Concrete implementations: [`HeapStackMem`] (plain heap) and
/// [`ArenaStackMem`] (arena cell).  Both are produced by their corresponding
/// [`StackAlloc`] implementation and converted into [`StackMem`] for storage
/// inside [`BasicTaskDesc`](crate::ult::desc::BasicTaskDesc).
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
// the process-wide registry of initialized arenas.  In practice there are
// only ever one or two (`UltStackArenaKind`, `AsyncTaskArenaKind`), so a
// linear scan under a lock is fine.
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

/// An allocated task stack stored inside [`BasicTaskDesc`](crate::ult::desc::BasicTaskDesc).
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

fn heap_layout(size: usize) -> Layout {
    Layout::from_size_align(size.max(16), 16).expect("cmpth: bad stack layout")
}

// ---------------------------------------------------------------------------
// StackAlloc policy
// ---------------------------------------------------------------------------

/// Stack allocation policy.  Selected per system via
/// [`UltSchedulerSystem::StackAlloc`](crate::UltSchedulerSystem::StackAlloc).
pub trait StackAlloc: Send + Sync + 'static {
    /// The concrete stack-memory type produced by this allocator.
    type Mem: UltStackMemory + Into<StackMem>;
    #[doc(hidden)]
    fn alloc_stack(size: usize) -> Self::Mem;
}

/// Plain heap-allocated stacks (16-byte aligned).
pub struct HeapStack;

impl StackAlloc for HeapStack {
    type Mem = HeapStackMem;
    fn alloc_stack(size: usize) -> HeapStackMem {
        let ptr = unsafe { std::alloc::alloc(heap_layout(size.max(16))) };
        assert!(!ptr.is_null(), "cmpth: stack allocation failed");
        HeapStackMem { ptr, size: size.max(16) }
    }
}

/// Stacks carved from one reserved mmap arena (see the module docs).
///
/// Required by [`SpCurrent`](crate::ult::lookup::SpCurrent).  The cell
/// stride is fixed by the first allocation; every cell provides
/// `stride - page - 16` bytes of stack (at least the requested size).
pub struct ArenaStack;

impl StackAlloc for ArenaStack {
    type Mem = ArenaStackMem;
    fn alloc_stack(size: usize) -> ArenaStackMem {
        alloc_arena_cell::<UltStackArenaKind>(size)
    }
}

/// Storage carved from a *second*, independent reserved mmap arena, used for
/// `spawn_async`/`recurse` task storage instead of ULT stacks (see
/// `crate::ult::pool`). Same mechanism as [`ArenaStack`], but keyed by
/// [`AsyncTaskArenaKind`] so its stride (sized for small `Future` payloads)
/// can never collide with a stackful system's (much larger) `STACK_SIZE`.
pub struct AsyncArenaStack;

impl StackAlloc for AsyncArenaStack {
    type Mem = ArenaStackMem;
    fn alloc_stack(size: usize) -> ArenaStackMem {
        alloc_arena_cell::<AsyncTaskArenaKind>(size)
    }
}

/// Shared allocation logic for any [`ArenaKind`] (see the module docs for
/// why this can't just be a generic function with function-local statics).
fn alloc_arena_cell<K: ArenaKind>(size: usize) -> ArenaStackMem {
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
    let color = ((cell / ar.stride) % 31) * 384;

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

const ARENA_META_INIT: ArenaMeta = ArenaMeta {
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

/// The original ULT-stack arena kind — identical behavior to before this
/// module supported more than one arena.
pub(crate) struct UltStackArenaKind;

static ULT_STACK_ARENA: std::sync::OnceLock<Arena> = std::sync::OnceLock::new();
static ULT_STACK_ARENA_META: ArenaMeta = ARENA_META_INIT;

impl ArenaKind for UltStackArenaKind {
    fn cell() -> &'static std::sync::OnceLock<Arena> { &ULT_STACK_ARENA }
    fn meta() -> &'static ArenaMeta { &ULT_STACK_ARENA_META }
}

/// `spawn_async`/`recurse` task storage arena kind (see `crate::ult::pool`
/// and `crate::ult::thread::recurse`) — independent stride from
/// [`UltStackArenaKind`], so a small async-task request can never collide
/// with a stackful system's much larger `STACK_SIZE`.
pub(crate) struct AsyncTaskArenaKind;

static ASYNC_TASK_ARENA: std::sync::OnceLock<Arena> = std::sync::OnceLock::new();
static ASYNC_TASK_ARENA_META: ArenaMeta = ARENA_META_INIT;

impl ArenaKind for AsyncTaskArenaKind {
    fn cell() -> &'static std::sync::OnceLock<Arena> { &ASYNC_TASK_ARENA }
    fn meta() -> &'static ArenaMeta { &ASYNC_TASK_ARENA_META }
}

fn arena_init<K: ArenaKind>(first_size: usize) -> &'static Arena {
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

/// Map a stack pointer to the lookup slot of the arena cell it lies on
/// (the [`UltStackArenaKind`] arena specifically).  Returns `None` when
/// `sp` is outside that arena (OS-thread stacks, heap-allocated ULT
/// stacks, external threads, or a different arena kind's cells).
#[inline]
pub(crate) fn slot_from_sp(sp: usize) -> Option<&'static CellSlot> {
    slot_from_addr::<UltStackArenaKind>(sp)
}

#[inline]
fn round_up(v: usize, to: usize) -> usize {
    (v + to - 1) & !(to - 1)
}

fn page_size() -> usize {
    static PAGE: AtomicUsize = AtomicUsize::new(0);
    let p = PAGE.load(Ordering::Relaxed);
    if p != 0 {
        return p;
    }
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    PAGE.store(p, Ordering::Relaxed);
    p
}

#[cfg(test)]
mod tests {
    use crate::ThreadSystem;
    use crate::UltSystem;

    crate::ult_system! {
        struct SpTestSystem {
            base:        crate::OsSystem,
            context:     crate::NativeContext,
            deque:       crate::CrossbeamDeque<crate::BasicTaskDesc>,
            stack_size:  64 * 1024,
            stack_alloc: crate::ArenaStack,
            lookup:      crate::SpCurrent,
        }
    }

    /// The sp lookup must actually hit (not silently fall back to TLS).
    #[test]
    fn sp_lookup_hits_on_arena_stack() {
        use crate::ult::lookup::system_id;
        fn current_sp() -> usize {
            let sp: usize;
            #[cfg(target_arch = "aarch64")]
            unsafe { core::arch::asm!("mov {}, sp", out(reg) sp) };
            #[cfg(target_arch = "x86_64")]
            unsafe { core::arch::asm!("mov {}, rsp", out(reg) sp) };
            sp
        }
        SpTestSystem::run(2, || {
            let h = SpTestSystem::spawn(|| {
                let slot = super::slot_from_sp(current_sp())
                    .expect("sp lookup missed on an arena stack");
                assert_eq!(
                    slot.system_id.get(),
                    system_id::<SpTestSystem>(),
                    "cell slot tagged with the wrong system"
                );
                assert!(!slot.worker.get().is_null(), "slot worker not maintained");
                42u64
            });
            assert_eq!(crate::JoinHandleLike::join(h), 42);
        });
    }
}
