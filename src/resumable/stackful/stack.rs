//! Real ULT stack allocation: [`ArenaStack`] (the [`UltStackArenaKind`]
//! instantiation of the generic arena mechanism in
//! [`common::stack`](crate::resumable::common::stack) — see that module's
//! docs for the shared machinery and cell layout this builds on, and that
//! module for [`HeapStack`](crate::resumable::common::stack::HeapStack),
//! which despite the name isn't stackful-specific: `BasicTaskDesc::alloc`'s
//! `spawn_async` oversized-request fallback uses it directly too).
//!
//! [`ArenaStack`] enables [`SpCurrent`](crate::resumable::stackful::lookup::SpCurrent):
//! any code running on an arena stack can find its own task descriptor from
//! the stack pointer alone, with no TLS access. Each cell also begins with
//! a PROT_NONE guard page, so stack overflow faults immediately instead of
//! silently corrupting a neighbor.

use crate::resumable::common::stack::{
    alloc_arena_cell, slot_from_addr, Arena, ArenaKind, ArenaMeta, ArenaStackMem, CellSlot, StackAlloc,
    ARENA_META_INIT,
};

/// Stacks carved from one reserved mmap arena (see
/// [`common::stack`](crate::resumable::common::stack)'s module docs).
///
/// Required by [`SpCurrent`](crate::resumable::stackful::lookup::SpCurrent).  The cell
/// stride is fixed by the first allocation; every cell provides
/// `stride - page - 16` bytes of stack (at least the requested size).
pub struct ArenaStack;

impl StackAlloc for ArenaStack {
    type Mem = ArenaStackMem;
    fn alloc_stack(size: usize) -> ArenaStackMem {
        alloc_arena_cell::<UltStackArenaKind>(size)
    }
}

/// The original ULT-stack arena kind — identical behavior to before
/// [`common::stack`](crate::resumable::common::stack) supported more than
/// one arena.
pub(crate) struct UltStackArenaKind;

static ULT_STACK_ARENA: std::sync::OnceLock<Arena> = std::sync::OnceLock::new();
static ULT_STACK_ARENA_META: ArenaMeta = ARENA_META_INIT;

impl ArenaKind for UltStackArenaKind {
    fn cell() -> &'static std::sync::OnceLock<Arena> { &ULT_STACK_ARENA }
    fn meta() -> &'static ArenaMeta { &ULT_STACK_ARENA_META }
}

/// Map a stack pointer to the lookup slot of the arena cell it lies on
/// (the [`UltStackArenaKind`] arena specifically).  Returns `None` when
/// `sp` is outside that arena (OS-thread stacks, heap-allocated ULT
/// stacks, external threads, or a different arena kind's cells).
#[inline]
pub(crate) fn slot_from_sp(sp: usize) -> Option<&'static CellSlot> {
    slot_from_addr::<UltStackArenaKind>(sp)
}

#[cfg(test)]
mod tests {
    use crate::ThreadSystem;
    use crate::ScopedStackfulTaskSystem;

    struct SpTestSystem;

    impl crate::UltIdentity for SpTestSystem {
        type Base = crate::OsSystem;
        type Ctx = crate::NativeContext;
        type Deque = crate::CrossbeamDeque<crate::BasicTaskDesc>;
        type Alloc = crate::ArenaStack;
        type Lookup = crate::SpCurrent;

        fn worker_tls_anchor() -> &'static <crate::OsSystem as ThreadSystem>::ThreadSpecific<crate::UltWorker<Self>> {
            static A: crate::TlsAnchor = crate::TlsAnchor::new();
            crate::TlsSlot::from_anchor(&A)
        }
    }

    /// The sp lookup must actually hit (not silently fall back to TLS).
    #[test]
    fn sp_lookup_hits_on_arena_stack() {
        use crate::resumable::common::lookup::system_id;
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
