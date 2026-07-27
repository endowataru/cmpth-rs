//! `spawn_async`/`recurse` task storage: [`AsyncArenaStack`], the
//! `AsyncTaskArenaKind` instantiation of the generic arena mechanism in
//! [`common::stack`](crate::resumable::common::stack) — see that module's
//! docs for the shared machinery and cell layout this builds on, and
//! [`crate::resumable::common::pool`] for how this storage is pooled.

use crate::resumable::common::stack::{alloc_arena_cell, Arena, ArenaKind, ArenaMeta, ArenaStackMem, StackAlloc, ARENA_META_INIT};

/// Storage carved from a *second*, independent reserved mmap arena, used for
/// `spawn_async`/`recurse` task storage instead of ULT stacks.
pub struct AsyncArenaStack;

impl StackAlloc for AsyncArenaStack {
    type Mem = ArenaStackMem;
    fn alloc_stack(size: usize) -> ArenaStackMem {
        alloc_arena_cell::<AsyncTaskArenaKind>(size)
    }
}

/// `spawn_async`/`recurse` task storage arena kind — its own reserved
/// region and stride, independent of any other `ArenaKind` that might be
/// registered.
pub(crate) struct AsyncTaskArenaKind;

static ASYNC_TASK_ARENA: std::sync::OnceLock<Arena> = std::sync::OnceLock::new();
static ASYNC_TASK_ARENA_META: ArenaMeta = ARENA_META_INIT;

impl ArenaKind for AsyncTaskArenaKind {
    fn cell() -> &'static std::sync::OnceLock<Arena> { &ASYNC_TASK_ARENA }
    fn meta() -> &'static ArenaMeta { &ASYNC_TASK_ARENA_META }
}
