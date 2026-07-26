//! `spawn_async`/`recurse` task storage: [`AsyncArenaStack`], the
//! [`AsyncTaskArenaKind`] instantiation of the generic arena mechanism in
//! [`common::stack`](crate::resumable::common::stack) — see that module's
//! docs for the shared machinery and cell layout this builds on, and
//! [`crate::resumable::common::pool`] for how this storage is pooled.
//!
//! Same mechanism as
//! [`stackful::stack::ArenaStack`](crate::resumable::stackful::stack::ArenaStack),
//! but keyed by its own [`ArenaKind`] so its stride (sized for small
//! `Future` payloads) can never collide with a stackful system's (much
//! larger) `STACK_SIZE`.

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

/// `spawn_async`/`recurse` task storage arena kind — independent stride from
/// [`stackful::stack::UltStackArenaKind`](crate::resumable::stackful::stack::UltStackArenaKind),
/// so a small async-task request can never collide with a stackful system's
/// much larger `STACK_SIZE`.
pub(crate) struct AsyncTaskArenaKind;

static ASYNC_TASK_ARENA: std::sync::OnceLock<Arena> = std::sync::OnceLock::new();
static ASYNC_TASK_ARENA_META: ArenaMeta = ARENA_META_INIT;

impl ArenaKind for AsyncTaskArenaKind {
    fn cell() -> &'static std::sync::OnceLock<Arena> { &ASYNC_TASK_ARENA }
    fn meta() -> &'static ArenaMeta { &ASYNC_TASK_ARENA_META }
}
