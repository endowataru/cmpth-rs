//! [`ScopedStackfulTaskSystem`]/[`ScopedStacklessTaskSystem`] — a rayon-
//! `join`-like binary divide-and-conquer primitive: run two function
//! objects (or futures), potentially in parallel via work stealing, and
//! either synchronously (stackful) or via `.await` (stackless) wait for
//! both results.
//!
//! Lives in [`crate::scoped`] (module name, not the method name — see that
//! module's docs) because the defining property here isn't really "fork" or
//! "join" at all: unlike `spawn`/`spawn_async`, which split off *one*
//! function and reify the caller's own continuation as the other,
//! separately schedulable half, `parallel_call` takes two already-closed
//! function objects and never exposes the caller's continuation as anything
//! stealable — control returns to the caller's own next line exactly like
//! an ordinary (if parallel) function call would. That's the same
//! "nothing spawned here outlives this call" property `std::thread::scope`
//! names after itself, just restricted to exactly two branches. This is a
//! *stricter* constraint than [`ThreadSystem`](crate::ThreadSystem)'s
//! spawn/join (whose spawned task may outlive the caller) — so anything
//! with `ThreadSystem`'s looser capability can trivially satisfy this one
//! too (spawn one branch, run the other inline, join). See
//! [`StackfulTaskSystem`](crate::traits::stackful::StackfulTaskSystem) for
//! that blanket derivation.
//!
//! The method was originally named after Intel TBB's `tbb::parallel_invoke`
//! (renamed `parallel_call` here to fit this trait family's naming, not
//! because the TBB precedent stopped applying) for this specific shape of
//! primitive — not "join" (collides in spirit with
//! [`JoinHandleLike::join`](crate::JoinHandleLike::join)), not "fork_join"
//! (rayon itself reserves "fork-join" language for its N-ary, heap-allocated
//! `scope()`/`Scope::spawn()` API, not for the binary, stack-only `join()`
//! this mirrors — confirmed against rayon's own docs: `join`'s description
//! never uses "fork-join"; `scope`'s does).
//!
//! Implemented directly (no `ThreadSystem`/`SchedulerSystem` involved) by
//! [`crate::scoped`]'s standalone engine for systems that want *only* this
//! capability, and blanket-derived for anything that already has
//! `ThreadSystem`/[`StacklessTaskSystem`](crate::StacklessTaskSystem) — see
//! [`crate::scoped`]'s docs for why the standalone engine stays independent
//! of `resumable`'s `SchedulerSystem`/`UltWorker` machinery.

pub use crate::traits::system::scoped::{ScopedStackfulTaskSystem, ScopedStacklessTaskSystem};
