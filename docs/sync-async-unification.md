# Sync/async unification design

Status: design settled, **no implementation yet**. This document is the
reference for implementing it. See also `ISSUES.md` for a related, blocking
prerequisite bug.

## Goal

Let `Mutex`/`Barrier`/`Delegator` (currently built only on stackful ULTs, via
`SuspendedThread`) be usable from both real ULTs (`spawn`) and stackless
`spawn_async` tasks — including a single instance being waited on by both
kinds simultaneously — without making pure-ULT-only code pay for
`std::task::Waker`'s dynamic dispatch anywhere on its path.

## System-level capability traits

Two independent traits; a "mixed" system just implements both (no third
trait needed):

```rust
pub trait UltSystem { .. }          // existing: real stackful ULTs
pub trait AsyncWorkerSystem { .. }  // new (renamed from an earlier AsyncTaskSystem):
                                     // stackless spawn_async-style tasks
```

Naming notes:
- `AsyncWorkerSystem`, not `AsyncTaskSystem`: "Task" was reserved for the
  neutral/either-kind concept at the WaitSlot layer (see below) to match
  cmpth-rs's own existing usage (`UltDesc::TaskResult`, `UltWorker::cur_task`
  already mean "whichever kind is running, thread or async"). "Worker" was
  chosen over "Future" because it conveys parallel-runtime-ness (matching
  the crate's existing `UltWorker` vocabulary); "Future" reads as a value
  type, not an execution engine.

## WaitSlot-level types

One minimal shared trait, three concrete types:

```rust
pub trait Suspended<S>: Default {
    fn is_set(&self) -> bool;
    fn notify(&self);
}
```

| Type | Bound | Storage | Notes |
|---|---|---|---|
| `SuspendedThread<S>` | `S: UltSystem` | `AtomicPtr<UltDesc>` | Unchanged from today's `SuspendedThread`/`BasicSuspendedThread` — no rename, no signature changes. |
| `SuspendedFuture<S>` | `S: AsyncWorkerSystem` | boxed `std::task::Waker` | Deliberately a *standard* `Waker`, not cmpth's internal `UltDesc` pointer — this is what lets `Mutex::lock_async()` compose with any executor (tokio, futures, cmpth's own), matching `cmpth::future::yield_now()`'s existing executor-agnostic philosophy. |
| `SuspendedTask<S>` | `S: UltSystem + AsyncWorkerSystem` | tagged `AtomicUsize` (Empty / Parked / Async) | The mixed case. `notify()` is the only method that branches on the tag; that branch is the only place indirect (`Waker::wake`) dispatch is ever paid, and only when the slot actually holds an async waiter. |

## Sync/async method flavors — same method names everywhere

Two more traits split the "wait"/"direct handoff" operations by calling
convention. Same pattern as `cmpth::future::yield_now()` vs
`ThreadSystem::yield_now()`: **same method name, disambiguated by which
trait you `use`**, never by a `_async` suffix.

```rust
pub trait SyncSuspended<S>: Suspended<S> {
    fn wait_with<F: FnOnce()>(&self, f: F);
    fn wait_with_cond<F: FnOnce() -> bool>(&self, f: F) -> bool;
    fn enter(&self) -> bool;             // false only possible on SuspendedTask
    fn swap(&self, next: &Self) -> bool; // (target held an async waiter, not Parked)
}

pub trait AsyncSuspended<S>: Suspended<S> {
    fn register(&self, cx: &mut Context<'_>) -> bool;
    async fn wait_with(&self, f: impl FnOnce());
    async fn enter(&self);   // always "succeeds" — built on notify(), which
    async fn swap(&self, next: &Self); // is valid for either kind of target
}
```

- `SuspendedThread` implements `SyncSuspended` only.
- `SuspendedFuture` implements `AsyncSuspended` only.
- `SuspendedTask` implements **both**.

`SyncSuspended::enter`/`swap` return `bool` because, on `SuspendedTask`
specifically, the target may hold an async registration instead of a real
continuation — a real context jump is only possible when it's actually
Parked. `SuspendedThread`'s impl always returns `true` (type-guaranteed).
`AsyncSuspended::enter`/`swap` never need this: they're built on `notify()`
(§ "enter/swap", below), which is valid for either kind of target, so they
have no failure case.

Same split applies one level up, to the actual primitives:

```rust
pub trait SyncLock<T>  { fn lock(&self) -> Guard<'_>; }
pub trait AsyncLock<T> { async fn lock(&self) -> Guard<'_>; }
// ...and the equivalent SyncBarrier/AsyncBarrier, SyncDelegator/AsyncDelegator, etc.
```

`impl SyncLock for McsMutex<S, T, N>` where `N: SyncSuspended<S>`;
`impl AsyncLock for McsMutex<S, T, N>` where `N: AsyncSuspended<S>`. A mutex
built on `SuspendedTask` gets both impls, so both `.lock()` and
`.lock().await` exist on the same value — pick the flavor via which trait is
`use`d at the call site. (Importing both into the same scope makes `.lock()`
ambiguous; use fully-qualified syntax if one call site genuinely needs both.)

Suggested module layout for bulk import:
```rust
use cmpth::traits::sync::*;      // SyncSuspended, SyncLock, SyncBarrier, ...
use cmpth::traits::r#async::*;   // AsyncSuspended, AsyncLock, ... (`async` is a
                                  // keyword; `r#async` works but reads awkwardly —
                                  // consider `blocking`/`cooperative` or similar instead)
```

## No `OnUlt` capability token — reuse `cur_task.is_root` instead

Earlier drafts of this design threaded an explicit `OnUlt<S>` proof token
through every `SyncSuspended` call, to statically rule out calling it from
async code. Dropped: too much ceremony for ordinary ULT code, and it was
only ever load-bearing for `SuspendedTask` (for `SuspendedThread`,
`S: UltSystem` alone already guarantees it, always trivially true).

The risk it was guarding against is real and worse than a clean panic: if
`SyncSuspended::wait_with` is (incorrectly) called from inside
`run_async_poll`, `UltWorker::<S>::current()` still returns `Some` (it's
keyed to the OS thread, which is legitimately a worker) — so the obvious
"is there a current worker" check doesn't catch it. A real `suspend_to_sched`
call at that point would try to save a continuation whose "stack" is really
just a point in the middle of the worker's own shared dispatch-loop call
stack (since `run_async_poll` never allocates a dedicated stack) — resuming
it later jumps back into a stack region likely long since overwritten by
unrelated work. That's memory corruption, not a clean abort — unlike, say,
calling into cmpth from a hand-rolled `std::thread::spawn` that was never
registered as a worker (TLS lookup cleanly returns `None` there).

Fix: reuse `UltWorker::cur_task`, which already exists and is already
correctly maintained. Verified this session (see
`src/ult/worker.rs`/`scheduler.rs`): `cur_task` is set to the worker's own
`root_desc` (`is_root: true`) at dispatch-loop start and every time a real
ULT suspends/exits back to the scheduler; it's set to the *task's* own desc
(`is_root: false`) only while that ULT is actually resumed and running;
`run_async_poll` never touches it, so it correctly stays `is_root: true`
throughout any async poll. So:

```rust
// at the top of every SyncSuspended method:
let wk = UltWorker::<S>::current().expect("not on a worker");
assert!(!unsafe { (*wk.cur_task.get()).is_root },
    "sync wait/enter/swap called outside a real ULT");
```

No new per-worker state needed — this is a straight reuse of an existing,
already-correct field.

## `enter`/`swap`: what they actually do, and why async doesn't need a
## dedicated primitive

Verified against `references/composablethreads/include/cmpth/wss/basic_suspended_thread.hpp`:
`enter()` is **not** exit-like/divergent. It does
`wk.suspend_to_cont::<on_enter>(move(this->cont_))` where `on_enter` does
`wk.local_push_top(move(cont))` — i.e. "notify with the roles reversed":
jump directly into the target's continuation, and push the *caller's own*
now-suspended continuation onto the local deque (mirrors `spawn`'s
child-first semantics, just for waking instead of spawning). It does
eventually return, exactly like `wait_with` — just later, whenever the
caller's pushed continuation is itself resumed.

They are **pure optimizations, never required for correctness** — `notify()`
alone is always a complete, correct fallback. Confirmed twice: C++
`basic_mcs_mutex.hpp:88-92` has `//next->sth.notify(wk);` commented out right
beside the `enter(wk)` call the author kept; Rust's own
`delegator.rs::unlock()` already branches between `swap` and `notify`
depending on `is_active`.

**Async code doesn't need a dedicated enter/swap primitive**, for a
structural reason, not just because `OnUlt`/context-switching isn't
available: stackful code needs `suspend_to_sched` specifically because plain
synchronous Rust can't otherwise "pause here, resume later" — but
`Future::poll()` already has that built in via `Poll::Pending`. So
`AsyncSuspended::enter`/`swap` can be plain combinators:

```rust
async fn enter(&self) {
    self.notify();          // push target to local-deque top
    yield_to_bottom().await; // defer *myself* to the FIFO/steal end
}
```

**This requires fixing the yield-fairness bug in `ISSUES.md` first.** If the
"defer myself" step pushes to `push_local_top` (LIFO) instead of
`push_local_bottom` (FIFO/steal end) — which is what `cmpth::future::yield_now()`
currently does, unconditionally, regardless of backend — the caller ends up
*ahead of* the target it just tried to hand off to (LIFO means whichever was
pushed last, i.e. the caller's own deferred continuation, gets popped
first), inverting the intended priority. `ThreadSystem::yield_now()` (the
stackful version) already gets this right via `push_local_bottom`; the async
self-wake path and `CrossbeamDeque::push_bottom` do not yet. `AsyncSuspended::enter`/`swap`
should not be implemented before that's fixed.

## Avoiding duplicated sync/async algorithm bodies

Concern: hand-maintaining two divergent implementations (one per flavor) for
every primitive risks drift and, more importantly, an unusable, sprawling
interface. Resolution: write the shared algorithm once in a `macro_rules!`,
parameterized only on the "how do I wait" step; each flavor's trait impl
supplies a different expression:

```rust
macro_rules! lock_body {
    ($wait:expr) => {{
        let (is_locked, prev, cur) = queue.start_lock();
        if is_locked {
            'done: { break 'done make_guard(); }
        } else {
            $wait;
            make_guard()
        }
    }};
}

impl<S: UltSystem, T, N: SyncSuspended<S>> SyncLock<T> for McsMutex<S, T, N> {
    fn lock(&self) -> Guard<'_> {
        lock_body!(sth.wait_with(|| { queue.set_next(prev, cur) }))
    }
}

impl<S: AsyncWorkerSystem, T, N: AsyncSuspended<S>> AsyncLock<T> for McsMutex<S, T, N> {
    async fn lock(&self) -> Guard<'_> {
        lock_body!(sth.wait_with(|| { queue.set_next(prev, cur) }).await)
    }
}
```

The sync expansion compiles to exactly today's direct-dispatch code; the
async expansion goes through `Poll`/`Waker`, but only because that
expansion was asked for. One source of truth for the queue-manipulation
logic either way.

**Gotcha:** use a labeled block (`'label: { break 'label val; }`) instead of
a bare `return` inside macro bodies shared between a plain `fn` and an
`async move { }` block — `return` inside `async move { }` returns from the
*enclosing function* (wrong type: `impl Future<Output = Guard>`, not
`Guard`), not from the async block itself.

## `spawn`/`spawn_async` and `JoinHandle`

`spawn`/`spawn_async` themselves are **not** worth unifying under one name —
unlike `wait_with`/`lock`, the sync/async difference here is in the
*argument type* (`FnOnce() -> T` vs `Future<Output = T>`), not the calling
convention, and the two bodies don't share much algorithm to begin with. If
attempted anyway (`SyncSpawn`/`AsyncSpawn` traits implemented by
`UltSystem`/`AsyncWorkerSystem` respectively, both providing a method named
`spawn`), note `S::spawn(f)` is an associated-function call, not a method
call on a value — Rust's overload resolution for same-named trait methods
does not appear to disambiguate by argument type when both traits are in
scope, same caveat as `SyncLock`/`AsyncLock`, untested.

`JoinHandle`'s existing design already supports both `.join()` (sync,
inherent method) and `.await` (async, via `impl Future for JoinHandle`) on
one type, using the `join_state` tagged-`AtomicUsize` encoding
(`ult/desc.rs`) to track whichever kind of waiter (if any) attaches. Note
this sidesteps the "same name" problem entirely — `.join()` and `.await` are
different syntax, not competing methods — so no `SyncJoin`/`AsyncJoin` trait
split was ever needed for *that* reason.

But it's still the same "pay for flexibility whether or not you use it"
shape this whole design has been trying to avoid elsewhere: *every* spawned
task's exit path carries the tagged-branch cost, regardless of whether that
particular handle is ever actually joined from the other side. Unlike a
`Mutex`, though, a `JoinHandle` has exactly one consumer, and that consumer
usually already knows — from their own calling context — whether they'll
`.join()` or `.await` it. The genuinely-don't-know-in-advance case (spawn
resolved from sync code, result consumed from async code, or vice versa) is
real but much rarer than `Mutex`'s "many independent, uncoordinated waiters"
situation.

Proposed split, mirroring `SuspendedThread`/`SuspendedFuture`/`SuspendedTask`:
- `SyncJoinHandle<S, T>` — `spawn()`'s default return type. `.join()` only,
  zero branching (same shape as `SuspendedThread`'s direct notify).
  - `AsyncJoinHandle<S, T>` — `spawn_async()`'s default return type. `.await`
  only, zero branching (same shape as `SuspendedFuture`).
- `MixedJoinHandle<S, T>` — today's `join_state`-tagged design, kept as an
  explicit opt-in (e.g. a distinct constructor/conversion) for the rarer
  cross case, not the default.

This is independent of which of `spawn`/`spawn_async` created the
underlying task — a real ULT's handle could in principle still be the
`Mixed` flavor if someone explicitly wants cross-context joining.

## Open / not yet decided

- Exact `Sync{Barrier,Condvar,Delegator}`/`Async{Barrier,Condvar,Delegator}`
  trait shapes — same pattern as `SyncLock`/`AsyncLock`, not spelled out in
  detail yet.
- Module name for the async-flavor trait bundle (`r#async` works but is
  awkward; want something cleaner).
- Whether `SuspendedTask`'s internal tagged `AtomicUsize` needs the same
  atomic-claim/cancel-race care that `join_state` (`ult/desc.rs`) has, or
  whether the simpler "fresh node, written once before publish" argument
  (established earlier in the design discussion) fully covers it for all of
  Mutex/Barrier/Delegator's actual usage patterns — not re-verified against
  Barrier/Condvar's code specifically.
- Whether to fix `CrossbeamDeque::push_bottom`'s degradation (`ISSUES.md`)
  as part of this work or file it as a fully separate task — it's now a
  hard prerequisite for `AsyncSuspended::enter`/`swap` specifically, even
  though it was originally found as an unrelated fairness bug.

## Suggested implementation order

1. Fix the `yield_now`/`push_bottom` fairness bug (`ISSUES.md`) — now a
   prerequisite, not just a nice-to-have.
2. `Suspended<S>` + `SuspendedThread<S>` (should be close to a pure rename/
   no-op relative to today's code) + `SyncSuspended<S>` trait, retrofit
   `McsMutex`'s existing sync path onto it with the macro, confirm no
   codegen regression.
3. `SuspendedFuture<S>` + `AsyncSuspended<S>`, add `McsMutex::lock_async()`
   for the pure-async instantiation.
4. `SuspendedTask<S>` (tagged union) + both trait impls together, including
   the `cur_task.is_root` guard, `try`-style `bool` returns on
   `SyncSuspended::enter`/`swap`.
5. Repeat for `Barrier`, then `Delegator` last (most structurally different;
   `enter`/`swap` usage there is also currently behind
   `#[allow(dead_code)]` dedicated-consumer-thread mode that isn't wired up
   yet).
