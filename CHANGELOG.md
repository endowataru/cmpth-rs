# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
but this crate is **experimental**: the API may change between releases
without a deprecation cycle (see README "Status").

## [Unreleased]

### Breaking

- **Delegator interface redesigned.** A consumer now declares
  `Request<'a>`/`Response` and runs a request with
  `execute(req, cx) -> Poll<Response>`: `Poll::Ready` finishes the call,
  `Poll::Pending` keeps it, with the `Completion` taken by `cx.defer()`;
  a completed call comes back to the delegator from `progress` as a
  `Ready`.  `progress`/`is_active` have defaults.  Callers use
  `Delegator::call(req)` (returns the response; requests may borrow from
  the caller's frame), `with_lock` and, for closures, `FnConsumer` with
  `DelegatorRun::run`.  `start` returns `Pin<Box<Self>>` and dropping the
  delegator stops it (no `stop`).  The `imm`/`del` closure pair,
  `DelegatorConsumer::Work` and the mpsc-style `delegator()`/`Producer`
  are removed.

### Added

- `CombiningDelegator`: no consumer thread; the lock holder runs the
  queued requests.

### Fixed

- `McsDelegator`: releasing the lock to a caller waiting for it parked the
  releasing thread in the consumer's slot; a caller whose call was kept
  could be left asleep with the lock held (`unlock_and_wait` now follows
  the C++ reference); a caller completed by `progress` just before the
  consumer went idle was never woken.
- `RingBufDelegator`: the lock was never released and the holder never
  found its successor.
- `OsDelegator`: every call runs inline under the mutex (it used to wake
  all waiters on each call).

## [0.4.0] - 2026-08-11

Large internal trait-ladder refactor (52 commits since 0.3.2). The
worker-pool axis (`Base`/`SuspendedToken`/`RunQueue`/`Lookup`/`Worker`) and
the task-descriptor-pooling axis (`Desc`/`Pool`/`AsyncPool`/`ExternalQueue`)
are now independent traits (`WorkerSystem` / `PoolSystem`) instead of one
combined `SchedulerSystem`, and `scoped::ScopedTaskSystem` implements
`WorkerSystem` directly — the first non-`resumable` implementation, proving
the axis is genuinely swappable rather than declared-but-single-implementation.

### Breaking

- **`ThreadSystem` removed**, split into independent capability traits:
  - `SpawnableStackfulTaskSystem` — `spawn`/`yield_now` (what `ThreadSystem`
    used to require of everyone).
  - `StackfulSyncSystem` — `Mutex`/`Barrier`.
  - `BlockOnSystem` — `block_on`.
  - `DelegationSystem` — `Delegator` (RDMA-style request delegation).
  - `NestableSystem` — `ThreadSpecific` (TLS, needed to nest one system
    inside another).

  Code that did `use cmpth::ThreadSystem;` and called `.spawn()`/`.join()`
  only needs `SpawnableStackfulTaskSystem` now. Code that also used
  `block_on`/`Mutex`/`Barrier`/`Delegator`/nesting needs the matching trait(s)
  imported alongside it.

- **`WorkerDeque` replaced by `WorkerRunQueue`.** The old trait's
  position-based `push_top`/`push_bottom`/`try_pop_top`/`try_steal_bottom`
  is replaced by intent-based `push` (run next)/`defer` (run after
  already-queued work)/`try_pop`/`try_steal`. `CrossbeamDeque` and
  `SpinDeque` are renamed to `HybridRunQueue` and `SpinRunQueue`. This also
  fixes stackful `yield_now()`, which previously silently degraded to LIFO
  re-queueing because no lock-free deque could actually implement
  `push_bottom` (see `HybridRunQueue`'s two-queue design).

- **`Worker` trait narrowed and renamed to `WorkerOps`.** It now only
  exposes `current()`. The descriptor-typed accessors it used to carry
  (`cur_task`, `external_queue`, `polling_async`, …) moved to an
  internal-only `DescWorkerOps` trait (not part of the public API), since a
  `WorkerSystem` with no pooled-descriptor concept (like `scoped`) has no
  use for them.

- **`DescPool::alloc` returns an owned `SuspendedTaskToken<D>` instead of a
  raw `*mut D`.** Affects anyone implementing a custom `DescPool`.

- **`run(num_workers, f)` free function replaced by a builder.** Use
  `StackfulBuilder`/`StacklessBuilder`, or `StackfulInitSystem`/
  `StacklessInitSystem` for standalone (non-blocking) pool initialization.
  ULT stack size is now configurable through the same builder.

- **`StackfulOnlyResumable` renamed to `StackfulOnlyResumableCore`.**

- **`ExternalQueue::on_start` removed.** A queue that needs a dedicated
  poller (`PollerUltQueue`) now declares `NEEDS_SERVICE = true` and
  implements `run_service`/`stop_service`; the pool's own `init` spawns and
  joins it as an ordinary task instead of the queue managing its own thread
  lifecycle.

- **`SchedulerSystem` split into `WorkerSystem` + `PoolSystem`.** Anyone
  implementing `SchedulerSystem` by hand (the documented escape hatch for
  fully custom policy composition) now implements both narrower traits
  instead of one combined one.

### Added

- `available_parallelism()` helper.
- `scoped::ScopedTaskSystem` now implements `WorkerSystem`, sharing the
  worker-pool/deque/lookup infrastructure with `resumable`'s stackful and
  stackless engines instead of a fully separate implementation. Task
  representation (`TaskRef`, stack-resident, no pooled allocation) is
  unchanged — this integration does not affect `parallel_call`'s
  measured performance advantage.
  `TaskExitSink`, `HandoffTaskDesc`, `WakerTaskDesc` — replace the previous
  `JoinState`/`WakeOutcome` enum-returning join protocol with a
  method-based one that doesn't expose flavor-irrelevant variants to
  descriptors that can't reach them.

### Fixed

- Stackless `yield_now()` was a no-op in some paths; it now actually
  yields.
- A `PollerUltQueue` hang (a pre-existing defect, not a regression from
  this release) is fixed; its previously-`#[ignore]`d regression test now
  runs.

## [0.3.2] and earlier

Not tracked in this file. See `git tag` / `git log` for history up to
`v0.3.2`.
