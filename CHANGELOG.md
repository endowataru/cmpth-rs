# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
but this crate is **experimental**: the API may change between releases
without a deprecation cycle (see README "Status").

## [0.5.0] - 2026-11-15

A system is now composed from more axes, each a trait with a default:
what idle workers do (with a parking policy that searches the way Go's
and Tokio's schedulers do), victim selection, dispatch order, duties,
observers and per-worker hooks; `impl_with_defaults!` lets a system name
only what it changes.  The delegator interface is redesigned around
requests and `Poll`, with new delegators beside it; the context switch is
inline assembly with no build script; and duties and post delegators
serve work that only polling can drive (61 commits since 0.4.0).

### Breaking

- **Systems name more axes.** `WorkerSystem` and the config traits gain
  `type Idle: IdlePolicy`, `type Steal: StealPolicy`,
  `type Dispatch: DispatchPolicy`, `type Duties: DutyDomain`,
  `type Observe: SchedObserver` and `type Hooks: WorkerHooks`;
  `UltConfig` and `UltDualConfig` also take `ExternalQueue` and
  `PostConfig`, `UltAsyncConfig` `ExternalQueue`.  Wrap a config impl in
  `impl_with_defaults!` to get the defaults for what it leaves out.  The
  old behaviour is `SpinIdle`, `SweepSteal`, `StandardDispatch`,
  `GlobalDuties`, `NoObserver` and `NoHooks`; the default steal policy is
  now `RandomSweepSteal` (see Changed).  A hand-written `LocalQueue` has
  one more method.
- **Config traits renamed.** `UltIdentity` and `UltAsyncIdentity` are
  `UltConfig` and `UltAsyncConfig`.  Names now follow
  one rule (see README): `*Config` for traits that choose types and
  constants, `*Options` for run-time value structs, `*Policy` for
  behaviour the scheduler calls.
- **A config type gets its TLS anchor through `WorkerTlsAnchor`**, an
  unsafe supertrait of the config traits that
  `cmpth::worker_tls_anchor!(T)` implements; the `worker_tls_anchor`
  member of the config traits is gone.
- **The dual systems are built through the new `UltDualConfig`.**
  `DefaultDualTaskSystem` and `DefaultNestedDualTaskSystem` are aliases
  of `UltDualSystem<M>` over hidden markers instead of structs.  Code
  that names them as types or calls their functions is unchanged; code
  that used one as a value or implemented a foreign trait for it is not.
- **`DefaultDualTaskSystem`'s `SuspendedThread` is `DualResumable`**, so
  its delegator, mutex and barrier can be shared with plain OS threads.
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
- **The stackful mutex, condvar and barrier are written against traits.**
  `Mutex`/`MutexCore`, `McsMutex`/`McsCondvar` and `Barrier`/`BarrierCore`
  take `S: StackfulParking` instead of the resumable engine, so a system
  with a scheduler of its own gets them by implementing the traits.
  `MutexCore` and `BarrierCore`'s associated type
  `StackfulSchedulerSystem` is renamed `System`.
- **`ContextPolicy` callbacks are traits that own the switch-out.** The
  `SwitchFn`/`CondSwitchFn` function-pointer types are replaced by one
  callback trait per switch primitive (`SwitchFnLike`, `SwapFnLike`,
  `CondSwitchFnLike`, `RestoreFnLike`); `EntryFn` stays.  A callback
  lands on its destination itself through the new `ContextPolicy::land`
  instead of returning a `Transfer` for the primitive to act on (only a
  conditional switch's callback returns, and only to cancel), so
  `CondTransfer` is removed.  Only custom `ContextPolicy` implementations
  and their direct callers are affected.
- **New system-level traits `DutySystem` and `PostingSystem`.** Every
  system in the crate implements them, as do the config traits' blanket
  impls; a fully hand-written system implements them to use duties and
  post delegators.

### Added

- **Idle policies** (`IdlePolicy`, in `traits`, over an `IdleContext`):
  `SpinIdle` (the previous behaviour, compiled away), `ParkIdle` (block
  the OS thread; `FutexWait` on Linux futex / macOS
  `os_sync_wait_on_address`, or `CondvarWait`), and
  `SearchIdle<C: SearchCap = GoCap>`, the standard parking policy: at most
  `C::cap` idle workers search (`GoCap`: half the busy workers, Go's rule;
  `HalfCap`: half the workers, Tokio's; `FixedCap<N>`) and the others park
  at once.  A searcher spins for twice its average idle period when that
  is below 100 us; producers wake a sleeper only when nobody searches.
- `ParkDualTaskSystem` (`SearchIdle`) and `ParkHalfDualTaskSystem`
  (`SearchIdle<HalfCap>`): `DefaultDualTaskSystem` with parking workers.
- **Steal policies** (`StealPolicy`): `SweepSteal` (every other worker
  once, starting one place further each round), `RandomSteal` (one
  random victim per round) and `RandomSweepSteal` (every other worker
  once from a random start).
- **Dispatch policies** (`DispatchPolicy`): the order of the worker's own
  queue, a steal round and the external queue, one policy for the three
  places a worker picks a task (`Site::Loop`, `Site::Switch`,
  `Site::Wait`).  `StandardDispatch` keeps the previous orders.
- **Scheduler observers** (`SchedObserver`): steal rounds, dispatches,
  idle rounds and parks; `NoObserver` compiles to nothing.
- **Worker hooks** (`WorkerHooks`): `start` runs on a worker's thread
  before it first looks for work and returns a `State` that `stop` takes
  back after it last did.  `NoHooks` (the default), `PinWorkers` (pins
  worker i to the i-th CPU the process may use, Linux) and
  `BackgroundWorkers` (macOS background QoS: efficiency cores at a low
  clock; on fib with 12 workers about a ninth of the energy for 5.5 times
  the time).
- `impl_with_defaults! { impl UltConfig for T { ... } }`: an ordinary
  impl block that gets the documented default for each type it leaves
  out, for `UltConfig`, `UltDualConfig`, `UltAsyncConfig` and
  `ScopedConfig`; a type the trait lacks is a compile error.
- `UltDualSystem<M: UltDualConfig>` and `ScopedSystem<M: ScopedConfig>`.
  `ScopedTaskSystem` is now an alias for `ScopedSystem<DefaultScopedMarker>`
  (`SpinIdle`), so existing code compiles unchanged.
- `LeanFrameContext`, a second `ContextPolicy`: it declares callee-saved
  registers as clobbers instead of saving them all, which wins on
  switch-heavy loops (about 9-13 % on a spawn/join microbenchmark) and
  loses narrowly on register-heavy recursion (1-2 % on fib/nqueens).
  `NativeContext` stays the default.
- **Delegators:** `CombiningDelegator`, with no consumer thread: the lock
  holder runs the requests queued behind it, so it works before any
  worker pool exists and for OS threads.  `CombiningDelegator<S, C, K>`
  takes a `CombiningConfig`; a nonzero `K::BOUND` makes the holder hand
  the lock to the next queued caller after serving that many requests of
  others (`DefaultCombining`: 0, no bound).
- Plain OS threads (not workers) can wait on `DualResumable`;
  `OsSuspendedThread` implements `StackfulResumable`, so the delegators
  also run on `OsSystem`.
- **Duties** (`DutySystem`, `DutyService`, `DutyLike`): work that must be
  checked every so often because nothing announces its result.  A duty is
  an obligation to run at least every period (`S::duty`, e.g. a
  transport's progress engine serving other processes), a wait
  (`S::on_demand`, `DutyLike::waiter`: run every wait period while tasks
  wait on it) or a deadline (`DutyLike::set_deadline`; `S::sleep`
  suspends a task until a timer duty wakes it).  A duty is run, cheapest
  first, by whoever makes the progress anyway (`DutyLike::touch`), by
  idle workers once it is half due, or by a keeper thread asleep until
  the earliest due time; `DutyPlace::Keeper` keeps blocking polls off the
  workers.  Busy workers never check.  While tasks wait on a duty, an
  idle worker searches for a while before it parks, and one parked worker
  sleeps until the duties' next due time so the keeper need not wake.
  A system's `Duties` picks the registry: `GlobalDuties` (the default,
  process-wide; `CMPTH_WAIT_SPIN_US` sets the search time, default 100,
  and `CMPTH_DUTY_QOS=background` lowers the keeper's QoS), `NoDuties`, or
  a static `DutyRegistry::new(DutyOptions)` with its own keeper.  Scoped
  tasks waiting on a join run the duties too.
- **Post delegators** (`PostDelegator`, `PostConsumer`, in
  `resumable::common::sync::post`, re-exported at the root): a post asks
  a consumer to catch up with everything the posting thread did so far
  and returns at once; the consumer serves later, once for many posts,
  and reports the `Ticket`s it covered.  The consumer states which cost
  to keep low (`FavorPosting` or `FavorServing`) and the system's
  `PostConfig` maps that to `SlotPostDelegator` (a post is a store to the
  thread's own slot; the server polls every slot) or `QueuePostDelegator`
  (the first post after a serve lists the slot and asks for a poll; an
  idle delegator never polls).  `PostDelegatorOf<S, C>` names the pick;
  `PostOptions` sets its latency and place.
- `MallocFrame`, a `StackAlloc` from the global allocator without guard
  pages, for stackless frames.
- `bench/`: `ctxbench` (context primitives) and a microbenchmark of one
  delegated call.

### Changed

- **The context switch is inline assembly** (`asm!`) on both AArch64 and
  x86-64, inlined into its call sites, and `make_context` is plain Rust.
  `asm/` and the `cc` build dependency are gone: no build script and no
  assembler are needed.
- The default steal policy is `RandomSweepSteal` instead of the fixed
  sweep: interleaved fib runs at 2/6/12 workers showed no consistent
  difference among the three policies, and it keeps every round a full
  search, which the parking protocol's last check relies on.
- Pool-backed `parallel_call` for `ScopedStackfulTaskSystem` and
  `ScopedStacklessTaskSystem` on the ULT systems (stackful-only,
  stackless-only and dual): `a` runs inline, `b` is a real pushed task;
  an un-stolen `b` is popped back and run as a plain call, with no context
  switch and no join protocol.
- The scoped engine shares its worker core, pool core and idle loop with
  `resumable` instead of keeping its own.
- The dual systems' stackless frames come from `MallocFrame`: about
  0.8 KiB per waiting task instead of 16.7 KiB (a guard-paged page per
  frame on Apple silicon), with no measurable change in spawn/await time.
  Spawned ULTs keep guard-paged stacks.
- `WorkerRunQueue`/`RunQueueStealer`/`Steal`, `IdlePolicy`, `StealPolicy`,
  `DispatchPolicy` and `SchedObserver` moved to `traits::component`;
  still re-exported from their old `resumable::common` modules.
- Workers take tasks woken from outside the pool before their own queue,
  not only when idle.

### Fixed

- Stackless `parallel_call` with a self-recursive `b` failed to compile
  (an infinitely sized future; `b` is now a pushed task).
- `McsDelegator`: releasing the lock to a caller waiting for it parked the
  releasing thread in the consumer's slot; a caller whose call was kept
  could be left asleep with the lock held (`unlock_and_wait` now follows
  the C++ reference); a caller completed by `progress` just before the
  consumer went idle was never woken.
- `RingBufDelegator`: the lock was never released and the holder never
  found its successor; a caller could queue behind a holder that had
  already released the lock, and sleep forever.
- `OsDelegator`: it crashed once moved (e.g. into an `Arc`), since its
  consumer thread kept the old address; a caller that missed the lock was
  never woken; the consumer held the lock while waiting, so every call
  went to the consumer thread.  Calls now run inline under the mutex.
- A task woken by an OS thread never ran while its worker's tasks kept
  yielding to each other (it waited in the external queue, which only the
  idle path drained); a lock handed to it from an OS thread deadlocked
  everything behind it.

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
