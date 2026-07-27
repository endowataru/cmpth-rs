# ComposableThreads (cmpth)

cmpth is a Rust parallelism library that provides **three** ways to
express parallel work — **stackful** (real user-level threads),
**stackless** (`spawn_async`/`.await`), and **scoped** (a Rayon-`join`-like
binary primitive) — built from the same small set of trait-based
components (context-switch policy, work-stealing deque, stack allocator,
…). Use one of the ready-made systems as-is, or swap out individual
components to build your own. It's designed carefully to be competitive
with other Rust parallelism runtimes (see [Performance](#performance)).

A Rust reimplementation of the C++ library
[ComposableThreads](https://doi.org/10.2197/ipsjjip.30.269)
(Endo, Sato, Taura: *ComposableThreads: Rethinking User-level Threads with
Composability and Parametricity in C++*, JIP 2022).

> **Status: experimental.**  The library is under active development and
> the API may change between releases without a deprecation cycle.

## Three parallelization models

All three models share the same worker pool and work-stealing deque; they
differ in what a "task" is and how the scheduler waits for one.

### Stackful — real user-level threads

Every task gets its own stack, so it can block — in a mutex, a barrier, a
blocking `join`, or an `.await` via `block_on` — without stalling the OS
thread underneath. `spawn` uses *work-first* (child-first) scheduling: it
switches to the child immediately and publishes the parent's continuation
for stealing (the Cilk discipline), which keeps the working set depth-first
and bounds memory.

```rust
use cmpth::DefaultStackfulOnlyTaskSystem;
use cmpth::traits::stackful::*;

fn fib<S: ThreadSystem>(n: u64) -> u64 {
    if n <= 1 { return n; }
    let h = S::spawn(move || fib::<S>(n - 1)); // child runs first; parent is stealable
    let r2 = fib::<S>(n - 2);                  // parent continues here (or on a thief)
    h.join() + r2
}

fn main() {
    DefaultStackfulOnlyTaskSystem::run(4, || {
        assert_eq!(fib::<DefaultStackfulOnlyTaskSystem>(34), 5_702_887);
    });
}
```

`DefaultStackfulOnlyTaskSystem` is cmpth's ready-made stackful system — see
[Trait-based components](#trait-based-components-not-monoliths) below for
why the example calls it through `S: ThreadSystem` rather than naming
`DefaultStackfulOnlyTaskSystem` inside `fib` itself. `cmpth::traits::stackful::*` /
`cmpth::traits::stackless::*` pull in every trait for that model in one
line, so you don't have to track down which single trait a given method
lives on.

Spawned tasks aren't limited to fork/join — they can block on a shared
`Mutex` too, since each one has its own stack to suspend:

```rust
use cmpth::DefaultStackfulOnlyTaskSystem;
use cmpth::traits::stackful::*;
use std::sync::Arc;

fn sum_concurrently<S: ThreadSystem>(n: u64) -> u64 {
    let total = Arc::new(S::Mutex::new(0u64));
    let handles: Vec<_> = (0..n)
        .map(|i| {
            let total = Arc::clone(&total);
            S::spawn(move || *total.lock() += i) // blocks the ULT, not the OS thread
        })
        .collect();
    for h in handles {
        h.join();
    }
    *total.lock()
}

fn main() {
    DefaultStackfulOnlyTaskSystem::run(4, || {
        assert_eq!(sum_concurrently::<DefaultStackfulOnlyTaskSystem>(100), (0..100).sum());
    });
}
```

Need both `spawn`/`.join()` *and* `spawn_async`/`.await` live on the same
tasks (e.g. a `Mutex` contended from both stackful and stackless callers)?
`DefaultDualTaskSystem` provides both calling conventions on one system —
the tradeoff, covered next, is a small per-task dispatch cost neither
single-flavor default pays.

### Stackless — `spawn_async` / `.await`

A task is a `Future`, driven by polling in place: no stack allocation, no
context switch per poll. `DefaultStacklessOnlyTaskSystem` is the ready-made
system for this model — no separate setup needed, and genuinely no stackful
capability at all (no context-switch policy, no stack allocator), so it
skips the dual-flavor dispatch `DefaultDualTaskSystem` pays on every task
(~10–15% faster on a pure-`spawn_async` workload):

```rust
use cmpth::DefaultStacklessOnlyTaskSystem;
use cmpth::traits::stackless::*;

fn fib<S: StacklessTaskSystem>(n: u64) -> impl std::future::Future<Output = u64> + Send {
    async move {
        if n <= 1 { return n; }
        let h = S::spawn(move || fib::<S>(n - 1)).await; // child runs concurrently
        let r2 = S::recurse(move || fib::<S>(n - 2)).await; // parent continues in place
        h.await + r2
    }
}

fn main() {
    DefaultStacklessOnlyTaskSystem::run_async(4, async {
        assert_eq!(fib::<DefaultStacklessOnlyTaskSystem>(34).await, 5_702_887);
    });
}
```

`DefaultDualTaskSystem` still works for `spawn_async` too, if a system needs both
calling conventions live at once — it's just the wrong default when a
system is only ever going to be async.

### Scoped — binary divide-and-conquer

`parallel_call` runs two closures, potentially in parallel, and returns
once both finish — like Rayon's `join`. Unlike `spawn`, the caller's own
continuation is never exposed as stealable work (nothing outlives the
call), so there's no task descriptor, no pool, and no heap allocation on
the un-stolen path — the cheapest of the three models.

```rust
use cmpth::ScopedStackfulTaskSystem;

fn fib<S: ScopedStackfulTaskSystem>(n: u64) -> u64 {
    if n <= 1 { return n; }
    let (a, b) = S::parallel_call(move || fib::<S>(n - 1), move || fib::<S>(n - 2));
    a + b
}

fn main() {
    let r = cmpth::ScopedTaskSystem::run(4, move || fib::<cmpth::ScopedTaskSystem>(34));
    assert_eq!(r, 5_702_887);
}
```

**Which one?** Stackful when tasks need to block on locks, barriers, or
other blocking work alongside spawned tasks (add `DefaultDualTaskSystem`
instead of `DefaultStackfulOnlyTaskSystem` if the same tasks also need
`spawn_async`). Stackless when integrating with existing `async`/`.await`
code, or to avoid paying for a stack per task. Scoped when the parallelism
is a pure recursive divide-and-conquer with no blocking involved — it has
the lowest overhead of the three.

## Trait-based components, not monoliths

Every axis of variation — context-switch policy, work-stealing deque,
stack allocator, and more — is a separate trait, not a hardcoded choice. A
concrete "system" is a struct that picks one type per axis via associated
types:

```rust
pub struct MySystem;

impl cmpth::UltIdentity for MySystem {
    type Base = cmpth::OsSystem;                          // what the workers run on
    type Ctx = cmpth::NativeContext;                       // context-switch implementation
    type Deque = cmpth::CrossbeamDeque<cmpth::BasicTaskDesc>; // work-stealing deque
    type Alloc = cmpth::HeapStack;                         // stack allocator
    type Lookup = cmpth::TlsCurrent;                       // current-worker lookup

    fn worker_tls_anchor() -> &'static <cmpth::OsSystem as cmpth::ThreadSystem>::ThreadSpecific<cmpth::UltWorker<Self>> {
        static A: cmpth::TlsAnchor = cmpth::TlsAnchor::new();
        cmpth::TlsSlot::from_anchor(&A)
    }
}
```

`UltIdentity` builds a stackful system this way (`DefaultStackfulOnlyTaskSystem`
is wired up exactly like `MySystem` above, just with `HeapStack`/`TlsCurrent`);
`UltAsyncIdentity` does the same for a stackless-only system. Both are config
traits — implement one for your own marker type and a blanket impl (written
inside `cmpth`) supplies `SchedulerSystem`/`StackfulSchedulerSystem`/`ThreadSystem`.
Both are shorthand for hand-writing the underlying trait implementations
yourself — the escape hatch for when their fixed shape doesn't fit, since
every component they wire up is a public trait you can implement directly.

Every stackful system implements `ThreadSystem` directly, so schedulers
**nest**: set `type Base = MySystem` in a second system and it runs ULTs
on top of ULTs. Nesting doubles as a correctness check for the abstraction
boundaries — the same code must work at every level.

All three models' traits share one root, `TaskSystem` — the declaration
that a system provides an efficient (work-stealing) scheduler as its
execution model. `ThreadSystem` (stackful spawn/join),
`ScopedStackfulTaskSystem`/`ScopedStacklessTaskSystem` (`parallel_call`),
and `StacklessTaskSystem` (`spawn`/`recurse`) each build on it with
their own capability. `StackfulTaskSystem`/`StacklessTaskSystem` are
empty *bundles* on top of those — "everything a complete stackful/
stackless system offers" as one bound — blanket-derived automatically
for any system with the right pieces; nothing implements them by hand.

The key internal design, inherited from ComposableThreads: every context
switch takes a callback that runs *after* the switch, on the destination
stack.  A suspended task's continuation therefore only comes into existence
once its context is fully saved, which eliminates "saving in progress"
flags and spin-wait handshakes throughout the scheduler.

### Program against the trait, not the concrete system

All three quickstart examples above already follow this rule: each `fib`
is generic over a trait (`S: ThreadSystem`, `S: StacklessTaskSystem`, `S:
ScopedStackfulTaskSystem`), and the concrete system
(`DefaultStackfulOnlyTaskSystem`, `DefaultStacklessOnlyTaskSystem`,
`ScopedTaskSystem`) only ever appears at the single "pick a system" call
site (`main`), e.g. `S::run(4, move || fib::<S>(34))` invoked with `S =
MySystem`. Naming a concrete system inside code that isn't itself that
call site — e.g. calling `cmpth::ScopedTaskSystem::parallel_call(...)`
directly from inside `fib`'s own body instead of `S::parallel_call(...)`
— locks that code to one scheduler, defeating the entire point of the
trait-based composition above: write reusable library code the same way
the examples do, not that way.

## Features

- `spawn` / `JoinHandle` (also usable as a `Future`), detach on drop —
  stackful
- `spawn_async`: run a `Future` as a task without allocating a stack —
  stackless
- `parallel_call`: binary divide-and-conquer with no task descriptor and
  no heap allocation on the un-stolen path — scoped, in both a blocking
  (`ScopedStackfulTaskSystem`) and an `.await`-based
  (`ScopedStacklessTaskSystem`) flavor
- `block_on` integration for driving futures from ULT context
- Mutex, condvar, barrier, and MCS-based delegation primitives, all generic
  over the threading system
- Pluggable task-descriptor pools (`SimplePool`, `ReturnPool`)
- Wakers callable from external (non-worker) OS threads

## Performance

cmpth is designed carefully to be competitive with other Rust parallelism
runtimes: the context-switch and spawn/join paths are hand-tuned, and the
`scoped` model in particular avoids any task descriptor or heap
allocation on the un-stolen path — the same technique Rayon's `join`
uses.

That said, this hasn't yet been benchmarked rigorously at scale — what
exists so far is spot-checking on individual machines, not a systematic
study. The workspace includes a benchmark harness (`bench/`) comparing
all three models against other threading and async runtimes (Rayon,
Tokio, MassiveThreads, …) behind common traits, for anyone who wants to
check for themselves.

## Platform support

x86-64 and AArch64, on macOS and Linux (hand-written context-switch
assembly for both, with the AArch64 v8–v15 callee-saved contract enforced
via inline-asm clobbers).  Requires Rust 1.85+ (edition 2024).

## License

Licensed under the Apache License, Version 2.0
([LICENSE-APACHE](LICENSE-APACHE)).
