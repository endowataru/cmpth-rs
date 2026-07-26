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
use cmpth::DualTaskSystem;
use cmpth::traits::stackful::*; // ThreadSystem, JoinHandleLike, StackfulTaskSystem, ...

fn fib<S: ThreadSystem>(n: u64) -> u64 {
    if n <= 1 { return n; }
    let h = S::spawn(move || fib::<S>(n - 1)); // child runs first; parent is stealable
    let r2 = fib::<S>(n - 2);                  // parent continues here (or on a thief)
    h.join() + r2
}

fn main() {
    DualTaskSystem::run(4, || assert_eq!(fib::<DualTaskSystem>(34), 5_702_887));
}
```

`DualTaskSystem` is cmpth's ready-made stackful system — see
[Trait-based components](#trait-based-components-not-monoliths) below for
why the example calls it through `S: ThreadSystem` rather than naming
`DualTaskSystem` inside `fib` itself. `cmpth::traits::stackful::*` /
`cmpth::traits::stackless::*` pull in every trait for that model in one
line, so you don't have to track down which single trait a given method
lives on.

### Stackless — `spawn_async` / `.await`

A task is a `Future`, driven by polling in place: no stack allocation, no
context switch per poll.

```rust
use cmpth::traits::stackless::*;

cmpth::ult_async_system! {
    struct MyAsyncSystem {
        base:  cmpth::OsSystem,
        deque: cmpth::CrossbeamDeque<cmpth::BasicTaskDesc>,
    }
}

fn main() {
    MyAsyncSystem::run_async(2, async {
        let h = MyAsyncSystem::spawn(|| async { 6 * 7 }).await;
        assert_eq!(h.await, 42);
    });
}
```

### Scoped — binary divide-and-conquer

`parallel_call` runs two closures, potentially in parallel, and returns
once both finish — like Rayon's `join`. Unlike `spawn`, the caller's own
continuation is never exposed as stealable work (nothing outlives the
call), so there's no task descriptor, no pool, and no heap allocation on
the un-stolen path — the cheapest of the three models.

```rust
use cmpth::ScopedStackfulTaskSystem as _;

fn fib(n: u64) -> u64 {
    if n <= 1 { return n; }
    let (a, b) = cmpth::ScopedTaskSystem::parallel_call(move || fib(n - 1), move || fib(n - 2));
    a + b
}

fn main() {
    let r = cmpth::ScopedTaskSystem::run(4, move || fib(34));
    assert_eq!(r, 5_702_887);
}
```

**Which one?** Stackful when tasks need to block on locks, barriers, or
other blocking work alongside spawned tasks. Stackless when integrating
with existing `async`/`.await` code, or to avoid paying for a stack per
task. Scoped when the parallelism is a pure recursive divide-and-conquer
with no blocking involved — it has the lowest overhead of the three.

## Trait-based components, not monoliths

Every axis of variation — context-switch policy, work-stealing deque,
stack allocator, and more — is a separate trait, not a hardcoded choice. A
concrete "system" is a struct that picks one type per axis via associated
types:

```rust
cmpth::ult_system! {
    pub struct MySystem {
        base:       cmpth::OsSystem,        // what the workers run on
        context:    cmpth::NativeContext,   // context-switch implementation
        deque:      cmpth::CrossbeamDeque<cmpth::BasicTaskDesc>,  // work-stealing deque
        stack_size: 64 * 1024,
    }
}
```

`ult_system!` builds a stackful system this way; `ult_async_system!` does
the same for a stackless-only system. Both are shorthand for hand-writing
the underlying trait implementations yourself — the escape hatch for when
a macro's fixed shape doesn't fit, since every component the macros wire
up is a public trait you can implement directly.

Every stackful system implements `ThreadSystem` directly, so schedulers
**nest**: set `base: MySystem` in a second system and it runs ULTs on top
of ULTs. Nesting doubles as a correctness check for the abstraction
boundaries — the same code must work at every level.

All three models' traits share one root, `TaskSystem` — the declaration
that a system provides an efficient (work-stealing) scheduler as its
execution model. `ThreadSystem` (stackful spawn/join),
`ScopedStackfulTaskSystem`/`ScopedStacklessTaskSystem` (`parallel_call`),
and `StacklessTaskSystem` (`spawn_async`/`recurse`) each build on it with
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

The stackful example above already follows this rule: `fib` is generic
over `S: ThreadSystem`, and `DualTaskSystem` only ever appears at the
single "pick a system" call site (`main`). The stackless and scoped
examples take a shortcut and call `MyAsyncSystem`/`ScopedTaskSystem`
directly throughout, for brevity — fine for a quickstart, but write
reusable library code the stackful example's way: generic over the trait
(`S: ThreadSystem`, `S: ScopedStackfulTaskSystem`, or `S:
StackfulTaskSystem` for both together, etc.), never against a concrete
system directly. Naming a concrete system inside code that isn't itself
the "pick a system" call site locks that code to one scheduler,
defeating the entire point of the trait-based composition above:

```rust
// Good: generic over the trait, works with any ScopedStackfulTaskSystem system.
fn fib_good<S: cmpth::ScopedStackfulTaskSystem>(n: u64) -> u64 {
    if n <= 1 { return n; }
    let (a, b) = S::parallel_call(move || fib_good::<S>(n - 1), move || fib_good::<S>(n - 2));
    a + b
}

// Bad: hardcodes one scheduler; can't be reused with a different system.
use cmpth::ScopedStackfulTaskSystem as _;
fn fib_bad(n: u64) -> u64 {
    if n <= 1 { return n; }
    let (a, b) = cmpth::ScopedTaskSystem::parallel_call(move || fib_bad(n - 1), move || fib_bad(n - 2));
    a + b
}
```

The concrete system name should only ever appear at the top-level call
site that picks which system to run, e.g. `S::run(4, move || fib::<S>(34))`
invoked with `S = MySystem`.

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
