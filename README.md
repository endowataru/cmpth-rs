# ComposableThreads (cmpth)

A trait-composable user-level threading (ULT) library for Rust, with
work-first (child-first) fork/join scheduling and work stealing.
A Rust reimplementation of the C++ library
[ComposableThreads](https://doi.org/10.2197/ipsjjip.30.269)
(Endo, Sato, Taura: *ComposableThreads: Rethinking User-level Threads with
Composability and Parametricity in C++*, JIP 2022).

> **Status: experimental.**  The library is under active development and
> the API may change between releases without a deprecation cycle.

```rust
use cmpth::default::{run, spawn};

fn fib(n: u64) -> u64 {
    if n <= 1 { return n; }
    let h = spawn(move || fib(n - 1)); // child runs first; parent is stealable
    let r2 = fib(n - 2);               // parent continues here (or on a thief)
    h.join().unwrap() + r2
}

fn main() {
    run(4, || assert_eq!(fib(34), 5_702_887));
}
```

## What it is

- **True user-level threads.** Every task has its own stack.  Tasks can
  block — in a mutex, a barrier, a `join`, or an `.await` — without
  blocking the OS thread underneath, so recursive fork/join and blocking
  synchronization compose freely.
- **Work-first scheduling.** `spawn` switches to the child immediately and
  publishes the parent's continuation for stealing (the Cilk discipline),
  which keeps the working set depth-first and bounds memory.
- **Low overhead.** A spawn+join round-trip is ~25 ns and a context switch
  ~7 ns on Apple Silicon (see `bench/` for the benchmark suite).

## Trait-based composition

Every axis of variation is a trait, chosen per system via associated types:

```rust
cmpth::ult_system! {
    pub struct MySystem {
        base:       cmpth::OsSystem,        // what the workers run on
        context:    cmpth::NativeContext,   // context-switch implementation
        deque:      cmpth::CrossbeamDeque,  // work-stealing deque
        stack_size: 64 * 1024,
    }
}
```

Because every `UltSystem` is itself a `ThreadSystem`, schedulers **nest**:
set `base: MySystem` in a second system and it runs ULTs on top of ULTs.
Nesting doubles as a correctness check for the abstraction boundaries — the
same code must work at every level.

The key internal design, inherited from ComposableThreads: every context
switch takes a callback that runs *after* the switch, on the destination
stack.  A suspended task's continuation therefore only comes into existence
once its context is fully saved, which eliminates "saving in progress"
flags and spin-wait handshakes throughout the scheduler.

## Features

- `spawn` / `JoinHandle` (also usable as a `Future`), detach on drop
- `spawn_async`: run a `Future` as a task without allocating a stack
- `block_on` integration for driving futures from ULT context
- Mutex, condvar, barrier, and MCS-based delegation primitives, all generic
  over the threading system
- Pluggable task-descriptor pools (`SimplePool`, `ReturnPool`)
- Wakers callable from external (non-worker) OS threads

## Performance

On an Apple M-series core (1 worker), a full spawn+join round-trip of an
empty task takes ~25 ns and fib(34) — 9,227,464 fork/join pairs — runs in
~335 ms (~36 ns per pair).  This is the cost of *real, suspendable stacks*:
tasks may block in mutexes, joins, or `.await` without stalling the OS
thread underneath.  The workspace includes a benchmark harness (`bench/`)
comparing against other threading runtimes behind a common trait.

## Platform support

x86-64 and AArch64, on macOS and Linux (hand-written context-switch
assembly for both, with the AArch64 v8–v15 callee-saved contract enforced
via inline-asm clobbers).  Requires Rust 1.85+ (edition 2024).

## License

Licensed under the Apache License, Version 2.0
([LICENSE-APACHE](LICENSE-APACHE)).
