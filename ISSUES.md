# Known issues

## `yield_now()` does not always yield fairly (LIFO vs FIFO deque end)

**Where:** `src/ult/worker.rs`, `src/ult/deque.rs`, `src/ult/waker.rs`, `src/future.rs`

**Background:** the work-stealing deque has two ends: `push_top`/`try_pop_top`
(LIFO, used by the owning worker to resume/wake things with cache locality)
and `push_bottom`/`try_steal_bottom` (FIFO from the owner's perspective, used
so a voluntary yield lets already-queued local work run first — the same
technique MassiveThreads uses). `WorkerDeque::yield_now()` (stackful ULTs)
correctly calls `push_local_bottom`:

```rust
fn yield_now(&self) -> &Self {
    self.suspend_to_sched(|wk, prev| wk.push_local_bottom(prev))
}
```

**Problem 1 — `CrossbeamDeque` silently degrades `push_bottom` to `push_top`.**
`CrossbeamDeque` (the backend used by `DefaultUltSystem`) has no owner-side
"push to the steal end" operation (crossbeam's Chase-Lev deque doesn't expose
one), so its `push_bottom` impl falls back to calling the same code as
`push_top`. Under the default configuration, stackful `ThreadSystem::yield_now()`
therefore does **not** achieve fair FIFO yielding — the yielding ULT is
pushed LIFO and is likely to be immediately re-popped by the same worker,
ahead of older queued work. Only `SpinDeque` implements true opposite-end
semantics today.

**Problem 2 — `cmpth::future::yield_now()` never uses the bottom end at all.**
The async self-wake path (`UltWorker::run_async_poll`'s `NOTIFIED` branch,
and `waker.rs::push_continuation`) unconditionally calls `push_local_top`,
regardless of deque backend:

```rust
// worker.rs, run_async_poll, on Poll::Pending with state == NOTIFIED
self.push_local_top(SuspendedUlt(desc));
```

This code path has no way to distinguish "this was a deliberate cooperative
yield" (`cmpth::future::yield_now()`, which just does
`cx.waker().wake_by_ref(); Poll::Pending`) from any other kind of
synchronous self-wake during `poll()`. So `future::yield_now()` always
re-queues LIFO, even with `SpinDeque` — it never gets the fairness
`ThreadSystem::yield_now()` was designed to have. Functionally it still
works (tests pass), but it's closer to "a cheap self-repoll loop" than a
real yield: other already-queued local work does not get priority over it.

**Possible directions (not decided):**
- Give `SuspendedUlt`/the NOTIFIED-during-poll path a way to distinguish
  "yield" wakes from other self-wakes (e.g. a distinct flag/state, or a
  separate vtable) so `run_async_poll` can route deliberate yields to
  `push_local_bottom`.
- Fix `CrossbeamDeque::push_bottom` to be a real opposite-end push, or
  document/accept the degradation and steer users who need fair yielding
  toward `SpinDeque`.

Found 2026-07-18 while discussing the `WaitSlot`/`SuspendedTask`
sync+async unification design; not yet triaged for priority.
