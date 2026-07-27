//! Test-only convenience shim bound to [`DefaultDualTaskSystem`], mirroring
//! what `cmpth::default` used to provide as public API. Kept here instead of
//! in the crate: it's genuinely useful for terse test bodies, but a
//! trait-based library shouldn't ship an opinionated "the default" entry
//! point in its public API when several equally-valid default systems exist
//! (`DefaultStackfulOnlyTaskSystem`, `DefaultStacklessOnlyTaskSystem`, …).

#![allow(dead_code)]

use cmpth::{DefaultDualTaskSystem, ScopedStackfulTaskSystem, ThreadSystem};

pub fn run<F, R>(num_workers: usize, root: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    DefaultDualTaskSystem::run(num_workers, root)
}

pub fn spawn<T, F>(f: F) -> <DefaultDualTaskSystem as ThreadSystem>::JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    DefaultDualTaskSystem::spawn(f)
}

/// Spawn a `Future` as a stackless task from stackful ULT context, blocking
/// until the spawn itself completes (not the task) and returning a handle
/// that can be `.join()`ed like any other.
pub fn spawn_async<T, F>(f: F) -> <DefaultDualTaskSystem as ThreadSystem>::JoinHandle<T>
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    DefaultDualTaskSystem::block_on(cmpth::resumable::stackless::thread::spawn_async::<
        DefaultDualTaskSystem,
        T,
        F,
        _,
    >(move || f))
}

pub fn yield_now() {
    DefaultDualTaskSystem::yield_now();
}

pub type Mutex<T> = cmpth::McsMutex<DefaultDualTaskSystem, T>;
pub type Condvar = cmpth::McsCondvar<DefaultDualTaskSystem>;
pub type Barrier = cmpth::UltBarrier<DefaultDualTaskSystem>;
