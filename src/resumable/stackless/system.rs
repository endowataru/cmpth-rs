//! The blanket [`StacklessTaskSystem`]/[`ScopedStacklessTaskSystem`] impls
//! for every async-capable [`SchedulerSystem`], and the
//! [`ult_async_system!`](crate::ult_async_system) macro that generates a
//! stackless-only system.
//!
//! The trait declarations themselves live in [`crate::traits::system`]/
//! [`crate::traits::scoped`] (pure interface, no `resumable`-layer types in
//! their own signatures); this module only supplies the bodies, which is
//! where naming `SchedulerSystem` and concrete resumable types
//! (`JoinHandle`, `spawn_async`, `recurse`, `run_async`) is fine.

use std::future::Future;

use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackless::desc::AsyncTaskDesc;
use crate::traits::scoped::ScopedStacklessTaskSystem;

// `StacklessSystem`/`StacklessTaskSystem` now live in `crate::traits::system`
// — re-exported below for callers that still spell out
// `resumable::stackless::system::StacklessSystem`/`StacklessTaskSystem`.
pub use crate::traits::system::{StacklessSystem, StacklessTaskSystem};

impl<S: SchedulerSystem> StacklessTaskSystem for S
where
    S::Desc: AsyncTaskDesc,
{
    type SpawnHandle<T: Send + 'static> = crate::resumable::common::thread::JoinHandle<S, T>;

    fn spawn<T, F, Mk>(mk: Mk) -> impl Future<Output = Self::SpawnHandle<T>> + Send
    where
        F: Future<Output = T> + Send + 'static,
        Mk: FnOnce() -> F + Send + 'static,
        T: Send + 'static,
    {
        crate::resumable::stackless::thread::spawn_async::<Self, T, F, Mk>(mk)
    }

    fn recurse<F, Mk>(mk: Mk) -> impl Future<Output = F::Output> + Send
    where
        F: Future + Send,
        Mk: FnOnce() -> F,
    {
        crate::resumable::stackless::thread::recurse::<Self, F, Mk>(mk)
    }
}

/// `parallel_call` for a system that already has `StacklessTaskSystem`'s
/// `spawn`: strictly cheaper capability, satisfied trivially by spawning
/// one branch and awaiting the other inline — same relationship as
/// `resumable::stackful::system`'s `ScopedStackfulTaskSystem` blanket. Calls
/// the same `spawn_async` free function `StacklessTaskSystem::spawn`'s
/// blanket does directly (rather than going through `S::spawn`) so this
/// impl doesn't need a `StacklessTaskSystem` bound of its own — the two
/// blankets are independent, both satisfied by the same `SchedulerSystem +
/// AsyncTaskDesc` condition.
impl<S: SchedulerSystem> ScopedStacklessTaskSystem for S
where
    S::Desc: AsyncTaskDesc,
{
    fn run_async<F>(num_workers: usize, root: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        crate::resumable::stackless::scheduler::run_async::<Self, F>(num_workers, root)
    }

    async fn parallel_call<Fa, Fb, Ra, Rb, MkA, MkB>(mk_a: MkA, mk_b: MkB) -> (Ra, Rb)
    where
        MkA: FnOnce() -> Fa + Send + 'static,
        MkB: FnOnce() -> Fb + Send + 'static,
        Fa: Future<Output = Ra> + Send + 'static,
        Fb: Future<Output = Rb> + Send + 'static,
        Ra: Send + 'static,
        Rb: Send + 'static,
    {
        let h = crate::resumable::stackless::thread::spawn_async::<Self, Ra, Fa, MkA>(mk_a).await;
        let rb = mk_b().await;
        (h.await, rb)
    }
}

// ---------------------------------------------------------------------------
// ult_async_system! macro
// ---------------------------------------------------------------------------

/// Define a complete **stackless-only** ULT system in one declaration.
///
/// Unlike [`ult_system!`](crate::ult_system), the generated marker struct implements only
/// [`SchedulerSystem`] — never [`StackfulSchedulerSystem`](crate::resumable::stackful::system::StackfulSchedulerSystem), so it never names a
/// context-switch policy or stack allocator, because it has none. Its only
/// entry points are [`crate::resumable::stackless::scheduler::run_async`] (run) and
/// [`crate::resumable::stackless::thread::spawn_async`] (spawn); there is no `spawn`, no
/// `block_on`, no `ThreadSystem` impl for it via the `StackfulSystem` blanket
/// (that blanket requires stackful capability).
///
/// `Worker::execute`'s dispatch is [`crate::resumable::stackful::worker::execute_stackful`]
/// -shaped in spirit but for polling instead of switching: it always polls,
/// with no `poll_fn`-tag branch, because every task on this system is one.
///
/// ```
/// use cmpth::SuspendedUlt;
/// use cmpth::{ScopedStacklessTaskSystem, StacklessTaskSystem};
///
/// cmpth::ult_async_system! {
///     struct MyAsyncSystem {
///         base:  cmpth::OsSystem,
///         deque: cmpth::CrossbeamDeque<cmpth::BasicTaskDesc>,
///     }
/// }
///
/// MyAsyncSystem::run_async(2, async {
///     let h = MyAsyncSystem::spawn(|| async { 6 * 7 }).await;
///     assert_eq!(h.await, 42);
/// });
/// ```
#[macro_export]
macro_rules! ult_async_system {
    ($(#[$meta:meta])* $vis:vis struct $name:ident {
        base:  $base:ty,
        deque: $deque:ty $(,)?
    }) => {
        $crate::ult_async_system! {
            $(#[$meta])* $vis struct $name {
                base:            $base,
                deque:           $deque,
                // Sound as the default here specifically: this macro's
                // output never implements StackfulSchedulerSystem, so it never
                // does a real context switch (see InlineTlsCurrent's doc
                // comment for the hazard that would otherwise apply).
                lookup:          $crate::resumable::stackless::lookup::InlineTlsCurrent,
                async_pool_size: 512,
            }
        }
    };
    ($(#[$meta:meta])* $vis:vis struct $name:ident {
        base:    $base:ty,
        deque:   $deque:ty,
        lookup:  $lookup:ty $(,)?
    }) => {
        $crate::ult_async_system! {
            $(#[$meta])* $vis struct $name {
                base:            $base,
                deque:           $deque,
                lookup:          $lookup,
                async_pool_size: 512,
            }
        }
    };
    ($(#[$meta:meta])* $vis:vis struct $name:ident {
        base:            $base:ty,
        deque:           $deque:ty,
        lookup:          $lookup:ty,
        async_pool_size: $async_pool_size:expr $(,)?
    }) => {
        $(#[$meta])*
        $vis struct $name;

        impl $crate::resumable::common::system::SchedulerSystem for $name {
            type Base  = $base;
            type Desc  = $crate::resumable::common::desc::BasicTaskDesc;
            type Deque = $deque;
            type ExternalQueue = $crate::resumable::common::external_queue::StealPathQueue<$crate::resumable::common::desc::BasicTaskDesc>;
            // Never actually allocated through: this flavor has no `spawn`,
            // only `spawn_async` (which goes through AsyncPool below).
            // SimplePool is the cheapest DescPool to instantiate for a type
            // that's never used.
            type Pool = $crate::resumable::common::pool::SimplePool<$crate::resumable::common::desc::BasicTaskDesc>;
            const ASYNC_POOL_SIZE: usize = $async_pool_size;
            type AsyncPool = $crate::resumable::common::pool::ReturnPool<$crate::resumable::common::desc::BasicTaskDesc, $crate::resumable::stackless::stack::AsyncArenaStack>;
            type RecursionPool = $crate::resumable::common::pool::ThresholdPool<$crate::resumable::common::pool::BlockPool>;
            type Lookup = $lookup;

            fn worker_tls()
            -> &'static <$base as $crate::ThreadSystem>::ThreadSpecific<$crate::UltWorker<$name>>
            {
                static A: $crate::TlsAnchor = $crate::TlsAnchor::new();
                $crate::TlsSlot::from_anchor(&A)
            }

            // Stackless-only: always poll, never switch — no poll_fn tag
            // check, because every task on this system is a poll_fn task.
            fn execute(
                wk: &$crate::UltWorker<Self>,
                cont: $crate::SuspendedUlt<$crate::resumable::common::desc::BasicTaskDesc>,
            ) {
                $crate::resumable::stackless::worker::execute_async(wk, cont)
            }

            fn free_finished_desc(wk: &$crate::UltWorker<Self>, desc: *mut $crate::resumable::common::desc::BasicTaskDesc) {
                $crate::resumable::stackless::worker::free_finished_desc_async(wk, desc)
            }
        }
    };
}
