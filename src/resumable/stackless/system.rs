//! [`StacklessTaskSystem`] — the blanket-implemented async-task capability
//! trait, and the [`ult_async_system!`](crate::ult_async_system) macro
//! that generates a stackless-only system.

use std::future::Future;

use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackless::desc::AsyncTaskDesc;

// `StacklessSystem` now lives in `crate::traits::system` —
// re-exported below for callers that still spell out
// `resumable::stackless::system::StacklessSystem`.
pub use crate::traits::system::StacklessSystem;

/// `S::spawn(...)`/`S::recurse(...)` associated-function form of
/// [`crate::resumable::stackless::thread::spawn_async`]/[`crate::resumable::stackless::thread::recurse`].
///
/// Blanket-implemented for every [`SchedulerSystem`] whose `Desc` supports
/// async tasks, exactly like [`ThreadSystem`](crate::ThreadSystem)'s blanket derivation from
/// [`StackfulSystem`](crate::resumable::stackful::system::StackfulSystem) — no concrete system ever writes `impl
/// StacklessTaskSystem for ...` by hand.
pub trait StacklessTaskSystem: SchedulerSystem
where
    Self::Desc: AsyncTaskDesc,
{
    /// See [`crate::resumable::stackless::thread::spawn_async`].
    fn spawn<T, F, Mk>(mk: Mk) -> crate::resumable::stackless::thread::SpawnAction<Self, T>
    where
        F: Future<Output = T> + Send + 'static,
        Mk: FnOnce() -> F + Send + 'static,
        T: Send + 'static,
    {
        crate::resumable::stackless::thread::spawn_async::<Self, T, F, Mk>(mk)
    }

    /// See [`crate::resumable::stackless::thread::recurse`].
    fn recurse<F, Mk>(mk: Mk) -> crate::resumable::stackless::thread::RecursionFrame<Self, F>
    where
        F: Future,
        Mk: FnOnce() -> F,
    {
        crate::resumable::stackless::thread::recurse::<Self, F, Mk>(mk)
    }

    /// See [`crate::resumable::stackless::scheduler::run_async`]. Named `run_async`, not
    /// `run`, so it never collides with [`StackfulSystem::run`](crate::StackfulSystem::run)
    /// on a dual system that implements both.
    fn run_async<F>(num_workers: usize, root: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        crate::resumable::stackless::scheduler::run_async::<Self, F>(num_workers, root)
    }

    /// Yield once to the executor from inside an async task on this system
    /// — see [`crate::future::yield_now`], which this just forwards to.
    /// Not generic over `Self` at all (unlike `spawn`/`recurse`/`run_async`):
    /// provided here purely so generic code bounded by `S: StacklessTaskSystem`
    /// can write `S::yield_now().await` instead of a separate
    /// `cmpth::future` import, matching this trait's other methods.
    ///
    /// Deliberately shares its name with
    /// [`ThreadSystem::yield_now`](crate::ThreadSystem::yield_now) (the
    /// stackful, synchronous, whole-ULT-suspending version) rather than being
    /// renamed to dodge the collision — on a dual system implementing both
    /// traits, calling `Concrete::yield_now()` is ambiguous by design (same
    /// resolution as `spawn` above) and must be disambiguated with
    /// `<Concrete as StacklessTaskSystem>::yield_now()` /
    /// `<Concrete as ThreadSystem>::yield_now()`; a generic caller bounded by
    /// only one of the two traits never sees the ambiguity.
    fn yield_now() -> impl Future<Output = ()> {
        crate::future::yield_now()
    }
}

impl<S: SchedulerSystem> StacklessTaskSystem for S where S::Desc: AsyncTaskDesc {}

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
/// use cmpth::resumable::stackless::system::StacklessTaskSystem;
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
