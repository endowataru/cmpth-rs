//! Macros for defining systems: [`worker_tls_anchor!`](crate::worker_tls_anchor)
//! gives a type its own TLS anchor, and
//! [`impl_with_defaults!`](crate::impl_with_defaults) fills in the members
//! a config-trait impl leaves out.

/// Implements [`WorkerTlsAnchor`](crate::WorkerTlsAnchor) for each listed
/// type with a `static` of its own.
///
/// ```
/// pub struct MyMarker;
/// cmpth::worker_tls_anchor!(MyMarker);
/// ```
///
/// The types must not be generic: a `static` cannot depend on a type
/// parameter, so every instance of a generic type would share one anchor.
#[macro_export]
macro_rules! worker_tls_anchor {
    ($($t:ty),+ $(,)?) => {
        $(
            // SAFETY: the static is local to this impl, so no other type
            // returns it.
            unsafe impl $crate::WorkerTlsAnchor for $t {
                fn anchor() -> &'static $crate::TlsAnchor {
                    static A: $crate::TlsAnchor = $crate::TlsAnchor::new();
                    &A
                }
            }
        )+
    };
}

/// Implements a config trait, filling in every associated type the body
/// leaves out with that trait's default.
///
/// The body is an ordinary impl block, so once associated type defaults
/// are stable the defaults can move into the traits and this wrapper can
/// be deleted without touching what is inside it:
///
/// ```
/// use cmpth::{JoinHandleLike, SpawnableStackfulTaskSystem, StackfulBuilder, StackfulInitSystem};
///
/// pub struct MyMarker;
/// cmpth::worker_tls_anchor!(MyMarker);
///
/// cmpth::impl_with_defaults! {
///     impl cmpth::UltDualConfig for MyMarker {
///         type Idle = cmpth::SearchIdle;
///         const STACK_SIZE: usize = 128 * 1024;
///     }
/// }
///
/// type MySystem = cmpth::UltDualSystem<MyMarker>;
///
/// MySystem::builder().workers(2).run(|| {
///     assert_eq!(JoinHandleLike::join(MySystem::spawn(|| 42)), 42);
/// });
/// ```
///
/// The trait is written `cmpth::Trait` or `Trait`.  The body may hold
/// `type` and `const` items; consts already have defaults in the traits
/// and pass through unchanged.  A type the trait does not have is a
/// compile error, and so is one given twice.  The type implementing the
/// trait also needs [`worker_tls_anchor!`](crate::worker_tls_anchor)
/// (not for [`ScopedConfig`](crate::ScopedConfig)).
///
/// The defaults:
///
/// | type | [`UltConfig`](crate::UltConfig) | [`UltDualConfig`](crate::UltDualConfig) | [`UltAsyncConfig`](crate::UltAsyncConfig) | [`ScopedConfig`](crate::ScopedConfig) |
/// |---|---|---|---|---|
/// | `Base` | [`OsSystem`](crate::OsSystem) | [`OsSystem`](crate::OsSystem) | [`OsSystem`](crate::OsSystem) | |
/// | `Ctx` | [`NativeContext`](crate::NativeContext) | [`NativeContext`](crate::NativeContext) | | |
/// | `Desc` | [`StackfulOnlyTaskDesc<Self>`](crate::StackfulOnlyTaskDesc) | | [`StacklessOnlyTaskDesc<UltAsyncSystem<Self>>`](crate::StacklessOnlyTaskDesc) | |
/// | `RunQueue` | [`HybridRunQueue`](crate::HybridRunQueue) of `Desc`'s tokens | [`HybridRunQueue`](crate::HybridRunQueue) of [`DualTaskDesc`](crate::DualTaskDesc) tokens | [`HybridRunQueue`](crate::HybridRunQueue) of `Desc`'s tokens | |
/// | `Alloc` | [`HeapStack`](crate::HeapStack) | [`HeapStack`](crate::HeapStack) | | |
/// | `Lookup` | [`TlsCurrent`](crate::TlsCurrent) | [`TlsCurrent`](crate::TlsCurrent) | [`InlineTlsCurrent`](crate::InlineTlsCurrent) | |
/// | `Idle` | [`SpinIdle`](crate::SpinIdle) | [`SpinIdle`](crate::SpinIdle) | [`SpinIdle`](crate::SpinIdle) | [`SpinIdle`](crate::SpinIdle) |
/// | `Steal` | [`RandomSweepSteal`](crate::RandomSweepSteal) | [`RandomSweepSteal`](crate::RandomSweepSteal) | [`RandomSweepSteal`](crate::RandomSweepSteal) | [`RandomSweepSteal`](crate::RandomSweepSteal) |
/// | `Dispatch` | [`StandardDispatch`](crate::StandardDispatch) | [`StandardDispatch`](crate::StandardDispatch) | [`StandardDispatch`](crate::StandardDispatch) | [`StandardDispatch`](crate::StandardDispatch) |
/// | `Duties` | [`GlobalDuties`](crate::resumable::common::duty::GlobalDuties) | [`GlobalDuties`](crate::resumable::common::duty::GlobalDuties) | [`GlobalDuties`](crate::resumable::common::duty::GlobalDuties) | [`GlobalDuties`](crate::resumable::common::duty::GlobalDuties) |
/// | `Observe` | [`NoObserver`](crate::NoObserver) | [`NoObserver`](crate::NoObserver) | [`NoObserver`](crate::NoObserver) | [`NoObserver`](crate::NoObserver) |
/// | `Hooks` | [`NoHooks`](crate::NoHooks) | [`NoHooks`](crate::NoHooks) | [`NoHooks`](crate::NoHooks) | [`NoHooks`](crate::NoHooks) |
/// | `ExternalQueue` | [`StealPathQueue`](crate::StealPathQueue) of `Desc` | [`StealPathQueue`](crate::StealPathQueue) of [`DualTaskDesc`](crate::DualTaskDesc) | [`StealPathQueue`](crate::StealPathQueue) of `Desc` | |
/// | `PostConfig` | [`DefaultPostConfig`](crate::DefaultPostConfig) | [`DefaultPostConfig`](crate::DefaultPostConfig) | | |
///
/// The type implementing the trait must not be generic (as for
/// `worker_tls_anchor!`).
///
/// A type the trait does not have, and a type given twice:
///
/// ```compile_fail
/// pub struct M;
/// cmpth::worker_tls_anchor!(M);
/// cmpth::impl_with_defaults! {
///     impl cmpth::UltDualConfig for M { type Desc = (); }
/// }
/// ```
///
/// ```compile_fail,E0201
/// pub struct M;
/// cmpth::worker_tls_anchor!(M);
/// cmpth::impl_with_defaults! {
///     impl cmpth::UltConfig for M { type Idle = cmpth::SpinIdle; type Idle = cmpth::ParkIdle; }
/// }
/// ```
#[macro_export]
macro_rules! impl_with_defaults {
    // --- entry: one pair of arms per config trait ------------------------
    //
    // The trait path is kept as written, so a bare name needs its import
    // like a plain impl.  The body is emitted unchanged, followed by one
    // `@or` per defaulted type: each scans the body and emits its default
    // only if the body does not give that type.

    (impl cmpth::UltConfig for $t:ty { $($b:tt)* }) => { $crate::impl_with_defaults!(@ult [$crate::UltConfig] $t; $($b)*); };
    (impl UltConfig for $t:ty { $($b:tt)* }) => { $crate::impl_with_defaults!(@ult [UltConfig] $t; $($b)*); };
    (impl cmpth::UltDualConfig for $t:ty { $($b:tt)* }) => { $crate::impl_with_defaults!(@dual [$crate::UltDualConfig] $t; $($b)*); };
    (impl UltDualConfig for $t:ty { $($b:tt)* }) => { $crate::impl_with_defaults!(@dual [UltDualConfig] $t; $($b)*); };
    (impl cmpth::UltAsyncConfig for $t:ty { $($b:tt)* }) => { $crate::impl_with_defaults!(@async [$crate::UltAsyncConfig] $t; $($b)*); };
    (impl UltAsyncConfig for $t:ty { $($b:tt)* }) => { $crate::impl_with_defaults!(@async [UltAsyncConfig] $t; $($b)*); };
    (impl cmpth::ScopedConfig for $t:ty { $($b:tt)* }) => { $crate::impl_with_defaults!(@scoped [$crate::ScopedConfig] $t; $($b)*); };
    (impl ScopedConfig for $t:ty { $($b:tt)* }) => { $crate::impl_with_defaults!(@scoped [ScopedConfig] $t; $($b)*); };

    // --- the defaults, per trait -----------------------------------------

    (@ult [$($tr:tt)*] $t:ty; $($b:tt)*) => {
        $crate::impl_with_defaults!(@check UltConfig; $($b)*);
        impl $($tr)* for $t {
            $($b)*
            $crate::impl_with_defaults!(@or Base [$($b)*] type Base = $crate::OsSystem;);
            $crate::impl_with_defaults!(@or Ctx [$($b)*] type Ctx = $crate::NativeContext;);
            $crate::impl_with_defaults!(@or Desc [$($b)*] type Desc = $crate::StackfulOnlyTaskDesc<Self>;);
            $crate::impl_with_defaults!(@or RunQueue [$($b)*] type RunQueue = $crate::HybridRunQueue<$crate::SuspendedTaskToken<<Self as $crate::UltConfig>::Desc>>;);
            $crate::impl_with_defaults!(@or Alloc [$($b)*] type Alloc = $crate::HeapStack;);
            $crate::impl_with_defaults!(@or Lookup [$($b)*] type Lookup = $crate::TlsCurrent;);
            $crate::impl_with_defaults!(@or Idle [$($b)*] type Idle = $crate::SpinIdle;);
            $crate::impl_with_defaults!(@or Steal [$($b)*] type Steal = $crate::RandomSweepSteal;);
            $crate::impl_with_defaults!(@or Dispatch [$($b)*] type Dispatch = $crate::StandardDispatch;);
            $crate::impl_with_defaults!(@or Duties [$($b)*] type Duties = $crate::resumable::common::duty::GlobalDuties;);
            $crate::impl_with_defaults!(@or Observe [$($b)*] type Observe = $crate::NoObserver;);
            $crate::impl_with_defaults!(@or Hooks [$($b)*] type Hooks = $crate::NoHooks;);
            $crate::impl_with_defaults!(@or ExternalQueue [$($b)*] type ExternalQueue = $crate::StealPathQueue<<Self as $crate::UltConfig>::Desc>;);
            $crate::impl_with_defaults!(@or PostConfig [$($b)*] type PostConfig = $crate::DefaultPostConfig;);
        }
    };
    (@dual [$($tr:tt)*] $t:ty; $($b:tt)*) => {
        $crate::impl_with_defaults!(@check UltDualConfig; $($b)*);
        impl $($tr)* for $t {
            $($b)*
            $crate::impl_with_defaults!(@or Base [$($b)*] type Base = $crate::OsSystem;);
            $crate::impl_with_defaults!(@or Ctx [$($b)*] type Ctx = $crate::NativeContext;);
            $crate::impl_with_defaults!(@or RunQueue [$($b)*] type RunQueue = $crate::HybridRunQueue<$crate::SuspendedTaskToken<$crate::DualTaskDesc<$crate::UltDualSystem<Self>>>>;);
            $crate::impl_with_defaults!(@or Alloc [$($b)*] type Alloc = $crate::HeapStack;);
            $crate::impl_with_defaults!(@or Lookup [$($b)*] type Lookup = $crate::TlsCurrent;);
            $crate::impl_with_defaults!(@or Idle [$($b)*] type Idle = $crate::SpinIdle;);
            $crate::impl_with_defaults!(@or Steal [$($b)*] type Steal = $crate::RandomSweepSteal;);
            $crate::impl_with_defaults!(@or Dispatch [$($b)*] type Dispatch = $crate::StandardDispatch;);
            $crate::impl_with_defaults!(@or Duties [$($b)*] type Duties = $crate::resumable::common::duty::GlobalDuties;);
            $crate::impl_with_defaults!(@or Observe [$($b)*] type Observe = $crate::NoObserver;);
            $crate::impl_with_defaults!(@or Hooks [$($b)*] type Hooks = $crate::NoHooks;);
            $crate::impl_with_defaults!(@or ExternalQueue [$($b)*] type ExternalQueue = $crate::StealPathQueue<$crate::DualTaskDesc<$crate::UltDualSystem<Self>>>;);
            $crate::impl_with_defaults!(@or PostConfig [$($b)*] type PostConfig = $crate::DefaultPostConfig;);
        }
    };
    (@async [$($tr:tt)*] $t:ty; $($b:tt)*) => {
        $crate::impl_with_defaults!(@check UltAsyncConfig; $($b)*);
        impl $($tr)* for $t {
            $($b)*
            $crate::impl_with_defaults!(@or Base [$($b)*] type Base = $crate::OsSystem;);
            $crate::impl_with_defaults!(@or Desc [$($b)*] type Desc = $crate::StacklessOnlyTaskDesc<$crate::UltAsyncSystem<Self>>;);
            $crate::impl_with_defaults!(@or RunQueue [$($b)*] type RunQueue = $crate::HybridRunQueue<$crate::SuspendedTaskToken<<Self as $crate::UltAsyncConfig>::Desc>>;);
            $crate::impl_with_defaults!(@or Lookup [$($b)*] type Lookup = $crate::InlineTlsCurrent;);
            $crate::impl_with_defaults!(@or Idle [$($b)*] type Idle = $crate::SpinIdle;);
            $crate::impl_with_defaults!(@or Steal [$($b)*] type Steal = $crate::RandomSweepSteal;);
            $crate::impl_with_defaults!(@or Dispatch [$($b)*] type Dispatch = $crate::StandardDispatch;);
            $crate::impl_with_defaults!(@or Duties [$($b)*] type Duties = $crate::resumable::common::duty::GlobalDuties;);
            $crate::impl_with_defaults!(@or Observe [$($b)*] type Observe = $crate::NoObserver;);
            $crate::impl_with_defaults!(@or Hooks [$($b)*] type Hooks = $crate::NoHooks;);
            $crate::impl_with_defaults!(@or ExternalQueue [$($b)*] type ExternalQueue = $crate::StealPathQueue<<Self as $crate::UltAsyncConfig>::Desc>;);
        }
    };
    (@scoped [$($tr:tt)*] $t:ty; $($b:tt)*) => {
        $crate::impl_with_defaults!(@check ScopedConfig; $($b)*);
        impl $($tr)* for $t {
            $($b)*
            $crate::impl_with_defaults!(@or Idle [$($b)*] type Idle = $crate::SpinIdle;);
            $crate::impl_with_defaults!(@or Steal [$($b)*] type Steal = $crate::RandomSweepSteal;);
            $crate::impl_with_defaults!(@or Dispatch [$($b)*] type Dispatch = $crate::StandardDispatch;);
            $crate::impl_with_defaults!(@or Duties [$($b)*] type Duties = $crate::resumable::common::duty::GlobalDuties;);
            $crate::impl_with_defaults!(@or Observe [$($b)*] type Observe = $crate::NoObserver;);
            $crate::impl_with_defaults!(@or Hooks [$($b)*] type Hooks = $crate::NoHooks;);
        }
    };

    // --- which types each trait has --------------------------------------

    (@has UltConfig Base) => {};
    (@has UltConfig Ctx) => {};
    (@has UltConfig Desc) => {};
    (@has UltConfig RunQueue) => {};
    (@has UltConfig Alloc) => {};
    (@has UltConfig Lookup) => {};
    (@has UltConfig Idle) => {};
    (@has UltConfig Steal) => {};
    (@has UltConfig Dispatch) => {};
    (@has UltConfig Duties) => {};
    (@has UltConfig Observe) => {};
    (@has UltConfig Hooks) => {};
    (@has UltConfig ExternalQueue) => {};
    (@has UltConfig PostConfig) => {};
    (@has UltDualConfig Base) => {};
    (@has UltDualConfig Ctx) => {};
    (@has UltDualConfig RunQueue) => {};
    (@has UltDualConfig Alloc) => {};
    (@has UltDualConfig Lookup) => {};
    (@has UltDualConfig Idle) => {};
    (@has UltDualConfig Steal) => {};
    (@has UltDualConfig Dispatch) => {};
    (@has UltDualConfig Duties) => {};
    (@has UltDualConfig Observe) => {};
    (@has UltDualConfig Hooks) => {};
    (@has UltDualConfig ExternalQueue) => {};
    (@has UltDualConfig PostConfig) => {};
    (@has UltAsyncConfig Base) => {};
    (@has UltAsyncConfig Desc) => {};
    (@has UltAsyncConfig RunQueue) => {};
    (@has UltAsyncConfig Lookup) => {};
    (@has UltAsyncConfig Idle) => {};
    (@has UltAsyncConfig Steal) => {};
    (@has UltAsyncConfig Dispatch) => {};
    (@has UltAsyncConfig Duties) => {};
    (@has UltAsyncConfig Observe) => {};
    (@has UltAsyncConfig Hooks) => {};
    (@has UltAsyncConfig ExternalQueue) => {};
    (@has ScopedConfig Idle) => {};
    (@has ScopedConfig Steal) => {};
    (@has ScopedConfig Dispatch) => {};
    (@has ScopedConfig Duties) => {};
    (@has ScopedConfig Observe) => {};
    (@has ScopedConfig Hooks) => {};
    (@has $tr:ident $k:ident) => {
        compile_error!(concat!("impl_with_defaults!: `", stringify!($tr), "` has no associated type `", stringify!($k), "`"));
    };

    // --- `@check`: every item is a known `type` or a `const` -------------

    (@check $tr:ident;) => {};
    (@check $tr:ident; type $k:ident = $v:ty; $($r:tt)*) => {
        $crate::impl_with_defaults!(@has $tr $k);
        $crate::impl_with_defaults!(@check $tr; $($r)*);
    };
    (@check $tr:ident; const $n:ident : $ct:ty = $e:expr; $($r:tt)*) => {
        $crate::impl_with_defaults!(@check $tr; $($r)*);
    };
    (@check $tr:ident; $($r:tt)*) => {
        compile_error!(concat!("impl_with_defaults!: expected `type Name = Type;` or `const NAME: Type = value;` in the `", stringify!($tr), "` impl, found `", stringify!($($r)*), "`"));
    };

    // --- `@or K [body] default`: the default unless the body gives `K` ---
    //
    // One arm per type name (any trait), since macro_rules cannot compare
    // two identifiers.

    (@or $k:ident [] $($d:tt)*) => { $($d)* };
    (@or Base [type Base = $v:ty; $($r:tt)*] $($d:tt)*) => {};
    (@or Ctx [type Ctx = $v:ty; $($r:tt)*] $($d:tt)*) => {};
    (@or Desc [type Desc = $v:ty; $($r:tt)*] $($d:tt)*) => {};
    (@or RunQueue [type RunQueue = $v:ty; $($r:tt)*] $($d:tt)*) => {};
    (@or Alloc [type Alloc = $v:ty; $($r:tt)*] $($d:tt)*) => {};
    (@or Lookup [type Lookup = $v:ty; $($r:tt)*] $($d:tt)*) => {};
    (@or Idle [type Idle = $v:ty; $($r:tt)*] $($d:tt)*) => {};
    (@or Steal [type Steal = $v:ty; $($r:tt)*] $($d:tt)*) => {};
    (@or Dispatch [type Dispatch = $v:ty; $($r:tt)*] $($d:tt)*) => {};
    (@or Duties [type Duties = $v:ty; $($r:tt)*] $($d:tt)*) => {};
    (@or Observe [type Observe = $v:ty; $($r:tt)*] $($d:tt)*) => {};
    (@or Hooks [type Hooks = $v:ty; $($r:tt)*] $($d:tt)*) => {};
    (@or ExternalQueue [type ExternalQueue = $v:ty; $($r:tt)*] $($d:tt)*) => {};
    (@or PostConfig [type PostConfig = $v:ty; $($r:tt)*] $($d:tt)*) => {};
    (@or $k:ident [type $o:ident = $v:ty; $($r:tt)*] $($d:tt)*) => {
        $crate::impl_with_defaults!(@or $k [$($r)*] $($d)*);
    };
    (@or $k:ident [const $n:ident : $ct:ty = $e:expr; $($r:tt)*] $($d:tt)*) => {
        $crate::impl_with_defaults!(@or $k [$($r)*] $($d)*);
    };
    // Anything else: `@check` has already reported it.
    (@or $k:ident [$($r:tt)*] $($d:tt)*) => {};

    ($($other:tt)*) => {
        compile_error!("impl_with_defaults!: expected `impl Trait for Type { ... }` with Trait one of UltConfig, UltDualConfig, UltAsyncConfig, ScopedConfig");
    };
}
