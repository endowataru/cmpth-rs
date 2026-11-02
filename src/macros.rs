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
///     impl cmpth::UltDualIdentity for MyMarker {
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
/// (not for [`ScopedIdentity`](crate::ScopedIdentity)).
///
/// The defaults:
///
/// | type | [`UltIdentity`](crate::UltIdentity) | [`UltDualIdentity`](crate::UltDualIdentity) | [`UltAsyncIdentity`](crate::UltAsyncIdentity) | [`ScopedIdentity`](crate::ScopedIdentity) |
/// |---|---|---|---|---|
/// | `Base` | [`OsSystem`](crate::OsSystem) | [`OsSystem`](crate::OsSystem) | [`OsSystem`](crate::OsSystem) | |
/// | `Ctx` | [`NativeContext`](crate::NativeContext) | [`NativeContext`](crate::NativeContext) | | |
/// | `Desc` | [`StackfulOnlyTaskDesc<Self>`](crate::StackfulOnlyTaskDesc) | | [`StacklessOnlyTaskDesc<UltAsyncSystem<Self>>`](crate::StacklessOnlyTaskDesc) | |
/// | `RunQueue` | [`HybridRunQueue`](crate::HybridRunQueue) of `Desc`'s tokens | [`HybridRunQueue`](crate::HybridRunQueue) of [`DualTaskDesc`](crate::DualTaskDesc) tokens | [`HybridRunQueue`](crate::HybridRunQueue) of `Desc`'s tokens | |
/// | `Alloc` | [`HeapStack`](crate::HeapStack) | [`HeapStack`](crate::HeapStack) | | |
/// | `Lookup` | [`TlsCurrent`](crate::TlsCurrent) | [`TlsCurrent`](crate::TlsCurrent) | [`InlineTlsCurrent`](crate::InlineTlsCurrent) | |
/// | `Idle` | [`SpinIdle`](crate::SpinIdle) | [`SpinIdle`](crate::SpinIdle) | [`SpinIdle`](crate::SpinIdle) | [`SpinIdle`](crate::SpinIdle) |
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
///     impl cmpth::UltDualIdentity for M { type Desc = (); }
/// }
/// ```
///
/// ```compile_fail,E0201
/// pub struct M;
/// cmpth::worker_tls_anchor!(M);
/// cmpth::impl_with_defaults! {
///     impl cmpth::UltIdentity for M { type Idle = cmpth::SpinIdle; type Idle = cmpth::ParkIdle; }
/// }
/// ```
#[macro_export]
macro_rules! impl_with_defaults {
    // --- entry: one pair of arms per config trait ------------------------
    //
    // The state is `[items given so far] [slots]`: one slot per defaulted
    // type, `(def)` until the body gives that type, then `(set)`.

    (impl cmpth::UltIdentity for $t:ty { $($body:tt)* }) => {
        $crate::impl_with_defaults!(@ult [$crate::UltIdentity] $t; []; [(def) (def) (def) (def) (def) (def) (def)]; $($body)*);
    };
    (impl UltIdentity for $t:ty { $($body:tt)* }) => {
        $crate::impl_with_defaults!(@ult [UltIdentity] $t; []; [(def) (def) (def) (def) (def) (def) (def)]; $($body)*);
    };
    (impl cmpth::UltDualIdentity for $t:ty { $($body:tt)* }) => {
        $crate::impl_with_defaults!(@dual [$crate::UltDualIdentity] $t; []; [(def) (def) (def) (def) (def) (def)]; $($body)*);
    };
    (impl UltDualIdentity for $t:ty { $($body:tt)* }) => {
        $crate::impl_with_defaults!(@dual [UltDualIdentity] $t; []; [(def) (def) (def) (def) (def) (def)]; $($body)*);
    };
    (impl cmpth::UltAsyncIdentity for $t:ty { $($body:tt)* }) => {
        $crate::impl_with_defaults!(@async [$crate::UltAsyncIdentity] $t; []; [(def) (def) (def) (def) (def)]; $($body)*);
    };
    (impl UltAsyncIdentity for $t:ty { $($body:tt)* }) => {
        $crate::impl_with_defaults!(@async [UltAsyncIdentity] $t; []; [(def) (def) (def) (def) (def)]; $($body)*);
    };
    (impl cmpth::ScopedIdentity for $t:ty { $($body:tt)* }) => {
        $crate::impl_with_defaults!(@scoped [$crate::ScopedIdentity] $t; []; [(def)]; $($body)*);
    };
    (impl ScopedIdentity for $t:ty { $($body:tt)* }) => {
        $crate::impl_with_defaults!(@scoped [ScopedIdentity] $t; []; [(def)]; $($body)*);
    };

    // --- a default, emitted only while its slot is still `(def)` --------

    (@d (def) $($item:tt)*) => { $($item)* };
    (@d (set) $($item:tt)*) => {};

    // --- UltIdentity: Base Ctx Desc RunQueue Alloc Lookup Idle -----------

    (@ult $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $c:tt $d:tt $r:tt $a:tt $l:tt $id:tt]; type Base = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@ult $tr $t; [$($i)* type Base = $v;]; [(set) $c $d $r $a $l $id]; $($rest)*);
    };
    (@ult $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $c:tt $d:tt $r:tt $a:tt $l:tt $id:tt]; type Ctx = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@ult $tr $t; [$($i)* type Ctx = $v;]; [$b (set) $d $r $a $l $id]; $($rest)*);
    };
    (@ult $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $c:tt $d:tt $r:tt $a:tt $l:tt $id:tt]; type Desc = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@ult $tr $t; [$($i)* type Desc = $v;]; [$b $c (set) $r $a $l $id]; $($rest)*);
    };
    (@ult $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $c:tt $d:tt $r:tt $a:tt $l:tt $id:tt]; type RunQueue = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@ult $tr $t; [$($i)* type RunQueue = $v;]; [$b $c $d (set) $a $l $id]; $($rest)*);
    };
    (@ult $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $c:tt $d:tt $r:tt $a:tt $l:tt $id:tt]; type Alloc = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@ult $tr $t; [$($i)* type Alloc = $v;]; [$b $c $d $r (set) $l $id]; $($rest)*);
    };
    (@ult $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $c:tt $d:tt $r:tt $a:tt $l:tt $id:tt]; type Lookup = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@ult $tr $t; [$($i)* type Lookup = $v;]; [$b $c $d $r $a (set) $id]; $($rest)*);
    };
    (@ult $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $c:tt $d:tt $r:tt $a:tt $l:tt $id:tt]; type Idle = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@ult $tr $t; [$($i)* type Idle = $v;]; [$b $c $d $r $a $l (set)]; $($rest)*);
    };
    (@ult $tr:tt $t:ty; [$($i:tt)*]; $s:tt; const $n:ident : $ct:ty = $e:expr; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@ult $tr $t; [$($i)* const $n: $ct = $e;]; $s; $($rest)*);
    };
    (@ult [$($trp:tt)*] $t:ty; [$($i:tt)*]; [$b:tt $c:tt $d:tt $r:tt $a:tt $l:tt $id:tt];) => {
        impl $($trp)* for $t {
            $($i)*
            $crate::impl_with_defaults!(@d $b type Base = $crate::OsSystem;);
            $crate::impl_with_defaults!(@d $c type Ctx = $crate::NativeContext;);
            $crate::impl_with_defaults!(@d $d type Desc = $crate::StackfulOnlyTaskDesc<Self>;);
            $crate::impl_with_defaults!(@d $r type RunQueue = $crate::HybridRunQueue<$crate::SuspendedTaskToken<<Self as $crate::UltIdentity>::Desc>>;);
            $crate::impl_with_defaults!(@d $a type Alloc = $crate::HeapStack;);
            $crate::impl_with_defaults!(@d $l type Lookup = $crate::TlsCurrent;);
            $crate::impl_with_defaults!(@d $id type Idle = $crate::SpinIdle;);
        }
    };
    (@ult $tr:tt $t:ty; $i:tt; $s:tt; $($rest:tt)+) => {
        $crate::impl_with_defaults!(@unknown UltIdentity; $($rest)+);
    };

    // --- UltDualIdentity: Base Ctx RunQueue Alloc Lookup Idle ------------

    (@dual $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $c:tt $r:tt $a:tt $l:tt $id:tt]; type Base = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@dual $tr $t; [$($i)* type Base = $v;]; [(set) $c $r $a $l $id]; $($rest)*);
    };
    (@dual $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $c:tt $r:tt $a:tt $l:tt $id:tt]; type Ctx = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@dual $tr $t; [$($i)* type Ctx = $v;]; [$b (set) $r $a $l $id]; $($rest)*);
    };
    (@dual $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $c:tt $r:tt $a:tt $l:tt $id:tt]; type RunQueue = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@dual $tr $t; [$($i)* type RunQueue = $v;]; [$b $c (set) $a $l $id]; $($rest)*);
    };
    (@dual $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $c:tt $r:tt $a:tt $l:tt $id:tt]; type Alloc = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@dual $tr $t; [$($i)* type Alloc = $v;]; [$b $c $r (set) $l $id]; $($rest)*);
    };
    (@dual $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $c:tt $r:tt $a:tt $l:tt $id:tt]; type Lookup = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@dual $tr $t; [$($i)* type Lookup = $v;]; [$b $c $r $a (set) $id]; $($rest)*);
    };
    (@dual $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $c:tt $r:tt $a:tt $l:tt $id:tt]; type Idle = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@dual $tr $t; [$($i)* type Idle = $v;]; [$b $c $r $a $l (set)]; $($rest)*);
    };
    (@dual $tr:tt $t:ty; [$($i:tt)*]; $s:tt; const $n:ident : $ct:ty = $e:expr; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@dual $tr $t; [$($i)* const $n: $ct = $e;]; $s; $($rest)*);
    };
    (@dual [$($trp:tt)*] $t:ty; [$($i:tt)*]; [$b:tt $c:tt $r:tt $a:tt $l:tt $id:tt];) => {
        impl $($trp)* for $t {
            $($i)*
            $crate::impl_with_defaults!(@d $b type Base = $crate::OsSystem;);
            $crate::impl_with_defaults!(@d $c type Ctx = $crate::NativeContext;);
            $crate::impl_with_defaults!(@d $r type RunQueue = $crate::HybridRunQueue<$crate::SuspendedTaskToken<$crate::DualTaskDesc<$crate::UltDualSystem<Self>>>>;);
            $crate::impl_with_defaults!(@d $a type Alloc = $crate::HeapStack;);
            $crate::impl_with_defaults!(@d $l type Lookup = $crate::TlsCurrent;);
            $crate::impl_with_defaults!(@d $id type Idle = $crate::SpinIdle;);
        }
    };
    (@dual $tr:tt $t:ty; $i:tt; $s:tt; $($rest:tt)+) => {
        $crate::impl_with_defaults!(@unknown UltDualIdentity; $($rest)+);
    };

    // --- UltAsyncIdentity: Base Desc RunQueue Lookup Idle ----------------

    (@async $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $d:tt $r:tt $l:tt $id:tt]; type Base = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@async $tr $t; [$($i)* type Base = $v;]; [(set) $d $r $l $id]; $($rest)*);
    };
    (@async $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $d:tt $r:tt $l:tt $id:tt]; type Desc = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@async $tr $t; [$($i)* type Desc = $v;]; [$b (set) $r $l $id]; $($rest)*);
    };
    (@async $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $d:tt $r:tt $l:tt $id:tt]; type RunQueue = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@async $tr $t; [$($i)* type RunQueue = $v;]; [$b $d (set) $l $id]; $($rest)*);
    };
    (@async $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $d:tt $r:tt $l:tt $id:tt]; type Lookup = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@async $tr $t; [$($i)* type Lookup = $v;]; [$b $d $r (set) $id]; $($rest)*);
    };
    (@async $tr:tt $t:ty; [$($i:tt)*]; [$b:tt $d:tt $r:tt $l:tt $id:tt]; type Idle = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@async $tr $t; [$($i)* type Idle = $v;]; [$b $d $r $l (set)]; $($rest)*);
    };
    (@async $tr:tt $t:ty; [$($i:tt)*]; $s:tt; const $n:ident : $ct:ty = $e:expr; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@async $tr $t; [$($i)* const $n: $ct = $e;]; $s; $($rest)*);
    };
    (@async [$($trp:tt)*] $t:ty; [$($i:tt)*]; [$b:tt $d:tt $r:tt $l:tt $id:tt];) => {
        impl $($trp)* for $t {
            $($i)*
            $crate::impl_with_defaults!(@d $b type Base = $crate::OsSystem;);
            $crate::impl_with_defaults!(@d $d type Desc = $crate::StacklessOnlyTaskDesc<$crate::UltAsyncSystem<Self>>;);
            $crate::impl_with_defaults!(@d $r type RunQueue = $crate::HybridRunQueue<$crate::SuspendedTaskToken<<Self as $crate::UltAsyncIdentity>::Desc>>;);
            $crate::impl_with_defaults!(@d $l type Lookup = $crate::InlineTlsCurrent;);
            $crate::impl_with_defaults!(@d $id type Idle = $crate::SpinIdle;);
        }
    };
    (@async $tr:tt $t:ty; $i:tt; $s:tt; $($rest:tt)+) => {
        $crate::impl_with_defaults!(@unknown UltAsyncIdentity; $($rest)+);
    };

    // --- ScopedIdentity: Idle --------------------------------------------

    (@scoped $tr:tt $t:ty; [$($i:tt)*]; [$id:tt]; type Idle = $v:ty; $($rest:tt)*) => {
        $crate::impl_with_defaults!(@scoped $tr $t; [$($i)* type Idle = $v;]; [(set)]; $($rest)*);
    };
    (@scoped [$($trp:tt)*] $t:ty; [$($i:tt)*]; [$id:tt];) => {
        impl $($trp)* for $t {
            $($i)*
            $crate::impl_with_defaults!(@d $id type Idle = $crate::SpinIdle;);
        }
    };
    (@scoped $tr:tt $t:ty; $i:tt; $s:tt; $($rest:tt)+) => {
        $crate::impl_with_defaults!(@unknown ScopedIdentity; $($rest)+);
    };

    // --- errors ----------------------------------------------------------

    (@unknown $tr:ident; type $k:ident $($rest:tt)*) => {
        compile_error!(concat!("impl_with_defaults!: `", stringify!($tr), "` has no associated type `", stringify!($k), "`"));
    };
    (@unknown $tr:ident; $($rest:tt)*) => {
        compile_error!(concat!("impl_with_defaults!: expected `type Name = Type;` or `const NAME: Type = value;` in the `", stringify!($tr), "` impl, found `", stringify!($($rest)*), "`"));
    };
    ($($other:tt)*) => {
        compile_error!("impl_with_defaults!: expected `impl Trait for Type { ... }` with Trait one of UltIdentity, UltDualIdentity, UltAsyncIdentity, ScopedIdentity");
    };
}
