//! `impl_with_defaults!`: omitted types take the documented defaults, given
//! ones win, and every config trait builds a working system.

use std::marker::PhantomData;

use cmpth::{
    DualTaskDesc, HybridRunQueue, InlineTlsCurrent, JoinHandleLike, NativeContext, OsSystem,
    ScopedStackfulTaskSystem, SpawnableStackfulTaskSystem, SpinIdle, StackfulBuilder,
    StackfulInitSystem, StackfulOnlyTaskDesc, StacklessBuilder, StacklessInitSystem,
    StacklessOnlyTaskDesc, StacklessTaskSystem, SuspendedTaskToken, TlsCurrent, UltAsyncConfig,
    UltAsyncSystem, UltDualConfig, UltDualSystem, UltConfig,
};

/// Compiles only when `A` and `B` are the same type.
fn same<A, B>()
where
    PhantomData<A>: Is<PhantomData<B>>,
{
}
trait Is<T> {}
impl<T> Is<T> for T {}

pub struct Stackful;
cmpth::worker_tls_anchor!(Stackful);
cmpth::impl_with_defaults! {
    impl cmpth::UltConfig for Stackful {
        type Idle = cmpth::SearchIdle;
        const STACK_SIZE: usize = 128 * 1024;
    }
}

pub struct Dual;
cmpth::worker_tls_anchor!(Dual);
cmpth::impl_with_defaults! {
    impl UltDualConfig for Dual {
        type Base = cmpth::DefaultDualTaskSystem;
        type Steal = cmpth::RandomSteal;
    }
}

pub struct Async;
cmpth::worker_tls_anchor!(Async);
cmpth::impl_with_defaults! {
    impl cmpth::UltAsyncConfig for Async {
        const ASYNC_POOL_SIZE: usize = 256;
        type Steal = cmpth::RandomSweepSteal;
    }
}

pub struct Scoped;
cmpth::impl_with_defaults! {
    impl cmpth::ScopedConfig for Scoped {
        type Idle = cmpth::ParkIdle;
        type Duties = cmpth::NoDuties;
    }
}

#[test]
fn defaults_and_overrides() {
    same::<<Stackful as UltConfig>::Base, OsSystem>();
    same::<<Stackful as UltConfig>::Ctx, NativeContext>();
    same::<<Stackful as UltConfig>::Desc, StackfulOnlyTaskDesc<Stackful>>();
    same::<<Stackful as UltConfig>::RunQueue, HybridRunQueue<SuspendedTaskToken<StackfulOnlyTaskDesc<Stackful>>>>();
    same::<<Stackful as UltConfig>::Alloc, cmpth::HeapStack>();
    same::<<Stackful as UltConfig>::Idle, cmpth::SearchIdle>();
    same::<<Stackful as UltConfig>::Steal, cmpth::RandomSweepSteal>();
    same::<<Stackful as UltConfig>::Dispatch, cmpth::StandardDispatch>();
    same::<<Stackful as UltConfig>::Duties, cmpth::GlobalDuties>();
    same::<<Stackful as UltConfig>::Observe, cmpth::NoObserver>();
    same::<<Stackful as UltConfig>::ExternalQueue, cmpth::StealPathQueue<StackfulOnlyTaskDesc<Stackful>>>();
    same::<<Stackful as UltConfig>::PostConfig, cmpth::DefaultPostConfig>();
    assert_eq!(<Stackful as UltConfig>::STACK_SIZE, 128 * 1024);

    same::<<Dual as UltDualConfig>::Base, cmpth::DefaultDualTaskSystem>();
    same::<<Dual as UltDualConfig>::RunQueue, HybridRunQueue<SuspendedTaskToken<DualTaskDesc<UltDualSystem<Dual>>>>>();
    same::<<Dual as UltDualConfig>::Idle, SpinIdle>();
    same::<<Dual as UltDualConfig>::Steal, cmpth::RandomSteal>();
    same::<<Dual as UltDualConfig>::ExternalQueue, cmpth::StealPathQueue<DualTaskDesc<UltDualSystem<Dual>>>>();
    assert_eq!(<Dual as UltDualConfig>::ASYNC_POOL_SIZE, 512);

    same::<<Async as UltAsyncConfig>::Desc, StacklessOnlyTaskDesc<UltAsyncSystem<Async>>>();
    same::<<Async as UltAsyncConfig>::Steal, cmpth::RandomSweepSteal>();
    assert_eq!(<Async as UltAsyncConfig>::ASYNC_POOL_SIZE, 256);
    same::<<Async as UltAsyncConfig>::RunQueue, HybridRunQueue<SuspendedTaskToken<StacklessOnlyTaskDesc<UltAsyncSystem<Async>>>>>();

    same::<<Scoped as cmpth::ScopedConfig>::Idle, cmpth::ParkIdle>();
    same::<<Scoped as cmpth::ScopedConfig>::Steal, cmpth::RandomSweepSteal>();
    same::<<Scoped as cmpth::ScopedConfig>::Duties, cmpth::NoDuties>();

    // `Lookup` carries a `where` clause, so it is checked through the system.
    same::<<Stackful as cmpth::WorkerSystem>::Lookup, TlsCurrent>();
    same::<<UltAsyncSystem<Async> as cmpth::WorkerSystem>::Lookup, InlineTlsCurrent>();
}

#[test]
fn systems_run() {
    Stackful::builder().workers(2).run(|| {
        assert_eq!(JoinHandleLike::join(Stackful::spawn(|| 1)), 1);
    });

    // `Dual` nests on DefaultDualTaskSystem.
    <cmpth::DefaultDualTaskSystem as StackfulInitSystem>::builder().workers(2).run(|| {
        type S = UltDualSystem<Dual>;
        <S as StackfulInitSystem>::builder().workers(2).run(|| {
            assert_eq!(JoinHandleLike::join(<S as SpawnableStackfulTaskSystem>::spawn(|| 2)), 2);
        });
    });

    UltAsyncSystem::<Async>::builder().workers(2).run_async(async {
        let h = UltAsyncSystem::<Async>::spawn(|| async { 3 }).await;
        assert_eq!(h.await, 3);
    });

    type Sc = cmpth::ScopedSystem<Scoped>;
    let r = Sc::builder().workers(2).run(|| Sc::parallel_call(|| 4, || 5));
    assert_eq!(r, (4, 5));
}
