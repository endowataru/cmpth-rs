//! `impl_with_defaults!`: omitted types take the documented defaults, given
//! ones win, and every config trait builds a working system.

use std::marker::PhantomData;

use cmpth::{
    DualTaskDesc, HybridRunQueue, InlineTlsCurrent, JoinHandleLike, NativeContext, OsSystem,
    ScopedStackfulTaskSystem, SpawnableStackfulTaskSystem, SpinIdle, StackfulBuilder,
    StackfulInitSystem, StackfulOnlyTaskDesc, StacklessBuilder, StacklessInitSystem,
    StacklessOnlyTaskDesc, StacklessTaskSystem, SuspendedTaskToken, TlsCurrent, UltAsyncIdentity,
    UltAsyncSystem, UltDualIdentity, UltDualSystem, UltIdentity,
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
    impl cmpth::UltIdentity for Stackful {
        type Idle = cmpth::SearchIdle;
        const STACK_SIZE: usize = 128 * 1024;
    }
}

pub struct Dual;
cmpth::worker_tls_anchor!(Dual);
cmpth::impl_with_defaults! {
    impl UltDualIdentity for Dual {
        type Base = cmpth::DefaultDualTaskSystem;
        type Steal = cmpth::RandomSteal;
    }
}

pub struct Async;
cmpth::worker_tls_anchor!(Async);
cmpth::impl_with_defaults! {
    impl cmpth::UltAsyncIdentity for Async {
        const ASYNC_POOL_SIZE: usize = 256;
        type Steal = cmpth::RandomSweepSteal;
    }
}

pub struct Scoped;
cmpth::impl_with_defaults! {
    impl cmpth::ScopedIdentity for Scoped {
        type Idle = cmpth::ParkIdle;
        type Duties = cmpth::duty::NoDuties;
    }
}

#[test]
fn defaults_and_overrides() {
    same::<<Stackful as UltIdentity>::Base, OsSystem>();
    same::<<Stackful as UltIdentity>::Ctx, NativeContext>();
    same::<<Stackful as UltIdentity>::Desc, StackfulOnlyTaskDesc<Stackful>>();
    same::<<Stackful as UltIdentity>::RunQueue, HybridRunQueue<SuspendedTaskToken<StackfulOnlyTaskDesc<Stackful>>>>();
    same::<<Stackful as UltIdentity>::Alloc, cmpth::HeapStack>();
    same::<<Stackful as UltIdentity>::Idle, cmpth::SearchIdle>();
    same::<<Stackful as UltIdentity>::Steal, cmpth::RandomSweepSteal>();
    same::<<Stackful as UltIdentity>::Dispatch, cmpth::StandardDispatch>();
    same::<<Stackful as UltIdentity>::Duties, cmpth::duty::GlobalDuties>();
    same::<<Stackful as UltIdentity>::Observe, cmpth::NoObserver>();
    same::<<Stackful as UltIdentity>::ExternalQueue, cmpth::StealPathQueue<StackfulOnlyTaskDesc<Stackful>>>();
    same::<<Stackful as UltIdentity>::PostChoice, cmpth::DefaultPostChoice>();
    assert_eq!(<Stackful as UltIdentity>::STACK_SIZE, 128 * 1024);

    same::<<Dual as UltDualIdentity>::Base, cmpth::DefaultDualTaskSystem>();
    same::<<Dual as UltDualIdentity>::RunQueue, HybridRunQueue<SuspendedTaskToken<DualTaskDesc<UltDualSystem<Dual>>>>>();
    same::<<Dual as UltDualIdentity>::Idle, SpinIdle>();
    same::<<Dual as UltDualIdentity>::Steal, cmpth::RandomSteal>();
    same::<<Dual as UltDualIdentity>::ExternalQueue, cmpth::StealPathQueue<DualTaskDesc<UltDualSystem<Dual>>>>();
    assert_eq!(<Dual as UltDualIdentity>::ASYNC_POOL_SIZE, 512);

    same::<<Async as UltAsyncIdentity>::Desc, StacklessOnlyTaskDesc<UltAsyncSystem<Async>>>();
    same::<<Async as UltAsyncIdentity>::Steal, cmpth::RandomSweepSteal>();
    assert_eq!(<Async as UltAsyncIdentity>::ASYNC_POOL_SIZE, 256);
    same::<<Async as UltAsyncIdentity>::RunQueue, HybridRunQueue<SuspendedTaskToken<StacklessOnlyTaskDesc<UltAsyncSystem<Async>>>>>();

    same::<<Scoped as cmpth::ScopedIdentity>::Idle, cmpth::ParkIdle>();
    same::<<Scoped as cmpth::ScopedIdentity>::Steal, cmpth::RandomSweepSteal>();
    same::<<Scoped as cmpth::ScopedIdentity>::Duties, cmpth::duty::NoDuties>();

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
