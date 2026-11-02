//! The conformance checks (`common/conformance.rs`) over the standard systems and over systems built
//! from each steal, idle and dispatch policy.

mod common;

use common::conformance;

macro_rules! dual {
    ($name:ident, $steal:ty, $idle:ty) => {
        pub struct $name;
        cmpth::worker_tls_anchor!($name);
        cmpth::impl_with_defaults! {
            impl cmpth::UltDualIdentity for $name {
                type Steal = $steal;
                type Idle = $idle;
            }
        }
    };
}

dual!(SweepSpin, cmpth::SweepSteal, cmpth::SpinIdle);
dual!(SweepPark, cmpth::SweepSteal, cmpth::ParkIdle);
dual!(SweepSearch, cmpth::SweepSteal, cmpth::SearchIdle);
dual!(RandomSpin, cmpth::RandomSteal, cmpth::SpinIdle);
dual!(RandomPark, cmpth::RandomSteal, cmpth::ParkIdle);
dual!(RandomSearch, cmpth::RandomSteal, cmpth::SearchIdle);
dual!(RandomSweepPark, cmpth::RandomSweepSteal, cmpth::ParkIdle);

/// Steals at task switches too.
pub struct StealOnSwitch;

impl cmpth::DispatchPolicy for StealOnSwitch {
    type State = ();
    fn next<Q: cmpth::TaskSources>(st: &(), q: &Q, site: cmpth::Site) -> cmpth::Steal<Q::Item> {
        use cmpth::{Site, StandardDispatch, Steal};
        match (site, StandardDispatch::next(st, q, site)) {
            (Site::Switch, Steal::Empty) => q.steal(),
            (_, s) => s,
        }
    }
}

pub struct SwitchStealer;
cmpth::worker_tls_anchor!(SwitchStealer);
cmpth::impl_with_defaults! {
    impl cmpth::UltIdentity for SwitchStealer {
        type Dispatch = StealOnSwitch;
        type Idle = cmpth::SearchIdle;
    }
}

const W: usize = 4;

fn stackful<S>()
where
    S: cmpth::StackfulInitSystem + cmpth::StackfulParking + cmpth::StackfulSyncSystem,
{
    conformance::stackful::<S>(W);
    conformance::parking::<S>(W);
    conformance::sync::<S>(W);
}

/// Dual systems also take wakeups from OS threads.
fn full<S>()
where
    S: cmpth::StackfulInitSystem + cmpth::StackfulParking + cmpth::StackfulSyncSystem,
{
    stackful::<S>();
    conformance::parking_from_os::<S>(W);
}

#[test]
fn standard_stackful() {
    stackful::<cmpth::DefaultStackfulOnlyTaskSystem>();
}

#[test]
fn standard_dual() {
    full::<cmpth::DefaultDualTaskSystem>();
    full::<cmpth::ParkDualTaskSystem>();
    full::<cmpth::ParkHalfDualTaskSystem>();
}

#[test]
fn steal_and_idle_policies() {
    type U<M> = cmpth::UltDualSystem<M>;
    full::<U<SweepSpin>>();
    full::<U<SweepPark>>();
    full::<U<SweepSearch>>();
    full::<U<RandomSpin>>();
    full::<U<RandomPark>>();
    full::<U<RandomSearch>>();
    full::<U<RandomSweepPark>>();
}

#[test]
fn dispatch_policy() {
    stackful::<SwitchStealer>();
}

#[test]
fn scoped() {
    conformance::scoped::<cmpth::ScopedTaskSystem>(W);
}

#[test]
fn stackless() {
    conformance::stackless::<cmpth::DefaultDualTaskSystem>(W);
}
