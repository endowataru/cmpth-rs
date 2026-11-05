//! QoS policies: systems with each policy run and pass the conformance
//! checks.

mod common;

use common::conformance;
use cmpth::{StackfulInitSystem, UltDualSystem};

macro_rules! dual {
    ($name:ident { $($body:tt)* }) => {
        pub struct $name;
        cmpth::worker_tls_anchor!($name);
        cmpth::impl_with_defaults! { impl cmpth::UltDualIdentity for $name { $($body)* } }
    };
}

dual!(Background { type Qos = cmpth::BackgroundQos; type Idle = cmpth::SearchIdle; });
dual!(Demote { type Qos = cmpth::DemoteIdleQos; });
dual!(Split { type Qos = cmpth::SplitQos<2>; type Idle = cmpth::SearchIdle; });

#[test]
fn policies_pass_conformance() {
    fn all<S>()
    where
        S: StackfulInitSystem + cmpth::StackfulParking + cmpth::StackfulSyncSystem,
    {
        conformance::stackful::<S>(4);
        conformance::parking::<S>(4);
        conformance::parking_from_os::<S>(4);
        conformance::sync::<S>(4);
    }
    all::<UltDualSystem<Background>>();
    all::<UltDualSystem<Demote>>();
    all::<UltDualSystem<Split>>();
}
