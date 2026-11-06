use std::time::Duration;

use crate::traits::component::duty::{DutyPlace, DutyService};

/// A system's duty service: where code running on the system registers
/// its duties, and which duties the system's idle workers run.
pub trait DutySystem: 'static {
    type Duties: DutyService;

    /// [`DutyService::duty`] of [`Self::Duties`].
    #[inline]
    fn duty(period: Duration, max_period: Duration, place: DutyPlace, poll: impl Fn() -> bool + Send + Sync + 'static) -> DutyOf<Self> {
        Self::Duties::duty(period, max_period, place, poll)
    }

    /// [`DutyService::on_demand`] of [`Self::Duties`].
    #[inline]
    fn on_demand(wait_period: Duration, place: DutyPlace, poll: impl Fn() -> bool + Send + Sync + 'static) -> DutyOf<Self> {
        Self::Duties::on_demand(wait_period, place, poll)
    }

    /// [`DutyService::sleep`] of [`Self::Duties`]; on a stackful task,
    /// through `block_on`.
    #[inline]
    fn sleep(d: Duration) -> <Self::Duties as DutyService>::Sleep {
        Self::Duties::sleep(d)
    }
}

/// The duty handle of system `S`.
pub type DutyOf<S> = <<S as DutySystem>::Duties as DutyService>::Duty;

/// The toucher of a duty of system `S`.
pub type DutyToucherOf<S> = <DutyOf<S> as crate::traits::component::duty::DutyLike>::Toucher;

/// A waiter of a duty of system `S`.
pub type DutyWaiterOf<S> = <DutyOf<S> as crate::traits::component::duty::DutyLike>::Waiter;
