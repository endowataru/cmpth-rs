//! Dispatch policies ([`DispatchPolicy`]).

use crate::resumable::common::deque::Steal;

pub use crate::traits::component::dispatch::{DispatchPolicy, Site, TaskSources};

/// The standard order:
///
/// * [`Site::Loop`] and [`Site::Wait`]: own queue, then the pool's yielded
///   tasks, then a steal round, then the external queue;
/// * [`Site::Switch`]: the external queue, then the own queue, then the
///   yielded tasks, never a steal (the worker steals from its loop).  The
///   external queue comes first because tasks that only yield to each
///   other never let the worker reach its loop.
#[derive(Default)]
pub struct StandardDispatch;

impl DispatchPolicy for StandardDispatch {
    type State = ();

    #[inline(always)]
    fn next<Q: TaskSources>(_state: &(), q: &Q, site: Site) -> Steal<Q::Item> {
        match site {
            Site::Loop | Site::Wait => {
                if let Some(c) = q.local() {
                    return Steal::Success(c);
                }
                if let Some(c) = q.yielded() {
                    return Steal::Success(c);
                }
                let s = q.steal();
                if let Steal::Success(_) = s {
                    return s;
                }
                if let Some(c) = q.external() {
                    return Steal::Success(c);
                }
                s
            }
            Site::Switch => {
                if let Some(c) = q.external() {
                    return Steal::Success(c);
                }
                if let Some(c) = q.local() {
                    return Steal::Success(c);
                }
                match q.yielded() {
                    Some(c) => Steal::Success(c),
                    None => Steal::Empty,
                }
            }
        }
    }
}
