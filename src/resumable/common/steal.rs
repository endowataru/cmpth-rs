//! Steal policies ([`StealPolicy`]).

use std::cell::Cell;

use crate::resumable::common::deque::{RunQueueStealer, Steal};

pub use crate::traits::component::steal::StealPolicy;

/// Tries every other worker once, starting one place further on each round.
#[derive(Default)]
pub struct SweepSteal;

impl StealPolicy for SweepSteal {
    type State = Cell<usize>;

    fn new_state(me: usize) -> Cell<usize> {
        Cell::new(me.wrapping_mul(0x9E37_79B9).wrapping_add(1))
    }

    fn try_steal<T: Send, St: RunQueueStealer<T>>(state: &Cell<usize>, me: usize, victims: &[St]) -> Steal<T> {
        let start = state.get();
        state.set(start.wrapping_add(1));
        sweep(start, me, victims)
    }
}

/// Tries one other worker chosen uniformly at random per round.
#[derive(Default)]
pub struct RandomSteal;

impl StealPolicy for RandomSteal {
    type State = Cell<u64>;

    fn new_state(me: usize) -> Cell<u64> {
        Cell::new(seed(me))
    }

    fn try_steal<T: Send, St: RunQueueStealer<T>>(state: &Cell<u64>, me: usize, victims: &[St]) -> Steal<T> {
        let n = victims.len();
        let mut v = (next(state) % (n as u64 - 1)) as usize;
        if v >= me {
            v += 1;
        }
        victims[v].try_steal()
    }
}

/// Tries every other worker once, starting at a random one each round.
#[derive(Default)]
pub struct RandomSweepSteal;

impl StealPolicy for RandomSweepSteal {
    type State = Cell<u64>;

    fn new_state(me: usize) -> Cell<u64> {
        Cell::new(seed(me))
    }

    fn try_steal<T: Send, St: RunQueueStealer<T>>(state: &Cell<u64>, me: usize, victims: &[St]) -> Steal<T> {
        let start = (next(state) % victims.len() as u64) as usize;
        sweep(start, me, victims)
    }
}

/// Every victim but `me`, from `start` (mod `n`) on.
#[inline]
fn sweep<T: Send, St: RunQueueStealer<T>>(start: usize, me: usize, victims: &[St]) -> Steal<T> {
    let n = victims.len();
    let mut saw_retry = false;
    for i in 0..n {
        let victim = start.wrapping_add(i) % n;
        if victim == me {
            continue;
        }
        match victims[victim].try_steal() {
            Steal::Success(c) => return Steal::Success(c),
            Steal::Retry => saw_retry = true,
            Steal::Empty => {}
        }
    }
    if saw_retry { Steal::Retry } else { Steal::Empty }
}

/// A distinct non-zero xorshift state per worker (splitmix64 of its index).
fn seed(me: usize) -> u64 {
    let mut z = (me as u64).wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) | 1
}

/// xorshift64*.
#[inline]
fn next(state: &Cell<u64>) -> u64 {
    let mut x = state.get();
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state.set(x);
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stealer that records being tried and is always empty.
    #[derive(Clone)]
    struct Probe<'a>(&'a Cell<u32>);
    // SAFETY: test-only, used from one thread.
    unsafe impl Send for Probe<'_> {}
    unsafe impl Sync for Probe<'_> {}
    impl RunQueueStealer<u32> for Probe<'static> {
        fn try_steal(&self) -> Steal<u32> {
            self.0.set(self.0.get() + 1);
            Steal::Empty
        }
    }

    fn probes(n: usize) -> Vec<Probe<'static>> {
        (0..n).map(|_| Probe(Box::leak(Box::new(Cell::new(0))))).collect()
    }

    #[test]
    fn sweeps_try_everyone_but_me() {
        let v = probes(5);
        let s = SweepSteal::new_state(2);
        assert!(matches!(SweepSteal::try_steal(&s, 2, &v), Steal::Empty));
        let r = RandomSweepSteal::new_state(2);
        assert!(matches!(RandomSweepSteal::try_steal(&r, 2, &v), Steal::Empty));
        let hits: Vec<u32> = v.iter().map(|p| p.0.get()).collect();
        assert_eq!(hits, [2, 2, 0, 2, 2]);
    }

    #[test]
    fn random_never_picks_me_and_reaches_everyone() {
        let v = probes(4);
        let s = RandomSteal::new_state(1);
        for _ in 0..400 {
            let _ = RandomSteal::try_steal(&s, 1, &v);
        }
        let hits: Vec<u32> = v.iter().map(|p| p.0.get()).collect();
        assert_eq!(hits[1], 0);
        assert_eq!(hits.iter().sum::<u32>(), 400);
        assert!(hits.iter().enumerate().all(|(i, &h)| i == 1 || h > 80), "{hits:?}");
    }
}
