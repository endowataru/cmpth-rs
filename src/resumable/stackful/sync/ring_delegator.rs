use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::traits::DelegatorConsumer;
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::common::system::PoolSystem;
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::stackful::worker::StackfulWorker;
use crate::traits::{SpawnableStackfulTaskSystem, SuspendableSystem};
use crate::resumable::common::worker::WorkerOps;

use super::delegator::{Delegator, DelegatorNode, SyncQueue};

// ---------------------------------------------------------------------------
// RingBufQueue
// ---------------------------------------------------------------------------

/// `tail` holds the number of claimed slots shifted left by one; its low
/// bit (`FREE`) says nobody holds the lock.  Releasing sets the bit
/// rather than moving `tail` back, so `tail` never repeats a value: a
/// caller that read `head` and then lost a race to a hand-over followed by
/// a release fails its compare-exchange instead of queueing behind a
/// holder that is gone.
pub struct RingBufQueue<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>, const N: usize> where <S as PoolSystem>::Desc: StackfulTaskDesc {
    head:  AtomicUsize,
    tail:  AtomicUsize,
    nodes: Box<[RingSlot<S, C>; N]>,
}

struct RingSlot<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>> where <S as PoolSystem>::Desc: StackfulTaskDesc {
    ready: AtomicBool,
    node:  std::cell::UnsafeCell<DelegatorNode<S, C>>,
}

/// The low bit of `RingBufQueue::tail`: the lock is free.
const FREE: usize = 1;

unsafe impl<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>, const N: usize> Send
    for RingBufQueue<S, C, N> where <S as PoolSystem>::Desc: StackfulTaskDesc {}
unsafe impl<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>, const N: usize> Sync
    for RingBufQueue<S, C, N> where <S as PoolSystem>::Desc: StackfulTaskDesc {}

impl<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>, const N: usize> Default
    for RingBufQueue<S, C, N> where <S as PoolSystem>::Desc: StackfulTaskDesc
{
    fn default() -> Self where <S as PoolSystem>::Desc: StackfulTaskDesc {
        assert!(N.is_power_of_two(), "RingBufQueue capacity must be a power of two");
        // SAFETY: array of UnsafeCell<DelegatorNode> initialized to Default.
        let nodes: Vec<RingSlot<S, C>> = (0..N)
            .map(|_| RingSlot {
                ready: AtomicBool::new(false),
                node:  std::cell::UnsafeCell::new(DelegatorNode::default()),
            })
            .collect();
        RingBufQueue {
            head:  AtomicUsize::new(0),
            tail:  AtomicUsize::new(FREE),
            nodes: nodes.into_boxed_slice().try_into().unwrap_or_else(|_| {
                panic!("RingBufQueue: size mismatch")
            }),
        }
    }
}

impl<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>, const N: usize> RingBufQueue<S, C, N> where <S as PoolSystem>::Desc: StackfulTaskDesc {
    fn mask(idx: usize) -> usize where <S as PoolSystem>::Desc: StackfulTaskDesc { idx & (N - 1) }

    fn slot_node(&self, idx: usize) -> *mut DelegatorNode<S, C> where <S as PoolSystem>::Desc: StackfulTaskDesc {
        self.nodes[Self::mask(idx)].node.get()
    }
}

impl<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>, const N: usize> SyncQueue<S, C>
    for RingBufQueue<S, C, N>
where
    <S as PoolSystem>::Desc: StackfulTaskDesc,
    S::Worker: StackfulWorker<S>,
{
    fn start_lock(
        &self,
    ) -> (bool, *mut DelegatorNode<S, C>, *mut DelegatorNode<S, C>) where <S as PoolSystem>::Desc: StackfulTaskDesc {
        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let idx = tail >> 1;
            let is_locked = tail & FREE != 0;

            // Queue full: yield and retry.  (A stale `head` only makes
            // this more cautious.)
            if !is_locked && idx.wrapping_sub(self.head.load(Ordering::Acquire)) >= N {
                if let Some(wk) = S::Worker::current() {
                    wk.yield_now();
                } else {
                    std::hint::spin_loop();
                }
                continue;
            }

            match self.tail.compare_exchange_weak(
                tail, idx.wrapping_add(1) << 1, Ordering::AcqRel, Ordering::Relaxed,
            ) {
                Ok(_) => {
                    if is_locked {
                        // The holder occupies the slot it claimed.
                        self.head.store(idx, Ordering::Release);
                        return (true, std::ptr::null_mut(), self.slot_node(idx));
                    }
                    return (false, self.slot_node(idx.wrapping_sub(1)), self.slot_node(idx));
                }
                Err(_) => continue,
            }
        }
    }

    fn set_next(
        &self,
        _prev: *mut DelegatorNode<S, C>,
        cur: *mut DelegatorNode<S, C>,
    ) where <S as PoolSystem>::Desc: StackfulTaskDesc {
        // For ring buffer, "set_next" means marking the slot as ready.
        // Find which slot `cur` belongs to.
        let slot_idx = self.slot_index(cur);
        self.nodes[Self::mask(slot_idx)].ready.store(true, Ordering::Release);
    }

    fn get_head(&self) -> *mut DelegatorNode<S, C> where <S as PoolSystem>::Desc: StackfulTaskDesc {
        self.slot_node(self.head.load(Ordering::Relaxed))
    }

    fn try_unlock(&self, _head: *mut DelegatorNode<S, C>) -> bool where <S as PoolSystem>::Desc: StackfulTaskDesc {
        // The holder occupies slot `head`; with no one behind it `tail`
        // counts `head + 1` slots.  Releasing marks the lock free: the next
        // caller claims slot `head + 1` and holds the lock.
        let head = self.head.load(Ordering::Relaxed);
        let tail = head.wrapping_add(1) << 1;
        self.tail
            .compare_exchange(tail, tail | FREE, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    }

    fn try_follow_head(
        &self,
        head: *mut DelegatorNode<S, C>,
    ) -> Option<*mut DelegatorNode<S, C>> where <S as PoolSystem>::Desc: StackfulTaskDesc {
        // The successor is the slot after the holder's, once its caller has
        // linked (marked it ready).
        let head_idx = self.head.load(Ordering::Relaxed);
        let next_idx = head_idx.wrapping_add(1);
        let slot = &self.nodes[Self::mask(next_idx)];
        if slot.ready.load(Ordering::Acquire) {
            slot.ready.store(false, Ordering::Relaxed);
            // Reset the old head for reuse.
            unsafe { *head = DelegatorNode::default() };
            self.head.store(next_idx, Ordering::Release);
            return Some(self.slot_node(next_idx));
        }
        None
    }
}

impl<S: StackfulSchedulerSystem + SpawnableStackfulTaskSystem + SuspendableSystem, C: DelegatorConsumer<S>, const N: usize> RingBufQueue<S, C, N> where <S as PoolSystem>::Desc: StackfulTaskDesc {
    fn slot_index(&self, node: *mut DelegatorNode<S, C>) -> usize where <S as PoolSystem>::Desc: StackfulTaskDesc {
        let base = self.nodes[0].node.get() as usize;
        let size = std::mem::size_of::<RingSlot<S, C>>();
        (node as usize - base) / size
    }
}

// ---------------------------------------------------------------------------
// Public type alias
// ---------------------------------------------------------------------------

pub type RingBufDelegator<S, C, const N: usize = 256> = Delegator<S, C, RingBufQueue<S, C, N>>;
