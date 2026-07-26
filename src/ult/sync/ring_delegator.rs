use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::traits::DelegatorConsumer;
use crate::ult::desc::{StackfulTaskDesc, WakerTaskDesc};
use crate::ult::system::{SchedulerSystem, UltSchedulerSystem};
use crate::ult::worker::StackfulWorker;
use crate::traits::UltSystem;
use crate::ult::worker::Worker;

use super::delegator::{Delegator, DelegatorNode, SyncQueue};

// ---------------------------------------------------------------------------
// RingBufQueue
// ---------------------------------------------------------------------------

pub struct RingBufQueue<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, const N: usize> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    head:  AtomicUsize,
    tail:  AtomicUsize,
    nodes: Box<[RingSlot<S, C>; N]>,
}

struct RingSlot<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    ready: AtomicBool,
    node:  std::cell::UnsafeCell<DelegatorNode<S, C>>,
}

unsafe impl<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, const N: usize> Send
    for RingBufQueue<S, C, N> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {}
unsafe impl<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, const N: usize> Sync
    for RingBufQueue<S, C, N> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {}

impl<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, const N: usize> Default
    for RingBufQueue<S, C, N> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc
{
    fn default() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
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
            tail:  AtomicUsize::new(0),
            nodes: nodes.into_boxed_slice().try_into().unwrap_or_else(|_| {
                panic!("RingBufQueue: size mismatch")
            }),
        }
    }
}

impl<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, const N: usize> RingBufQueue<S, C, N> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    fn mask(idx: usize) -> usize where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc { idx & (N - 1) }

    fn slot_node(&self, idx: usize) -> *mut DelegatorNode<S, C> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        self.nodes[Self::mask(idx)].node.get()
    }
}

impl<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, const N: usize> SyncQueue<S, C>
    for RingBufQueue<S, C, N> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc
{
    fn start_lock(
        &self,
    ) -> (bool, *mut DelegatorNode<S, C>, *mut DelegatorNode<S, C>) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        loop {
            let tail = self.tail.load(Ordering::Relaxed);
            let head = self.head.load(Ordering::Acquire);

            // Queue full: yield and retry.
            if tail.wrapping_sub(head) >= N {
                if let Some(wk) = crate::ult::worker::UltWorker::<S>::current() {
                    wk.yield_now();
                } else {
                    std::hint::spin_loop();
                }
                continue;
            }

            let is_locked = tail == head;
            match self.tail.compare_exchange_weak(
                tail, tail.wrapping_add(1), Ordering::AcqRel, Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let prev_node = if is_locked {
                        std::ptr::null_mut()
                    } else {
                        self.slot_node(tail.wrapping_sub(1))
                    };
                    return (is_locked, prev_node, self.slot_node(tail));
                }
                Err(_) => continue,
            }
        }
    }

    fn set_next(
        &self,
        _prev: *mut DelegatorNode<S, C>,
        cur: *mut DelegatorNode<S, C>,
    ) where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        // For ring buffer, "set_next" means marking the slot as ready.
        // Find which slot `cur` belongs to.
        let slot_idx = self.slot_index(cur);
        self.nodes[Self::mask(slot_idx)].ready.store(true, Ordering::Release);
    }

    fn get_head(&self) -> *mut DelegatorNode<S, C> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        self.slot_node(self.head.load(Ordering::Relaxed))
    }

    fn try_unlock(&self, _head: *mut DelegatorNode<S, C>) -> bool where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return true; // empty, effectively unlocked
        }
        false
    }

    fn try_follow_head(
        &self,
        head: *mut DelegatorNode<S, C>,
    ) -> Option<*mut DelegatorNode<S, C>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        let head_idx = self.head.load(Ordering::Relaxed);
        let slot = &self.nodes[Self::mask(head_idx)];
        if slot.ready.load(Ordering::Acquire) {
            slot.ready.store(false, Ordering::Relaxed);
            // Reset node to default for reuse.
            unsafe { *head = DelegatorNode::default() };
            let next_idx = head_idx.wrapping_add(1);
            self.head.store(next_idx, Ordering::Release);
            return Some(self.slot_node(next_idx));
        }
        None
    }
}

impl<S: UltSchedulerSystem + UltSystem, C: DelegatorConsumer<S>, const N: usize> RingBufQueue<S, C, N> where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
    fn slot_index(&self, node: *mut DelegatorNode<S, C>) -> usize where <S as SchedulerSystem>::Desc: StackfulTaskDesc + WakerTaskDesc {
        let base = self.nodes[0].node.get() as usize;
        let size = std::mem::size_of::<RingSlot<S, C>>();
        (node as usize - base) / size
    }
}

// ---------------------------------------------------------------------------
// Public type alias
// ---------------------------------------------------------------------------

pub type RingBufDelegator<S, C, const N: usize = 256> = Delegator<S, C, RingBufQueue<S, C, N>>;
