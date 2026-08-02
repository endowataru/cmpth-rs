use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::traits::DelegatorConsumer;
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::stackful::suspended::StackfulOnlyResumable;
use crate::resumable::stackful::worker::StackfulWorker;
use crate::traits::ThreadSystem;
use crate::resumable::common::worker::Worker;

use super::delegator::{Delegator, DelegatorNode, SyncQueue};

// ---------------------------------------------------------------------------
// RingBufQueue
// ---------------------------------------------------------------------------

pub struct RingBufQueue<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>, const N: usize> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    head:  AtomicUsize,
    tail:  AtomicUsize,
    nodes: Box<[RingSlot<S, C>; N]>,
}

struct RingSlot<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    ready: AtomicBool,
    node:  std::cell::UnsafeCell<DelegatorNode<S, C>>,
}

unsafe impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>, const N: usize> Send
    for RingBufQueue<S, C, N> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {}
unsafe impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>, const N: usize> Sync
    for RingBufQueue<S, C, N> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {}

impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>, const N: usize> Default
    for RingBufQueue<S, C, N> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable
{
    fn default() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
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

impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>, const N: usize> RingBufQueue<S, C, N> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    fn mask(idx: usize) -> usize where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable { idx & (N - 1) }

    fn slot_node(&self, idx: usize) -> *mut DelegatorNode<S, C> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        self.nodes[Self::mask(idx)].node.get()
    }
}

impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>, const N: usize> SyncQueue<S, C>
    for RingBufQueue<S, C, N> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable
{
    fn start_lock(
        &self,
    ) -> (bool, *mut DelegatorNode<S, C>, *mut DelegatorNode<S, C>) where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        loop {
            let tail = self.tail.load(Ordering::Relaxed);
            let head = self.head.load(Ordering::Acquire);

            // Queue full: yield and retry.
            if tail.wrapping_sub(head) >= N {
                if let Some(wk) = crate::resumable::common::worker::UltWorker::<S>::current() {
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
    ) where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        // For ring buffer, "set_next" means marking the slot as ready.
        // Find which slot `cur` belongs to.
        let slot_idx = self.slot_index(cur);
        self.nodes[Self::mask(slot_idx)].ready.store(true, Ordering::Release);
    }

    fn get_head(&self) -> *mut DelegatorNode<S, C> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        self.slot_node(self.head.load(Ordering::Relaxed))
    }

    fn try_unlock(&self, _head: *mut DelegatorNode<S, C>) -> bool where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        // `head` is an *occupied* position (the slot `start_lock` handed
        // out), and `tail` is one-past-the-last-allocated slot — so "empty,
        // nobody joined after me" is `tail == head + 1`, not `tail == head`
        // (that can never be true here: whoever currently holds `head`
        // already counted themselves into `tail`). Mirrors McsQueue's own
        // `try_unlock`: CAS `tail` from "just me" back to "nothing new since
        // head", not a same-value CAS — and, same as there, a plain
        // load-then-compare here would race a concurrent `start_lock`
        // between the read and this function's caller committing to
        // "unlocked" (two threads could both conclude they're now in
        // charge). The CAS makes the transition atomic: it only succeeds if
        // `tail` is still exactly `head + 1` at the moment of the swap.
        let head = self.head.load(Ordering::Relaxed);
        self.tail.compare_exchange(
            head.wrapping_add(1),
            head,
            Ordering::AcqRel,
            Ordering::Acquire,
        ).is_ok()
    }

    fn try_follow_head(
        &self,
        head: *mut DelegatorNode<S, C>,
    ) -> Option<*mut DelegatorNode<S, C>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        // The successor's own slot is what `set_next` marks ready (it always
        // operates on `cur`, never `prev` — see `set_next` above), not the
        // current head's slot: the head's own `ready` flag was consumed (or,
        // for the very first/lock-winning holder, never set at all) long
        // before it became head. Checking the wrong slot here used to mean
        // `try_follow_head` reported "no successor yet" even when one was
        // genuinely ready, silently falling through to the slower
        // consumer_sth-notify path instead of the direct hand-off.
        let head_idx = self.head.load(Ordering::Relaxed);
        let next_idx = head_idx.wrapping_add(1);
        let slot = &self.nodes[Self::mask(next_idx)];
        if slot.ready.load(Ordering::Acquire) {
            slot.ready.store(false, Ordering::Relaxed);
            // Reset node to default for reuse.
            unsafe { *head = DelegatorNode::default() };
            self.head.store(next_idx, Ordering::Release);
            return Some(self.slot_node(next_idx));
        }
        None
    }
}

impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>, const N: usize> RingBufQueue<S, C, N> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    fn slot_index(&self, node: *mut DelegatorNode<S, C>) -> usize where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        let base = self.nodes[0].node.get() as usize;
        let size = std::mem::size_of::<RingSlot<S, C>>();
        (node as usize - base) / size
    }
}

// ---------------------------------------------------------------------------
// Public type alias
// ---------------------------------------------------------------------------

pub type RingBufDelegator<S, C, const N: usize = 256> = Delegator<S, C, RingBufQueue<S, C, N>>;
