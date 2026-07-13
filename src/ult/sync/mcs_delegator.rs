use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::traits::DelegatorConsumer;
use crate::ult::system::UltSystem;

use super::delegator::{Delegator, DelegatorNode, SyncQueue};

// ---------------------------------------------------------------------------
// MCS queue node wrapper
// ---------------------------------------------------------------------------

struct McsEntry<S: UltSystem, C: DelegatorConsumer<S>> {
    next: AtomicPtr<McsEntry<S, C>>,
    node: DelegatorNode<S, C>,
}

// ---------------------------------------------------------------------------
// McsQueue
// ---------------------------------------------------------------------------

pub struct McsQueue<S: UltSystem, C: DelegatorConsumer<S>> {
    tail: AtomicPtr<McsEntry<S, C>>,
    // head tracks the current lock holder's entry
    head: std::cell::Cell<*mut McsEntry<S, C>>,
}

unsafe impl<S: UltSystem, C: DelegatorConsumer<S>> Send for McsQueue<S, C> {}
unsafe impl<S: UltSystem, C: DelegatorConsumer<S>> Sync for McsQueue<S, C> {}

impl<S: UltSystem, C: DelegatorConsumer<S>> Default for McsQueue<S, C> {
    fn default() -> Self {
        // Allocate a sentinel head node so start_lock always has a cur to return.
        let sentinel = Box::into_raw(Box::new(McsEntry {
            next: AtomicPtr::new(null_mut()),
            node: DelegatorNode::default(),
        }));
        McsQueue {
            tail: AtomicPtr::new(sentinel),
            head: std::cell::Cell::new(sentinel),
        }
    }
}

impl<S: UltSystem, C: DelegatorConsumer<S>> Drop for McsQueue<S, C> {
    fn drop(&mut self) {
        // Free the sentinel (and any remaining nodes, though normally none).
        let mut ptr = self.head.get();
        while !ptr.is_null() {
            let next = unsafe { (*ptr).next.load(Ordering::Acquire) };
            unsafe { drop(Box::from_raw(ptr)) };
            ptr = next;
        }
    }
}

impl<S: UltSystem, C: DelegatorConsumer<S>> SyncQueue<S, C> for McsQueue<S, C> {
    fn start_lock(
        &self,
    ) -> (bool, *mut DelegatorNode<S, C>, *mut DelegatorNode<S, C>) {
        let new_entry = Box::into_raw(Box::new(McsEntry {
            next: AtomicPtr::new(null_mut()),
            node: DelegatorNode::default(),
        }));
        let prev_tail = self.tail.swap(new_entry, Ordering::AcqRel);
        let is_locked = prev_tail == self.head.get();
        let prev_node = if is_locked {
            null_mut()
        } else {
            unsafe { &mut (*prev_tail).node as *mut DelegatorNode<S, C> }
        };
        let cur_node = unsafe { &mut (*new_entry).node as *mut DelegatorNode<S, C> };
        if is_locked {
            // We are the new holder: update head.
            self.head.set(new_entry);
        }
        (is_locked, prev_node, cur_node)
    }

    fn set_next(
        &self,
        prev: *mut DelegatorNode<S, C>,
        cur: *mut DelegatorNode<S, C>,
    ) {
        let prev_entry = entry_of(prev);
        let cur_entry = entry_of(cur);
        unsafe { (*prev_entry).next.store(cur_entry, Ordering::Release) };
    }

    fn get_head(&self) -> *mut DelegatorNode<S, C> {
        unsafe { &mut (*self.head.get()).node }
    }

    fn try_unlock(&self, head: *mut DelegatorNode<S, C>) -> bool {
        let head_entry = entry_of(head);
        self.tail
            .compare_exchange(head_entry, head_entry, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    fn try_follow_head(
        &self,
        head: *mut DelegatorNode<S, C>,
    ) -> Option<*mut DelegatorNode<S, C>> {
        let head_entry = entry_of(head);
        let next = unsafe { (*head_entry).next.load(Ordering::Acquire) };
        if next.is_null() {
            return None;
        }
        // Advance head.
        let old = self.head.get();
        self.head.set(next);
        unsafe { drop(Box::from_raw(old)) };
        Some(unsafe { &mut (*next).node })
    }
}

fn entry_of<S: UltSystem, C: DelegatorConsumer<S>>(
    node: *mut DelegatorNode<S, C>,
) -> *mut McsEntry<S, C> {
    // DelegatorNode is the `node` field of McsEntry; compute the container ptr.
    let offset = std::mem::offset_of!(McsEntry<S, C>, node);
    (node as *mut u8).wrapping_sub(offset) as *mut McsEntry<S, C>
}

// ---------------------------------------------------------------------------
// Public type alias
// ---------------------------------------------------------------------------

pub type McsDelegator<S, C> = Delegator<S, C, McsQueue<S, C>>;
