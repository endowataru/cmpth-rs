use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::traits::DelegatorConsumer;
use crate::resumable::stackful::desc::StackfulTaskDesc;
use crate::resumable::common::system::SchedulerSystem;
use crate::resumable::stackful::system::StackfulSchedulerSystem;
use crate::resumable::stackful::suspended::StackfulOnlyResumable;
use crate::traits::ThreadSystem;

use super::delegator::{Delegator, DelegatorNode, SyncQueue};

// ---------------------------------------------------------------------------
// MCS queue node wrapper
// ---------------------------------------------------------------------------

struct McsEntry<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    next: AtomicPtr<McsEntry<S, C>>,
    node: DelegatorNode<S, C>,
}

// ---------------------------------------------------------------------------
// McsQueue
// ---------------------------------------------------------------------------

pub struct McsQueue<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    tail: AtomicPtr<McsEntry<S, C>>,
    // head tracks the current lock holder's entry
    head: std::cell::Cell<*mut McsEntry<S, C>>,
}

unsafe impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>> Send for McsQueue<S, C> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {}
unsafe impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>> Sync for McsQueue<S, C> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {}

impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>> Default for McsQueue<S, C> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    fn default() -> Self where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        // No sentinel: `tail`/`head` start genuinely null, matching the C++
        // reference (`basic_mcs_core.hpp`: `tail_{nullptr}`, `head_` defaults
        // null). A prior version pre-allocated a sentinel and compared
        // against it in `start_lock`/`try_unlock` instead of against null —
        // that's a different (and wrong) condition: it let a second caller
        // "win" the lock immediately by coincidentally matching the
        // sentinel's address, and it never let a fully-drained queue return
        // to a state where a future caller *could* win immediately. See
        // `start_lock`/`try_unlock` below.
        McsQueue {
            tail: AtomicPtr::new(null_mut()),
            head: std::cell::Cell::new(null_mut()),
        }
    }
}

impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>> Drop for McsQueue<S, C> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    fn drop(&mut self) where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        // Free the sentinel (and any remaining nodes, though normally none).
        let mut ptr = self.head.get();
        while !ptr.is_null() {
            let next = unsafe { (*ptr).next.load(Ordering::Acquire) };
            unsafe { drop(Box::from_raw(ptr)) };
            ptr = next;
        }
    }
}

impl<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>> SyncQueue<S, C> for McsQueue<S, C> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    fn start_lock(
        &self,
    ) -> (bool, *mut DelegatorNode<S, C>, *mut DelegatorNode<S, C>) where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        let new_entry = Box::into_raw(Box::new(McsEntry {
            next: AtomicPtr::new(null_mut()),
            node: DelegatorNode::default(),
        }));
        let prev_tail = self.tail.swap(new_entry, Ordering::AcqRel);
        // Win immediately iff the queue was genuinely empty (tail was null),
        // matching `basic_mcs_core.hpp::start_lock`'s `prev == nullptr` — NOT
        // "prev happens to equal the current head", which is a different,
        // incorrect condition (see the comment on `Default` above).
        let is_locked = prev_tail.is_null();
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
    ) where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        let prev_entry = entry_of(prev);
        let cur_entry = entry_of(cur);
        unsafe { (*prev_entry).next.store(cur_entry, Ordering::Release) };
    }

    fn get_head(&self) -> *mut DelegatorNode<S, C> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        unsafe { &mut (*self.head.get()).node }
    }

    fn try_unlock(&self, head: *mut DelegatorNode<S, C>) -> bool where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
        let head_entry = entry_of(head);
        // Standard MCS unlock (`basic_mcs_core.hpp::try_unlock`): CAS `tail`
        // from `head_entry` to *null* (not to `head_entry` again — a
        // same-value CAS never actually releases the queue, so no future
        // caller could ever win the fast path again after the first
        // lock/unlock cycle, which was the second half of this bug).
        // `self.head` is proactively cleared first and restored on failure,
        // matching C++ exactly — `try_follow_head` unconditionally reads and
        // frees whatever `self.head` currently holds, so it must be correct
        // (either null, meaning "fully unlocked", or the original
        // `head_entry`) by the time either function is next called.
        self.head.set(null_mut());
        match self.tail.compare_exchange(
            head_entry,
            null_mut(),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => true,
            Err(_) => {
                self.head.set(head_entry);
                false
            }
        }
    }

    fn try_follow_head(
        &self,
        head: *mut DelegatorNode<S, C>,
    ) -> Option<*mut DelegatorNode<S, C>> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
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

fn entry_of<S: StackfulSchedulerSystem + ThreadSystem, C: DelegatorConsumer<S>>(
    node: *mut DelegatorNode<S, C>,
) -> *mut McsEntry<S, C> where <S as SchedulerSystem>::Desc: StackfulTaskDesc, <S as ThreadSystem>::SuspendedThread: StackfulOnlyResumable {
    // DelegatorNode is the `node` field of McsEntry; compute the container ptr.
    let offset = std::mem::offset_of!(McsEntry<S, C>, node);
    (node as *mut u8).wrapping_sub(offset) as *mut McsEntry<S, C>
}

// ---------------------------------------------------------------------------
// Public type alias
// ---------------------------------------------------------------------------

pub type McsDelegator<S, C> = Delegator<S, C, McsQueue<S, C>>;
