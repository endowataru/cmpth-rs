//! Post delegators: [`PostDelegator`] implementations, and the default
//! choice between them.
//!
//! Both keep one slot per posting OS thread, registered at its first post
//! and kept until the delegator is dropped; a post bumps the slot's count,
//! which only that thread writes.  They differ in how a server finds the
//! posts:
//!
//! - [`SlotPostDelegator`] (ffwd-like) looks at every slot.  A post is a
//!   load and a store to the thread's own cache line; a serve reads every
//!   producer's line, and the server polls on a period (backing off while
//!   nothing is posted), so an idle delegator still costs its polls.
//! - [`QueuePostDelegator`] keeps a list of the slots posted since the
//!   last serve.  A post adds an atomic swap on its own line, and the
//!   first post after a serve links the slot into the list and asks for a
//!   poll; a serve touches only listed slots, and nothing polls while
//!   nothing is posted.
//!
//! Unprompted serves run as a [`Duty`].

use std::cell::{RefCell, UnsafeCell};
use std::pin::Pin;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use crate::duty::Duty;
use crate::traits::{PostChoice, PostConfig, PostConsumer, PostDelegator, Ticket};

// ---------------------------------------------------------------------------
// Producers: one slot per posting thread
// ---------------------------------------------------------------------------

/// The part of a slot its thread writes.
#[repr(align(128))]
struct Hot {
    /// Post count, written only by the slot's thread (`Release`).
    mark: AtomicU64,
    /// Listed (or being listed) in the pending list (queue variant).
    queued: AtomicBool,
    next_pending: AtomicPtr<Slot>,
}

/// The part only the server touches.
#[repr(align(128))]
struct Cold {
    /// The count the last serve covered.
    seen: UnsafeCell<u64>,
}

struct Slot {
    hot: Hot,
    cold: Cold,
    producer: u32,
    /// Registration list; fixed once linked.
    next: *const Slot,
}

/// Unique per delegator, never reused: a thread's slot cache keys on it.
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// This thread's slot in each delegator it posted to.  Entries of
    /// dropped delegators stay, unmatched (their ids are never reused).
    static MINE: RefCell<Vec<(u64, *const Slot)>> = const { RefCell::new(Vec::new()) };
}

struct Producers {
    id: u64,
    head: AtomicPtr<Slot>,
    count: AtomicU32,
}

impl Producers {
    fn new() -> Self {
        Producers { id: NEXT_ID.fetch_add(1, Ordering::Relaxed), head: AtomicPtr::new(null_mut()), count: AtomicU32::new(0) }
    }

    /// This OS thread's slot.  Not inlined: a ULT may resume on another
    /// thread, and an inlined thread-local address could be reused across
    /// a switch in the caller.
    #[inline(never)]
    fn mine(&self) -> &Slot {
        let p = MINE.with(|m| {
            let mut m = m.borrow_mut();
            if let Some(&(_, s)) = m.iter().find(|(id, _)| *id == self.id) {
                return s;
            }
            let s = self.register();
            m.push((self.id, s));
            s
        });
        unsafe { &*p }
    }

    #[cold]
    fn register(&self) -> *const Slot {
        let s = Box::into_raw(Box::new(Slot {
            hot: Hot { mark: AtomicU64::new(0), queued: AtomicBool::new(false), next_pending: AtomicPtr::new(null_mut()) },
            cold: Cold { seen: UnsafeCell::new(0) },
            producer: self.count.fetch_add(1, Ordering::Relaxed),
            next: std::ptr::null(),
        }));
        let mut h = self.head.load(Ordering::Relaxed);
        loop {
            unsafe { (*s).next = h };
            match self.head.compare_exchange_weak(h, s, Ordering::Release, Ordering::Relaxed) {
                Ok(_) => return s,
                Err(cur) => h = cur,
            }
        }
    }

    /// A post: the ticket, and the slot.
    #[inline]
    fn bump(&self) -> (Ticket, &Slot) {
        let s = self.mine();
        let seq = s.hot.mark.load(Ordering::Relaxed) + 1;
        s.hot.mark.store(seq, Ordering::Release);
        (Ticket { producer: s.producer, seq }, s)
    }

    fn iter(&self) -> impl Iterator<Item = &Slot> {
        let mut p = self.head.load(Ordering::Acquire) as *const Slot;
        std::iter::from_fn(move || {
            let s = unsafe { p.as_ref()? };
            p = s.next;
            Some(s)
        })
    }
}

impl Drop for Producers {
    fn drop(&mut self) {
        let mut p = *self.head.get_mut();
        while !p.is_null() {
            let s = unsafe { Box::from_raw(p) };
            p = s.next as *mut Slot;
        }
    }
}

// ---------------------------------------------------------------------------
// Server: the consumer behind a try-lock
// ---------------------------------------------------------------------------

struct Server<C> {
    busy: AtomicBool,
    consumer: UnsafeCell<C>,
    batch: UnsafeCell<Vec<Ticket>>,
}

impl<C> Server<C> {
    fn new(consumer: C) -> Self {
        Server { busy: AtomicBool::new(false), consumer: UnsafeCell::new(consumer), batch: UnsafeCell::new(Vec::new()) }
    }

    #[inline]
    fn try_lock(&self) -> bool {
        !self.busy.load(Ordering::Relaxed)
            && self.busy.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok()
    }

    fn lock(&self) {
        let mut spins = 0u32;
        while !self.try_lock() {
            if spins < 64 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
    }

    fn unlock(&self) {
        self.busy.store(false, Ordering::Release);
    }
}

/// Note `s` in the batch if it has posts the last serve did not cover.
/// Server only.
#[inline]
fn collect(s: &Slot, batch: &mut Vec<Ticket>) {
    let m = s.hot.mark.load(Ordering::Acquire);
    let seen = unsafe { &mut *s.cold.seen.get() };
    if m != *seen {
        *seen = m;
        batch.push(Ticket { producer: s.producer, seq: m });
    }
}

struct SendPtr<T>(*const T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

// ---------------------------------------------------------------------------
// SlotPostDelegator
// ---------------------------------------------------------------------------

/// Posting first (see the module doc comment).
pub struct SlotPostDelegator<C: PostConsumer> {
    producers: Producers,
    server: Server<C>,
    duty: OnceLock<Duty>,
}

// SAFETY: the consumer and the slots' server parts are touched only under
// the server lock.
unsafe impl<C: PostConsumer> Send for SlotPostDelegator<C> {}
unsafe impl<C: PostConsumer> Sync for SlotPostDelegator<C> {}

/// How far the polls of an idle [`SlotPostDelegator`] back off, in
/// multiples of its latency.
const SLOT_BACKOFF: u32 = 16;

impl<C: PostConsumer> PostDelegator<C> for SlotPostDelegator<C> {
    fn start(consumer: C, cfg: PostConfig) -> Pin<Box<Self>> {
        let d = Box::pin(SlotPostDelegator { producers: Producers::new(), server: Server::new(consumer), duty: OnceLock::new() });
        let p = SendPtr(&*d as *const Self);
        // A post seen late still waits up to the backed-off period.
        let duty = Duty::new(cfg.latency, cfg.latency * SLOT_BACKOFF, cfg.place, move || {
            let p = &p;
            unsafe { (*p.0).serve_now() }
        });
        let _ = d.duty.set(duty);
        d
    }

    #[inline]
    fn post(&self) -> Ticket {
        self.producers.bump().0
    }

    fn serve_now(&self) -> bool {
        if !self.server.try_lock() {
            return false;
        }
        let batch = unsafe { &mut *self.server.batch.get() };
        batch.clear();
        for s in self.producers.iter() {
            collect(s, batch);
        }
        let did = !batch.is_empty();
        if did {
            unsafe { (*self.server.consumer.get()).serve(batch) };
        }
        self.server.unlock();
        did
    }

    fn with_lock<R>(&self, f: impl FnOnce(&mut C) -> R) -> R {
        self.server.lock();
        let r = f(unsafe { &mut *self.server.consumer.get() });
        self.server.unlock();
        r
    }
}

impl<C: PostConsumer> Drop for SlotPostDelegator<C> {
    fn drop(&mut self) {
        // No poll runs once the duty is dropped.
        drop(self.duty.take());
    }
}

// ---------------------------------------------------------------------------
// QueuePostDelegator
// ---------------------------------------------------------------------------

/// Serving first (see the module doc comment).
pub struct QueuePostDelegator<C: PostConsumer> {
    producers: Producers,
    /// Slots posted since they were last served (a Treiber stack the
    /// server takes whole).
    pending: AtomicPtr<Slot>,
    server: Server<C>,
    latency: std::time::Duration,
    duty: OnceLock<Duty>,
}

// SAFETY: as for `SlotPostDelegator`.
unsafe impl<C: PostConsumer> Send for QueuePostDelegator<C> {}
unsafe impl<C: PostConsumer> Sync for QueuePostDelegator<C> {}

impl<C: PostConsumer> QueuePostDelegator<C> {
    /// Ask for a poll within the latency.
    #[cold]
    fn ask(&self) {
        if let Some(d) = self.duty.get() {
            d.set_deadline(Some(Instant::now() + self.latency));
        }
    }
}

impl<C: PostConsumer> PostDelegator<C> for QueuePostDelegator<C> {
    fn start(consumer: C, cfg: PostConfig) -> Pin<Box<Self>> {
        let d = Box::pin(QueuePostDelegator {
            producers: Producers::new(),
            pending: AtomicPtr::new(null_mut()),
            server: Server::new(consumer),
            latency: cfg.latency,
            duty: OnceLock::new(),
        });
        let p = SendPtr(&*d as *const Self);
        let duty = Duty::on_demand(cfg.latency, cfg.place, move || {
            let p = &p;
            let d = unsafe { &*p.0 };
            let did = d.serve_now();
            // Posted during the serve (or someone else was serving): the
            // poll cleared the deadline, so ask again.
            if !d.pending.load(Ordering::Relaxed).is_null() {
                d.ask();
            }
            did
        });
        let _ = d.duty.set(duty);
        d
    }

    #[inline]
    fn post(&self) -> Ticket {
        let (t, s) = self.producers.bump();
        // `swap`, not a load: the server clears `queued` with a swap too,
        // so one of the two reads the other's write.  Either the server's
        // clear comes later and its `Acquire` sees our count, or ours does
        // and we list the slot again.
        if !s.hot.queued.swap(true, Ordering::AcqRel) {
            let sp = s as *const Slot as *mut Slot;
            let mut h = self.pending.load(Ordering::Relaxed);
            loop {
                s.hot.next_pending.store(h, Ordering::Relaxed);
                match self.pending.compare_exchange_weak(h, sp, Ordering::Release, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(cur) => h = cur,
                }
            }
            if h.is_null() {
                self.ask();
            }
        }
        t
    }

    fn serve_now(&self) -> bool {
        if self.pending.load(Ordering::Relaxed).is_null() || !self.server.try_lock() {
            return false;
        }
        let batch = unsafe { &mut *self.server.batch.get() };
        batch.clear();
        let mut p = self.pending.swap(null_mut(), Ordering::Acquire);
        while let Some(s) = unsafe { p.as_ref() } {
            // Read the link before clearing `queued`: the slot may be
            // listed again right after.
            p = s.hot.next_pending.load(Ordering::Relaxed);
            s.hot.queued.swap(false, Ordering::AcqRel);
            collect(s, batch);
        }
        let did = !batch.is_empty();
        if did {
            unsafe { (*self.server.consumer.get()).serve(batch) };
        }
        self.server.unlock();
        did
    }

    fn with_lock<R>(&self, f: impl FnOnce(&mut C) -> R) -> R {
        self.server.lock();
        let r = f(unsafe { &mut *self.server.consumer.get() });
        self.server.unlock();
        r
    }
}

impl<C: PostConsumer> Drop for QueuePostDelegator<C> {
    fn drop(&mut self) {
        drop(self.duty.take());
    }
}

// ---------------------------------------------------------------------------
// Default choice
// ---------------------------------------------------------------------------

/// [`SlotPostDelegator`] when posting comes first, [`QueuePostDelegator`]
/// when serving does.
pub struct DefaultPostChoice;

impl PostChoice for DefaultPostChoice {
    type Posting<C: PostConsumer> = SlotPostDelegator<C>;
    type Serving<C: PostConsumer> = QueuePostDelegator<C>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duty::DutyPlace;
    use crate::traits::{FavorPosting, FavorServing, PostDelegatorOf, PostingSystem};
    use std::sync::Arc;
    use std::time::Duration;

    /// Records, per producer, the highest count served; checks that the
    /// data each producer wrote before posting is visible at the serve.
    struct Check<F> {
        data: Arc<Vec<AtomicU64>>,
        served: Arc<Vec<AtomicU64>>,
        serves: Arc<AtomicU64>,
        _f: std::marker::PhantomData<fn() -> F>,
    }

    impl<F: crate::traits::PostFavor> PostConsumer for Check<F> {
        type Favor = F;
        fn serve(&mut self, upto: &[Ticket]) {
            self.serves.fetch_add(1, Ordering::Relaxed);
            for t in upto {
                let p = t.producer as usize;
                assert!(t.seq > self.served[p].load(Ordering::Relaxed), "seq went back");
                // from its second post on, the producer wrote data[p] = seq
                // (relaxed) before posting
                assert!(t.seq < 2 || self.data[p].load(Ordering::Relaxed) >= t.seq, "post covered unseen data");
                self.served[p].store(t.seq, Ordering::Release);
            }
        }
    }

    struct Sys;
    impl PostingSystem for Sys {
        type PostChoice = DefaultPostChoice;
    }

    fn run<F: crate::traits::PostFavor>(serve_inline: bool) -> u64 {
        const N: usize = 4;
        const POSTS: u64 = 20_000;
        let data: Arc<Vec<AtomicU64>> = Arc::new((0..N).map(|_| AtomicU64::new(0)).collect());
        let served: Arc<Vec<AtomicU64>> = Arc::new((0..N).map(|_| AtomicU64::new(0)).collect());
        let serves = Arc::new(AtomicU64::new(0));
        let c = Check::<F> { data: data.clone(), served: served.clone(), serves: serves.clone(), _f: Default::default() };
        let cfg = PostConfig { latency: Duration::from_micros(50), place: DutyPlace::Anywhere };
        let d: Pin<Box<PostDelegatorOf<Sys, Check<F>>>> = PostDelegator::start(c, cfg);
        let d = &*d;
        std::thread::scope(|sc| {
            for _ in 0..N {
                let data = data.clone();
                sc.spawn(move || {
                    // the first post names the producer
                    let p = d.post().producer as usize;
                    for seq in 2..=POSTS {
                        data[p].store(seq, Ordering::Relaxed);
                        let t = d.post();
                        assert_eq!(t, Ticket { producer: p as u32, seq });
                        if serve_inline && seq % 64 == 0 {
                            d.serve_now();
                        }
                    }
                });
            }
        });
        // Every last post is served unprompted within a while.
        let until = Instant::now() + Duration::from_secs(10);
        while served.iter().any(|s| s.load(Ordering::Acquire) != POSTS) {
            assert!(Instant::now() < until, "posts never served");
            std::thread::sleep(Duration::from_micros(100));
        }
        serves.load(Ordering::Relaxed)
    }

    #[test]
    fn slot_serves_every_post_unprompted() {
        let n = run::<FavorPosting>(false);
        assert!(n > 0);
    }

    #[test]
    fn queue_serves_every_post_unprompted() {
        let n = run::<FavorServing>(false);
        assert!(n > 0);
    }

    #[test]
    fn slot_serves_every_post_inline() {
        run::<FavorPosting>(true);
    }

    #[test]
    fn queue_serves_every_post_inline() {
        run::<FavorServing>(true);
    }

    #[test]
    fn default_choice_follows_favor() {
        fn is<T: 'static, U: 'static>() -> bool {
            std::any::TypeId::of::<T>() == std::any::TypeId::of::<U>()
        }
        assert!(is::<PostDelegatorOf<Sys, Check<FavorPosting>>, SlotPostDelegator<Check<FavorPosting>>>());
        assert!(is::<PostDelegatorOf<Sys, Check<FavorServing>>, QueuePostDelegator<Check<FavorServing>>>());
    }
}
