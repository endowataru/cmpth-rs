//! Lifecycle hooks run once per worker, on the worker's own thread, and
//! pools of differently configured systems run side by side and wake each
//! other's tasks.

use std::cell::Cell;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

mod common;

use common::{conformance, fib};
use cmpth::{
    LifecyclePolicy, Resumable, SpawnableStackfulTaskSystem, StackfulBuilder, StackfulInitSystem,
    StackfulResumable, SuspendableSystem, UltDualSystem,
};

static STARTS: AtomicUsize = AtomicUsize::new(0);
static STOPS: AtomicUsize = AtomicUsize::new(0);
static THREADS: Mutex<Vec<std::thread::ThreadId>> = Mutex::new(Vec::new());

#[derive(Default)]
pub struct Counting;

impl LifecyclePolicy for Counting {
    fn start(&self, index: usize, num: usize, st: &Cell<u64>) {
        assert!(index < num);
        st.set(index as u64 + 1);
        STARTS.fetch_add(1, Ordering::SeqCst);
        THREADS.lock().unwrap().push(std::thread::current().id());
    }
    fn stop(&self, st: &Cell<u64>) {
        assert_ne!(st.get(), 0, "stop without start");
        STOPS.fetch_add(1, Ordering::SeqCst);
    }
}

pub struct Hooked;
cmpth::worker_tls_anchor!(Hooked);
cmpth::impl_with_defaults! {
    impl cmpth::UltDualIdentity for Hooked {
        type Lifecycle = Counting;
        type Idle = cmpth::SearchIdle;
    }
}

pub struct Pinned;
cmpth::worker_tls_anchor!(Pinned);
cmpth::impl_with_defaults! {
    impl cmpth::UltDualIdentity for Pinned {
        type Lifecycle = cmpth::PinWorkers;
    }
}

#[test]
fn hooks_run_once_per_worker_on_its_thread() {
    type S = UltDualSystem<Hooked>;
    let r = <S as StackfulInitSystem>::builder().workers(4).run(|| fib::<S>(20));
    assert_eq!(r, 6765);
    assert_eq!(STARTS.load(Ordering::SeqCst), 4);
    assert_eq!(STOPS.load(Ordering::SeqCst), 4);
    let ids: HashSet<_> = THREADS.lock().unwrap().iter().cloned().collect();
    assert_eq!(ids.len(), 4, "one start per worker thread");
}

#[test]
fn pinned_workers_run() {
    type S = UltDualSystem<Pinned>;
    conformance::stackful::<S>(4);
}

/// A spinning pool and a parking pool, each on its own caller thread; a
/// task of each parks and is woken by a task of the other (an OS-thread
/// wakeup for the waker's pool), round after round.
#[test]
fn pools_side_by_side_wake_each_other() {
    type A = cmpth::DefaultDualTaskSystem;
    type B = cmpth::ParkDualTaskSystem;

    let rounds = 100;
    let slots_a: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let slots_b: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let (sa, sb) = (Arc::clone(&slots_a), Arc::clone(&slots_b));
    let (sa2, sb2) = (Arc::clone(&slots_a), Arc::clone(&slots_b));

    // Park a task on its own system's handle and publish it; wake the
    // peer's published handle, alternating.
    fn side<S, P>(mine: Arc<Mutex<Vec<usize>>>, peer: Arc<Mutex<Vec<usize>>>, rounds: usize, first: bool)
    where
        S: SpawnableStackfulTaskSystem + SuspendableSystem,
        S::SuspendedThread: StackfulResumable<S>,
        P: SuspendableSystem,
    {
        for i in 0..rounds {
            if first ^ (i % 2 == 1) {
                let slot = S::SuspendedThread::default();
                let p = &slot as *const S::SuspendedThread as usize;
                slot.wait_with(|| mine.lock().unwrap().push(p));
            } else {
                let p = loop {
                    if let Some(p) = peer.lock().unwrap().pop() {
                        break p;
                    }
                    S::yield_now();
                };
                let slot = unsafe { &*(p as *const P::SuspendedThread) };
                while !slot.is_set() {
                    S::yield_now();
                }
                slot.notify();
            }
        }
    }

    let ta = std::thread::spawn(move || {
        <A as StackfulInitSystem>::builder().workers(2).run(move || side::<A, B>(sa, sb, rounds, true));
    });
    let tb = std::thread::spawn(move || {
        <B as StackfulInitSystem>::builder().workers(2).run(move || side::<B, A>(sb2, sa2, rounds, false));
    });
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        ta.join().unwrap();
        tb.join().unwrap();
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(60)).expect("pools side by side hung");
}
