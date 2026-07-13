//! Argobots ULT backend for the benchmark harness.
//!
//! Enabled with `--features argobots`.  Requires the Argobots C library
//! (`libabt`).  Set `ABT_ROOT=/path/to/argobots-install` or ensure
//! pkg-config finds it.
//!
//! # Building Argobots
//!
//! ```sh
//! git clone https://github.com/pmodels/argobots
//! cd argobots && ./autogen.sh
//! ./configure --prefix="$ABT_ROOT"
//! make -j$(nproc) && make install
//! ```
//!
//! # Worker count
//!
//! Argobots is initialised once per process.  All execution streams share a
//! single MPMC FIFO pool; every `spawn` pushes a ULT into that pool.
//! The worker count is fixed at the first call to [`ArgobotsBench::run`].

use std::os::raw::{c_int, c_void};
use std::sync::OnceLock;

use crate::BenchSystem;

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

type AbtXstream    = *mut c_void;
type AbtThread     = *mut c_void;
type AbtPool       = *mut c_void;
type AbtSched      = *mut c_void;
type AbtSchedConf  = *mut c_void;
type AbtThreadAttr = *mut c_void;

const ABT_SUCCESS:          c_int        = 0;
const ABT_POOL_FIFO:        c_int        = 0;
const ABT_POOL_ACCESS_MPMC: c_int        = 3;
const ABT_FALSE:            c_int        = 0;
const ABT_SCHED_DEFAULT:    c_int        = 0;
const ABT_SCHED_CONFIG_NULL: AbtSchedConf = std::ptr::null_mut();
const ABT_THREAD_ATTR_NULL:  AbtThreadAttr = std::ptr::null_mut();

unsafe extern "C" {
    fn ABT_init(argc: c_int, argv: *mut *mut i8) -> c_int;
    fn ABT_xstream_self(xstream: *mut AbtXstream) -> c_int;
    fn ABT_xstream_set_main_sched(xstream: AbtXstream, sched: AbtSched) -> c_int;
    fn ABT_xstream_create(sched: AbtSched, xstream: *mut AbtXstream) -> c_int;
    fn ABT_pool_create_basic(kind: c_int, access: c_int, automatic: c_int,
                             pool: *mut AbtPool) -> c_int;
    fn ABT_sched_create_basic(predef: c_int, num_pools: c_int, pools: *mut AbtPool,
                              config: AbtSchedConf, sched: *mut AbtSched) -> c_int;
    fn ABT_thread_create(pool: AbtPool, func: unsafe extern "C" fn(*mut c_void),
                         arg: *mut c_void, attr: AbtThreadAttr,
                         thread: *mut AbtThread) -> c_int;
    fn ABT_thread_join(thread: AbtThread) -> c_int;
    fn ABT_thread_free(thread: *mut AbtThread) -> c_int;
}

// ---------------------------------------------------------------------------
// One-time init: shared MPMC pool + N execution streams
// ---------------------------------------------------------------------------

struct AbtState {
    pool:            AbtPool,
    #[allow(dead_code)]
    extra_xstreams:  Vec<AbtXstream>,
}
unsafe impl Send for AbtState {}
unsafe impl Sync for AbtState {}

static ABT_STATE: OnceLock<AbtState> = OnceLock::new();
static ABT_WORKERS: OnceLock<usize>  = OnceLock::new();

fn abt_pool() -> AbtPool {
    ABT_STATE.get().expect("ArgobotsBench::spawn called before run").pool
}

fn abt_ensure_init(num_workers: usize) {
    let &stored = ABT_WORKERS.get_or_init(|| {
        ABT_STATE.get_or_init(|| unsafe {
            let ret = ABT_init(0, std::ptr::null_mut());
            assert_eq!(ret, ABT_SUCCESS, "ABT_init failed (ret={ret})");

            // Shared MPMC pool.
            let mut pool: AbtPool = std::ptr::null_mut();
            let ret = ABT_pool_create_basic(
                ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_FALSE, &mut pool,
            );
            assert_eq!(ret, ABT_SUCCESS, "ABT_pool_create_basic failed");

            // Redirect the primary xstream's scheduler to the shared pool.
            let mut primary: AbtXstream = std::ptr::null_mut();
            ABT_xstream_self(&mut primary);
            let mut sched: AbtSched = std::ptr::null_mut();
            let ret = ABT_sched_create_basic(
                ABT_SCHED_DEFAULT, 1, &mut pool, ABT_SCHED_CONFIG_NULL, &mut sched,
            );
            assert_eq!(ret, ABT_SUCCESS, "ABT_sched_create_basic (primary) failed");
            let ret = ABT_xstream_set_main_sched(primary, sched);
            assert_eq!(ret, ABT_SUCCESS, "ABT_xstream_set_main_sched failed");

            // Extra xstreams, each with their own scheduler on the shared pool.
            let mut extra = Vec::with_capacity(num_workers.saturating_sub(1));
            for _ in 1..num_workers {
                let mut sched_i: AbtSched = std::ptr::null_mut();
                let ret = ABT_sched_create_basic(
                    ABT_SCHED_DEFAULT, 1, &mut pool, ABT_SCHED_CONFIG_NULL, &mut sched_i,
                );
                assert_eq!(ret, ABT_SUCCESS, "ABT_sched_create_basic (extra) failed");
                let mut xs: AbtXstream = std::ptr::null_mut();
                let ret = ABT_xstream_create(sched_i, &mut xs);
                assert_eq!(ret, ABT_SUCCESS, "ABT_xstream_create failed");
                extra.push(xs);
            }

            AbtState { pool, extra_xstreams: extra }
        });
        num_workers
    });

    if stored != num_workers {
        eprintln!(
            "argobots: worker count fixed at {stored}, ignoring requested {num_workers}. \
             Run `cargo bench -- 'argobots/{num_workers}'` in a fresh process for that count."
        );
    }
}

// ---------------------------------------------------------------------------
// Trampoline + result slot
// ---------------------------------------------------------------------------

struct Slot<T> {
    func:   Box<dyn FnOnce() -> T + Send>,
    result: Option<T>,
}

unsafe extern "C" fn trampoline<T: Send + 'static>(arg: *mut c_void) {
    let slot = &mut *(arg as *mut Slot<T>);
    let f = std::mem::replace(&mut slot.func, Box::new(|| unreachable!()));
    slot.result = Some(f());
}

// ---------------------------------------------------------------------------
// JoinHandle
// ---------------------------------------------------------------------------

pub struct AbtJoinHandle<T> {
    thread: AbtThread,
    slot:   Box<Slot<T>>,
}

unsafe impl<T: Send> Send for AbtJoinHandle<T> {}

impl<T: Send + 'static> cmpth::JoinHandleLike<T> for AbtJoinHandle<T> {
    fn join(mut self) -> T {
        unsafe {
            let ret = ABT_thread_join(self.thread);
            assert_eq!(ret, ABT_SUCCESS, "ABT_thread_join failed");
            ABT_thread_free(&mut self.thread);
        }
        self.slot.result.take().expect("argobots trampoline did not set result")
    }
}

// ---------------------------------------------------------------------------
// ArgobotsBench
// ---------------------------------------------------------------------------

pub struct ArgobotsBench;

impl BenchSystem for ArgobotsBench {
    type JoinHandle<T: Send + 'static> = AbtJoinHandle<T>;

    fn run(num_workers: usize, f: impl FnOnce() + Send + 'static) {
        abt_ensure_init(num_workers);
        let h = Self::spawn(f);
        cmpth::JoinHandleLike::join(h);
    }

    fn spawn<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> AbtJoinHandle<T> {
        let pool = abt_pool();
        let mut slot = Box::new(Slot::<T> {
            func:   Box::new(f),
            result: None,
        });
        let arg = &mut *slot as *mut Slot<T> as *mut c_void;
        let mut thread: AbtThread = std::ptr::null_mut();
        unsafe {
            let ret = ABT_thread_create(pool, trampoline::<T>, arg, ABT_THREAD_ATTR_NULL, &mut thread);
            assert_eq!(ret, ABT_SUCCESS, "ABT_thread_create failed");
        }
        AbtJoinHandle { thread, slot }
    }
}
