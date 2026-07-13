//! MassiveThreads (MYTH) backend for the benchmark harness.
//!
//! Enabled with `--features massivethreads`.  Requires the `myth` library;
//! set `MYTH_ROOT=/path/to/massivethreads-install` or ensure pkg-config finds it.
//!
//! # Worker count
//!
//! `myth_init` initialises the worker pool once per process.  [`MythBench::run`]
//! uses `myth_init_ex` with an explicit worker count on the first call and
//! records that count.  Subsequent calls must use the same count; otherwise the
//! function panics.  Run one benchmark binary per desired worker count, or set
//! `MYTH_WORKERS=<n>` to fix the count used by the first call.

use std::marker::PhantomData;
use std::os::raw::{c_int, c_void};
use std::sync::OnceLock;

use crate::BenchSystem;

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

type MythThread = *mut c_void;
type MythFunc   = unsafe extern "C" fn(*mut c_void) -> *mut c_void;

unsafe extern "C" {
    fn myth_init() -> c_int;
    fn myth_create(func: MythFunc, arg: *mut c_void) -> MythThread;
    fn myth_join(th: MythThread, retval: *mut *mut c_void) -> c_int;
    fn myth_yield();
    fn myth_fini();
}

// ---------------------------------------------------------------------------
// Initialise once via MYTH_WORKER_NUM env var
// ---------------------------------------------------------------------------

static MYTH_WORKERS: OnceLock<usize> = OnceLock::new();

fn myth_ensure_init(num_workers: usize) {
    let &stored = MYTH_WORKERS.get_or_init(|| {
        // MYTH_WORKER_NUM is read by myth_init().
        // SAFETY: benchmark runner is single-threaded at this point.
        unsafe { std::env::set_var("MYTH_WORKER_NUM", num_workers.to_string()) };
        let ret = unsafe { myth_init() };
        // myth_init returns 1 on success (all paths in myth_init_ex_body return 1)
        assert_eq!(ret, 1, "myth_init failed (ret={ret})");
        num_workers
    });
    assert_eq!(
        stored, num_workers,
        "MassiveThreads worker count is fixed at first myth_init call ({stored}); \
         cannot change to {num_workers} mid-process. \
         Run separate benchmark binaries per worker count."
    );
}

// ---------------------------------------------------------------------------
// JoinHandle
// ---------------------------------------------------------------------------

pub struct MythJoinHandle<T> {
    thread:  MythThread,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for MythJoinHandle<T> {}

impl<T: Send + 'static> cmpth::JoinHandleLike<T> for MythJoinHandle<T> {
    fn join(self) -> T {
        let mut retval: *mut c_void = std::ptr::null_mut();
        unsafe {
            myth_join(self.thread, &mut retval);
            *Box::from_raw(retval as *mut T)
        }
    }
}

// ---------------------------------------------------------------------------
// Trampoline: void*(*)(void*) → Rust closure
// ---------------------------------------------------------------------------

unsafe extern "C" fn trampoline<T: Send + 'static>(arg: *mut c_void) -> *mut c_void {
    let f = *Box::from_raw(arg as *mut Box<dyn FnOnce() -> T + Send>);
    let result = f();
    Box::into_raw(Box::new(result)) as *mut c_void
}

// ---------------------------------------------------------------------------
// MythBench
// ---------------------------------------------------------------------------

pub struct MythBench;

impl BenchSystem for MythBench {
    type JoinHandle<T: Send + 'static> = MythJoinHandle<T>;

    fn run(num_workers: usize, f: impl FnOnce() + Send + 'static) {
        myth_ensure_init(num_workers);
        f();
    }

    fn spawn<T: Send + 'static>(
        f: impl FnOnce() -> T + Send + 'static,
    ) -> MythJoinHandle<T> {
        let boxed: Box<Box<dyn FnOnce() -> T + Send>> = Box::new(Box::new(f));
        let arg = Box::into_raw(boxed) as *mut c_void;
        let thread = unsafe { myth_create(trampoline::<T>, arg) };
        assert!(!thread.is_null(), "myth_create failed");
        MythJoinHandle { thread, _marker: PhantomData }
    }
}
