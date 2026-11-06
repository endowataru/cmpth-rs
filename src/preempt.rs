//! Timer-driven preemption of ULTs inside marked regions (AArch64 macOS
//! and Linux; opt-in: nothing happens before [`enable`]).
//!
//! Without preemption a ULT that computes for long without yielding keeps
//! every task queued on its worker waiting.  [`preemptible`] marks a region
//! of such code; once [`enable`] has started the timer, a region that has
//! run for a whole quantum is interrupted and yields.
//!
//! The mechanism follows Go's asynchronous preemption rather than switching
//! inside the signal handler:
//!
//! 1. A timer thread sends `SIGURG` to a worker thread whose region has not
//!    changed since the previous tick.
//! 2. The handler copies the interrupted machine context onto the ULT's own
//!    stack, below the handler's frame, and rewrites the context so that,
//!    when the handler returns, the thread "calls" a trampoline with that
//!    copy, in normal (not signal) context.
//! 3. The trampoline yields with
//!    [`yield_pinned`](crate::traits::SpawnableStackfulTaskSystem::yield_pinned):
//!    other tasks run, and the preempted one comes back on the same OS
//!    thread (the interrupted code may hold thread-specific state).
//! 4. Back from the yield, the trampoline signals its own thread; the
//!    handler copies the saved context into the new signal frame, and the
//!    kernel's return from that signal resumes the interrupted code with
//!    every register restored.
//!
//! No context switch happens inside a handler, so neither `SA_NODEFER` nor
//! a return on another thread (which macOS refuses) is needed.
//!
//! # Contract of a region
//!
//! The code in a region may be stopped between any two instructions, and
//! other tasks of its worker then run.  It must therefore not hold a lock
//! another task could take, not allocate or free (the allocator's locks),
//! and not call into the scheduler (spawn, yield, block): plain
//! computation on memory it owns.  Each ULT stack needs about 6 KiB (macOS)
//! or 10 KiB (Linux) free below the interrupted frame.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicU64, Ordering, compiler_fence};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use crate::traits::SpawnableStackfulTaskSystem;

const OUT: u8 = 0;
const IN_REGION: u8 = 1;
const PREEMPTED: u8 = 2;
const RESTORING: u8 = 3;

/// One per OS thread that ever entered a region; never freed.
struct Slot {
    thread: libc::pthread_t,
    state: AtomicU8,
    /// Bumped whenever the region starts or resumes; the timer preempts a
    /// thread whose `seq` did not move for a whole tick.
    seq: AtomicU64,
    restore: AtomicPtr<u8>,
    yield_fn: Cell<fn()>,
    preemptions: AtomicU64,
}

// SAFETY: `yield_fn` is only touched by the slot's own thread.
unsafe impl Sync for Slot {}

static SLOTS: Mutex<Vec<&'static Slot>> = Mutex::new(Vec::new());
static ENABLED: AtomicBool = AtomicBool::new(false);
static TIMER: OnceLock<()> = OnceLock::new();

thread_local! {
    static SLOT: Cell<*const Slot> = const { Cell::new(std::ptr::null()) };
}

fn slot() -> &'static Slot {
    let p = SLOT.with(|s| s.get());
    if !p.is_null() {
        return unsafe { &*p };
    }
    let s: &'static Slot = Box::leak(Box::new(Slot {
        thread: unsafe { libc::pthread_self() },
        state: AtomicU8::new(OUT),
        seq: AtomicU64::new(0),
        restore: AtomicPtr::new(std::ptr::null_mut()),
        yield_fn: Cell::new(|| {}),
        preemptions: AtomicU64::new(0),
    }));
    SLOT.with(|c| c.set(s));
    SLOTS.lock().unwrap().push(s);
    s
}

/// Run `f` as a preemptible region of the current ULT (see the module docs
/// for what `f` may do).  Regions nest; only the innermost one's system is
/// used to yield.
pub fn preemptible<S: SpawnableStackfulTaskSystem, R>(f: impl FnOnce() -> R) -> R {
    let s = slot();
    let prev_fn = s.yield_fn.replace(S::yield_pinned);
    let prev = s.state.load(Ordering::Relaxed);
    s.seq.fetch_add(1, Ordering::Relaxed);
    compiler_fence(Ordering::SeqCst);
    s.state.store(IN_REGION, Ordering::Relaxed);
    compiler_fence(Ordering::SeqCst);
    let r = f();
    compiler_fence(Ordering::SeqCst);
    s.state.store(prev, Ordering::Relaxed);
    compiler_fence(Ordering::SeqCst);
    s.yield_fn.set(prev_fn);
    r
}

/// Number of times regions on the current OS thread were preempted.
pub fn preemptions_here() -> u64 {
    slot().preemptions.load(Ordering::Relaxed)
}

/// Install the handler and start the timer thread (once); `quantum` is how
/// long a region may run before it is preempted.  Later calls only switch
/// preemption back on.
pub fn enable(quantum: Duration) {
    ENABLED.store(true, Ordering::Relaxed);
    TIMER.get_or_init(|| {
        install_handler();
        std::thread::Builder::new()
            .name("cmpth-preempt".into())
            .spawn(move || timer(quantum))
            .expect("cmpth: preemption timer thread");
    });
}

/// Stop preempting (the timer thread keeps running idle).
pub fn disable() {
    ENABLED.store(false, Ordering::Relaxed);
}

fn timer(quantum: Duration) {
    let mut seen: Vec<u64> = Vec::new();
    loop {
        std::thread::sleep(quantum);
        if !ENABLED.load(Ordering::Relaxed) {
            continue;
        }
        let slots = SLOTS.lock().unwrap();
        seen.resize(slots.len(), u64::MAX);
        for (s, last) in slots.iter().zip(seen.iter_mut()) {
            let seq = s.seq.load(Ordering::Relaxed);
            if s.state.load(Ordering::Relaxed) == IN_REGION && seq == *last {
                unsafe { libc::pthread_kill(s.thread, libc::SIGURG) };
            }
            *last = seq;
        }
    }
}

fn install_handler() {
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = handler as *const () as usize;
        sa.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
        libc::sigemptyset(&mut sa.sa_mask);
        let r = libc::sigaction(libc::SIGURG, &sa, std::ptr::null_mut());
        assert_eq!(r, 0, "cmpth: sigaction(SIGURG)");
    }
}

extern "C" fn handler(_sig: libc::c_int, _info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
    let p = SLOT.with(|s| s.get());
    if p.is_null() {
        return;
    }
    let s = unsafe { &*p };
    match s.state.load(Ordering::Relaxed) {
        RESTORING => unsafe {
            let buf = s.restore.load(Ordering::Relaxed);
            std::ptr::copy_nonoverlapping(buf, arch::mcontext(uc), arch::MCONTEXT_SIZE);
            s.state.store(IN_REGION, Ordering::Relaxed);
            s.seq.fetch_add(1, Ordering::Relaxed);
        },
        IN_REGION => unsafe {
            // The copy goes below this frame, leaving room for the calls
            // the handler still makes; the trampoline's stack starts below it.
            let here = &s as *const _ as usize;
            let buf = ((here - 2048 - arch::MCONTEXT_SIZE) & !63) as *mut u8;
            std::ptr::copy_nonoverlapping(arch::mcontext(uc), buf, arch::MCONTEXT_SIZE);
            s.state.store(PREEMPTED, Ordering::Relaxed);
            s.preemptions.fetch_add(1, Ordering::Relaxed);
            arch::redirect(uc, trampoline as *const () as usize, buf as usize, buf as usize);
        },
        _ => {}
    }
}

/// Entered from the signal handler's rewritten context, in normal context,
/// with the stack pointer just below `saved`.
extern "C" fn trampoline(saved: *mut u8) -> ! {
    let p = SLOT.with(|s| s.get());
    let s = unsafe { &*p };
    (s.yield_fn.get())();
    // Same OS thread (pinned yield): `s` is still ours.
    s.restore.store(saved, Ordering::Relaxed);
    compiler_fence(Ordering::SeqCst);
    s.state.store(RESTORING, Ordering::Relaxed);
    compiler_fence(Ordering::SeqCst);
    unsafe { libc::pthread_kill(libc::pthread_self(), libc::SIGURG) };
    // The handler resumed the interrupted code instead of returning here.
    std::process::abort()
}

#[cfg(target_os = "macos")]
mod arch {
    // `struct __darwin_ucontext` up to `uc_mcontext` (a pointer), and the
    // AArch64 `__darwin_mcontext64`: exception state (16 bytes), thread
    // state (x0..x28, fp, lr, sp, pc, cpsr, flags), NEON state.
    const UC_MCONTEXT: usize = 48;
    const SS: usize = 16;
    pub const MCONTEXT_SIZE: usize = 16 + 272 + 528;

    pub unsafe fn mcontext(uc: *mut libc::c_void) -> *mut u8 {
        unsafe { *((uc as *mut u8).add(UC_MCONTEXT) as *mut *mut u8) }
    }

    pub unsafe fn redirect(uc: *mut libc::c_void, pc: usize, sp: usize, x0: usize) {
        unsafe {
            let m = mcontext(uc);
            *(m.add(SS) as *mut usize) = x0;
            // x0..x28 (232 bytes), fp, lr, sp, pc.
            *(m.add(SS + 248) as *mut usize) = sp;
            *(m.add(SS + 256) as *mut usize) = pc;
        }
    }
}

#[cfg(target_os = "linux")]
mod arch {
    pub const MCONTEXT_SIZE: usize = std::mem::size_of::<libc::mcontext_t>();

    pub unsafe fn mcontext(uc: *mut libc::c_void) -> *mut u8 {
        unsafe { &raw mut (*(uc as *mut libc::ucontext_t)).uc_mcontext as *mut u8 }
    }

    pub unsafe fn redirect(uc: *mut libc::c_void, pc: usize, sp: usize, x0: usize) {
        unsafe {
            let m = &mut (*(uc as *mut libc::ucontext_t)).uc_mcontext;
            m.regs[0] = x0 as u64;
            m.sp = sp as u64;
            m.pc = pc as u64;
        }
    }
}
