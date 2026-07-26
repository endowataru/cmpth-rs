//! Count heap allocations for fib(n) under each scheduler, to explain *why*
//! stackless-only is so much slower than rayon/stackful-only for this
//! workload (not run as part of the criterion suite — a scratch example).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

struct CountingAlloc;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static DEALLOCS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn reset() -> u64 {
    ALLOCS.store(0, Ordering::Relaxed);
    DEALLOCS.store(0, Ordering::Relaxed);
    std::time::Instant::now().elapsed().as_nanos() as u64 // dummy to force a read
}

fn report(label: &str, n: u64, result: u64) {
    let a = ALLOCS.load(Ordering::Relaxed);
    let d = DEALLOCS.load(Ordering::Relaxed);
    println!("{label:24} fib({n})={result:<10} allocs={a:<12} deallocs={d}");
}

fn run_fib<S: cmpth_bench::BenchSystem>(n: u64) -> u64 {
    use std::sync::{Arc, Mutex};
    let result = Arc::new(Mutex::new(0u64));
    let result2 = Arc::clone(&result);
    S::run(1, move || {
        *result2.lock().unwrap() = cmpth_bench::fib::<S>(n);
    });
    let v = *result.lock().unwrap();
    v
}

fn main() {
    let n = 28;

    reset();
    let r = run_fib::<cmpth_bench::CmpthBench>(n);
    report("cmpth-dual", n, r);

    reset();
    let r = run_fib::<cmpth_bench::StackfulOnlyBench>(n);
    report("cmpth-stackful-only", n, r);

    reset();
    let r = cmpth_bench::run_fib_async::<cmpth_bench::AsyncOnlySystem>(1, n);
    report("cmpth-stackless-only", n, r);

    reset();
    let r = run_fib::<cmpth_bench::RayonBench>(n);
    report("rayon", n, r);
}
