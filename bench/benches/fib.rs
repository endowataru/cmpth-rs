use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cmpth_bench::{
    fib, run_fib_async, run_fib_forkjoin, AsyncOnlySystem, BenchSystem, CmpthBench, RayonBench,
    StackfulOnlyBench,
};
#[cfg(feature = "massivethreads")]
use cmpth_bench::MythBench;
#[cfg(feature = "may")]
use cmpth_bench::MayBench;
#[cfg(feature = "argobots")]
use cmpth_bench::ArgobotsBench;

fn bench_fib_system<S: BenchSystem>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    label: &str,
) {
    for workers in [1, 2, 4] {
        group.bench_with_input(BenchmarkId::new(label, workers), &workers, |b, &w| {
            b.iter(|| S::run(w, || assert_eq!(fib::<S>(34), 5_702_887)));
        });
    }
}

/// Stackless-only fib doesn't fit `BenchSystem` (no blocking join exists for
/// a pure stackless-only system) — driven directly through `run_fib_async`.
fn bench_fib_async(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    for workers in [1, 2, 4] {
        group.bench_with_input(
            BenchmarkId::new("cmpth-stackless-only", workers),
            &workers,
            |b, &w| {
                b.iter(|| assert_eq!(run_fib_async::<AsyncOnlySystem>(w, 34), 5_702_887));
            },
        );
    }
}

/// cmpth's experimental rayon-style scheduler (`cmpth::fork_join`) doesn't
/// fit `BenchSystem` either (no `spawn`/`JoinHandle`, only scoped `join`).
fn bench_fib_forkjoin(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    for workers in [1, 2, 4] {
        group.bench_with_input(
            BenchmarkId::new("cmpth-forkjoin", workers),
            &workers,
            |b, &w| {
                b.iter(|| assert_eq!(run_fib_forkjoin(w, 34), 5_702_887));
            },
        );
    }
}

fn bench_fib(c: &mut Criterion) {
    let mut group = c.benchmark_group("fib");
    // Three cmpth configurations side by side: today's dual system
    // (poll_fn tag check on every dispatch), stackful-only (execute_stackful,
    // no tag check at all), and stackless-only (execute_async, no tag check,
    // spawn_async/.await instead of spawn/.join()).
    bench_fib_system::<CmpthBench>(&mut group, "cmpth-dual");
    bench_fib_system::<StackfulOnlyBench>(&mut group, "cmpth-stackful-only");
    bench_fib_async(&mut group);
    bench_fib_forkjoin(&mut group);
    bench_fib_system::<RayonBench>(&mut group, "rayon");
    #[cfg(feature = "massivethreads")]
    bench_fib_system::<MythBench>(&mut group, "myth");
    #[cfg(feature = "may")]
    bench_fib_system::<MayBench>(&mut group, "may");
    // Tokio: synchronous join inside block_in_place deadlocks under recursive
    // fork-join regardless of worker count.  Tokio's design assumes async/await
    // throughout; synchronous joins are not a supported use case.
    // See spawn_overhead bench for Tokio's task-launch overhead numbers.
    #[cfg(feature = "argobots")]
    bench_fib_system::<ArgobotsBench>(&mut group, "argobots");
    group.finish();
}

criterion_group!(benches, bench_fib);
criterion_main!(benches);
