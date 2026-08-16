use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cmpth_bench::{nqueens, run_nqueens_parallel_invoke, BenchSystem, CmpthBench, RayonBench};
#[cfg(feature = "massivethreads")]
use cmpth_bench::MythBench;
#[cfg(feature = "may")]
use cmpth_bench::MayBench;
#[cfg(feature = "argobots")]
use cmpth_bench::ArgobotsBench;

fn bench_nqueens_system<S: BenchSystem>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    label: &str,
) {
    for workers in 1..=cmpth::available_parallelism() {
        group.bench_with_input(BenchmarkId::new(label, workers), &workers, |b, &w| {
            b.iter(|| {
                S::run(w, || assert_eq!(nqueens::<S>(vec![], 13, 3), 73_712));
            });
        });
    }
}

/// N-Queens via `ScopedStackfulTaskSystem::parallel_call` instead of
/// `BenchSystem::par_join` — mirrors `bench_fib_parallel_invoke` in
/// `fib.rs`. Generic over `S: ScopedStackfulTaskSystem + StackfulInitSystem`.
fn bench_nqueens_parallel_invoke<S: cmpth::ScopedStackfulTaskSystem + cmpth::StackfulInitSystem>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    label: &str,
) {
    for workers in 1..=cmpth::available_parallelism() {
        group.bench_with_input(BenchmarkId::new(label, workers), &workers, |b, &w| {
            b.iter(|| assert_eq!(run_nqueens_parallel_invoke::<S>(w, 13, 3), 73_712));
        });
    }
}

fn bench_nqueens(c: &mut Criterion) {
    let mut group = c.benchmark_group("nqueens");
    bench_nqueens_system::<CmpthBench>(&mut group, "cmpth");
    // The make_context-backed fork-parent-first `parallel_call`
    // (`docs/scoped-ult-promotion.md` §9.8.4), on a real ULT system, next to
    // the standalone scoped engine it's trying to approach.
    bench_nqueens_parallel_invoke::<cmpth::ScopedTaskSystem>(&mut group, "cmpth-parallel-invoke");
    bench_nqueens_parallel_invoke::<cmpth::DefaultStackfulOnlyTaskSystem>(&mut group, "cmpth-stackful-ult-parallel-invoke");
    bench_nqueens_parallel_invoke::<cmpth::DefaultDualTaskSystem>(&mut group, "cmpth-dual-parallel-invoke");
    bench_nqueens_system::<RayonBench>(&mut group, "rayon");
    #[cfg(feature = "massivethreads")]
    bench_nqueens_system::<MythBench>(&mut group, "myth");
    #[cfg(feature = "may")]
    bench_nqueens_system::<MayBench>(&mut group, "may");
    // Tokio excluded: synchronous join deadlocks under recursive fork-join.
    // See spawn_overhead bench for Tokio overhead numbers.
    #[cfg(feature = "argobots")]
    bench_nqueens_system::<ArgobotsBench>(&mut group, "argobots");
    group.finish();
}

criterion_group!(benches, bench_nqueens);
criterion_main!(benches);
