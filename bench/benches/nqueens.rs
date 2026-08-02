use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cmpth_bench::{nqueens, BenchSystem, CmpthBench, RayonBench};
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

fn bench_nqueens(c: &mut Criterion) {
    let mut group = c.benchmark_group("nqueens");
    bench_nqueens_system::<CmpthBench>(&mut group, "cmpth");
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
