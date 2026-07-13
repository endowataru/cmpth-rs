use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cmpth_bench::{fib, BenchSystem, CmpthBench, RayonBench};
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

fn bench_fib(c: &mut Criterion) {
    let mut group = c.benchmark_group("fib");
    bench_fib_system::<CmpthBench>(&mut group, "cmpth");
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
