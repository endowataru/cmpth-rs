use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cmpth::JoinHandleLike as _;
use cmpth_bench::{BenchSystem, CmpthBench, OsThreadBench, RayonBench};
#[cfg(feature = "massivethreads")]
use cmpth_bench::MythBench;
#[cfg(feature = "may")]
use cmpth_bench::MayBench;
#[cfg(feature = "tokio-rt")]
use cmpth_bench::TokioBench;
#[cfg(feature = "argobots")]
use cmpth_bench::ArgobotsBench;

fn bench_spawn_system<S: BenchSystem>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    label: &str,
) {
    for workers in [1, 2, 4] {
        group.bench_with_input(
            BenchmarkId::new(format!("{label}/single"), workers),
            &workers,
            |b, &w| {
                b.iter(|| S::run(w, || S::spawn(|| ()).join()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{label}/bulk-1000"), workers),
            &workers,
            |b, &w| {
                b.iter(|| {
                    S::run(w, || {
                        let handles: Vec<_> = (0u64..1000).map(|i| S::spawn(move || i)).collect();
                        let _: u64 = handles.into_iter().map(|h| h.join()).sum();
                    });
                });
            },
        );
    }
}

fn bench_spawn_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn_overhead");
    bench_spawn_system::<CmpthBench>(&mut group, "cmpth");
    bench_spawn_system::<OsThreadBench>(&mut group, "os");
    bench_spawn_system::<RayonBench>(&mut group, "rayon");
    #[cfg(feature = "massivethreads")]
    bench_spawn_system::<MythBench>(&mut group, "myth");
    #[cfg(feature = "may")]
    bench_spawn_system::<MayBench>(&mut group, "may");
    #[cfg(feature = "tokio-rt")]
    bench_spawn_system::<TokioBench>(&mut group, "tokio");
    #[cfg(feature = "argobots")]
    bench_spawn_system::<ArgobotsBench>(&mut group, "argobots");
    group.finish();
}

criterion_group!(benches, bench_spawn_overhead);
criterion_main!(benches);
