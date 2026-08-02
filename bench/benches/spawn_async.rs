//! Async-task benchmarks — cmpth only (async is not part of BenchSystem).
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cmpth_bench::dual::{run, spawn, spawn_async};

fn bench_spawn_async(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn_async");

    for workers in 1..=cmpth::available_parallelism() {
        group.bench_with_input(BenchmarkId::new("single", workers), &workers, |b, &w| {
            b.iter(|| run(w, || { spawn_async(async { () }).join().unwrap(); }));
        });

        group.bench_with_input(BenchmarkId::new("bulk-1000", workers), &workers, |b, &w| {
            b.iter(|| {
                run(w, || {
                    let handles: Vec<_> =
                        (0u64..1000).map(|i| spawn_async(async move { i })).collect();
                    let _: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
                });
            });
        });

        group.bench_with_input(BenchmarkId::new("sync-single-cmp", workers), &workers, |b, &w| {
            b.iter(|| run(w, || { spawn(|| ()).join().unwrap(); }));
        });

        group.bench_with_input(BenchmarkId::new("sync-bulk-1000-cmp", workers), &workers, |b, &w| {
            b.iter(|| {
                run(w, || {
                    let handles: Vec<_> = (0u64..1000).map(|i| spawn(move || i)).collect();
                    let _: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
                });
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_spawn_async);
criterion_main!(benches);
