use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use image::{DynamicImage, ImageBuffer, Rgba};
use xeno_lib::transforms::Interpolation;
use xeno_lib::{crop, flip_horizontal, flip_vertical, rotate, rotate_90, rotate_180, rotate_270};

const LARGE_WIDTH: u32 = 4000;
const LARGE_HEIGHT: u32 = 2500;

fn large_rgba_image() -> DynamicImage {
    let buffer = ImageBuffer::from_fn(LARGE_WIDTH, LARGE_HEIGHT, |x, y| {
        let r = (x % 256) as u8;
        let g = (y % 256) as u8;
        let b = ((x + y) % 256) as u8;
        let a = 255u8;
        Rgba([r, g, b, a])
    });
    DynamicImage::ImageRgba8(buffer)
}

fn benchmarks(c: &mut Criterion) {
    let image = large_rgba_image();

    let mut group = c.benchmark_group("transforms_rgba8_10mp");
    group.sample_size(10);

    group.bench_function(BenchmarkId::new("flip_horizontal", "rgba8"), |b| {
        b.iter(|| {
            let _ = flip_horizontal(black_box(&image)).expect("flip horizontal");
        });
    });

    group.bench_function(BenchmarkId::new("flip_vertical", "rgba8"), |b| {
        b.iter(|| {
            let _ = flip_vertical(black_box(&image)).expect("flip vertical");
        });
    });

    group.bench_function(BenchmarkId::new("rotate_90", "rgba8"), |b| {
        b.iter(|| {
            let _ = rotate_90(black_box(&image)).expect("rotate 90");
        });
    });

    group.bench_function(BenchmarkId::new("rotate_180", "rgba8"), |b| {
        b.iter(|| {
            let _ = rotate_180(black_box(&image)).expect("rotate 180");
        });
    });

    group.bench_function(BenchmarkId::new("rotate_270", "rgba8"), |b| {
        b.iter(|| {
            let _ = rotate_270(black_box(&image)).expect("rotate 270");
        });
    });

    group.bench_function(BenchmarkId::new("rotate_bilinear_33deg", "rgba8"), |b| {
        b.iter(|| {
            let _ = rotate(black_box(&image), 33.0, Interpolation::Bilinear).expect("rotate");
        });
    });

    group.bench_function(BenchmarkId::new("crop_center", "rgba8"), |b| {
        b.iter(|| {
            let _ = crop(black_box(&image), 500, 400, 3000, 1700).expect("crop");
        });
    });

    group.finish();
}

criterion_group!(transform_benches, benchmarks);
criterion_main!(transform_benches);
