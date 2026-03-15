use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use image::{DynamicImage, ImageBuffer, Rgba};
use xeno_lib::transforms::Interpolation;
use xeno_lib::{
    crop, flip_horizontal, flip_vertical, rotate, rotate_90, rotate_180, rotate_270,
    // Adjustments
    adjust_brightness, adjust_contrast, adjust_hue,
    // Filters
    gaussian_blur, unsharp_mask, edge_detect, emboss, sepia,
    // Analysis
    histogram,
    // Resize
    resize_exact,
};

// ---------------------------------------------------------------------------
// Helper: generate a deterministic RGBA test image of arbitrary size
// ---------------------------------------------------------------------------

fn make_rgba_image(width: u32, height: u32) -> DynamicImage {
    let buffer = ImageBuffer::from_fn(width, height, |x, y| {
        let r = (x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13))) as u8;
        let g = (y.wrapping_mul(11).wrapping_add(x.wrapping_mul(3))) as u8;
        let b = (x.wrapping_add(y).wrapping_mul(17)) as u8;
        Rgba([r, g, b, 255u8])
    });
    DynamicImage::ImageRgba8(buffer)
}

// ---------------------------------------------------------------------------
// Benchmark group: geometric transforms (existing + expanded)
// ---------------------------------------------------------------------------

fn bench_transforms(c: &mut Criterion) {
    let img_2048 = make_rgba_image(2048, 2048);
    let img_512 = make_rgba_image(512, 512);

    let mut group = c.benchmark_group("transforms");
    group.sample_size(10);

    // Flip
    group.bench_function(BenchmarkId::new("flip_horizontal", "2048x2048"), |b| {
        b.iter(|| flip_horizontal(black_box(&img_2048)).expect("flip_h"));
    });

    group.bench_function(BenchmarkId::new("flip_vertical", "2048x2048"), |b| {
        b.iter(|| flip_vertical(black_box(&img_2048)).expect("flip_v"));
    });

    // Rotate
    group.bench_function(BenchmarkId::new("rotate_90", "2048x2048"), |b| {
        b.iter(|| rotate_90(black_box(&img_2048)).expect("rot90"));
    });

    group.bench_function(BenchmarkId::new("rotate_180", "2048x2048"), |b| {
        b.iter(|| rotate_180(black_box(&img_2048)).expect("rot180"));
    });

    group.bench_function(BenchmarkId::new("rotate_270", "2048x2048"), |b| {
        b.iter(|| rotate_270(black_box(&img_2048)).expect("rot270"));
    });

    group.bench_function(
        BenchmarkId::new("rotate_bilinear_33deg", "2048x2048"),
        |b| {
            b.iter(|| {
                rotate(black_box(&img_2048), 33.0, Interpolation::Bilinear).expect("rotate33")
            });
        },
    );

    // Crop
    group.bench_function(BenchmarkId::new("crop_center", "2048x2048"), |b| {
        b.iter(|| crop(black_box(&img_2048), 256, 256, 1792, 1792).expect("crop"));
    });

    // Resize
    group.bench_function(BenchmarkId::new("resize_2048_to_512", "downscale"), |b| {
        b.iter(|| {
            resize_exact(black_box(&img_2048), 512, 512, Interpolation::Bilinear)
                .expect("downscale")
        });
    });

    group.bench_function(BenchmarkId::new("resize_512_to_2048", "upscale"), |b| {
        b.iter(|| {
            resize_exact(black_box(&img_512), 2048, 2048, Interpolation::Bilinear)
                .expect("upscale")
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group: image processing (blur, sharpen, filters)
// ---------------------------------------------------------------------------

fn bench_image_processing(c: &mut Criterion) {
    let img_512 = make_rgba_image(512, 512);
    let img_1024 = make_rgba_image(1024, 1024);
    let img_2048 = make_rgba_image(2048, 2048);

    let mut group = c.benchmark_group("image_processing");
    group.sample_size(10);

    // Gaussian blur
    group.bench_function(BenchmarkId::new("gaussian_blur", "512x512"), |b| {
        b.iter(|| gaussian_blur(black_box(&img_512), 3.0).expect("blur512"));
    });

    group.bench_function(BenchmarkId::new("gaussian_blur", "2048x2048"), |b| {
        b.iter(|| gaussian_blur(black_box(&img_2048), 3.0).expect("blur2048"));
    });

    // Sharpen (unsharp mask)
    group.bench_function(BenchmarkId::new("sharpen", "1024x1024"), |b| {
        b.iter(|| unsharp_mask(black_box(&img_1024), 2.0, 5).expect("sharpen"));
    });

    // Edge detect
    group.bench_function(BenchmarkId::new("edge_detect", "1024x1024"), |b| {
        b.iter(|| edge_detect(black_box(&img_1024), 1.0).expect("edge"));
    });

    // Emboss
    group.bench_function(BenchmarkId::new("emboss", "1024x1024"), |b| {
        b.iter(|| emboss(black_box(&img_1024), 1.0).expect("emboss"));
    });

    // Sepia
    group.bench_function(BenchmarkId::new("sepia", "1024x1024"), |b| {
        b.iter(|| sepia(black_box(&img_1024)).expect("sepia"));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group: color adjustments
// ---------------------------------------------------------------------------

fn bench_adjustments(c: &mut Criterion) {
    let img_1024 = make_rgba_image(1024, 1024);

    let mut group = c.benchmark_group("adjustments");
    group.sample_size(10);

    group.bench_function(BenchmarkId::new("brightness", "1024x1024"), |b| {
        b.iter(|| adjust_brightness(black_box(&img_1024), 25.0).expect("brightness"));
    });

    group.bench_function(BenchmarkId::new("contrast", "1024x1024"), |b| {
        b.iter(|| adjust_contrast(black_box(&img_1024), 25.0).expect("contrast"));
    });

    group.bench_function(BenchmarkId::new("hue_shift", "1024x1024"), |b| {
        b.iter(|| adjust_hue(black_box(&img_1024), 90.0).expect("hue"));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group: image encoding (PNG, JPEG, WebP)
// ---------------------------------------------------------------------------

fn bench_encoding(c: &mut Criterion) {
    let img_1024 = make_rgba_image(1024, 1024);

    let mut group = c.benchmark_group("encoding");
    group.sample_size(10);

    // PNG encoding
    group.bench_function(BenchmarkId::new("encode_png", "1024x1024"), |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(4 * 1024 * 1024);
            let cursor = std::io::Cursor::new(&mut buf);
            black_box(&img_1024)
                .write_to(cursor, image::ImageFormat::Png)
                .expect("encode png");
            black_box(buf);
        });
    });

    // JPEG encoding
    group.bench_function(BenchmarkId::new("encode_jpeg_q85", "1024x1024"), |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(2 * 1024 * 1024);
            let cursor = std::io::Cursor::new(&mut buf);
            let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(cursor, 85);
            black_box(&img_1024)
                .write_with_encoder(encoder)
                .expect("encode jpeg");
            black_box(buf);
        });
    });

    // WebP encoding
    group.bench_function(BenchmarkId::new("encode_webp", "1024x1024"), |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(2 * 1024 * 1024);
            let cursor = std::io::Cursor::new(&mut buf);
            black_box(&img_1024)
                .write_to(cursor, image::ImageFormat::WebP)
                .expect("encode webp");
            black_box(buf);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group: color conversion (YUV ↔ RGBA)
// ---------------------------------------------------------------------------

fn bench_color_conversion(c: &mut Criterion) {
    // 1920x1080 (1080p) test data for YUV420->RGBA
    let width: usize = 1920;
    let height: usize = 1080;
    let y_plane: Vec<u8> = (0..width * height).map(|i| (i % 235 + 16) as u8).collect();
    let u_plane: Vec<u8> = (0..(width / 2) * (height / 2))
        .map(|i| (i % 224 + 16) as u8)
        .collect();
    let v_plane: Vec<u8> = (0..(width / 2) * (height / 2))
        .map(|i| ((i + 64) % 224 + 16) as u8)
        .collect();

    // 1080p RGBA for RGBA->NV12
    let rgba_data: Vec<u8> = (0..width * height * 4)
        .map(|i| (i % 256) as u8)
        .collect();

    let mut group = c.benchmark_group("color_conversion");
    group.sample_size(10);

    group.bench_function(BenchmarkId::new("yuv420_to_rgba", "1080p"), |b| {
        b.iter(|| {
            // YUV420->RGBA conversion (BT.709, same algorithm as video::decode::hevc)
            let mut rgba = vec![0u8; width * height * 4];
            for row in 0..height {
                for col in 0..width {
                    let y_val = black_box(y_plane[row * width + col]) as f32;
                    let u_val = black_box(u_plane[(row / 2) * (width / 2) + (col / 2)]) as f32
                        - 128.0;
                    let v_val = black_box(v_plane[(row / 2) * (width / 2) + (col / 2)]) as f32
                        - 128.0;
                    let r = (y_val + 1.5748 * v_val).clamp(0.0, 255.0) as u8;
                    let g = (y_val - 0.1873 * u_val - 0.4681 * v_val).clamp(0.0, 255.0) as u8;
                    let b = (y_val + 1.8556 * u_val).clamp(0.0, 255.0) as u8;
                    let offset = (row * width + col) * 4;
                    rgba[offset] = r;
                    rgba[offset + 1] = g;
                    rgba[offset + 2] = b;
                    rgba[offset + 3] = 255;
                }
            }
            black_box(rgba);
        });
    });

    group.bench_function(BenchmarkId::new("rgba_to_nv12", "1080p"), |b| {
        b.iter(|| {
            // RGBA->NV12 conversion (BT.709, same algorithm as video::encode::nvenc)
            let y_size = width * height;
            let uv_size = width * (height / 2);
            let mut nv12 = vec![0u8; y_size + uv_size];
            let (y_out, uv_out) = nv12.split_at_mut(y_size);

            for row in 0..height {
                for col in 0..width {
                    let idx = (row * width + col) * 4;
                    let r = black_box(rgba_data[idx]) as f32;
                    let g = black_box(rgba_data[idx + 1]) as f32;
                    let b_val = black_box(rgba_data[idx + 2]) as f32;
                    let y = (0.2126 * r + 0.7152 * g + 0.0722 * b_val).clamp(0.0, 255.0);
                    y_out[row * width + col] = y as u8;
                }
            }

            for row in (0..height).step_by(2) {
                for col in (0..width).step_by(2) {
                    let mut sum_r = 0.0f32;
                    let mut sum_g = 0.0f32;
                    let mut sum_b = 0.0f32;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let idx = ((row + dy) * width + (col + dx)) * 4;
                            sum_r += rgba_data[idx] as f32;
                            sum_g += rgba_data[idx + 1] as f32;
                            sum_b += rgba_data[idx + 2] as f32;
                        }
                    }
                    let r = sum_r / 4.0;
                    let g = sum_g / 4.0;
                    let b_val = sum_b / 4.0;
                    let cb =
                        (-0.1146 * r - 0.3854 * g + 0.5 * b_val + 128.0).clamp(0.0, 255.0) as u8;
                    let cr =
                        (0.5 * r - 0.4542 * g - 0.0458 * b_val + 128.0).clamp(0.0, 255.0) as u8;
                    let uv_idx = (row / 2) * width + col;
                    uv_out[uv_idx] = cb;
                    uv_out[uv_idx + 1] = cr;
                }
            }

            black_box(nv12);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group: analysis (histogram)
// ---------------------------------------------------------------------------

fn bench_analysis(c: &mut Criterion) {
    let img_1024 = make_rgba_image(1024, 1024);

    let mut group = c.benchmark_group("analysis");
    group.sample_size(20);

    group.bench_function(BenchmarkId::new("histogram", "1024x1024"), |b| {
        b.iter(|| histogram(black_box(&img_1024)));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_transforms,
    bench_image_processing,
    bench_adjustments,
    bench_encoding,
    bench_color_conversion,
    bench_analysis,
);
criterion_main!(benches);
