use image::{DynamicImage, ImageBuffer, Rgba};
use xeno_lib::transforms::{flip_horizontal, flip_vertical};

fn create_test_image(width: u32, height: u32) -> DynamicImage {
    let buffer = ImageBuffer::from_fn(width, height, |x, y| {
        let r = (x % 256) as u8;
        let g = (y % 256) as u8;
        let b = ((x + y) % 256) as u8;
        let a = 255u8;
        Rgba([r, g, b, a])
    });
    DynamicImage::ImageRgba8(buffer)
}

fn main() {
    println!("Testing SIMD usage in xeno_lib transforms");
    println!("==========================================\n");

    // Test with a moderately sized image (4000x2500 = 10MP like the benchmark)
    let width = 4000u32;
    let height = 2500u32;
    println!(
        "Creating {}x{} RGBA8 test image ({} pixels)...",
        width,
        height,
        width * height
    );
    let image = create_test_image(width, height);

    println!("\nImage details:");
    println!("  Color type: {:?}", image.color());
    println!("  Bytes per pixel: 4 (RGBA)");
    println!("  Total bytes: {}", width * height * 4);

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            println!("\n✓ AVX2 is available - SIMD acceleration should be active");
        } else {
            println!("\n✗ AVX2 is NOT available - will use scalar fallback");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("\n✓ NEON is available (always on aarch64) - SIMD acceleration should be active");
    }

    println!("\nPerforming horizontal flip...");
    let start = std::time::Instant::now();
    match flip_horizontal(&image) {
        Ok(_result) => {
            let duration = start.elapsed();
            println!("  ✓ Completed in {:.2}ms", duration.as_secs_f64() * 1000.0);
        }
        Err(e) => println!("  ✗ Error: {:?}", e),
    }

    println!("\nPerforming vertical flip...");
    let start = std::time::Instant::now();
    match flip_vertical(&image) {
        Ok(_result) => {
            let duration = start.elapsed();
            println!("  ✓ Completed in {:.2}ms", duration.as_secs_f64() * 1000.0);
        }
        Err(e) => println!("  ✗ Error: {:?}", e),
    }

    println!("\n===========================================");
    println!("Notes:");
    println!("- Horizontal flip (RGBA): Should use AVX2 (8 pixels/32 bytes per vector)");
    println!("- Vertical flip: Should use AVX2 copy_row (64 bytes at a time)");
    println!("- If times are much slower than expected, SIMD may not be working");
}
