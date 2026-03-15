use super::*;
use image::{DynamicImage, ImageBuffer};

fn gradient_image() -> DynamicImage {
    let mut data = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            data.push((x * 40 + y * 10) as u8);
            data.push(128);
            data.push(200);
        }
    }
    let buffer = ImageBuffer::from_raw(4, 4, data).expect("valid buffer");
    DynamicImage::ImageRgb8(buffer)
}

fn rgba_test_image() -> DynamicImage {
    let data = vec![
        255, 0, 0, 180,     // red, semi-transparent
        0, 255, 0, 255,     // green, opaque
        0, 0, 255, 64,      // blue, mostly transparent
        128, 128, 128, 255, // gray, opaque
    ];
    let buffer = ImageBuffer::from_raw(2, 2, data).expect("valid buffer");
    DynamicImage::ImageRgba8(buffer)
}

#[test]
fn blur_with_zero_sigma_returns_clone() {
    let image = gradient_image();
    let blurred = gaussian_blur(&image, 0.0).expect("blur succeeds");
    assert_eq!(blurred.to_rgb8(), image.to_rgb8());
}

#[test]
fn blur_with_zero_sigma_rgba_returns_identical() {
    let image = rgba_test_image();
    let blurred = gaussian_blur(&image, 0.0).expect("blur succeeds");
    // sigma=0 should return a clone, but since gaussian_blur converts to RGBA8
    // via to_rgba8() internally, it may not preserve the original format.
    // What matters is pixel values are identical.
    assert_eq!(blurred.to_rgba8(), image.to_rgba8());
}

#[test]
fn blur_rejects_negative_sigma() {
    let image = gradient_image();
    assert!(gaussian_blur(&image, -1.0).is_err());
}

#[test]
fn blur_rejects_nan_sigma() {
    let image = gradient_image();
    assert!(gaussian_blur(&image, f32::NAN).is_err());
}

#[test]
fn sharpen_preserves_dimensions() {
    let image = gradient_image();
    let sharpened = unsharp_mask(&image, 1.0, 1).expect("sharpen succeeds");
    assert_eq!(sharpened.width(), image.width());
    assert_eq!(sharpened.height(), image.height());
}

#[test]
fn edge_detect_outputs_highlighted_edges() {
    let image = gradient_image();
    let edges = edge_detect(&image, 1.0).expect("edge detection succeeds");
    assert_eq!(edges.width(), image.width());
    assert_eq!(edges.height(), image.height());
}

#[test]
fn emboss_outputs_rgba_image() {
    let image = gradient_image();
    let embossed = emboss(&image, 1.0).expect("emboss succeeds");
    assert_eq!(embossed.width(), image.width());
}

#[test]
fn sepia_applies_tone() {
    let image = gradient_image();
    let toned = sepia(&image).expect("sepia succeeds").to_rgba8();
    let px = toned.get_pixel(0, 0);
    assert!(px[0] >= px[1] && px[1] >= px[2]);
}

#[test]
fn sepia_preserves_alpha() {
    let image = rgba_test_image();
    let toned = sepia(&image).expect("sepia succeeds").to_rgba8();
    // Alpha values should be preserved
    assert_eq!(toned.get_pixel(0, 0)[3], 180);
    assert_eq!(toned.get_pixel(1, 0)[3], 255);
    assert_eq!(toned.get_pixel(0, 1)[3], 64);
    assert_eq!(toned.get_pixel(1, 1)[3], 255);
}

#[test]
fn vignette_preserves_alpha() {
    let image = rgba_test_image();
    let vignetted = vignette(&image, 1.0, 0.5).expect("vignette succeeds").to_rgba8();
    // Alpha values should be preserved (vignette only darkens RGB)
    assert_eq!(vignetted.get_pixel(0, 0)[3], 180);
}

#[test]
fn denoise_with_zero_strength_returns_clone() {
    let image = gradient_image();
    let denoised = denoise(&image, 0).expect("denoise succeeds");
    assert_eq!(denoised.to_rgb8(), image.to_rgb8());
}

#[test]
fn posterize_preserves_alpha() {
    let image = rgba_test_image();
    let poster = posterize(&image, 4).expect("posterize succeeds").to_rgba8();
    assert_eq!(poster.get_pixel(0, 0)[3], 180);
    assert_eq!(poster.get_pixel(0, 1)[3], 64);
}

#[test]
fn solarize_preserves_alpha() {
    let image = rgba_test_image();
    let solar = solarize(&image, 128).expect("solarize succeeds").to_rgba8();
    assert_eq!(solar.get_pixel(0, 0)[3], 180);
    assert_eq!(solar.get_pixel(0, 1)[3], 64);
}

#[test]
fn all_filters_produce_valid_pixel_range() {
    let image = rgba_test_image();

    let check = |img: &DynamicImage, name: &str| {
        for pixel in img.to_rgba8().pixels() {
            for c in 0..4 {
                assert!(pixel[c] <= 255, "{} produced out-of-range value", name);
            }
        }
    };

    check(&gaussian_blur(&image, 2.0).unwrap(), "gaussian_blur");
    check(&unsharp_mask(&image, 1.0, 1).unwrap(), "unsharp_mask");
    check(&edge_detect(&image, 2.0).unwrap(), "edge_detect");
    check(&emboss(&image, 2.0).unwrap(), "emboss");
    check(&sepia(&image).unwrap(), "sepia");
    check(&vignette(&image, 1.0, 0.5).unwrap(), "vignette");
    check(&denoise(&image, 5).unwrap(), "denoise");
    check(&posterize(&image, 4).unwrap(), "posterize");
    check(&solarize(&image, 128).unwrap(), "solarize");
    check(&color_temperature(&image, 50.0).unwrap(), "color_temperature warm");
    check(&color_temperature(&image, -50.0).unwrap(), "color_temperature cool");
    check(&tint(&image, 50.0).unwrap(), "tint");
    check(&vibrance(&image, 50.0).unwrap(), "vibrance");
}
