use super::*;
use image::{DynamicImage, ImageBuffer, Rgba};
use pretty_assertions::assert_eq;

fn rgb_image() -> DynamicImage {
    let data = vec![
        255, 0, 0, //
        0, 255, 0, //
        0, 0, 255, //
    ];
    let buffer = ImageBuffer::from_raw(3, 1, data).expect("valid buffer");
    DynamicImage::ImageRgb8(buffer)
}

fn rgba_pixel_image() -> DynamicImage {
    let data = vec![128, 64, 32, 200];
    let buffer = ImageBuffer::from_raw(1, 1, data).expect("valid buffer");
    DynamicImage::ImageRgba8(buffer)
}

#[test]
fn grayscale_preserves_alpha() {
    let image = rgba_pixel_image();
    let out = grayscale(&image).expect("grayscale succeeds");
    let px = out.to_rgba8().get_pixel(0, 0).0;
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 200);
}

#[test]
fn invert_flips_channels_only() {
    let image = rgba_pixel_image();
    let out = invert(&image).expect("invert succeeds");
    let px = out.to_rgba8().get_pixel(0, 0).0;
    assert_eq!(px, [127, 191, 223, 200]);
}

#[test]
fn brightness_clamps() {
    let image = rgba_pixel_image();
    let out = adjust_brightness(&image, 100.0).expect("brighten succeeds");
    let px = out.to_rgba8().get_pixel(0, 0).0;
    assert_eq!(px, [255, 255, 255, 200]);
}

#[test]
fn contrast_decreases_midtones() {
    let mut data = vec![128, 128, 128, 128];
    let buffer = ImageBuffer::from_raw(1, 1, data.clone()).expect("buffer");
    let image = DynamicImage::ImageRgba8(buffer);
    let out = adjust_contrast(&image, -50.0).expect("contrast succeeds");
    let px = out.to_rgba8().get_pixel(0, 0).0;
    assert!(px[0] > 60 && px[0] < 200);
    assert_eq!(px[3], 128);
    data.copy_from_slice(&px);
}

#[test]
fn saturation_zero_produces_gray() {
    let image = rgb_image();
    let out = adjust_saturation(&image, -100.0).expect("desaturate succeeds");
    let buffer = out.to_rgb8();
    let pixel = buffer.get_pixel(0, 0);
    assert_eq!(pixel[0], pixel[1]);
    assert_eq!(pixel[1], pixel[2]);
}

#[test]
fn hue_rotation_cycles_channels() {
    let image = rgb_image();
    let out = adjust_hue(&image, 120.0).expect("hue succeeds");
    let buffer = out.to_rgb8();
    let rotated = buffer.get_pixel(0, 0);
    assert!(rotated[1] > rotated[0]);
}

#[test]
fn exposure_increases_values() {
    let image = rgba_pixel_image();
    let out = adjust_exposure(&image, 1.0).expect("exposure succeeds");
    let px = out.to_rgba8().get_pixel(0, 0).0;
    assert!(px[0] > 128);
    assert_eq!(px[3], 200);
}

#[test]
fn gamma_less_than_one_brightens() {
    let image = rgba_pixel_image();
    let out = adjust_gamma(&image, 0.5).expect("gamma succeeds");
    let px = out.to_rgba8().get_pixel(0, 0).0;
    assert!(px[0] > 128);
    assert_eq!(px[3], 200);
}

// =========================================================================
// Deep testing: identity adjustments
// =========================================================================

#[test]
fn brightness_zero_is_identity() {
    let image = rgba_pixel_image();
    let out = adjust_brightness(&image, 0.0).expect("brightness succeeds");
    assert_eq!(
        image.to_rgba8().get_pixel(0, 0),
        out.to_rgba8().get_pixel(0, 0),
        "brightness(0) must be identity"
    );
}

#[test]
fn contrast_zero_is_identity() {
    let image = rgba_pixel_image();
    let out = adjust_contrast(&image, 0.0).expect("contrast succeeds");
    let input_px = image.to_rgba8().get_pixel(0, 0).0;
    let out_px = out.to_rgba8().get_pixel(0, 0).0;
    // contrast(0) uses factor = (259 * 255) / (255 * 259) = 1.0, so it should be identity
    assert_eq!(input_px, out_px, "contrast(0) must be identity");
}

#[test]
fn hue_zero_is_identity() {
    let image = rgba_pixel_image();
    let out = adjust_hue(&image, 0.0).expect("hue succeeds");
    let input_px = image.to_rgba8().get_pixel(0, 0).0;
    let out_px = out.to_rgba8().get_pixel(0, 0).0;
    // Allow +/- 1 for floating point rounding
    for i in 0..4 {
        assert!(
            (input_px[i] as i16 - out_px[i] as i16).abs() <= 1,
            "hue(0) channel {} differs: input={}, output={}",
            i,
            input_px[i],
            out_px[i]
        );
    }
}

#[test]
fn hue_360_is_identity() {
    let image = rgba_pixel_image();
    let out = adjust_hue(&image, 360.0).expect("hue succeeds");
    let input_px = image.to_rgba8().get_pixel(0, 0).0;
    let out_px = out.to_rgba8().get_pixel(0, 0).0;
    // Allow +/- 1 for floating point rounding
    for i in 0..4 {
        assert!(
            (input_px[i] as i16 - out_px[i] as i16).abs() <= 1,
            "hue(360) channel {} differs: input={}, output={}",
            i,
            input_px[i],
            out_px[i]
        );
    }
}

#[test]
fn saturation_zero_is_identity() {
    let image = rgba_pixel_image();
    let out = adjust_saturation(&image, 0.0).expect("saturation succeeds");
    let input_px = image.to_rgba8().get_pixel(0, 0).0;
    let out_px = out.to_rgba8().get_pixel(0, 0).0;
    // saturation(0) uses factor=1.0, so lum + (channel - lum) * 1.0 = channel
    assert_eq!(input_px, out_px, "saturation(0) must be identity");
}

#[test]
fn saturation_minus100_produces_grayscale() {
    // Use a more colorful pixel
    let data = vec![255, 0, 0, 255]; // pure red
    let buffer = ImageBuffer::from_raw(1, 1, data).expect("valid buffer");
    let image = DynamicImage::ImageRgba8(buffer);

    let out = adjust_saturation(&image, -100.0).expect("saturation succeeds");
    let px = out.to_rgba8().get_pixel(0, 0).0;
    // All color channels should be equal (grayscale)
    assert_eq!(px[0], px[1], "R and G should be equal after full desaturation");
    assert_eq!(px[1], px[2], "G and B should be equal after full desaturation");
    assert_eq!(px[3], 255, "alpha should be preserved");
}

// =========================================================================
// Deep testing: invert(invert(x)) = x
// =========================================================================

#[test]
fn double_invert_is_identity() {
    let data = vec![
        100, 150, 200, 180,
        0, 255, 128, 64,
        50, 75, 100, 255,
        200, 100, 50, 0,
    ];
    let buffer = ImageBuffer::from_raw(2, 2, data.clone()).expect("valid buffer");
    let image = DynamicImage::ImageRgba8(buffer);

    let once = invert(&image).expect("invert succeeds");
    let twice = invert(&once).expect("invert succeeds");
    let out = twice.to_rgba8().into_raw();

    assert_eq!(out, data, "invert(invert(x)) must equal x");
}

// =========================================================================
// Deep testing: grayscale uses Rec.709 coefficients
// =========================================================================

#[test]
fn grayscale_uses_rec709_coefficients() {
    // Pure white (255, 255, 255) -> should remain 255
    let data = vec![255, 255, 255, 255];
    let buffer = ImageBuffer::from_raw(1, 1, data).expect("valid buffer");
    let image = DynamicImage::ImageRgba8(buffer);
    let out = grayscale(&image).expect("grayscale succeeds");
    let px = out.to_rgba8().get_pixel(0, 0).0;
    assert_eq!(px[0], 255);

    // Pure red (255, 0, 0) -> Rec.709: 0.2126 * 255 = 54.2 -> 54
    let data = vec![255, 0, 0, 255];
    let buffer = ImageBuffer::from_raw(1, 1, data).expect("valid buffer");
    let image = DynamicImage::ImageRgba8(buffer);
    let out = grayscale(&image).expect("grayscale succeeds");
    let px = out.to_rgba8().get_pixel(0, 0).0;
    assert_eq!(px[0], 54, "grayscale of pure red should be ~54 (Rec.709)");

    // Pure green (0, 255, 0) -> Rec.709: 0.7152 * 255 = 182.4 -> 182
    let data = vec![0, 255, 0, 255];
    let buffer = ImageBuffer::from_raw(1, 1, data).expect("valid buffer");
    let image = DynamicImage::ImageRgba8(buffer);
    let out = grayscale(&image).expect("grayscale succeeds");
    let px = out.to_rgba8().get_pixel(0, 0).0;
    assert_eq!(px[0], 182, "grayscale of pure green should be ~182 (Rec.709)");

    // Pure blue (0, 0, 255) -> Rec.709: 0.0722 * 255 = 18.4 -> 18
    let data = vec![0, 0, 255, 255];
    let buffer = ImageBuffer::from_raw(1, 1, data).expect("valid buffer");
    let image = DynamicImage::ImageRgba8(buffer);
    let out = grayscale(&image).expect("grayscale succeeds");
    let px = out.to_rgba8().get_pixel(0, 0).0;
    assert_eq!(px[0], 18, "grayscale of pure blue should be ~18 (Rec.709)");
}

// =========================================================================
// Deep testing: all adjustments preserve alpha
// =========================================================================

#[test]
fn all_adjustments_preserve_alpha() {
    let data = vec![128, 64, 32, 42]; // alpha = 42
    let buffer = ImageBuffer::from_raw(1, 1, data).expect("valid buffer");
    let image = DynamicImage::ImageRgba8(buffer);

    let check_alpha = |img: &DynamicImage, name: &str| {
        let px = img.to_rgba8().get_pixel(0, 0).0;
        assert_eq!(px[3], 42, "{} did not preserve alpha: got {}", name, px[3]);
    };

    check_alpha(&adjust_brightness(&image, 50.0).unwrap(), "brightness");
    check_alpha(&adjust_brightness(&image, -50.0).unwrap(), "brightness neg");
    check_alpha(&adjust_contrast(&image, 50.0).unwrap(), "contrast");
    check_alpha(&adjust_contrast(&image, -50.0).unwrap(), "contrast neg");
    check_alpha(&adjust_saturation(&image, 50.0).unwrap(), "saturation");
    check_alpha(&adjust_saturation(&image, -50.0).unwrap(), "saturation neg");
    check_alpha(&adjust_hue(&image, 90.0).unwrap(), "hue");
    check_alpha(&adjust_exposure(&image, 1.0).unwrap(), "exposure");
    check_alpha(&adjust_gamma(&image, 2.0).unwrap(), "gamma");
    check_alpha(&grayscale(&image).unwrap(), "grayscale");
    check_alpha(&invert(&image).unwrap(), "invert");
}

// =========================================================================
// Deep testing: all adjustments produce pixels in [0, 255]
// =========================================================================

#[test]
fn all_adjustments_clamp_to_valid_range() {
    // Extreme pixel values to stress clamping
    let data = vec![
        0, 0, 0, 255,       // black
        255, 255, 255, 255,  // white
        255, 0, 0, 255,      // red
        0, 255, 0, 255,      // green
    ];
    let buffer = ImageBuffer::from_raw(2, 2, data).expect("valid buffer");
    let image = DynamicImage::ImageRgba8(buffer);

    let check = |img: &DynamicImage, name: &str| {
        for pixel in img.to_rgba8().pixels() {
            for c in 0..4 {
                assert!(pixel[c] <= 255, "{} produced out-of-range value", name);
            }
        }
    };

    check(&adjust_brightness(&image, 100.0).unwrap(), "brightness +100");
    check(&adjust_brightness(&image, -100.0).unwrap(), "brightness -100");
    check(&adjust_contrast(&image, 100.0).unwrap(), "contrast +100");
    check(&adjust_contrast(&image, -100.0).unwrap(), "contrast -100");
    check(&adjust_saturation(&image, 100.0).unwrap(), "saturation +100");
    check(&adjust_saturation(&image, -100.0).unwrap(), "saturation -100");
    check(&adjust_hue(&image, 180.0).unwrap(), "hue 180");
    check(&adjust_exposure(&image, 3.0).unwrap(), "exposure +3");
    check(&adjust_exposure(&image, -3.0).unwrap(), "exposure -3");
    check(&adjust_gamma(&image, 0.1).unwrap(), "gamma 0.1");
    check(&adjust_gamma(&image, 5.0).unwrap(), "gamma 5.0");
}
