use super::*;
use image::{DynamicImage, ImageBuffer};
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
