use super::{
    recenter, recenter_with_alpha_threshold,
    Interpolation, crop, flip_horizontal, flip_vertical, resize_by_percent, resize_exact,
    resize_to_fit, resize_to_height, resize_to_width, rotate, rotate_90, rotate_180, rotate_270,
    thumbnail,
};
use crate::error::TransformError;
use image::{DynamicImage, ImageBuffer, Luma, Rgb, Rgba};
use pretty_assertions::assert_eq;
use tempfile::tempdir;

fn rgb_image() -> DynamicImage {
    let data = vec![
        255, 0, 0, //
        0, 255, 0, //
        0, 0, 255, //
    ];
    let buffer = ImageBuffer::from_raw(3, 1, data).expect("valid buffer");
    DynamicImage::ImageRgb8(buffer)
}

fn rgba_image() -> DynamicImage {
    let data = vec![
        255, 0, 0, 255, //
        0, 255, 0, 255, //
        0, 0, 255, 255, //
        255, 255, 255, 255,
    ];
    let buffer = ImageBuffer::from_raw(2, 2, data).expect("valid buffer");
    DynamicImage::ImageRgba8(buffer)
}

fn luma_image() -> DynamicImage {
    let data = vec![
        10, 20, //
        30, 40, //
        50, 60, //
    ];
    let buffer = ImageBuffer::from_raw(2, 3, data).expect("valid buffer");
    DynamicImage::ImageLuma8(buffer)
}

#[test]
fn horizontal_flip_rgb() {
    let image = rgb_image();
    let flipped = flip_horizontal(&image).expect("flip succeeds");
    let buffer = flipped.to_rgb8();
    assert_eq!(buffer.get_pixel(0, 0), &Rgb([0, 0, 255]));
    assert_eq!(buffer.get_pixel(1, 0), &Rgb([0, 255, 0]));
    assert_eq!(buffer.get_pixel(2, 0), &Rgb([255, 0, 0]));
}

#[test]
fn vertical_flip_luma() {
    let image = luma_image();
    let flipped = flip_vertical(&image).expect("flip succeeds");
    let buffer = flipped.to_luma8();

    assert_eq!(buffer.get_pixel(0, 0), &Luma([50]));
    assert_eq!(buffer.get_pixel(1, 0), &Luma([60]));
    assert_eq!(buffer.get_pixel(0, 2), &Luma([10]));
    assert_eq!(buffer.get_pixel(1, 2), &Luma([20]));
}

#[test]
fn rotate_90_rgb() {
    let image = rgba_image();
    let rotated = rotate_90(&image).expect("rotation succeeds");
    let buffer = rotated.to_rgba8();
    assert_eq!(buffer.width(), 2);
    assert_eq!(buffer.height(), 2);
    assert_eq!(buffer.get_pixel(0, 0), &Rgba([0, 0, 255, 255]));
    assert_eq!(buffer.get_pixel(1, 0), &Rgba([255, 0, 0, 255]));
    assert_eq!(buffer.get_pixel(0, 1), &Rgba([255, 255, 255, 255]));
    assert_eq!(buffer.get_pixel(1, 1), &Rgba([0, 255, 0, 255]));
}

#[test]
fn rotate_180_round_trip() {
    let image = rgb_image();
    let rotated_once = rotate_180(&image).expect("rotation succeeds");
    let rotated_twice = rotate_180(&rotated_once).expect("rotation succeeds");
    assert_eq!(image.to_rgb8(), rotated_twice.to_rgb8());
}

#[test]
fn rotate_270_dimensions() {
    let image = luma_image();
    let rotated = rotate_270(&image).expect("rotation succeeds");
    assert_eq!(rotated.width(), image.height());
    assert_eq!(rotated.height(), image.width());
}

#[test]
fn arbitrary_rotation_nearest_changes_dimensions() {
    let image = rgba_image();
    let rotated = rotate(&image, 33.0, Interpolation::Nearest).expect("rotate succeeds");
    assert!(rotated.width() > image.width());
    assert!(rotated.height() > image.height());
}

#[test]
fn arbitrary_rotation_bilinear_preserves_zero_angle() {
    let image = rgba_image();
    let rotated = rotate(&image, 360.0, Interpolation::Bilinear).expect("rotate succeeds");
    assert_eq!(rotated.to_rgba8(), image.to_rgba8());
}

#[test]
fn rotate_rejects_invalid_angle() {
    let image = rgba_image();
    let err = rotate(&image, f32::NAN, Interpolation::Bilinear).expect_err("invalid angle");
    assert!(matches!(err, TransformError::InvalidAngle { .. }));
}

#[test]
fn crop_extracts_subregion() {
    let image = rgba_image();
    let cropped = crop(&image, 0, 0, 1, 2).expect("crop succeeds");
    assert_eq!(cropped.width(), 1);
    assert_eq!(cropped.height(), 2);
    assert_eq!(cropped.to_rgba8().get_pixel(0, 1), &Rgba([0, 0, 255, 255]));
}

#[test]
fn crop_out_of_bounds_errors() {
    let image = rgba_image();
    let err = crop(&image, 1, 1, 2, 2).expect_err("crop should fail");
    assert!(matches!(err, TransformError::CropOutOfBounds { .. }));
}

#[test]
fn resize_exact_nearest_downscale() {
    let image = rgba_image();
    let resized = resize_exact(&image, 1, 1, Interpolation::Nearest).expect("resize");
    assert_eq!(resized.width(), 1);
    assert_eq!(resized.height(), 1);
    assert_eq!(resized.to_rgba8().get_pixel(0, 0), &Rgba([255, 0, 0, 255]));
}

#[test]
fn resize_to_width_preserves_aspect_ratio() {
    let image = luma_image();
    let resized = resize_to_width(&image, 4, Interpolation::Bilinear).expect("resize");
    assert_eq!(resized.width(), 4);
    assert_eq!(resized.height(), 6);
}

#[test]
fn resize_to_height_preserves_aspect_ratio() {
    let image = luma_image();
    let resized = resize_to_height(&image, 9, Interpolation::Nearest).expect("resize");
    assert_eq!(resized.height(), 9);
    assert_eq!(resized.width(), 6);
}

#[test]
fn resize_to_fit_skips_when_smaller() {
    let image = DynamicImage::new_rgb8(64, 32);
    let resized = resize_to_fit(&image, 128, 128, Interpolation::Nearest).expect("resize");
    assert_eq!(resized.width(), 64);
    assert_eq!(resized.height(), 32);
}

#[test]
fn resize_by_percent_rejects_zero() {
    let image = rgba_image();
    let err =
        resize_by_percent(&image, 0.0, Interpolation::Nearest).expect_err("should reject factor");
    assert!(matches!(err, TransformError::InvalidScaleFactor { .. }));
}

#[test]
fn thumbnail_clamps_long_edge() {
    let image = DynamicImage::new_rgba8(400, 200);
    let resized = thumbnail(&image, 128, Interpolation::Bilinear).expect("resize");
    assert_eq!(resized.width(), 128);
    assert_eq!(resized.height(), 64);
}

#[test]
fn disk_round_trip_uses_tempfile() {
    let dir = tempdir().expect("tempdir");
    let input_path = dir.path().join("input.png");
    let output_path = dir.path().join("output.png");

    let image = rgba_image();
    image.save(&input_path).expect("save input");

    let loaded = image::open(&input_path).expect("load input");
    let flipped = flip_horizontal(&loaded).expect("flip");
    flipped.save(&output_path).expect("save output");

    let reloaded = image::open(&output_path).expect("load output");
    assert_eq!(flipped.to_rgba8(), reloaded.to_rgba8());
}

#[test]
fn recenter_transparent_content_centers_subject() {
    let mut img = image::RgbaImage::new(10, 10);
    // Subject starts near the left side.
    img.put_pixel(1, 4, Rgba([255, 0, 0, 255]));
    img.put_pixel(2, 4, Rgba([255, 0, 0, 255]));
    img.put_pixel(1, 5, Rgba([255, 0, 0, 255]));
    img.put_pixel(2, 5, Rgba([255, 0, 0, 255]));

    let input = DynamicImage::ImageRgba8(img);
    let centered = recenter(&input).expect("recenter");
    let out = centered.to_rgba8();

    // 2x2 subject should land centered at x=[4,5], y=[4,5] on a 10x10 canvas.
    assert_eq!(out.get_pixel(4, 4), &Rgba([255, 0, 0, 255]));
    assert_eq!(out.get_pixel(5, 4), &Rgba([255, 0, 0, 255]));
    assert_eq!(out.get_pixel(4, 5), &Rgba([255, 0, 0, 255]));
    assert_eq!(out.get_pixel(5, 5), &Rgba([255, 0, 0, 255]));
}

#[test]
fn recenter_threshold_ignores_faint_alpha() {
    let mut img = image::RgbaImage::new(8, 8);
    // Faint pixel at corner should be ignored with threshold.
    img.put_pixel(0, 0, Rgba([255, 255, 255, 4]));
    // Real subject.
    img.put_pixel(6, 3, Rgba([0, 255, 0, 255]));

    let input = DynamicImage::ImageRgba8(img);
    let centered = recenter_with_alpha_threshold(&input, 8).expect("recenter threshold");
    let out = centered.to_rgba8();

    // Single-pixel subject should move to the same center convention as
    // `center_on_canvas` (floor on even-sized canvases).
    assert_eq!(out.get_pixel(3, 3), &Rgba([0, 255, 0, 255]));
}
