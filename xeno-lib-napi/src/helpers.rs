//! Internal helpers for converting between N-API types and xeno-lib types.
//!
//! These functions form the bridge between raw JavaScript `Buffer` objects
//! (flat byte arrays) and the `image` crate's typed image representations.
//! All conversions validate dimensions and buffer sizes before proceeding.

use image::{DynamicImage, RgbaImage};
use napi::bindgen_prelude::*;

use crate::validation::validate_image_buffer;

/// Reconstruct an RGBA `DynamicImage` from a raw RGBA u8 buffer.
///
/// # Arguments
/// * `buffer` - Raw pixel data, expected to be RGBA u8 (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
///
/// # Returns
/// A `DynamicImage::ImageRgba8` containing the decoded pixel data.
///
/// # Errors
/// - If `width` or `height` is zero
/// - If `buffer.len()` does not equal `width * height * 4`
/// - If the `image` crate fails to construct the image (should not happen after validation)
pub fn buffer_to_image(
    buffer: &[u8],
    width: u32,
    height: u32,
) -> Result<DynamicImage> {
    validate_image_buffer(buffer, width, height)?;

    let rgba = RgbaImage::from_raw(width, height, buffer.to_vec()).ok_or_else(|| {
        Error::new(
            Status::GenericFailure,
            "Failed to construct RGBA image from buffer".to_string(),
        )
    })?;
    Ok(DynamicImage::ImageRgba8(rgba))
}

/// Convert a `DynamicImage` to a flat RGBA u8 `Vec<u8>`.
///
/// The output is always RGBA u8 regardless of the input image type.
/// This ensures consistency with the N-API contract that all image
/// buffers are RGBA u8 (4 bytes per pixel, row-major).
pub fn image_to_vec(image: &DynamicImage) -> Vec<u8> {
    image.to_rgba8().into_raw()
}

/// Map a xeno-lib `TransformError` into a napi `Error`.
///
/// This provides a clean error message to JavaScript callers without
/// exposing internal Rust type details.
pub fn transform_err(e: xeno_lib::TransformError) -> Error {
    Error::new(Status::GenericFailure, format!("{e}"))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_1x1_red_pixel() {
        let red_pixel: Vec<u8> = vec![255, 0, 0, 255];
        let img = buffer_to_image(&red_pixel, 1, 1).expect("should create 1x1 image");
        let out = image_to_vec(&img);
        assert_eq!(out, red_pixel);
    }

    #[test]
    fn roundtrip_2x2_image() {
        // 2x2 RGBA image: red, green, blue, white
        let pixels: Vec<u8> = vec![
            255, 0, 0, 255,     // red
            0, 255, 0, 255,     // green
            0, 0, 255, 255,     // blue
            255, 255, 255, 255, // white
        ];
        let img = buffer_to_image(&pixels, 2, 2).expect("should create 2x2 image");
        let out = image_to_vec(&img);
        assert_eq!(out, pixels);
    }

    #[test]
    fn zero_width_returns_error() {
        let err = buffer_to_image(&[], 0, 10).unwrap_err();
        assert!(err.reason.contains("non-zero"));
    }

    #[test]
    fn zero_height_returns_error() {
        let err = buffer_to_image(&[], 10, 0).unwrap_err();
        assert!(err.reason.contains("non-zero"));
    }

    #[test]
    fn wrong_buffer_length_returns_error() {
        let buf = vec![0u8; 10];
        let err = buffer_to_image(&buf, 4, 4).unwrap_err();
        assert!(err.reason.contains("Buffer size mismatch"));
    }

    #[test]
    fn empty_buffer_with_zero_dims_returns_error() {
        let err = buffer_to_image(&[], 0, 0).unwrap_err();
        assert!(err.reason.contains("non-zero"));
    }

    #[test]
    fn transparent_pixels_preserved() {
        // Semi-transparent pixel
        let pixel: Vec<u8> = vec![128, 64, 32, 127];
        let img = buffer_to_image(&pixel, 1, 1).expect("should create image");
        let out = image_to_vec(&img);
        assert_eq!(out, pixel);
    }
}
