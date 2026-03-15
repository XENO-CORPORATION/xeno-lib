//! Internal helpers for converting between N-API types and xeno-lib types.

use image::{DynamicImage, RgbaImage};
use napi::bindgen_prelude::*;

/// Reconstruct an RGBA `DynamicImage` from a raw RGBA u8 buffer.
///
/// Returns an error if the buffer length does not match `width * height * 4`.
pub fn buffer_to_image(
    buffer: &[u8],
    width: u32,
    height: u32,
) -> Result<DynamicImage> {
    let expected = (width as usize) * (height as usize) * 4;
    if buffer.len() != expected {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Buffer length mismatch: expected {} bytes ({}x{}x4), got {}",
                expected, width, height, buffer.len()
            ),
        ));
    }
    let rgba = RgbaImage::from_raw(width, height, buffer.to_vec()).ok_or_else(|| {
        Error::new(
            Status::GenericFailure,
            "Failed to construct RGBA image from buffer".to_string(),
        )
    })?;
    Ok(DynamicImage::ImageRgba8(rgba))
}

/// Convert a `DynamicImage` to a flat RGBA u8 `Vec<u8>`.
pub fn image_to_vec(image: &DynamicImage) -> Vec<u8> {
    image.to_rgba8().into_raw()
}

/// Map a xeno-lib `TransformError` into a napi `Error`.
pub fn transform_err(e: xeno_lib::TransformError) -> Error {
    Error::new(Status::GenericFailure, format!("{e}"))
}
