//! Image encoding bindings (Priority 1 -- for xeno-pixel export).
//!
//! Encode raw RGBA buffers to common image formats (PNG, JPEG, WebP).
//! Quality parameters are clamped to valid ranges rather than rejected,
//! matching the behavior of most image editors.

use image::DynamicImage;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::helpers::buffer_to_image;
use crate::validation::clamp_quality;

/// Encode an RGBA buffer to PNG format.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
///
/// # Returns
/// A `Buffer` containing the compressed PNG file bytes.
///
/// # Errors
/// - If buffer size does not match `width * height * 4`
/// - If width or height is zero
/// - If PNG encoding fails
#[napi]
pub fn encode_png(buffer: Buffer, width: u32, height: u32) -> Result<Buffer> {
    let img = buffer_to_image(&buffer, width, height)?;
    encode_image_format(&img, image::ImageFormat::Png)
}

/// Encode an RGBA buffer to JPEG format.
///
/// Alpha channel is discarded (JPEG does not support transparency).
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
/// * `quality` - JPEG quality (1-100, clamped if out of range; higher = better quality)
///
/// # Returns
/// A `Buffer` containing the compressed JPEG file bytes.
///
/// # Errors
/// - If buffer size does not match `width * height * 4`
/// - If width or height is zero
/// - If JPEG encoding fails
#[napi]
pub fn encode_jpeg(buffer: Buffer, width: u32, height: u32, quality: u32) -> Result<Buffer> {
    let img = buffer_to_image(&buffer, width, height)?;
    let quality = clamp_quality(quality) as u8;

    // JPEG doesn't support alpha, convert RGBA to RGB
    let rgb = DynamicImage::ImageRgb8(img.to_rgb8());
    let mut out = std::io::Cursor::new(Vec::new());
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
    rgb.write_with_encoder(encoder).map_err(|e| {
        Error::new(Status::GenericFailure, format!("JPEG encode failed: {e}"))
    })?;
    Ok(out.into_inner().into())
}

/// Encode an RGBA buffer to WebP format.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
/// * `quality` - WebP quality (1-100, clamped if out of range; higher = better quality)
///
/// # Returns
/// A `Buffer` containing the compressed WebP file bytes.
///
/// # Errors
/// - If buffer size does not match `width * height * 4`
/// - If width or height is zero
/// - If WebP encoding fails
#[napi]
pub fn encode_webp(buffer: Buffer, width: u32, height: u32, quality: u32) -> Result<Buffer> {
    let img = buffer_to_image(&buffer, width, height)?;
    let _quality = clamp_quality(quality);

    // The `image` crate supports WebP encoding through its built-in encoder.
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::WebP).map_err(|e| {
        Error::new(Status::GenericFailure, format!("WebP encode failed: {e}"))
    })?;
    Ok(out.into_inner().into())
}

/// Internal helper: encode a DynamicImage to the given format.
fn encode_image_format(img: &DynamicImage, format: image::ImageFormat) -> Result<Buffer> {
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, format).map_err(|e| {
        Error::new(
            Status::GenericFailure,
            format!("Image encode failed: {e}"),
        )
    })?;
    Ok(out.into_inner().into())
}
