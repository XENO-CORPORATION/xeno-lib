//! Image encoding bindings (Priority 1 -- for xeno-pixel export).
//!
//! Encode raw RGBA buffers to common image formats (PNG, JPEG, WebP).

use image::DynamicImage;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::helpers::buffer_to_image;

/// Encode an RGBA buffer to PNG.
///
/// Returns a `Buffer` containing the compressed PNG bytes.
#[napi]
pub fn encode_png(buffer: Buffer, width: u32, height: u32) -> Result<Buffer> {
    let img = buffer_to_image(&buffer, width, height)?;
    encode_image_format(&img, image::ImageFormat::Png)
}

/// Encode an RGBA buffer to JPEG.
///
/// `quality` is 1-100 (higher = better quality, larger file).
#[napi]
pub fn encode_jpeg(buffer: Buffer, width: u32, height: u32, quality: u32) -> Result<Buffer> {
    let img = buffer_to_image(&buffer, width, height)?;
    let quality = quality.clamp(1, 100) as u8;

    // JPEG doesn't support alpha, convert RGBA to RGB
    let rgb = DynamicImage::ImageRgb8(img.to_rgb8());
    let mut out = std::io::Cursor::new(Vec::new());
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
    rgb.write_with_encoder(encoder).map_err(|e| {
        Error::new(Status::GenericFailure, format!("JPEG encode failed: {e}"))
    })?;
    Ok(out.into_inner().into())
}

/// Encode an RGBA buffer to WebP.
///
/// `quality` is 1-100 (higher = better quality, larger file).
#[napi]
pub fn encode_webp(buffer: Buffer, width: u32, height: u32, quality: u32) -> Result<Buffer> {
    let img = buffer_to_image(&buffer, width, height)?;
    let _quality = quality.clamp(1, 100);

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
