//! Image processing bindings (Priority 1 -- for xeno-pixel).
//!
//! All functions accept and return RGBA u8 buffers (4 bytes per pixel).

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::helpers::{buffer_to_image, image_to_vec, transform_err};

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

/// Apply a Gaussian blur to an RGBA image.
///
/// `radius` is the blur sigma (> 0).
#[napi]
pub fn apply_gaussian_blur(
    buffer: Buffer,
    width: u32,
    height: u32,
    radius: f64,
) -> Result<Buffer> {
    let img = buffer_to_image(&buffer, width, height)?;
    let result = xeno_lib::gaussian_blur(&img, radius as f32).map_err(transform_err)?;
    Ok(image_to_vec(&result).into())
}

/// Sharpen an RGBA image using unsharp mask.
///
/// `strength` controls the sharpening sigma (> 0).
#[napi]
pub fn apply_sharpen(
    buffer: Buffer,
    width: u32,
    height: u32,
    strength: f64,
) -> Result<Buffer> {
    let img = buffer_to_image(&buffer, width, height)?;
    // unsharp_mask(sigma, threshold) -- use threshold=0 for uniform sharpen
    let result = xeno_lib::unsharp_mask(&img, strength as f32, 0).map_err(transform_err)?;
    Ok(image_to_vec(&result).into())
}

// ---------------------------------------------------------------------------
// Adjustments
// ---------------------------------------------------------------------------

/// Adjust brightness of an RGBA image.
///
/// `amount` is in the range \[-100, 100\].
#[napi]
pub fn adjust_brightness(
    buffer: Buffer,
    width: u32,
    height: u32,
    amount: f64,
) -> Result<Buffer> {
    let img = buffer_to_image(&buffer, width, height)?;
    let result = xeno_lib::adjust_brightness(&img, amount as f32).map_err(transform_err)?;
    Ok(image_to_vec(&result).into())
}

/// Adjust contrast of an RGBA image.
///
/// `amount` is in the range \[-100, 100\].
#[napi]
pub fn adjust_contrast(
    buffer: Buffer,
    width: u32,
    height: u32,
    amount: f64,
) -> Result<Buffer> {
    let img = buffer_to_image(&buffer, width, height)?;
    let result = xeno_lib::adjust_contrast(&img, amount as f32).map_err(transform_err)?;
    Ok(image_to_vec(&result).into())
}

/// Rotate the hue of an RGBA image.
///
/// `degrees` is an arbitrary angle (e.g., 180 inverts hue).
#[napi]
pub fn adjust_hue(
    buffer: Buffer,
    width: u32,
    height: u32,
    degrees: f64,
) -> Result<Buffer> {
    let img = buffer_to_image(&buffer, width, height)?;
    let result = xeno_lib::adjust_hue(&img, degrees as f32).map_err(transform_err)?;
    Ok(image_to_vec(&result).into())
}

/// Adjust saturation of an RGBA image.
///
/// `amount` is in the range \[-100, 100\].
#[napi]
pub fn adjust_saturation(
    buffer: Buffer,
    width: u32,
    height: u32,
    amount: f64,
) -> Result<Buffer> {
    let img = buffer_to_image(&buffer, width, height)?;
    let result = xeno_lib::adjust_saturation(&img, amount as f32).map_err(transform_err)?;
    Ok(image_to_vec(&result).into())
}

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------

/// Resize an RGBA image.
///
/// `method` is one of: `"nearest"`, `"bilinear"` (default if unrecognised).
#[napi]
pub fn resize_image(
    buffer: Buffer,
    width: u32,
    height: u32,
    new_width: u32,
    new_height: u32,
    method: String,
) -> Result<Buffer> {
    let img = buffer_to_image(&buffer, width, height)?;
    let interpolation = match method.to_lowercase().as_str() {
        "nearest" => xeno_lib::Interpolation::Nearest,
        _ => xeno_lib::Interpolation::Bilinear,
    };
    let result =
        xeno_lib::resize_exact(&img, new_width, new_height, interpolation).map_err(transform_err)?;
    Ok(image_to_vec(&result).into())
}
