//! Image processing bindings (Priority 1 -- for xeno-pixel).
//!
//! All functions accept and return RGBA u8 buffers (4 bytes per pixel, row-major).
//! Input validation is performed before any processing to ensure descriptive
//! error messages are returned to JavaScript callers.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::helpers::{buffer_to_image, image_to_vec, transform_err};
use crate::validation::{validate_finite_f64, validate_positive_f64, validate_resize_dimensions};

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

/// Apply a Gaussian blur to an RGBA image.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
/// * `radius` - Blur sigma (must be > 0, not NaN or Infinity)
///
/// # Returns
/// RGBA u8 buffer of the blurred image (same dimensions as input).
///
/// # Errors
/// - If buffer size does not match `width * height * 4`
/// - If width or height is zero
/// - If radius is <= 0, NaN, or Infinity
/// - If the underlying blur operation fails
#[napi]
pub fn apply_gaussian_blur(
    buffer: Buffer,
    width: u32,
    height: u32,
    radius: f64,
) -> Result<Buffer> {
    validate_positive_f64(radius, "Blur radius")?;
    let img = buffer_to_image(&buffer, width, height)?;
    let result = xeno_lib::gaussian_blur(&img, radius as f32).map_err(transform_err)?;
    Ok(image_to_vec(&result).into())
}

/// Sharpen an RGBA image using unsharp mask.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
/// * `strength` - Sharpening sigma (must be > 0, not NaN or Infinity)
///
/// # Returns
/// RGBA u8 buffer of the sharpened image (same dimensions as input).
///
/// # Errors
/// - If buffer size does not match `width * height * 4`
/// - If width or height is zero
/// - If strength is <= 0, NaN, or Infinity
/// - If the underlying sharpen operation fails
#[napi]
pub fn apply_sharpen(
    buffer: Buffer,
    width: u32,
    height: u32,
    strength: f64,
) -> Result<Buffer> {
    validate_positive_f64(strength, "Sharpen strength")?;
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
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
/// * `amount` - Brightness adjustment (typical range: -100 to 100, must be finite)
///
/// # Returns
/// RGBA u8 buffer with adjusted brightness (same dimensions as input).
///
/// # Errors
/// - If buffer size does not match `width * height * 4`
/// - If width or height is zero
/// - If amount is NaN or Infinity
#[napi]
pub fn adjust_brightness(
    buffer: Buffer,
    width: u32,
    height: u32,
    amount: f64,
) -> Result<Buffer> {
    validate_finite_f64(amount, "Brightness amount")?;
    let img = buffer_to_image(&buffer, width, height)?;
    let result = xeno_lib::adjust_brightness(&img, amount as f32).map_err(transform_err)?;
    Ok(image_to_vec(&result).into())
}

/// Adjust contrast of an RGBA image.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
/// * `amount` - Contrast adjustment (typical range: -100 to 100, must be finite)
///
/// # Returns
/// RGBA u8 buffer with adjusted contrast (same dimensions as input).
///
/// # Errors
/// - If buffer size does not match `width * height * 4`
/// - If width or height is zero
/// - If amount is NaN or Infinity
#[napi]
pub fn adjust_contrast(
    buffer: Buffer,
    width: u32,
    height: u32,
    amount: f64,
) -> Result<Buffer> {
    validate_finite_f64(amount, "Contrast amount")?;
    let img = buffer_to_image(&buffer, width, height)?;
    let result = xeno_lib::adjust_contrast(&img, amount as f32).map_err(transform_err)?;
    Ok(image_to_vec(&result).into())
}

/// Rotate the hue of an RGBA image.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
/// * `degrees` - Hue rotation angle in degrees (must be finite; wraps naturally)
///
/// # Returns
/// RGBA u8 buffer with rotated hue (same dimensions as input).
///
/// # Errors
/// - If buffer size does not match `width * height * 4`
/// - If width or height is zero
/// - If degrees is NaN or Infinity
#[napi]
pub fn adjust_hue(
    buffer: Buffer,
    width: u32,
    height: u32,
    degrees: f64,
) -> Result<Buffer> {
    validate_finite_f64(degrees, "Hue degrees")?;
    let img = buffer_to_image(&buffer, width, height)?;
    let result = xeno_lib::adjust_hue(&img, degrees as f32).map_err(transform_err)?;
    Ok(image_to_vec(&result).into())
}

/// Adjust saturation of an RGBA image.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
/// * `amount` - Saturation adjustment (typical range: -100 to 100, must be finite)
///
/// # Returns
/// RGBA u8 buffer with adjusted saturation (same dimensions as input).
///
/// # Errors
/// - If buffer size does not match `width * height * 4`
/// - If width or height is zero
/// - If amount is NaN or Infinity
#[napi]
pub fn adjust_saturation(
    buffer: Buffer,
    width: u32,
    height: u32,
    amount: f64,
) -> Result<Buffer> {
    validate_finite_f64(amount, "Saturation amount")?;
    let img = buffer_to_image(&buffer, width, height)?;
    let result = xeno_lib::adjust_saturation(&img, amount as f32).map_err(transform_err)?;
    Ok(image_to_vec(&result).into())
}

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------

/// Resize an RGBA image to new dimensions.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Current image width in pixels (must be > 0)
/// * `height` - Current image height in pixels (must be > 0)
/// * `new_width` - Target width in pixels (must be > 0)
/// * `new_height` - Target height in pixels (must be > 0)
/// * `method` - Interpolation method: `"nearest"` or `"bilinear"` (default)
///
/// # Returns
/// RGBA u8 buffer of the resized image (`new_width * new_height * 4` bytes).
///
/// # Errors
/// - If buffer size does not match `width * height * 4`
/// - If any dimension (width, height, new_width, new_height) is zero
/// - If the underlying resize operation fails
#[napi]
pub fn resize_image(
    buffer: Buffer,
    width: u32,
    height: u32,
    new_width: u32,
    new_height: u32,
    method: String,
) -> Result<Buffer> {
    validate_resize_dimensions(new_width, new_height)?;
    let img = buffer_to_image(&buffer, width, height)?;
    let interpolation = match method.to_lowercase().as_str() {
        "nearest" => xeno_lib::Interpolation::Nearest,
        _ => xeno_lib::Interpolation::Bilinear,
    };
    let result =
        xeno_lib::resize_exact(&img, new_width, new_height, interpolation).map_err(transform_err)?;
    Ok(image_to_vec(&result).into())
}
