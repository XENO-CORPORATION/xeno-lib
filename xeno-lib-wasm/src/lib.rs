//! WASM bindings for xeno-lib pure-compute image processing functions.
//!
//! This crate exposes a subset of xeno-lib's functionality to WebAssembly,
//! targeting `wasm32-unknown-unknown` for browser usage via `wasm-bindgen`.
//!
//! Only pure-compute functions are exposed — no file system, no network,
//! no ONNX Runtime, no GPU/CUDA dependencies.
//!
//! ## Supported operations
//!
//! - **Transforms:** flip (horizontal/vertical), rotate (90/180/270), crop, resize
//! - **Adjustments:** brightness, contrast, saturation, hue, grayscale, invert
//! - **Filters:** gaussian blur, sharpen (unsharp mask), edge detect, emboss, sepia
//! - **Analysis:** histogram computation
//! - **Color conversion:** YUV420 to RGBA, RGBA to NV12
//!
//! ## Data format
//!
//! All image data is passed as raw RGBA `u8` byte arrays with explicit
//! `width` and `height` parameters. Callers are responsible for ensuring
//! `data.len() == width * height * 4`.

use image::{DynamicImage, ImageBuffer, Rgba};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Reconstruct a DynamicImage from raw RGBA bytes + dimensions.
fn rgba_to_dynamic(data: &[u8], width: u32, height: u32) -> Result<DynamicImage, JsValue> {
    let expected = (width as usize) * (height as usize) * 4;
    if data.len() != expected {
        return Err(JsValue::from_str(&format!(
            "Invalid data length: expected {} ({}x{}x4), got {}",
            expected, width, height, data.len()
        )));
    }
    ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, data.to_vec())
        .map(DynamicImage::ImageRgba8)
        .ok_or_else(|| JsValue::from_str("Failed to create image from raw data"))
}

/// Extract raw RGBA bytes from a DynamicImage.
fn dynamic_to_rgba(img: &DynamicImage) -> Vec<u8> {
    img.to_rgba8().into_raw()
}

/// Map a xeno-lib TransformError to a JsValue.
fn map_err(e: xeno_lib::TransformError) -> JsValue {
    JsValue::from_str(&format!("{}", e))
}

// ===========================================================================
// Transforms
// ===========================================================================

/// Flip an RGBA image horizontally (mirror across vertical axis).
#[wasm_bindgen]
pub fn flip_horizontal(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::flip_horizontal(&img).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Flip an RGBA image vertically (mirror across horizontal axis).
#[wasm_bindgen]
pub fn flip_vertical(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::flip_vertical(&img).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Rotate an RGBA image 90 degrees clockwise.
///
/// Returns pixel data with swapped dimensions (new width = old height, new height = old width).
#[wasm_bindgen]
pub fn rotate_90(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::rotate_90(&img).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Rotate an RGBA image 180 degrees.
#[wasm_bindgen]
pub fn rotate_180(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::rotate_180(&img).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Rotate an RGBA image 270 degrees clockwise (90 degrees counter-clockwise).
///
/// Returns pixel data with swapped dimensions (new width = old height, new height = old width).
#[wasm_bindgen]
pub fn rotate_270(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::rotate_270(&img).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Crop a rectangular region from an RGBA image.
///
/// Returns the cropped pixel data. Output dimensions: `(x2 - x1) x (y2 - y1)`.
#[wasm_bindgen]
pub fn crop(
    data: &[u8],
    width: u32,
    height: u32,
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::crop(&img, x1, y1, x2, y2).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Resize an RGBA image to exact dimensions using bilinear interpolation.
///
/// Returns pixel data at the new dimensions.
#[wasm_bindgen]
pub fn resize(
    data: &[u8],
    width: u32,
    height: u32,
    new_width: u32,
    new_height: u32,
) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::resize_exact(
        &img,
        new_width,
        new_height,
        xeno_lib::transforms::Interpolation::Bilinear,
    )
    .map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

// ===========================================================================
// Filters
// ===========================================================================

/// Apply a gaussian blur to an RGBA image.
///
/// `sigma` controls blur radius (typical values: 0.5 to 10.0).
#[wasm_bindgen]
pub fn gaussian_blur(data: &[u8], width: u32, height: u32, sigma: f64) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::gaussian_blur(&img, sigma as f32).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Sharpen an RGBA image using an unsharp mask.
///
/// `sigma` controls the blur radius used for unsharp masking.
/// `threshold` controls the minimum difference for sharpening to apply.
#[wasm_bindgen]
pub fn sharpen(
    data: &[u8],
    width: u32,
    height: u32,
    sigma: f64,
    threshold: i32,
) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::unsharp_mask(&img, sigma as f32, threshold).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Detect edges in an RGBA image using a Laplacian kernel.
///
/// `strength` scales the edge detection kernel (typical: 0.5 to 2.0).
#[wasm_bindgen]
pub fn edge_detect(data: &[u8], width: u32, height: u32, strength: f64) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::edge_detect(&img, strength as f32).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Apply an emboss effect to an RGBA image.
///
/// `strength` scales the emboss kernel (typical: 0.5 to 2.0).
#[wasm_bindgen]
pub fn emboss(data: &[u8], width: u32, height: u32, strength: f64) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::emboss(&img, strength as f32).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Apply a sepia tone effect to an RGBA image.
#[wasm_bindgen]
pub fn sepia(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::sepia(&img).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

// ===========================================================================
// Adjustments
// ===========================================================================

/// Adjust brightness of an RGBA image.
///
/// `amount` is a percentage in [-100, 100]. Positive values brighten, negative darken.
#[wasm_bindgen]
pub fn adjust_brightness(data: &[u8], width: u32, height: u32, amount: f64) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::adjust_brightness(&img, amount as f32).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Adjust contrast of an RGBA image.
///
/// `amount` is a percentage in [-100, 100].
#[wasm_bindgen]
pub fn adjust_contrast(data: &[u8], width: u32, height: u32, amount: f64) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::adjust_contrast(&img, amount as f32).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Adjust saturation of an RGBA image.
///
/// `amount` is a percentage in [-100, 100]. -100 desaturates fully, +100 doubles saturation.
#[wasm_bindgen]
pub fn adjust_saturation(data: &[u8], width: u32, height: u32, amount: f64) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::adjust_saturation(&img, amount as f32).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Shift hue of an RGBA image.
///
/// `degrees` rotates the hue wheel (any value, wraps at 360).
#[wasm_bindgen]
pub fn adjust_hue(data: &[u8], width: u32, height: u32, degrees: f64) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::adjust_hue(&img, degrees as f32).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Convert an RGBA image to grayscale (preserves alpha).
#[wasm_bindgen]
pub fn grayscale(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::grayscale(&img).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

/// Invert the colors of an RGBA image (preserves alpha).
#[wasm_bindgen]
pub fn invert(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    let img = rgba_to_dynamic(data, width, height)?;
    let result = xeno_lib::invert(&img).map_err(map_err)?;
    Ok(dynamic_to_rgba(&result))
}

// ===========================================================================
// Color Conversion (standalone, no DynamicImage needed)
// ===========================================================================

/// Convert YUV420 planar data to RGBA.
///
/// Expects separate Y, U, V planes with standard 4:2:0 chroma subsampling.
/// Width and height must be even. Output length = width * height * 4.
#[wasm_bindgen]
pub fn yuv420_to_rgba(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, JsValue> {
    let w = width as usize;
    let h = height as usize;

    if w == 0 || h == 0 || w % 2 != 0 || h % 2 != 0 {
        return Err(JsValue::from_str("Width and height must be positive and even"));
    }
    if y_plane.len() < w * h {
        return Err(JsValue::from_str("Y plane too small"));
    }
    if u_plane.len() < (w / 2) * (h / 2) || v_plane.len() < (w / 2) * (h / 2) {
        return Err(JsValue::from_str("U/V planes too small"));
    }

    let mut rgba = vec![0u8; w * h * 4];
    for row in 0..h {
        for col in 0..w {
            let y_val = y_plane[row * w + col] as f32;
            let u_val = u_plane[(row / 2) * (w / 2) + (col / 2)] as f32 - 128.0;
            let v_val = v_plane[(row / 2) * (w / 2) + (col / 2)] as f32 - 128.0;

            // BT.709 YUV to RGB
            let r = (y_val + 1.5748 * v_val).clamp(0.0, 255.0) as u8;
            let g = (y_val - 0.1873 * u_val - 0.4681 * v_val).clamp(0.0, 255.0) as u8;
            let b = (y_val + 1.8556 * u_val).clamp(0.0, 255.0) as u8;

            let offset = (row * w + col) * 4;
            rgba[offset] = r;
            rgba[offset + 1] = g;
            rgba[offset + 2] = b;
            rgba[offset + 3] = 255;
        }
    }
    Ok(rgba)
}

/// Convert RGBA data to NV12 (Y plane + interleaved UV plane).
///
/// Width and height must be even. Output length = width * height * 3 / 2.
#[wasm_bindgen]
pub fn rgba_to_nv12(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    let w = width as usize;
    let h = height as usize;

    if w == 0 || h == 0 || w % 2 != 0 || h % 2 != 0 {
        return Err(JsValue::from_str("Width and height must be positive and even"));
    }
    if rgba.len() < w * h * 4 {
        return Err(JsValue::from_str("RGBA data too small"));
    }

    let y_size = w * h;
    let uv_size = w * (h / 2);
    let mut nv12 = vec![0u8; y_size + uv_size];
    let (y_out, uv_out) = nv12.split_at_mut(y_size);

    // BT.709 RGB to Y
    for row in 0..h {
        for col in 0..w {
            let idx = (row * w + col) * 4;
            let r = rgba[idx] as f32;
            let g = rgba[idx + 1] as f32;
            let b = rgba[idx + 2] as f32;
            y_out[row * w + col] = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 255.0) as u8;
        }
    }

    // BT.709 RGB to CbCr (2x2 block average)
    for row in (0..h).step_by(2) {
        for col in (0..w).step_by(2) {
            let mut sum_r = 0.0f32;
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let idx = ((row + dy) * w + (col + dx)) * 4;
                    sum_r += rgba[idx] as f32;
                    sum_g += rgba[idx + 1] as f32;
                    sum_b += rgba[idx + 2] as f32;
                }
            }
            let r = sum_r / 4.0;
            let g = sum_g / 4.0;
            let b = sum_b / 4.0;
            let cb = (-0.1146 * r - 0.3854 * g + 0.5 * b + 128.0).clamp(0.0, 255.0) as u8;
            let cr = (0.5 * r - 0.4542 * g - 0.0458 * b + 128.0).clamp(0.0, 255.0) as u8;
            let uv_idx = (row / 2) * w + col;
            uv_out[uv_idx] = cb;
            uv_out[uv_idx + 1] = cr;
        }
    }

    Ok(nv12)
}
