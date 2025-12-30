//! Image filter effects built on top of convolution kernels and gaussian blur.
//!
//! This module provides FFmpeg-equivalent video/image filters in pure Rust:
//! - Blur, sharpen (unsharp mask)
//! - Edge detection, emboss
//! - Sepia, vignette
//! - Denoise (spatial)
//! - Chromakey (green screen removal)
//! - Deinterlace (basic)

use crate::error::TransformError;
use image::{DynamicImage, Rgba, RgbaImage};
use image::imageops;

const EDGE_KERNEL: [f32; 9] = [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];
const EMBOSS_KERNEL: [f32; 9] = [-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0];

/// Apply a gaussian blur with the provided sigma (> 0).
pub fn gaussian_blur(image: &DynamicImage, sigma: f32) -> Result<DynamicImage, TransformError> {
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(TransformError::InvalidParameter {
            name: "sigma",
            value: sigma,
        });
    }
    if sigma.abs() < f32::EPSILON {
        return Ok(image.clone());
    }
    let rgba = image.to_rgba8();
    Ok(DynamicImage::ImageRgba8(imageops::blur(&rgba, sigma)))
}

/// Apply an unsharp mask (sharpen) using gaussian blur radius and threshold.
pub fn unsharp_mask(
    image: &DynamicImage,
    sigma: f32,
    threshold: i32,
) -> Result<DynamicImage, TransformError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(TransformError::InvalidParameter {
            name: "sigma",
            value: sigma,
        });
    }
    if threshold < 0 {
        return Err(TransformError::InvalidParameter {
            name: "threshold",
            value: threshold as f32,
        });
    }
    let rgba = image.to_rgba8();
    Ok(DynamicImage::ImageRgba8(imageops::unsharpen(
        &rgba, sigma, threshold,
    )))
}

/// Highlight edges using a Laplacian kernel scaled by `strength`.
pub fn edge_detect(image: &DynamicImage, strength: f32) -> Result<DynamicImage, TransformError> {
    if !strength.is_finite() {
        return Err(TransformError::InvalidParameter {
            name: "strength",
            value: strength,
        });
    }
    let scale = strength.max(0.0);
    if scale.abs() < f32::EPSILON {
        return Ok(image.clone());
    }
    let kernel: [f32; 9] = EDGE_KERNEL.map(|v| v * scale);
    let rgba = image.to_rgba8();
    Ok(DynamicImage::ImageRgba8(imageops::filter3x3(
        &rgba, &kernel,
    )))
}

/// Apply a classic emboss kernel to simulate relief shading.
pub fn emboss(image: &DynamicImage, strength: f32) -> Result<DynamicImage, TransformError> {
    if !strength.is_finite() {
        return Err(TransformError::InvalidParameter {
            name: "strength",
            value: strength,
        });
    }
    let kernel: [f32; 9] = EMBOSS_KERNEL.map(|v| v * strength);
    if strength.abs() < f32::EPSILON {
        return Ok(image.clone());
    }
    let rgba = image.to_rgba8();
    Ok(DynamicImage::ImageRgba8(imageops::filter3x3(
        &rgba, &kernel,
    )))
}

/// Apply a sepia tone effect while preserving alpha.
pub fn sepia(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    let mut rgba = image.to_rgba8();
    for pixel in rgba.pixels_mut() {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        pixel[0] = clamp_to_u8(0.393 * r + 0.769 * g + 0.189 * b);
        pixel[1] = clamp_to_u8(0.349 * r + 0.686 * g + 0.168 * b);
        pixel[2] = clamp_to_u8(0.272 * r + 0.534 * g + 0.131 * b);
    }
    Ok(DynamicImage::ImageRgba8(rgba))
}

/// Apply a vignette effect (darkened corners).
///
/// # Arguments
/// * `image` - Input image
/// * `strength` - Vignette strength (0.0 = none, 1.0 = strong)
/// * `radius` - Vignette radius relative to image size (0.0-1.0, 0.5 = half size)
pub fn vignette(
    image: &DynamicImage,
    strength: f32,
    radius: f32,
) -> Result<DynamicImage, TransformError> {
    if !strength.is_finite() || !radius.is_finite() {
        return Err(TransformError::InvalidParameter {
            name: "vignette parameters",
            value: strength,
        });
    }

    let strength = strength.clamp(0.0, 2.0);
    let radius = radius.clamp(0.1, 1.0);

    if strength.abs() < f32::EPSILON {
        return Ok(image.clone());
    }

    let mut rgba = image.to_rgba8();
    let (width, height) = (rgba.width() as f32, rgba.height() as f32);
    let center_x = width / 2.0;
    let center_y = height / 2.0;
    let max_dist = ((center_x * center_x + center_y * center_y) as f32).sqrt();
    let vignette_radius = max_dist * radius;

    for (x, y, pixel) in rgba.enumerate_pixels_mut() {
        let dx = x as f32 - center_x;
        let dy = y as f32 - center_y;
        let dist = (dx * dx + dy * dy).sqrt();

        // Calculate vignette factor
        let factor = if dist < vignette_radius {
            1.0
        } else {
            let normalized = (dist - vignette_radius) / (max_dist - vignette_radius);
            let vignette = 1.0 - (normalized * strength).min(1.0);
            vignette.max(0.0)
        };

        pixel[0] = clamp_to_u8(pixel[0] as f32 * factor);
        pixel[1] = clamp_to_u8(pixel[1] as f32 * factor);
        pixel[2] = clamp_to_u8(pixel[2] as f32 * factor);
    }

    Ok(DynamicImage::ImageRgba8(rgba))
}

/// Apply spatial denoise filter using a simple median-like approach.
///
/// # Arguments
/// * `image` - Input image
/// * `strength` - Denoise strength (1-10, higher = more smoothing)
pub fn denoise(image: &DynamicImage, strength: u32) -> Result<DynamicImage, TransformError> {
    if strength == 0 {
        return Ok(image.clone());
    }

    let strength = strength.min(10);

    // Use bilateral-like filtering: blur + edge preservation
    // For simplicity, we use a combination of blur and edge-aware blending
    let rgba = image.to_rgba8();
    let (width, height) = (rgba.width(), rgba.height());

    // Apply a light blur
    let blur_sigma = (strength as f32) * 0.5;
    let blurred = imageops::blur(&rgba, blur_sigma);

    // Blend based on local contrast (edge preservation)
    let mut output = RgbaImage::new(width, height);
    let threshold = 30.0 * (strength as f32 / 5.0); // Edge threshold

    for y in 0..height {
        for x in 0..width {
            let orig = rgba.get_pixel(x, y);
            let blur = blurred.get_pixel(x, y);

            // Calculate local variance to detect edges
            let mut variance = 0.0f32;
            let radius = 1i32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;
                    let neighbor = rgba.get_pixel(nx, ny);

                    let diff_r = (orig[0] as f32 - neighbor[0] as f32).abs();
                    let diff_g = (orig[1] as f32 - neighbor[1] as f32).abs();
                    let diff_b = (orig[2] as f32 - neighbor[2] as f32).abs();
                    variance += diff_r + diff_g + diff_b;
                }
            }
            variance /= 9.0 * 3.0;

            // High variance = edge, keep original; low variance = smooth area, use blur
            let blend = (variance / threshold).min(1.0);

            let out_r = orig[0] as f32 * blend + blur[0] as f32 * (1.0 - blend);
            let out_g = orig[1] as f32 * blend + blur[1] as f32 * (1.0 - blend);
            let out_b = orig[2] as f32 * blend + blur[2] as f32 * (1.0 - blend);

            output.put_pixel(x, y, Rgba([
                clamp_to_u8(out_r),
                clamp_to_u8(out_g),
                clamp_to_u8(out_b),
                orig[3],
            ]));
        }
    }

    Ok(DynamicImage::ImageRgba8(output))
}

/// Remove green screen (chromakey) from image.
///
/// # Arguments
/// * `image` - Input image
/// * `key_color` - The color to remove (e.g., green screen)
/// * `tolerance` - Color tolerance (0.0-1.0, higher = more removal)
/// * `softness` - Edge softness (0.0-1.0, higher = smoother edges)
pub fn chromakey(
    image: &DynamicImage,
    key_color: Rgba<u8>,
    tolerance: f32,
    softness: f32,
) -> Result<DynamicImage, TransformError> {
    if !tolerance.is_finite() || !softness.is_finite() {
        return Err(TransformError::InvalidParameter {
            name: "chromakey parameters",
            value: tolerance,
        });
    }

    let tolerance = tolerance.clamp(0.0, 1.0);
    let softness = softness.clamp(0.0, 1.0);

    let mut rgba = image.to_rgba8();
    let key_r = key_color[0] as f32;
    let key_g = key_color[1] as f32;
    let key_b = key_color[2] as f32;

    // Tolerance in color distance
    let max_dist = 441.67; // sqrt(255^2 * 3)
    let tolerance_dist = tolerance * max_dist;
    let softness_dist = softness * max_dist * 0.2;

    for pixel in rgba.pixels_mut() {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;

        // Calculate color distance from key
        let dist = ((r - key_r).powi(2) + (g - key_g).powi(2) + (b - key_b).powi(2)).sqrt();

        // Calculate alpha based on distance
        let alpha = if dist < tolerance_dist {
            0.0 // Fully transparent
        } else if dist < tolerance_dist + softness_dist {
            // Soft edge
            let t = (dist - tolerance_dist) / softness_dist;
            t
        } else {
            1.0 // Fully opaque
        };

        // Apply alpha
        pixel[3] = (pixel[3] as f32 * alpha) as u8;
    }

    Ok(DynamicImage::ImageRgba8(rgba))
}

/// Convenience function for green screen removal.
pub fn remove_green_screen(
    image: &DynamicImage,
    tolerance: f32,
    softness: f32,
) -> Result<DynamicImage, TransformError> {
    // Standard chroma green: RGB(0, 177, 64) or similar
    chromakey(image, Rgba([0, 177, 64, 255]), tolerance, softness)
}

/// Convenience function for blue screen removal.
pub fn remove_blue_screen(
    image: &DynamicImage,
    tolerance: f32,
    softness: f32,
) -> Result<DynamicImage, TransformError> {
    // Standard chroma blue: RGB(0, 71, 187) or similar
    chromakey(image, Rgba([0, 71, 187, 255]), tolerance, softness)
}

/// Apply basic deinterlacing using line blending.
///
/// This is a simple bob-style deinterlacer that blends adjacent lines.
/// For proper motion-adaptive deinterlacing, you would need temporal information
/// from multiple frames.
///
/// # Arguments
/// * `image` - Input interlaced frame
/// * `field` - Which field to keep: 0 = top field (even lines), 1 = bottom field (odd lines)
pub fn deinterlace(image: &DynamicImage, field: u8) -> Result<DynamicImage, TransformError> {
    let rgba = image.to_rgba8();
    let (width, height) = (rgba.width(), rgba.height());

    if height < 2 {
        return Ok(image.clone());
    }

    let mut output = RgbaImage::new(width, height);
    let keep_even = field == 0;

    for y in 0..height {
        let is_kept_line = if keep_even { y % 2 == 0 } else { y % 2 == 1 };

        for x in 0..width {
            let pixel = if is_kept_line {
                // Keep this line
                *rgba.get_pixel(x, y)
            } else {
                // Interpolate from adjacent lines
                let above = if y > 0 { rgba.get_pixel(x, y - 1) } else { rgba.get_pixel(x, y) };
                let below = if y < height - 1 { rgba.get_pixel(x, y + 1) } else { rgba.get_pixel(x, y) };

                Rgba([
                    ((above[0] as u16 + below[0] as u16) / 2) as u8,
                    ((above[1] as u16 + below[1] as u16) / 2) as u8,
                    ((above[2] as u16 + below[2] as u16) / 2) as u8,
                    ((above[3] as u16 + below[3] as u16) / 2) as u8,
                ])
            };

            output.put_pixel(x, y, pixel);
        }
    }

    Ok(DynamicImage::ImageRgba8(output))
}

/// Apply posterize effect (reduce color levels).
///
/// # Arguments
/// * `image` - Input image
/// * `levels` - Number of color levels per channel (2-256)
pub fn posterize(image: &DynamicImage, levels: u8) -> Result<DynamicImage, TransformError> {
    let levels = levels.max(2);
    let step = 255.0 / (levels - 1) as f32;

    let mut rgba = image.to_rgba8();

    for pixel in rgba.pixels_mut() {
        for i in 0..3 {
            let quantized = ((pixel[i] as f32 / step).round() * step) as u8;
            pixel[i] = quantized;
        }
    }

    Ok(DynamicImage::ImageRgba8(rgba))
}

/// Apply solarize effect (invert colors above threshold).
///
/// # Arguments
/// * `image` - Input image
/// * `threshold` - Threshold value (0-255)
pub fn solarize(image: &DynamicImage, threshold: u8) -> Result<DynamicImage, TransformError> {
    let mut rgba = image.to_rgba8();

    for pixel in rgba.pixels_mut() {
        for i in 0..3 {
            if pixel[i] > threshold {
                pixel[i] = 255 - pixel[i];
            }
        }
    }

    Ok(DynamicImage::ImageRgba8(rgba))
}

/// Apply color temperature adjustment (warm/cool).
///
/// # Arguments
/// * `image` - Input image
/// * `temperature` - Temperature shift (-100 to 100, negative = cooler/blue, positive = warmer/orange)
pub fn color_temperature(
    image: &DynamicImage,
    temperature: f32,
) -> Result<DynamicImage, TransformError> {
    if !temperature.is_finite() {
        return Err(TransformError::InvalidParameter {
            name: "temperature",
            value: temperature,
        });
    }

    let temperature = temperature.clamp(-100.0, 100.0);

    if temperature.abs() < f32::EPSILON {
        return Ok(image.clone());
    }

    let mut rgba = image.to_rgba8();

    // Temperature adjusts red/blue balance
    let r_adjust = temperature * 0.5;  // Warm adds red
    let b_adjust = -temperature * 0.5; // Warm removes blue

    for pixel in rgba.pixels_mut() {
        pixel[0] = clamp_to_u8(pixel[0] as f32 + r_adjust);
        pixel[2] = clamp_to_u8(pixel[2] as f32 + b_adjust);
    }

    Ok(DynamicImage::ImageRgba8(rgba))
}

/// Apply tint adjustment (green/magenta balance).
///
/// # Arguments
/// * `image` - Input image
/// * `tint` - Tint shift (-100 to 100, negative = green, positive = magenta)
pub fn tint(image: &DynamicImage, tint_value: f32) -> Result<DynamicImage, TransformError> {
    if !tint_value.is_finite() {
        return Err(TransformError::InvalidParameter {
            name: "tint",
            value: tint_value,
        });
    }

    let tint_value = tint_value.clamp(-100.0, 100.0);

    if tint_value.abs() < f32::EPSILON {
        return Ok(image.clone());
    }

    let mut rgba = image.to_rgba8();

    // Tint adjusts green/magenta balance
    let g_adjust = -tint_value * 0.5; // Magenta removes green

    for pixel in rgba.pixels_mut() {
        pixel[1] = clamp_to_u8(pixel[1] as f32 + g_adjust);
    }

    Ok(DynamicImage::ImageRgba8(rgba))
}

/// Apply vibrance adjustment (smart saturation that protects skin tones).
///
/// # Arguments
/// * `image` - Input image
/// * `amount` - Vibrance amount (-100 to 100)
pub fn vibrance(image: &DynamicImage, amount: f32) -> Result<DynamicImage, TransformError> {
    if !amount.is_finite() {
        return Err(TransformError::InvalidParameter {
            name: "vibrance",
            value: amount,
        });
    }

    let amount = amount.clamp(-100.0, 100.0) / 100.0;

    if amount.abs() < f32::EPSILON {
        return Ok(image.clone());
    }

    let mut rgba = image.to_rgba8();

    for pixel in rgba.pixels_mut() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;

        // Find the max and min values
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let saturation = if max > 0.0 { (max - min) / max } else { 0.0 };

        // Apply vibrance: less saturated colors get more boost
        let factor = 1.0 + amount * (1.0 - saturation);

        // Calculate luminance
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        // Apply factor to each channel
        let new_r = lum + (r - lum) * factor;
        let new_g = lum + (g - lum) * factor;
        let new_b = lum + (b - lum) * factor;

        pixel[0] = clamp_to_u8(new_r * 255.0);
        pixel[1] = clamp_to_u8(new_g * 255.0);
        pixel[2] = clamp_to_u8(new_b * 255.0);
    }

    Ok(DynamicImage::ImageRgba8(rgba))
}

#[inline]
fn clamp_to_u8(value: f32) -> u8 {
    value.max(0.0).min(255.0).round() as u8
}

#[cfg(test)]
mod tests;
