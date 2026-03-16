//! Postprocessing utilities for background removal.
//!
//! Applies the predicted alpha mask to the original image to create
//! an RGBA output with transparent background.

use image::{DynamicImage, RgbaImage};
use ndarray::Array2;
use rayon::prelude::*;

use crate::error::TransformError;

/// Applies an alpha mask to the original image, creating an RGBA output.
///
/// # Processing Steps
///
/// 1. Resize mask from model output size to original image dimensions
/// 2. Apply guided filtering using original image edges
/// 3. Composite original image with refined alpha channel
///
/// # Arguments
///
/// * `original` - The original input image
/// * `mask` - The predicted mask from the model (values in [0, 1])
/// * `original_width` - Original image width
/// * `original_height` - Original image height
/// * `_threshold` - Confidence threshold (unused, kept for API compatibility)
///
/// # Returns
///
/// An RGBA image with transparent background.
pub fn apply_mask(
    original: &DynamicImage,
    mask: &Array2<f32>,
    original_width: u32,
    original_height: u32,
    _threshold: f32,
) -> Result<DynamicImage, TransformError> {
    // Resize mask to original dimensions
    let resized_mask = resize_mask(mask, original_width as usize, original_height as usize);

    // Convert original to RGBA
    let rgba = original.to_rgba8();

    // Apply guided filter to refine mask edges using original image
    let refined_mask = guided_filter(&rgba, &resized_mask, 15, 0.01);

    // Apply mask to create output
    let output = apply_mask_to_image(&rgba, &refined_mask);

    Ok(DynamicImage::ImageRgba8(output))
}

/// Guided filter implementation for edge-aware mask refinement.
/// Uses the original image as a guide to preserve edges in the mask.
///
/// This is the key technique that makes Photoshop-level quality possible.
/// The filter smooths the mask while preserving edges that exist in the guide image.
fn guided_filter(guide: &RgbaImage, mask: &Array2<f32>, radius: i32, eps: f32) -> Array2<f32> {
    let width = guide.width() as usize;
    let height = guide.height() as usize;

    // Convert guide to grayscale float
    let mut guide_gray = Array2::<f32>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let pixel = guide.get_pixel(x as u32, y as u32);
            // Luminance formula
            guide_gray[[y, x]] = (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) / 255.0;
        }
    }

    // Compute box filter means
    let mean_i = box_filter(&guide_gray, radius);
    let mean_p = box_filter(mask, radius);

    // Compute correlations
    let mut ip = Array2::<f32>::zeros((height, width));
    let mut ii = Array2::<f32>::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            ip[[y, x]] = guide_gray[[y, x]] * mask[[y, x]];
            ii[[y, x]] = guide_gray[[y, x]] * guide_gray[[y, x]];
        }
    }

    let mean_ip = box_filter(&ip, radius);
    let mean_ii = box_filter(&ii, radius);

    // Compute covariance and variance
    let mut cov_ip = Array2::<f32>::zeros((height, width));
    let mut var_i = Array2::<f32>::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            cov_ip[[y, x]] = mean_ip[[y, x]] - mean_i[[y, x]] * mean_p[[y, x]];
            var_i[[y, x]] = mean_ii[[y, x]] - mean_i[[y, x]] * mean_i[[y, x]];
        }
    }

    // Compute a and b coefficients
    let mut a = Array2::<f32>::zeros((height, width));
    let mut b = Array2::<f32>::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            a[[y, x]] = cov_ip[[y, x]] / (var_i[[y, x]] + eps);
            b[[y, x]] = mean_p[[y, x]] - a[[y, x]] * mean_i[[y, x]];
        }
    }

    // Compute mean of a and b
    let mean_a = box_filter(&a, radius);
    let mean_b = box_filter(&b, radius);

    // Compute output
    let mut output = Array2::<f32>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            output[[y, x]] = (mean_a[[y, x]] * guide_gray[[y, x]] + mean_b[[y, x]]).clamp(0.0, 1.0);
        }
    }

    output
}

/// Fast box filter using integral images (summed area table).
fn box_filter(input: &Array2<f32>, radius: i32) -> Array2<f32> {
    let (height, width) = input.dim();

    // Build integral image
    let mut integral = Array2::<f64>::zeros((height + 1, width + 1));

    for y in 0..height {
        for x in 0..width {
            integral[[y + 1, x + 1]] = input[[y, x]] as f64
                + integral[[y, x + 1]]
                + integral[[y + 1, x]]
                - integral[[y, x]];
        }
    }

    // Apply box filter using integral image
    let mut output = Array2::<f32>::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let y0 = (y as i32 - radius).max(0) as usize;
            let y1 = ((y as i32 + radius + 1) as usize).min(height);
            let x0 = (x as i32 - radius).max(0) as usize;
            let x1 = ((x as i32 + radius + 1) as usize).min(width);

            let area = ((y1 - y0) * (x1 - x0)) as f64;
            let sum = integral[[y1, x1]] - integral[[y0, x1]] - integral[[y1, x0]] + integral[[y0, x0]];

            output[[y, x]] = (sum / area) as f32;
        }
    }

    output
}

/// Resizes a mask array using bilinear interpolation.
fn resize_mask(mask: &Array2<f32>, target_width: usize, target_height: usize) -> Array2<f32> {
    let (src_height, src_width) = mask.dim();

    // If dimensions match, return a clone
    if src_width == target_width && src_height == target_height {
        return mask.clone();
    }

    let mut resized = Array2::<f32>::zeros((target_height, target_width));

    // Calculate scaling factors
    let scale_x = src_width as f32 / target_width as f32;
    let scale_y = src_height as f32 / target_height as f32;

    // Bilinear interpolation using parallel iteration
    // SAFETY: Array2::zeros always produces a standard-layout (contiguous) array,
    // so as_slice_mut will never return None.
    resized
        .as_slice_mut()
        .expect("Array2::zeros always produces a contiguous array")
        .par_chunks_mut(target_width)
        .enumerate()
        .for_each(|(y, row)| {
            let src_y = y as f32 * scale_y;
            let y0 = (src_y.floor() as usize).min(src_height - 1);
            let y1 = (y0 + 1).min(src_height - 1);
            let y_frac = src_y - y0 as f32;

            for (x, pixel) in row.iter_mut().enumerate() {
                let src_x = x as f32 * scale_x;
                let x0 = (src_x.floor() as usize).min(src_width - 1);
                let x1 = (x0 + 1).min(src_width - 1);
                let x_frac = src_x - x0 as f32;

                // Bilinear interpolation
                let v00 = mask[[y0, x0]];
                let v01 = mask[[y0, x1]];
                let v10 = mask[[y1, x0]];
                let v11 = mask[[y1, x1]];

                let v0 = v00 * (1.0 - x_frac) + v01 * x_frac;
                let v1 = v10 * (1.0 - x_frac) + v11 * x_frac;

                *pixel = v0 * (1.0 - y_frac) + v1 * y_frac;
            }
        });

    resized
}

/// Applies the mask to an RGBA image, modifying the alpha channel.
fn apply_mask_to_image(image: &RgbaImage, mask: &Array2<f32>) -> RgbaImage {
    let width = image.width();
    let height = image.height();

    // Create output buffer
    let mut output_data: Vec<u8> = vec![0; (width * height * 4) as usize];

    // Get input data
    let input_data = image.as_raw();

    // Process pixels in parallel (by rows)
    output_data
        .par_chunks_mut((width * 4) as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let input_row_start = y * (width as usize) * 4;

            for x in 0..width as usize {
                let pixel_offset = x * 4;
                let input_offset = input_row_start + pixel_offset;

                // Get mask value (0.0 = background, 1.0 = foreground)
                let mask_value = mask[[y, x]].clamp(0.0, 1.0);

                // Use smooth alpha directly from mask value
                let alpha = (mask_value * 255.0).round() as u8;

                // Copy RGB, set alpha
                row[pixel_offset] = input_data[input_offset]; // R
                row[pixel_offset + 1] = input_data[input_offset + 1]; // G
                row[pixel_offset + 2] = input_data[input_offset + 2]; // B
                row[pixel_offset + 3] = alpha; // A
            }
        });

    RgbaImage::from_raw(width, height, output_data).expect("output buffer size is correct")
}

/// Applies morphological dilation to expand the mask slightly.
/// Useful for ensuring hair and fine details are included.
#[allow(dead_code)]
pub fn dilate_mask(mask: &Array2<f32>, iterations: u32) -> Array2<f32> {
    if iterations == 0 {
        return mask.clone();
    }

    let (height, width) = mask.dim();
    let mut current = mask.clone();

    for _ in 0..iterations {
        let mut next = Array2::<f32>::zeros((height, width));

        for y in 0..height {
            for x in 0..width {
                let mut max_val = current[[y, x]];

                // Check 3x3 neighborhood
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                        let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                        max_val = max_val.max(current[[ny, nx]]);
                    }
                }

                next[[y, x]] = max_val;
            }
        }

        current = next;
    }

    current
}

/// Applies morphological erosion to shrink the mask slightly.
/// Useful for removing edge artifacts.
#[allow(dead_code)]
pub fn erode_mask(mask: &Array2<f32>, iterations: u32) -> Array2<f32> {
    if iterations == 0 {
        return mask.clone();
    }

    let (height, width) = mask.dim();
    let mut current = mask.clone();

    for _ in 0..iterations {
        let mut next = Array2::<f32>::zeros((height, width));

        for y in 0..height {
            for x in 0..width {
                let mut min_val = current[[y, x]];

                // Check 3x3 neighborhood
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                        let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                        min_val = min_val.min(current[[ny, nx]]);
                    }
                }

                next[[y, x]] = min_val;
            }
        }

        current = next;
    }

    current
}

/// Applies Gaussian blur to smooth mask edges.
#[allow(dead_code)]
pub fn blur_mask(mask: &Array2<f32>, sigma: f32) -> Array2<f32> {
    if sigma <= 0.0 {
        return mask.clone();
    }

    let (height, width) = mask.dim();

    // Create Gaussian kernel
    let kernel_radius = (sigma * 3.0).ceil() as i32;
    let kernel_size = (kernel_radius * 2 + 1) as usize;
    let mut kernel = vec![0.0f32; kernel_size];

    let sigma_sq = sigma * sigma;
    let mut sum = 0.0f32;

    for (i, k) in kernel.iter_mut().enumerate() {
        let x = i as i32 - kernel_radius;
        *k = (-((x * x) as f32) / (2.0 * sigma_sq)).exp();
        sum += *k;
    }

    // Normalize kernel
    for k in &mut kernel {
        *k /= sum;
    }

    // Horizontal pass
    let mut temp = Array2::<f32>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0f32;
            for (i, &k) in kernel.iter().enumerate() {
                let sx = (x as i32 + i as i32 - kernel_radius).clamp(0, width as i32 - 1) as usize;
                value += mask[[y, sx]] * k;
            }
            temp[[y, x]] = value;
        }
    }

    // Vertical pass
    let mut result = Array2::<f32>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0f32;
            for (i, &k) in kernel.iter().enumerate() {
                let sy = (y as i32 + i as i32 - kernel_radius).clamp(0, height as i32 - 1) as usize;
                value += temp[[sy, x]] * k;
            }
            result[[y, x]] = value;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mask(width: usize, height: usize) -> Array2<f32> {
        let mut mask = Array2::<f32>::zeros((height, width));
        // Create a circle in the center
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let radius = (width.min(height) as f32) / 3.0;

        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                mask[[y, x]] = if dist < radius { 1.0 } else { 0.0 };
            }
        }
        mask
    }

    #[test]
    fn test_resize_mask_same_size() {
        let mask = create_test_mask(64, 64);
        let resized = resize_mask(&mask, 64, 64);
        assert_eq!(resized.dim(), (64, 64));
    }

    #[test]
    fn test_resize_mask_upscale() {
        let mask = create_test_mask(32, 32);
        let resized = resize_mask(&mask, 64, 64);
        assert_eq!(resized.dim(), (64, 64));
    }

    #[test]
    fn test_resize_mask_downscale() {
        let mask = create_test_mask(128, 128);
        let resized = resize_mask(&mask, 64, 64);
        assert_eq!(resized.dim(), (64, 64));
    }

    #[test]
    fn test_dilate_preserves_shape() {
        let mask = create_test_mask(64, 64);
        let dilated = dilate_mask(&mask, 2);
        assert_eq!(dilated.dim(), mask.dim());
    }

    #[test]
    fn test_erode_preserves_shape() {
        let mask = create_test_mask(64, 64);
        let eroded = erode_mask(&mask, 2);
        assert_eq!(eroded.dim(), mask.dim());
    }

    #[test]
    fn test_apply_mask_output_dimensions() {
        let image = RgbaImage::new(100, 80);
        let mask = create_test_mask(100, 80);
        let output = apply_mask_to_image(&image, &mask);
        assert_eq!(output.width(), 100);
        assert_eq!(output.height(), 80);
    }
}
