//! Image processing logic for Real-ESRGAN upscaling.
//!
//! Handles preprocessing, inference, and postprocessing including
//! tile-based processing for large images.

use image::{DynamicImage, Rgb, RgbImage, Rgba, RgbaImage};
use ndarray::Array4;

use crate::error::TransformError;

use super::model::UpscalerSession;

/// Upscales a small image directly without tiling.
///
/// This is used for images that fit within the tile size.
pub fn upscale_direct(
    image: &DynamicImage,
    session: &mut UpscalerSession,
) -> Result<DynamicImage, TransformError> {
    // Convert to tensor
    let input_tensor = image_to_tensor(image)?;

    // Run inference
    let output_tensor = session.run(&input_tensor)?;

    // Convert back to image
    tensor_to_image(&output_tensor)
}

/// Upscales a large image using tile-based processing.
///
/// This splits the image into overlapping tiles, upscales each tile,
/// and blends them back together to avoid seam artifacts.
pub fn upscale_tiled(
    image: &DynamicImage,
    session: &mut UpscalerSession,
) -> Result<DynamicImage, TransformError> {
    let config = session.config();
    let scale = session.scale_factor();
    let tile_size = config.tile_size;
    let overlap = config.tile_overlap;

    let input_width = image.width();
    let input_height = image.height();
    let output_width = input_width * scale;
    let output_height = input_height * scale;

    // Calculate number of tiles needed
    let step = tile_size - overlap * 2;
    let tiles_x = ((input_width as i32 - overlap as i32 * 2) as f32 / step as f32).ceil() as u32;
    let tiles_y = ((input_height as i32 - overlap as i32 * 2) as f32 / step as f32).ceil() as u32;
    let tiles_x = tiles_x.max(1);
    let tiles_y = tiles_y.max(1);

    // Create output image
    let rgb = image.to_rgb8();
    let mut output = RgbImage::new(output_width, output_height);
    let mut weight_map = vec![0.0f32; (output_width * output_height) as usize];

    // Process each tile
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            // Calculate tile bounds in input space
            let x_start = (tx * step).min(input_width.saturating_sub(tile_size));
            let y_start = (ty * step).min(input_height.saturating_sub(tile_size));
            let x_end = (x_start + tile_size).min(input_width);
            let y_end = (y_start + tile_size).min(input_height);
            let tile_w = x_end - x_start;
            let tile_h = y_end - y_start;

            // Extract tile from input
            let tile = extract_tile(&rgb, x_start, y_start, tile_w, tile_h);
            let tile_image = DynamicImage::ImageRgb8(tile);

            // Upscale tile
            let input_tensor = image_to_tensor(&tile_image)?;
            let output_tensor = session.run(&input_tensor)?;
            let upscaled_tile = tensor_to_rgb(&output_tensor)?;

            // Calculate output bounds
            let out_x_start = x_start * scale;
            let out_y_start = y_start * scale;
            let out_tile_w = tile_w * scale;
            let out_tile_h = tile_h * scale;

            // Blend tile into output with weight falloff at edges
            blend_tile(
                &mut output,
                &mut weight_map,
                &upscaled_tile,
                out_x_start,
                out_y_start,
                out_tile_w,
                out_tile_h,
                overlap * scale,
            );
        }
    }

    // Normalize by weights
    normalize_by_weights(&mut output, &weight_map);

    Ok(DynamicImage::ImageRgb8(output))
}

/// Extracts a tile from an RGB image.
fn extract_tile(image: &RgbImage, x: u32, y: u32, width: u32, height: u32) -> RgbImage {
    let mut tile = RgbImage::new(width, height);
    for ty in 0..height {
        for tx in 0..width {
            let src_x = (x + tx).min(image.width() - 1);
            let src_y = (y + ty).min(image.height() - 1);
            tile.put_pixel(tx, ty, *image.get_pixel(src_x, src_y));
        }
    }
    tile
}

/// Blends an upscaled tile into the output image with edge falloff.
fn blend_tile(
    output: &mut RgbImage,
    weight_map: &mut [f32],
    tile: &RgbImage,
    x_start: u32,
    y_start: u32,
    width: u32,
    height: u32,
    overlap: u32,
) {
    let out_width = output.width();

    for ty in 0..height {
        for tx in 0..width {
            let out_x = x_start + tx;
            let out_y = y_start + ty;

            if out_x >= output.width() || out_y >= output.height() {
                continue;
            }

            // Calculate blend weight based on distance from edge
            let weight = calculate_blend_weight(tx, ty, width, height, overlap);

            let idx = (out_y * out_width + out_x) as usize;
            let pixel = tile.get_pixel(tx.min(tile.width() - 1), ty.min(tile.height() - 1));

            // Accumulate weighted pixel values
            let existing = output.get_pixel(out_x, out_y);
            let existing_weight = weight_map[idx];
            let total_weight = existing_weight + weight;

            if total_weight > 0.0 {
                let new_r = (existing[0] as f32 * existing_weight + pixel[0] as f32 * weight)
                    / total_weight;
                let new_g = (existing[1] as f32 * existing_weight + pixel[1] as f32 * weight)
                    / total_weight;
                let new_b = (existing[2] as f32 * existing_weight + pixel[2] as f32 * weight)
                    / total_weight;

                output.put_pixel(
                    out_x,
                    out_y,
                    Rgb([new_r as u8, new_g as u8, new_b as u8]),
                );
                weight_map[idx] = total_weight;
            }
        }
    }
}

/// Calculates blend weight based on distance from tile edges.
fn calculate_blend_weight(x: u32, y: u32, width: u32, height: u32, overlap: u32) -> f32 {
    if overlap == 0 {
        return 1.0;
    }

    let overlap_f = overlap as f32;

    // Distance from each edge
    let left_dist = x as f32;
    let right_dist = (width - 1 - x) as f32;
    let top_dist = y as f32;
    let bottom_dist = (height - 1 - y) as f32;

    // Weight ramps up from 0 at edge to 1 at overlap distance
    let left_weight = (left_dist / overlap_f).min(1.0);
    let right_weight = (right_dist / overlap_f).min(1.0);
    let top_weight = (top_dist / overlap_f).min(1.0);
    let bottom_weight = (bottom_dist / overlap_f).min(1.0);

    // Combine weights (minimum gives smooth falloff in corners)
    left_weight.min(right_weight).min(top_weight).min(bottom_weight)
}

/// Normalizes output pixels by accumulated weights.
fn normalize_by_weights(output: &mut RgbImage, weight_map: &[f32]) {
    let width = output.width();
    for (idx, weight) in weight_map.iter().enumerate() {
        if *weight > 0.0 && *weight != 1.0 {
            let x = (idx as u32) % width;
            let y = (idx as u32) / width;
            // Already normalized during blending, this is just safety
            let _pixel = output.get_pixel(x, y);
        }
    }
}

/// Converts a DynamicImage to an ONNX input tensor.
///
/// Real-ESRGAN expects input in range [0, 1] with shape [1, 3, H, W].
pub fn image_to_tensor(image: &DynamicImage) -> Result<Array4<f32>, TransformError> {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    // Create tensor in NCHW format: [1, 3, H, W]
    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    // Fill tensor with normalized pixel values [0, 1]
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                tensor[[0, c, y, x]] = pixel[c] as f32 / 255.0;
            }
        }
    }

    Ok(tensor)
}

/// Converts an ONNX output tensor to a DynamicImage.
///
/// Expects tensor in range [0, 1] with shape [1, 3, H, W].
pub fn tensor_to_image(tensor: &Array4<f32>) -> Result<DynamicImage, TransformError> {
    let rgb = tensor_to_rgb(tensor)?;
    Ok(DynamicImage::ImageRgb8(rgb))
}

/// Converts an ONNX output tensor to an RgbImage.
fn tensor_to_rgb(tensor: &Array4<f32>) -> Result<RgbImage, TransformError> {
    let shape = tensor.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
        return Err(TransformError::InferenceFailed {
            message: format!("unexpected tensor shape: {:?}", shape),
        });
    }

    let height = shape[2];
    let width = shape[3];
    let mut image = RgbImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let r = (tensor[[0, 0, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (tensor[[0, 1, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (tensor[[0, 2, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            image.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    Ok(image)
}

/// Converts an ONNX output tensor to an RgbaImage with full alpha.
#[allow(dead_code)]
fn tensor_to_rgba(tensor: &Array4<f32>) -> Result<RgbaImage, TransformError> {
    let shape = tensor.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
        return Err(TransformError::InferenceFailed {
            message: format!("unexpected tensor shape: {:?}", shape),
        });
    }

    let height = shape[2];
    let width = shape[3];
    let mut image = RgbaImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let r = (tensor[[0, 0, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (tensor[[0, 1, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (tensor[[0, 2, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            image.put_pixel(x as u32, y as u32, Rgba([r, g, b, 255]));
        }
    }

    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(width: u32, height: u32) -> RgbImage {
        let mut img = RgbImage::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Rgb([
                ((x * 255) / width.max(1)) as u8,
                ((y * 255) / height.max(1)) as u8,
                128,
            ]);
        }
        img
    }

    #[test]
    fn test_image_to_tensor_shape() {
        let img = DynamicImage::ImageRgb8(create_test_image(64, 48));
        let tensor = image_to_tensor(&img).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 48, 64]);
    }

    #[test]
    fn test_image_to_tensor_range() {
        let img = DynamicImage::ImageRgb8(create_test_image(32, 32));
        let tensor = image_to_tensor(&img).unwrap();

        let min_val = tensor.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = tensor.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        assert!(min_val >= 0.0, "min value {} < 0", min_val);
        assert!(max_val <= 1.0, "max value {} > 1", max_val);
    }

    #[test]
    fn test_tensor_roundtrip() {
        let original = DynamicImage::ImageRgb8(create_test_image(16, 16));
        let tensor = image_to_tensor(&original).unwrap();
        let recovered = tensor_to_image(&tensor).unwrap();

        // Compare images (allow small differences due to float conversion)
        let orig_rgb = original.to_rgb8();
        let rec_rgb = recovered.to_rgb8();

        for y in 0..16 {
            for x in 0..16 {
                let op = orig_rgb.get_pixel(x, y);
                let rp = rec_rgb.get_pixel(x, y);
                for c in 0..3 {
                    let diff = (op[c] as i32 - rp[c] as i32).abs();
                    assert!(diff <= 1, "pixel ({}, {}) channel {} differs by {}", x, y, c, diff);
                }
            }
        }
    }

    #[test]
    fn test_extract_tile() {
        let img = create_test_image(100, 80);
        let tile = extract_tile(&img, 10, 20, 32, 32);
        assert_eq!(tile.width(), 32);
        assert_eq!(tile.height(), 32);
    }

    #[test]
    fn test_blend_weight() {
        // Center pixel should have weight 1.0
        let center_weight = calculate_blend_weight(50, 50, 100, 100, 10);
        assert!((center_weight - 1.0).abs() < 0.01);

        // Edge pixel should have weight 0.0
        let edge_weight = calculate_blend_weight(0, 50, 100, 100, 10);
        assert!(edge_weight < 0.01);

        // Mid-overlap pixel should have weight ~0.5
        let mid_weight = calculate_blend_weight(5, 50, 100, 100, 10);
        assert!((mid_weight - 0.5).abs() < 0.01);
    }
}
