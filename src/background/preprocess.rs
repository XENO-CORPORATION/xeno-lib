//! Image preprocessing for ONNX inference.
//!
//! Converts `DynamicImage` to normalized tensors in the format expected by
//! the BiRefNet model.

use image::{imageops::FilterType, DynamicImage};
use ndarray::Array4;

use crate::error::TransformError;

/// BiRefNet normalization parameters (ImageNet standard).
/// The model expects input normalized with ImageNet mean/std.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Converts a `DynamicImage` to a normalized ONNX input tensor.
///
/// # Processing Steps
///
/// 1. Resize to target dimensions (bilinear interpolation)
/// 2. Convert to RGB f32 in range [0, 1]
/// 3. Normalize using BiRefNet/ImageNet mean/std values
/// 4. Transpose from HWC to CHW format
/// 5. Add batch dimension
///
/// # Arguments
///
/// * `image` - The input image to preprocess
/// * `target_size` - The target dimensions (width, height) for the model
///
/// # Returns
///
/// A 4D tensor of shape `[1, 3, H, W]` ready for inference.
pub fn image_to_tensor(
    image: &DynamicImage,
    target_size: (u32, u32),
) -> Result<Array4<f32>, TransformError> {
    let (target_width, target_height) = target_size;

    // Resize image to model input size
    let resized = image.resize_exact(target_width, target_height, FilterType::Lanczos3);

    // Convert to RGB8
    let rgb = resized.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    // Create tensor in CHW format with batch dimension: [1, 3, H, W]
    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    // Fill tensor with normalized pixel values
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);

            // Normalize: (pixel / 255.0 - mean) / std
            for c in 0..3 {
                let value = pixel[c] as f32 / 255.0;
                let normalized = (value - MEAN[c]) / STD[c];
                tensor[[0, c, y, x]] = normalized;
            }
        }
    }

    Ok(tensor)
}

/// Converts a `DynamicImage` to a tensor with parallel processing.
///
/// This is an optimized version that uses rayon for parallel pixel processing,
/// beneficial for large images.
#[allow(dead_code)]
pub fn image_to_tensor_parallel(
    image: &DynamicImage,
    target_size: (u32, u32),
) -> Result<Array4<f32>, TransformError> {
    use rayon::prelude::*;

    let (target_width, target_height) = target_size;

    // Resize image to model input size
    let resized = image.resize_exact(target_width, target_height, FilterType::Lanczos3);
    let rgb = resized.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    // Get raw pixel data
    let raw_pixels = rgb.as_raw();

    // Create flattened arrays for each channel
    let pixels_per_channel = height * width;
    let mut r_channel = vec![0.0f32; pixels_per_channel];
    let mut g_channel = vec![0.0f32; pixels_per_channel];
    let mut b_channel = vec![0.0f32; pixels_per_channel];

    // Parallel processing of pixels
    r_channel
        .par_iter_mut()
        .zip(g_channel.par_iter_mut())
        .zip(b_channel.par_iter_mut())
        .enumerate()
        .for_each(|(idx, ((r, g), b))| {
            let pixel_offset = idx * 3;
            *r = (raw_pixels[pixel_offset] as f32 / 255.0 - MEAN[0]) / STD[0];
            *g = (raw_pixels[pixel_offset + 1] as f32 / 255.0 - MEAN[1]) / STD[1];
            *b = (raw_pixels[pixel_offset + 2] as f32 / 255.0 - MEAN[2]) / STD[2];
        });

    // Combine channels into tensor
    let mut data = Vec::with_capacity(3 * pixels_per_channel);
    data.extend(r_channel);
    data.extend(g_channel);
    data.extend(b_channel);

    // Reshape to [1, 3, H, W]
    Array4::from_shape_vec((1, 3, height, width), data).map_err(|_| {
        TransformError::AllocationFailed {
            width: width as u32,
            height: height as u32,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn create_test_rgb_image(width: u32, height: u32) -> DynamicImage {
        let mut img = RgbImage::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Rgb([
                ((x * 255) / width) as u8,
                ((y * 255) / height) as u8,
                128,
            ]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_tensor_shape() {
        let img = create_test_rgb_image(100, 80);
        let tensor = image_to_tensor(&img, (1024, 1024)).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 1024, 1024]);
    }

    #[test]
    fn test_tensor_range() {
        let img = create_test_rgb_image(64, 64);
        let tensor = image_to_tensor(&img, (64, 64)).unwrap();

        // After normalization to [0, 1] range
        let min_val = tensor.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = tensor.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        assert!(min_val >= -0.01, "min value {} is out of range", min_val);
        assert!(max_val <= 1.01, "max value {} is out of range", max_val);
    }

    #[test]
    fn test_parallel_produces_same_result() {
        let img = create_test_rgb_image(128, 128);
        let tensor1 = image_to_tensor(&img, (128, 128)).unwrap();
        let tensor2 = image_to_tensor_parallel(&img, (128, 128)).unwrap();

        // Compare tensors (allow small floating point differences)
        for (a, b) in tensor1.iter().zip(tensor2.iter()) {
            assert!((a - b).abs() < 1e-5, "tensors differ: {} vs {}", a, b);
        }
    }
}
