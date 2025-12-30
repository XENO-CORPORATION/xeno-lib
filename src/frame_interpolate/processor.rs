//! Frame interpolation processing logic.

use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};
use ndarray::Array4;

use crate::error::TransformError;
use super::model::InterpolatorSession;

/// Interpolates a single frame between two input frames.
pub fn interpolate_frame_impl(
    frame0: &DynamicImage,
    frame1: &DynamicImage,
    timestep: f32,
    session: &mut InterpolatorSession,
) -> Result<DynamicImage, TransformError> {
    // Ensure frames are same size
    if frame0.width() != frame1.width() || frame0.height() != frame1.height() {
        return Err(TransformError::InferenceFailed {
            message: "frames must have the same dimensions".to_string(),
        });
    }

    let width = frame0.width();
    let height = frame0.height();

    // Pad to multiple of 32 for model compatibility
    let pad_w = ((width + 31) / 32) * 32;
    let pad_h = ((height + 31) / 32) * 32;

    let frame0_padded = if pad_w != width || pad_h != height {
        frame0.resize_exact(pad_w, pad_h, FilterType::Lanczos3)
    } else {
        frame0.clone()
    };

    let frame1_padded = if pad_w != width || pad_h != height {
        frame1.resize_exact(pad_w, pad_h, FilterType::Lanczos3)
    } else {
        frame1.clone()
    };

    // Convert to tensors
    let tensor0 = image_to_tensor(&frame0_padded)?;
    let tensor1 = image_to_tensor(&frame1_padded)?;

    // Run interpolation
    let output_tensor = session.run(&tensor0, &tensor1, timestep)?;

    // Convert back to image
    let mut result = tensor_to_image(&output_tensor)?;

    // Crop back to original size if padded
    if pad_w != width || pad_h != height {
        result = result.crop_imm(0, 0, width, height);
    }

    Ok(result)
}

/// Interpolates multiple frames between two input frames.
pub fn interpolate_frames_impl(
    frame0: &DynamicImage,
    frame1: &DynamicImage,
    num_frames: u32,
    session: &mut InterpolatorSession,
) -> Result<Vec<DynamicImage>, TransformError> {
    let mut results = Vec::with_capacity(num_frames as usize);

    for i in 1..=num_frames {
        let timestep = i as f32 / (num_frames + 1) as f32;
        let interpolated = interpolate_frame_impl(frame0, frame1, timestep, session)?;
        results.push(interpolated);
    }

    Ok(results)
}

/// Detects if there's a scene change between two frames.
pub fn detect_scene_change(
    frame0: &DynamicImage,
    frame1: &DynamicImage,
    threshold: f32,
) -> bool {
    let rgb0 = frame0.to_rgb8();
    let rgb1 = frame1.to_rgb8();

    // Downsample for faster comparison
    let scale = 8;
    let w = (rgb0.width() / scale).max(1);
    let h = (rgb0.height() / scale).max(1);

    let small0 = image::imageops::resize(&rgb0, w, h, FilterType::Nearest);
    let small1 = image::imageops::resize(&rgb1, w, h, FilterType::Nearest);

    // Calculate mean absolute difference
    let mut total_diff = 0u64;
    let pixel_count = (w * h) as u64;

    for (p0, p1) in small0.pixels().zip(small1.pixels()) {
        for c in 0..3 {
            total_diff += (p0[c] as i32 - p1[c] as i32).unsigned_abs() as u64;
        }
    }

    let mean_diff = total_diff as f32 / (pixel_count * 3) as f32 / 255.0;
    mean_diff > threshold
}

/// Converts image to tensor normalized to [0, 1].
fn image_to_tensor(image: &DynamicImage) -> Result<Array4<f32>, TransformError> {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

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

/// Converts tensor from [0, 1] range to image.
fn tensor_to_image(tensor: &Array4<f32>) -> Result<DynamicImage, TransformError> {
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

    Ok(DynamicImage::ImageRgb8(image))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32, offset: u8) -> DynamicImage {
        let mut img = RgbImage::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Rgb([
                (((x * 255) / width.max(1)) as u8).wrapping_add(offset),
                (((y * 255) / height.max(1)) as u8).wrapping_add(offset),
                128u8.wrapping_add(offset),
            ]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_tensor_roundtrip() {
        let img = create_test_frame(64, 64, 0);
        let tensor = image_to_tensor(&img).unwrap();
        let recovered = tensor_to_image(&tensor).unwrap();

        assert_eq!(recovered.width(), 64);
        assert_eq!(recovered.height(), 64);
    }

    #[test]
    fn test_scene_detection_similar_frames() {
        let frame0 = create_test_frame(64, 64, 0);
        let frame1 = create_test_frame(64, 64, 5); // Small offset

        let is_scene_change = detect_scene_change(&frame0, &frame1, 0.3);
        assert!(!is_scene_change, "similar frames should not trigger scene change");
    }

    #[test]
    fn test_scene_detection_different_frames() {
        let frame0 = create_test_frame(64, 64, 0);

        // Create very different frame
        let mut img = RgbImage::new(64, 64);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([255, 0, 0]); // All red
        }
        let frame1 = DynamicImage::ImageRgb8(img);

        let is_scene_change = detect_scene_change(&frame0, &frame1, 0.3);
        assert!(is_scene_change, "very different frames should trigger scene change");
    }

    #[test]
    fn test_tensor_range() {
        let img = create_test_frame(32, 32, 0);
        let tensor = image_to_tensor(&img).unwrap();

        let min_val = tensor.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = tensor.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        assert!(min_val >= 0.0, "min {} < 0", min_val);
        assert!(max_val <= 1.0, "max {} > 1", max_val);
    }
}
