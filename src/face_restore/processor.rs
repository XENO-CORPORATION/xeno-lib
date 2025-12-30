//! Face restoration processing logic.

use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};
use ndarray::Array4;

use crate::error::TransformError;
use super::model::FaceRestorerSession;

/// Restores faces in an image.
pub fn restore_faces_impl(
    image: &DynamicImage,
    session: &mut FaceRestorerSession,
) -> Result<DynamicImage, TransformError> {
    let (input_w, input_h) = session.input_size();
    let original_width = image.width();
    let original_height = image.height();

    // For now, process the entire image as a face
    // In a full implementation, we'd detect faces first and process each one

    // Resize to model input size
    let resized = image.resize_exact(input_w, input_h, FilterType::Lanczos3);

    // Convert to tensor (normalized to [-1, 1] for GFPGAN)
    let input_tensor = image_to_tensor(&resized)?;

    // Run restoration
    let output_tensor = session.run(&input_tensor)?;

    // Convert back to image
    let restored = tensor_to_image(&output_tensor)?;

    // Resize back to original dimensions
    let final_image = restored.resize_exact(original_width, original_height, FilterType::Lanczos3);

    Ok(final_image)
}

/// Converts image to tensor normalized to [-1, 1].
fn image_to_tensor(image: &DynamicImage) -> Result<Array4<f32>, TransformError> {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                // Normalize to [-1, 1] (GFPGAN expects this range)
                tensor[[0, c, y, x]] = (pixel[c] as f32 / 127.5) - 1.0;
            }
        }
    }

    Ok(tensor)
}

/// Converts tensor from [-1, 1] range to image.
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
            // Convert from [-1, 1] to [0, 255]
            let r = ((tensor[[0, 0, y, x]] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            let g = ((tensor[[0, 1, y, x]] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            let b = ((tensor[[0, 2, y, x]] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            image.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    Ok(DynamicImage::ImageRgb8(image))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(width: u32, height: u32) -> DynamicImage {
        let mut img = RgbImage::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Rgb([
                ((x * 255) / width.max(1)) as u8,
                ((y * 255) / height.max(1)) as u8,
                128,
            ]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_tensor_roundtrip() {
        let img = create_test_image(64, 64);
        let tensor = image_to_tensor(&img).unwrap();
        let recovered = tensor_to_image(&tensor).unwrap();

        // Check dimensions
        assert_eq!(recovered.width(), 64);
        assert_eq!(recovered.height(), 64);
    }

    #[test]
    fn test_tensor_range() {
        let img = create_test_image(32, 32);
        let tensor = image_to_tensor(&img).unwrap();

        let min_val = tensor.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = tensor.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Should be in [-1, 1] range
        assert!(min_val >= -1.01, "min {} < -1", min_val);
        assert!(max_val <= 1.01, "max {} > 1", max_val);
    }
}
