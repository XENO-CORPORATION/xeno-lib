//! Inpainting processing logic.

use image::{DynamicImage, Rgb, RgbImage, GrayImage, Luma, imageops::FilterType};
use ndarray::Array4;

use crate::error::TransformError;
use super::model::InpainterSession;

/// Inpaints an image using a binary mask.
pub fn inpaint_impl(
    image: &DynamicImage,
    mask: &DynamicImage,
    session: &mut InpainterSession,
) -> Result<DynamicImage, TransformError> {
    let (input_w, input_h) = session.input_size();
    let original_width = image.width();
    let original_height = image.height();

    // Resize image to model input size
    let resized_img = image.resize_exact(input_w, input_h, FilterType::Lanczos3);
    let resized_mask = mask.resize_exact(input_w, input_h, FilterType::Nearest);

    // Optionally dilate mask
    let processed_mask = if session.config().dilate_mask {
        dilate_mask(&resized_mask.to_luma8(), session.config().dilation_radius)
    } else {
        resized_mask.to_luma8()
    };

    // Convert to tensors
    let image_tensor = image_to_tensor(&resized_img)?;
    let mask_tensor = mask_to_tensor(&processed_mask)?;

    // Run inpainting
    let output_tensor = session.run(&image_tensor, &mask_tensor)?;

    // Convert back to image
    let inpainted = tensor_to_image(&output_tensor)?;

    // Resize back to original dimensions
    let final_image = inpainted.resize_exact(original_width, original_height, FilterType::Lanczos3);

    Ok(final_image)
}

/// Dilates a binary mask by the specified radius.
fn dilate_mask(mask: &GrayImage, radius: u32) -> GrayImage {
    let width = mask.width();
    let height = mask.height();
    let mut dilated = GrayImage::new(width, height);
    let r = radius as i32;

    for y in 0..height {
        for x in 0..width {
            let mut max_val = 0u8;

            for dy in -r..=r {
                for dx in -r..=r {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;

                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let val = mask.get_pixel(nx as u32, ny as u32)[0];
                        max_val = max_val.max(val);
                    }
                }
            }

            dilated.put_pixel(x, y, Luma([max_val]));
        }
    }

    dilated
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

/// Converts grayscale mask to tensor (1 channel).
fn mask_to_tensor(mask: &GrayImage) -> Result<Array4<f32>, TransformError> {
    let (width, height) = (mask.width() as usize, mask.height() as usize);

    let mut tensor = Array4::<f32>::zeros((1, 1, height, width));

    for y in 0..height {
        for x in 0..width {
            let pixel = mask.get_pixel(x as u32, y as u32);
            // Binary mask: >127 = 1.0, else = 0.0
            tensor[[0, 0, y, x]] = if pixel[0] > 127 { 1.0 } else { 0.0 };
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

    fn create_test_mask(width: u32, height: u32) -> GrayImage {
        let mut mask = GrayImage::new(width, height);
        // Create a circular mask in the center
        let cx = width / 2;
        let cy = height / 2;
        let radius = width.min(height) / 4;

        for y in 0..height {
            for x in 0..width {
                let dx = x as i32 - cx as i32;
                let dy = y as i32 - cy as i32;
                let dist = ((dx * dx + dy * dy) as f32).sqrt();

                if dist < radius as f32 {
                    mask.put_pixel(x, y, Luma([255]));
                } else {
                    mask.put_pixel(x, y, Luma([0]));
                }
            }
        }
        mask
    }

    #[test]
    fn test_image_tensor_roundtrip() {
        let img = create_test_image(64, 64);
        let tensor = image_to_tensor(&img).unwrap();
        let recovered = tensor_to_image(&tensor).unwrap();

        assert_eq!(recovered.width(), 64);
        assert_eq!(recovered.height(), 64);
    }

    #[test]
    fn test_mask_tensor_binary() {
        let mask = create_test_mask(64, 64);
        let tensor = mask_to_tensor(&mask).unwrap();

        // Check all values are 0 or 1
        for &val in tensor.iter() {
            assert!(val == 0.0 || val == 1.0, "mask values should be binary");
        }
    }

    #[test]
    fn test_mask_dilation() {
        let mask = create_test_mask(64, 64);
        let dilated = dilate_mask(&mask, 3);

        // Dilated mask should have more white pixels
        let original_white: u32 = mask.pixels().map(|p| if p[0] > 127 { 1 } else { 0 }).sum();
        let dilated_white: u32 = dilated.pixels().map(|p| if p[0] > 127 { 1 } else { 0 }).sum();

        assert!(dilated_white >= original_white, "dilation should not reduce mask area");
    }
}
