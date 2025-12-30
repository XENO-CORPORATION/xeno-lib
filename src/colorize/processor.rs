//! Colorization processing logic.

use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};
use ndarray::Array4;

use crate::error::TransformError;
use super::model::ColorizerSession;

/// Colorizes a grayscale image.
pub fn colorize_impl(
    image: &DynamicImage,
    session: &mut ColorizerSession,
) -> Result<DynamicImage, TransformError> {
    let (input_w, input_h) = session.input_size();
    let original_width = image.width();
    let original_height = image.height();

    // Convert to grayscale luminance
    let gray = image.to_luma8();

    // Resize to model input size
    let resized_gray = image::imageops::resize(&gray, input_w, input_h, FilterType::Lanczos3);

    // Convert grayscale to tensor (L channel in LAB-like format)
    let input_tensor = grayscale_to_tensor(&resized_gray)?;

    // Run colorization (model outputs AB channels)
    let output_tensor = session.run(&input_tensor)?;

    // Combine L with predicted AB and convert to RGB
    let colorized = tensor_to_rgb(&output_tensor, &resized_gray, session.config().saturation)?;

    // Resize back to original dimensions
    let final_image = colorized.resize_exact(original_width, original_height, FilterType::Lanczos3);

    Ok(final_image)
}

/// Converts grayscale image to tensor for colorization model.
fn grayscale_to_tensor(gray: &image::GrayImage) -> Result<Array4<f32>, TransformError> {
    let (width, height) = (gray.width() as usize, gray.height() as usize);

    // DDColor expects L channel normalized to [0, 1]
    let mut tensor = Array4::<f32>::zeros((1, 1, height, width));

    for y in 0..height {
        for x in 0..width {
            let pixel = gray.get_pixel(x as u32, y as u32);
            tensor[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
        }
    }

    Ok(tensor)
}

/// Converts model output (AB channels) combined with L to RGB image.
fn tensor_to_rgb(
    ab_tensor: &Array4<f32>,
    gray: &image::GrayImage,
    saturation: f32,
) -> Result<DynamicImage, TransformError> {
    let shape = ab_tensor.shape();

    // Handle different output shapes
    let (height, width) = if shape.len() == 4 {
        (shape[2], shape[3])
    } else {
        return Err(TransformError::InferenceFailed {
            message: format!("unexpected tensor shape: {:?}", shape),
        });
    };

    let mut image = RgbImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            // Get original luminance
            let l = gray.get_pixel(
                (x as u32).min(gray.width() - 1),
                (y as u32).min(gray.height() - 1),
            )[0] as f32 / 255.0;

            // Get predicted color channels
            // Models typically output RGB directly or AB channels
            let (r, g, b) = if shape[1] >= 3 {
                // RGB output
                let r = ab_tensor[[0, 0, y, x]];
                let g = ab_tensor[[0, 1, y, x]];
                let b = ab_tensor[[0, 2, y, x]];
                (r, g, b)
            } else if shape[1] == 2 {
                // AB output - convert LAB to RGB
                let a = ab_tensor[[0, 0, y, x]] * saturation;
                let b_ch = ab_tensor[[0, 1, y, x]] * saturation;
                lab_to_rgb(l, a, b_ch)
            } else {
                // Single channel - treat as grayscale
                (l, l, l)
            };

            // Clamp and convert to u8
            let r_u8 = (r * 255.0).clamp(0.0, 255.0) as u8;
            let g_u8 = (g * 255.0).clamp(0.0, 255.0) as u8;
            let b_u8 = (b * 255.0).clamp(0.0, 255.0) as u8;

            image.put_pixel(x as u32, y as u32, Rgb([r_u8, g_u8, b_u8]));
        }
    }

    Ok(DynamicImage::ImageRgb8(image))
}

/// Converts LAB color to RGB.
fn lab_to_rgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    // Convert LAB to XYZ
    let l_scaled = l * 100.0;
    let a_scaled = a * 128.0;
    let b_scaled = b * 128.0;

    let y = (l_scaled + 16.0) / 116.0;
    let x = a_scaled / 500.0 + y;
    let z = y - b_scaled / 200.0;

    let x = if x.powi(3) > 0.008856 {
        x.powi(3)
    } else {
        (x - 16.0 / 116.0) / 7.787
    };
    let y = if y.powi(3) > 0.008856 {
        y.powi(3)
    } else {
        (y - 16.0 / 116.0) / 7.787
    };
    let z = if z.powi(3) > 0.008856 {
        z.powi(3)
    } else {
        (z - 16.0 / 116.0) / 7.787
    };

    // Reference white D65
    let x = x * 0.95047;
    let z = z * 1.08883;

    // XYZ to RGB (sRGB)
    let r = x * 3.2406 - y * 1.5372 - z * 0.4986;
    let g = -x * 0.9689 + y * 1.8758 + z * 0.0415;
    let b = x * 0.0557 - y * 0.2040 + z * 1.0570;

    // Gamma correction
    let r = if r > 0.0031308 {
        1.055 * r.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * r
    };
    let g = if g > 0.0031308 {
        1.055 * g.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * g
    };
    let b = if b > 0.0031308 {
        1.055 * b.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * b
    };

    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grayscale_tensor_shape() {
        let gray = image::GrayImage::new(64, 64);
        let tensor = grayscale_to_tensor(&gray).unwrap();
        assert_eq!(tensor.shape(), &[1, 1, 64, 64]);
    }

    #[test]
    fn test_lab_to_rgb_white() {
        let (r, g, b) = lab_to_rgb(1.0, 0.0, 0.0);
        assert!(r > 0.9 && g > 0.9 && b > 0.9);
    }

    #[test]
    fn test_lab_to_rgb_black() {
        let (r, g, b) = lab_to_rgb(0.0, 0.0, 0.0);
        assert!(r < 0.1 && g < 0.1 && b < 0.1);
    }
}
