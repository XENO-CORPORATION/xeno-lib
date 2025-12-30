//! Style transfer processing logic.

use image::{DynamicImage, imageops::FilterType, Rgba, RgbaImage};
use ndarray::Array4;

use crate::error::TransformError;
use super::model::StyleSession;

/// Apply neural style transfer to an image.
pub fn stylize(
    image: &DynamicImage,
    session: &mut StyleSession,
) -> Result<DynamicImage, TransformError> {
    let config = session.config();
    let max_dim = config.max_dimension;
    let (orig_w, orig_h) = (image.width(), image.height());

    // Resize if needed for memory/performance
    let (work_w, work_h) = if orig_w > max_dim || orig_h > max_dim {
        let scale = max_dim as f32 / orig_w.max(orig_h) as f32;
        ((orig_w as f32 * scale) as u32, (orig_h as f32 * scale) as u32)
    } else {
        (orig_w, orig_h)
    };

    let resized = if work_w != orig_w || work_h != orig_h {
        image.resize_exact(work_w, work_h, FilterType::Lanczos3)
    } else {
        image.clone()
    };

    // Optionally extract LAB L channel for color preservation
    let original_lab = if config.preserve_colors {
        Some(extract_lab_channels(&resized))
    } else {
        None
    };

    // Convert to tensor (NCHW format, normalized to [0, 255])
    let input_tensor = image_to_tensor(&resized)?;

    // Run inference
    let output_tensor = session.run(&input_tensor)?;

    // Convert back to image
    let mut styled = tensor_to_image(&output_tensor, work_w, work_h)?;

    // Restore original colors if requested
    if let Some((l_original, a_original, b_original)) = original_lab {
        styled = restore_colors(&styled, &l_original, &a_original, &b_original);
    }

    // Apply strength blending
    let strength = config.strength;
    if strength < 1.0 {
        styled = blend_images(&resized, &styled, strength);
    }

    // Resize back to original dimensions if needed
    let final_image = if work_w != orig_w || work_h != orig_h {
        DynamicImage::ImageRgba8(styled).resize_exact(orig_w, orig_h, FilterType::Lanczos3)
    } else {
        DynamicImage::ImageRgba8(styled)
    };

    Ok(final_image)
}

/// Apply style transfer with custom blending.
pub fn stylize_blended(
    image: &DynamicImage,
    session: &mut StyleSession,
    strength: f32,
) -> Result<DynamicImage, TransformError> {
    let styled = stylize(image, session)?;
    let strength = strength.clamp(0.0, 1.0);

    if strength >= 1.0 {
        return Ok(styled);
    }

    let original = image.to_rgba8();
    let styled_rgba = styled.to_rgba8();
    let blended = blend_images(
        &DynamicImage::ImageRgba8(original),
        &DynamicImage::ImageRgba8(styled_rgba),
        strength,
    );

    Ok(DynamicImage::ImageRgba8(blended))
}

/// Convert image to model input tensor.
fn image_to_tensor(image: &DynamicImage) -> Result<Array4<f32>, TransformError> {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            // Style transfer models typically expect [0, 255] range
            tensor[[0, 0, y, x]] = pixel[0] as f32;
            tensor[[0, 1, y, x]] = pixel[1] as f32;
            tensor[[0, 2, y, x]] = pixel[2] as f32;
        }
    }

    Ok(tensor)
}

/// Convert model output tensor to image.
fn tensor_to_image(tensor: &Array4<f32>, width: u32, height: u32) -> Result<RgbaImage, TransformError> {
    let mut img = RgbaImage::new(width, height);
    let shape = tensor.shape();

    if shape.len() != 4 {
        return Err(TransformError::InferenceFailed {
            message: "Invalid tensor shape".to_string(),
        });
    }

    let h = shape[2].min(height as usize);
    let w = shape[3].min(width as usize);

    for y in 0..h {
        for x in 0..w {
            // Clamp to [0, 255]
            let r = tensor[[0, 0, y, x]].clamp(0.0, 255.0) as u8;
            let g = tensor[[0, 1, y, x]].clamp(0.0, 255.0) as u8;
            let b = tensor[[0, 2, y, x]].clamp(0.0, 255.0) as u8;
            img.put_pixel(x as u32, y as u32, Rgba([r, g, b, 255]));
        }
    }

    Ok(img)
}

/// Extract LAB channels from image.
fn extract_lab_channels(image: &DynamicImage) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut l = vec![0.0f32; w * h];
    let mut a = vec![0.0f32; w * h];
    let mut b = vec![0.0f32; w * h];

    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            let (l_val, a_val, b_val) = rgb_to_lab(pixel[0], pixel[1], pixel[2]);
            let idx = y * w + x;
            l[idx] = l_val;
            a[idx] = a_val;
            b[idx] = b_val;
        }
    }

    (l, a, b)
}

/// Restore original colors using LAB color space.
fn restore_colors(
    styled: &RgbaImage,
    _l_original: &[f32],
    a_original: &[f32],
    b_original: &[f32],
) -> RgbaImage {
    let (w, h) = (styled.width() as usize, styled.height() as usize);
    let mut result = RgbaImage::new(w as u32, h as u32);

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let pixel = styled.get_pixel(x as u32, y as u32);

            // Get L from styled, AB from original
            let (l_styled, _, _) = rgb_to_lab(pixel[0], pixel[1], pixel[2]);
            let (r, g, b) = lab_to_rgb(l_styled, a_original[idx], b_original[idx]);

            result.put_pixel(x as u32, y as u32, Rgba([r, g, b, pixel[3]]));
        }
    }

    result
}

/// Blend two images together.
fn blend_images(original: &DynamicImage, styled: &DynamicImage, alpha: f32) -> RgbaImage {
    let orig = original.to_rgba8();
    let styl = styled.to_rgba8();
    let (w, h) = (orig.width(), orig.height());
    let mut result = RgbaImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let p1 = orig.get_pixel(x, y);
            let p2 = styl.get_pixel(x, y);

            let r = ((p1[0] as f32 * (1.0 - alpha)) + (p2[0] as f32 * alpha)) as u8;
            let g = ((p1[1] as f32 * (1.0 - alpha)) + (p2[1] as f32 * alpha)) as u8;
            let b = ((p1[2] as f32 * (1.0 - alpha)) + (p2[2] as f32 * alpha)) as u8;
            let a = p1[3].max(p2[3]);

            result.put_pixel(x, y, Rgba([r, g, b, a]));
        }
    }

    result
}

/// Convert RGB to LAB color space.
fn rgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    // RGB to XYZ
    let r = pivot_rgb(r as f32 / 255.0);
    let g = pivot_rgb(g as f32 / 255.0);
    let b = pivot_rgb(b as f32 / 255.0);

    let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

    // XYZ to LAB (D65 illuminant)
    let x = pivot_xyz(x / 0.95047);
    let y = pivot_xyz(y / 1.00000);
    let z = pivot_xyz(z / 1.08883);

    let l = 116.0 * y - 16.0;
    let a = 500.0 * (x - y);
    let b = 200.0 * (y - z);

    (l, a, b)
}

/// Convert LAB to RGB color space.
fn lab_to_rgb(l: f32, a: f32, b: f32) -> (u8, u8, u8) {
    // LAB to XYZ
    let y = (l + 16.0) / 116.0;
    let x = a / 500.0 + y;
    let z = y - b / 200.0;

    let x = unpivot_xyz(x) * 0.95047;
    let y = unpivot_xyz(y) * 1.00000;
    let z = unpivot_xyz(z) * 1.08883;

    // XYZ to RGB
    let r = x * 3.2404542 - y * 1.5371385 - z * 0.4985314;
    let g = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b = x * 0.0556434 - y * 0.2040259 + z * 1.0572252;

    let r = unpivot_rgb(r);
    let g = unpivot_rgb(g);
    let b = unpivot_rgb(b);

    (
        (r * 255.0).clamp(0.0, 255.0) as u8,
        (g * 255.0).clamp(0.0, 255.0) as u8,
        (b * 255.0).clamp(0.0, 255.0) as u8,
    )
}

fn pivot_rgb(v: f32) -> f32 {
    if v > 0.04045 {
        ((v + 0.055) / 1.055).powf(2.4)
    } else {
        v / 12.92
    }
}

fn unpivot_rgb(v: f32) -> f32 {
    if v > 0.0031308 {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * v
    }
}

fn pivot_xyz(v: f32) -> f32 {
    if v > 0.008856 {
        v.powf(1.0 / 3.0)
    } else {
        (7.787 * v) + (16.0 / 116.0)
    }
}

fn unpivot_xyz(v: f32) -> f32 {
    let v3 = v * v * v;
    if v3 > 0.008856 {
        v3
    } else {
        (v - 16.0 / 116.0) / 7.787
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_lab_roundtrip() {
        let (r, g, b) = (128, 64, 192);
        let (l, a, lab_b) = rgb_to_lab(r, g, b);
        let (r2, g2, b2) = lab_to_rgb(l, a, lab_b);

        // Allow small rounding errors
        assert!((r as i32 - r2 as i32).abs() <= 1);
        assert!((g as i32 - g2 as i32).abs() <= 1);
        assert!((b as i32 - b2 as i32).abs() <= 1);
    }

    #[test]
    fn test_blend_images() {
        let img1 = DynamicImage::new_rgba8(10, 10);
        let img2 = DynamicImage::new_rgba8(10, 10);
        let blended = blend_images(&img1, &img2, 0.5);
        assert_eq!(blended.width(), 10);
        assert_eq!(blended.height(), 10);
    }

    #[test]
    fn test_image_to_tensor() {
        let img = DynamicImage::new_rgb8(64, 64);
        let tensor = image_to_tensor(&img).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 64, 64]);
    }
}
