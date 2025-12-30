use crate::error::TransformError;
use image::{DynamicImage, RgbaImage};

/// Numerical comparison metrics between two images.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonMetrics {
    pub width: u32,
    pub height: u32,
    pub mean_squared_error: f32,
    pub peak_signal_to_noise_ratio: f32,
    pub differing_pixels: u32,
    pub max_channel_delta: u8,
}

/// Compare two images of identical dimensions, returning aggregate metrics.
pub fn compare(
    left: &DynamicImage,
    right: &DynamicImage,
) -> Result<ComparisonMetrics, TransformError> {
    let left_rgba: RgbaImage = left.to_rgba8();
    let right_rgba: RgbaImage = right.to_rgba8();
    if (left_rgba.width(), left_rgba.height()) != (right_rgba.width(), right_rgba.height()) {
        return Err(TransformError::DimensionMismatch {
            left_width: left_rgba.width(),
            left_height: left_rgba.height(),
            right_width: right_rgba.width(),
            right_height: right_rgba.height(),
        });
    }
    let total_pixels = (left_rgba.width() * left_rgba.height()) as usize;
    let total_channels = total_pixels * 4;

    let mut squared_sum = 0.0_f64;
    let mut differing = 0_u32;
    let mut max_delta = 0_u8;

    for (a, b) in left_rgba.pixels().zip(right_rgba.pixels()) {
        let mut pixel_diff = false;
        for channel in 0..4 {
            let delta = a[channel].abs_diff(b[channel]);
            if delta > 0 {
                pixel_diff = true;
                max_delta = max_delta.max(delta);
            }
            squared_sum += (delta as f64).powi(2);
        }
        if pixel_diff {
            differing += 1;
        }
    }

    let mse = if total_channels == 0 {
        0.0
    } else {
        (squared_sum / total_channels as f64) as f32
    };
    let psnr = if mse == 0.0 {
        f32::INFINITY
    } else {
        10.0 * (255.0_f32.powi(2) / mse).log10()
    };

    Ok(ComparisonMetrics {
        width: left_rgba.width(),
        height: left_rgba.height(),
        mean_squared_error: mse,
        peak_signal_to_noise_ratio: psnr,
        differing_pixels: differing,
        max_channel_delta: max_delta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbaImage;

    #[test]
    fn identical_images() {
        let left =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, image::Rgba([0, 0, 0, 255])));
        let right = left.clone();
        let metrics = compare(&left, &right).expect("compare");
        assert_eq!(metrics.mean_squared_error, 0.0);
        assert!(metrics.peak_signal_to_noise_ratio.is_infinite());
        assert_eq!(metrics.differing_pixels, 0);
    }

    #[test]
    fn detects_differences() {
        let left = RgbaImage::from_pixel(1, 1, image::Rgba([0, 0, 0, 255]));
        let right = RgbaImage::from_pixel(1, 1, image::Rgba([255, 255, 255, 255]));
        let metrics = compare(
            &DynamicImage::ImageRgba8(left.clone()),
            &DynamicImage::ImageRgba8(right.clone()),
        )
        .expect("compare");
        assert!(metrics.mean_squared_error > 0.0);
        assert_eq!(metrics.differing_pixels, 1);
        assert_eq!(metrics.max_channel_delta, 255);
    }

    #[test]
    fn mismatch_dimensions_error() {
        let left = DynamicImage::new_rgba8(2, 2);
        let right = DynamicImage::new_rgba8(1, 1);
        assert!(matches!(
            compare(&left, &right),
            Err(TransformError::DimensionMismatch { .. })
        ));
    }
}
