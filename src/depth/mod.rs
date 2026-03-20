// TODO: MIGRATE TO XENO-RT — this module belongs in the inference runtime, not the processing library.
// All ONNX Runtime inference (model loading, depth inference, pre/postprocessing) should move
// to xeno-rt. The apply_depth_blur() function is pure image processing and STAYS in xeno-lib.
// The DepthMap struct should be a shared type.
//!
//! AI-powered depth estimation using MiDaS.
//!
//! This module provides monocular depth estimation to generate
//! depth maps from single images.
//!
//! # Features
//!
//! - Generate depth maps from single images
//! - Multiple output formats (grayscale, colored)
//! - GPU acceleration via CUDA
//! - Applications: 3D effects, bokeh, AR/VR
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::depth::{estimate_depth, load_depth_estimator, DepthConfig};
//!
//! let config = DepthConfig::default();
//! let mut estimator = load_depth_estimator(&config)?;
//!
//! let image = image::open("photo.jpg")?;
//! let depth_map = estimate_depth(&image, &mut estimator)?;
//!
//! // Save as grayscale
//! depth_map.to_grayscale().save("depth.png")?;
//!
//! // Save as colored visualization
//! depth_map.to_colored().save("depth_colored.png")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Model Download
//!
//! Download MiDaS ONNX model:
//! - MiDaS: [GitHub](https://github.com/isl-org/MiDaS)
//!
//! Default path: `~/.xeno-lib/models/midas_v31_large.onnx`

mod config;
mod model;
mod processor;

use image::DynamicImage;

pub use config::{DepthConfig, DepthModel};
pub use model::{load_depth_estimator, DepthSession};
pub use processor::DepthMap;

use crate::error::TransformError;

/// Estimates depth from an image.
///
/// # Arguments
///
/// * `image` - The input image
/// * `session` - A loaded depth estimator model session
///
/// # Returns
///
/// A depth map with relative depth values.
pub fn estimate_depth(
    image: &DynamicImage,
    session: &mut DepthSession,
) -> Result<DepthMap, TransformError> {
    processor::estimate_depth_impl(image, session)
}

/// Quick depth estimation that loads model and processes in one call.
pub fn estimate_depth_quick(image: &DynamicImage) -> Result<DepthMap, TransformError> {
    let config = DepthConfig::default();
    let mut session = load_depth_estimator(&config)?;
    estimate_depth(image, &mut session)
}

/// Creates a depth-based blur effect (fake bokeh).
pub fn apply_depth_blur(
    image: &DynamicImage,
    depth_map: &DepthMap,
    focus_depth: f32,
    blur_strength: f32,
) -> DynamicImage {
    use image::{Rgb, RgbImage};

    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    let mut output = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let depth = depth_map.depth_at(x as usize, y as usize).unwrap_or(0.5);
            let blur_amount = ((depth - focus_depth).abs() * blur_strength) as i32;

            if blur_amount <= 0 {
                output.put_pixel(x, y, *rgb.get_pixel(x, y));
            } else {
                // Simple box blur based on depth
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                let mut count = 0u32;

                let radius = blur_amount.min(10);
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;

                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let p = rgb.get_pixel(nx as u32, ny as u32);
                            r_sum += p[0] as u32;
                            g_sum += p[1] as u32;
                            b_sum += p[2] as u32;
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    output.put_pixel(x, y, Rgb([
                        (r_sum / count) as u8,
                        (g_sum / count) as u8,
                        (b_sum / count) as u8,
                    ]));
                }
            }
        }
    }

    DynamicImage::ImageRgb8(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DepthConfig::default();
        assert_eq!(config.model, DepthModel::MidasLarge);
        assert!(config.use_gpu);
        assert!(config.normalize_output);
    }

    #[test]
    fn test_config_builder() {
        let config = DepthConfig::new(DepthModel::MidasSmall)
            .with_gpu(false)
            .with_inverted(true);

        assert_eq!(config.model, DepthModel::MidasSmall);
        assert!(!config.use_gpu);
        assert!(config.invert_depth);
    }
}
