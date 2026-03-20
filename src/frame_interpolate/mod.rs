// TODO: MIGRATE TO XENO-RT — this module belongs in the inference runtime, not the processing library.
// All ONNX Runtime inference (model loading, RIFE inference, pre/postprocessing) should move to
// xeno-rt. The is_scene_change() function uses pixel comparison (not AI) and could STAY in
// xeno-lib as a pure processing utility.
//!
//! AI-powered frame interpolation using RIFE.
//!
//! This module provides smooth frame interpolation for video frame rate
//! conversion, slow motion, and temporal upscaling.
//!
//! # Features
//!
//! - Interpolate frames for smooth slow motion
//! - Double, quadruple, or 8x frame rates
//! - Scene detection to avoid artifacts at cuts
//! - GPU acceleration via CUDA
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::frame_interpolate::{interpolate_frame, load_interpolator, InterpolationConfig};
//!
//! let config = InterpolationConfig::default();
//! let mut interpolator = load_interpolator(&config)?;
//!
//! let frame0 = image::open("frame_0000.png")?;
//! let frame1 = image::open("frame_0001.png")?;
//!
//! // Interpolate a frame at the midpoint
//! let mid_frame = interpolate_frame(&frame0, &frame1, 0.5, &mut interpolator)?;
//! mid_frame.save("frame_0000_5.png")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Model Download
//!
//! Download RIFE ONNX model:
//! - RIFE v4.6: [GitHub](https://github.com/hzwer/ECCV2022-RIFE)
//!
//! Default path: `~/.xeno-lib/models/rife-v4.6.onnx`

mod config;
mod model;
mod processor;

use image::DynamicImage;

pub use config::{InterpolationConfig, InterpolationModel};
pub use model::{load_interpolator, InterpolatorSession};

use crate::error::TransformError;

/// Interpolates a single frame between two input frames.
///
/// # Arguments
///
/// * `frame0` - The first (earlier) frame
/// * `frame1` - The second (later) frame
/// * `timestep` - Position between frames (0.0 = frame0, 1.0 = frame1)
/// * `session` - A loaded interpolator model session
///
/// # Returns
///
/// The interpolated frame at the specified timestep.
pub fn interpolate_frame(
    frame0: &DynamicImage,
    frame1: &DynamicImage,
    timestep: f32,
    session: &mut InterpolatorSession,
) -> Result<DynamicImage, TransformError> {
    processor::interpolate_frame_impl(frame0, frame1, timestep, session)
}

/// Interpolates multiple frames between two input frames.
///
/// # Arguments
///
/// * `frame0` - The first (earlier) frame
/// * `frame1` - The second (later) frame
/// * `num_frames` - Number of intermediate frames to generate
/// * `session` - A loaded interpolator model session
///
/// # Returns
///
/// A vector of interpolated frames evenly spaced between frame0 and frame1.
pub fn interpolate_frames(
    frame0: &DynamicImage,
    frame1: &DynamicImage,
    num_frames: u32,
    session: &mut InterpolatorSession,
) -> Result<Vec<DynamicImage>, TransformError> {
    processor::interpolate_frames_impl(frame0, frame1, num_frames, session)
}

/// Detects if there's a scene change between two frames.
///
/// Useful for avoiding interpolation artifacts at scene cuts.
pub fn is_scene_change(
    frame0: &DynamicImage,
    frame1: &DynamicImage,
    threshold: f32,
) -> bool {
    processor::detect_scene_change(frame0, frame1, threshold)
}

/// Quick interpolation that loads model and processes in one call.
pub fn interpolate_quick(
    frame0: &DynamicImage,
    frame1: &DynamicImage,
    timestep: f32,
) -> Result<DynamicImage, TransformError> {
    let config = InterpolationConfig::default();
    let mut session = load_interpolator(&config)?;
    interpolate_frame(frame0, frame1, timestep, &mut session)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = InterpolationConfig::default();
        assert_eq!(config.model, InterpolationModel::RifeV4);
        assert!(config.use_gpu);
        assert_eq!(config.multiplier, 2);
    }

    #[test]
    fn test_config_builder() {
        let config = InterpolationConfig::new(InterpolationModel::RifeV4HD)
            .with_gpu(false)
            .with_multiplier(4);

        assert_eq!(config.model, InterpolationModel::RifeV4HD);
        assert!(!config.use_gpu);
        assert_eq!(config.multiplier, 4);
    }
}
