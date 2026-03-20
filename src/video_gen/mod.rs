// TODO: MIGRATE TO XENO-RT — this module belongs in the inference runtime, not the processing library.
// All ONNX Runtime inference (model loading, Stable Video Diffusion / AnimateDiff inference)
// should move to xeno-rt. The GeneratedVideo struct is a data type that should be shared.
//!
//! AI-powered video generation from images or text.
//!
//! Supports Stable Video Diffusion (image-to-video) and AnimateDiff (text-to-video).
//!
//! # Output Contract
//!
//! Video outputs are sequences of RGBA u8 `DynamicImage` frames.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::video_gen::{generate_video, VideoGenConfig};
//!
//! let config = VideoGenConfig::default().with_frames(14);
//! let input = image::open("photo.jpg")?;
//! let video = generate_video(&input, &config)?;
//! println!("Generated {} frames at {} fps", video.frames.len(), video.fps);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod config;
pub mod model;
pub mod processor;

use image::DynamicImage;

pub use config::{VideoGenConfig, VideoGenModel};
pub use model::{load_video_gen_model, VideoGenSession};

use crate::error::TransformError;

/// Generated video output.
#[derive(Debug, Clone)]
pub struct GeneratedVideo {
    /// Sequence of RGBA u8 frames.
    pub frames: Vec<DynamicImage>,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Frames per second.
    pub fps: f32,
    /// Total duration in seconds.
    pub duration_secs: f32,
}

/// Generates video from an input image (image-to-video).
pub fn generate_video(
    image: &DynamicImage,
    config: &VideoGenConfig,
) -> Result<GeneratedVideo, TransformError> {
    processor::generate_from_image(image, config)
}

/// Generates video from a text prompt (text-to-video, AnimateDiff).
pub fn generate_video_from_text(
    prompt: &str,
    config: &VideoGenConfig,
) -> Result<GeneratedVideo, TransformError> {
    processor::generate_from_text(prompt, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_generate_video() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(128, 128));
        let config = VideoGenConfig::default();
        let result = generate_video(&img, &config).unwrap();
        assert_eq!(result.frames.len(), 14);
    }
}
