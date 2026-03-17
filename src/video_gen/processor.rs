//! Video generation processing.

use image::DynamicImage;

use crate::error::TransformError;
use super::config::VideoGenConfig;
use super::GeneratedVideo;

/// Generates video frames from a single image.
///
/// # Arguments
///
/// * `image` - Input conditioning image.
/// * `config` - Generation configuration.
///
/// # Returns
///
/// `GeneratedVideo` with RGBA u8 frames.
pub fn generate_from_image(
    image: &DynamicImage,
    config: &VideoGenConfig,
) -> Result<GeneratedVideo, TransformError> {
    // Resize input to model resolution
    let resized = image.resize_exact(
        config.width,
        config.height,
        image::imageops::FilterType::Lanczos3,
    );

    // Stub: in production, encode image, run diffusion model, decode latents to frames
    let mut frames = Vec::with_capacity(config.num_frames as usize);
    for _i in 0..config.num_frames {
        // Each frame is a copy of the input (placeholder)
        frames.push(resized.clone());
    }

    Ok(GeneratedVideo {
        frames,
        width: config.width,
        height: config.height,
        fps: config.fps,
        duration_secs: config.num_frames as f32 / config.fps,
    })
}

/// Generates video frames from a text prompt (AnimateDiff).
///
/// # Arguments
///
/// * `prompt` - Text description of desired video.
/// * `config` - Generation configuration.
pub fn generate_from_text(
    prompt: &str,
    config: &VideoGenConfig,
) -> Result<GeneratedVideo, TransformError> {
    if prompt.is_empty() {
        return Err(TransformError::InvalidParameter {
            name: "prompt",
            value: 0.0,
        });
    }

    // Stub: tokenize prompt, run text-conditioned diffusion
    let mut frames = Vec::with_capacity(config.num_frames as usize);
    for _i in 0..config.num_frames {
        let frame = DynamicImage::new_rgba8(config.width, config.height);
        frames.push(frame);
    }

    Ok(GeneratedVideo {
        frames,
        width: config.width,
        height: config.height,
        fps: config.fps,
        duration_secs: config.num_frames as f32 / config.fps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_generate_from_image() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(256, 256));
        let config = VideoGenConfig::default();
        let result = generate_from_image(&img, &config).unwrap();
        assert_eq!(result.frames.len(), config.num_frames as usize);
        assert_eq!(result.width, config.width);
    }

    #[test]
    fn test_generate_from_text_empty() {
        let config = VideoGenConfig::default();
        let result = generate_from_text("", &config);
        assert!(result.is_err());
    }
}
