//! AI-powered image upscaling using Real-ESRGAN.
//!
//! This module provides high-quality image super-resolution powered by the Real-ESRGAN
//! deep learning model. It supports 2x, 4x, and 8x upscaling with both CUDA (GPU) and
//! CPU execution.
//!
//! # Advantages Over Traditional Upscaling
//!
//! Unlike interpolation-based methods (bicubic, lanczos), Real-ESRGAN:
//! - **Generates realistic details** instead of just blurring
//! - **Recovers textures** like skin pores, hair, fabric patterns
//! - **Removes compression artifacts** while upscaling
//! - **Produces sharper edges** without ringing artifacts
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::upscale::{upscale, load_upscaler, UpscaleConfig, UpscaleModel};
//!
//! // Load the model once (reuse for multiple images)
//! let config = UpscaleConfig::default();  // 4x upscale
//! let mut upscaler = load_upscaler(&config)?;
//!
//! // Upscale an image: 480p -> 4K!
//! let input = image::open("low_res.jpg")?;
//! let output = upscale(&input, &mut upscaler)?;
//! output.save("high_res.png")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Model Download
//!
//! Real-ESRGAN models must be downloaded separately:
//!
//! | Model | Scale | URL |
//! |-------|-------|-----|
//! | RealESRGAN_x4plus | 4x | [HuggingFace](https://huggingface.co/ai-forever/Real-ESRGAN) |
//! | RealESRGAN_x2 | 2x | [HuggingFace](https://huggingface.co/ai-forever/Real-ESRGAN) |
//! | RealESRGAN_x4plus_anime | 4x | [GitHub](https://github.com/xinntao/Real-ESRGAN) |
//!
//! Default path: `~/.xeno-lib/models/realesrgan_x4plus.onnx`
//!
//! # Features
//!
//! - `upscale`: CPU-only support
//! - `upscale-cuda`: GPU acceleration via CUDA
//!
//! # Tile-Based Processing
//!
//! For large images, the upscaler automatically splits the image into tiles
//! to avoid GPU memory exhaustion. You can configure tile size:
//!
//! ```rust,no_run
//! use xeno_lib::upscale::{UpscaleConfig, UpscaleModel};
//!
//! let config = UpscaleConfig::new(UpscaleModel::RealEsrganX4Plus)
//!     .with_tile_size(256)  // Process in 256x256 tiles
//!     .with_tile_overlap(16);  // 16px overlap to avoid seams
//! ```

mod config;
mod model;
mod processor;

use image::DynamicImage;

pub use config::{UpscaleConfig, UpscaleModel};
pub use model::{load_upscaler, UpscalerSession};

use crate::error::TransformError;

/// Upscales an image using AI super-resolution.
///
/// # Arguments
///
/// * `image` - The input image to upscale
/// * `session` - A loaded upscaler model session
///
/// # Returns
///
/// A new `DynamicImage` with dimensions multiplied by the model's scale factor.
///
/// # Example
///
/// ```rust,no_run
/// use xeno_lib::upscale::{upscale, load_upscaler, UpscaleConfig};
///
/// let mut session = load_upscaler(&UpscaleConfig::default())?;
/// let input = image::open("photo_480p.jpg")?;
/// let result = upscale(&input, &mut session)?;  // Now 4x larger!
/// result.save("photo_4k.png")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn upscale(
    image: &DynamicImage,
    session: &mut UpscalerSession,
) -> Result<DynamicImage, TransformError> {
    let config = session.config();

    // For small images, process directly
    if image.width() <= config.tile_size && image.height() <= config.tile_size {
        return processor::upscale_direct(image, session);
    }

    // For large images, use tile-based processing
    processor::upscale_tiled(image, session)
}

/// Upscales multiple images sequentially.
///
/// # Arguments
///
/// * `images` - A slice of input images to process
/// * `session` - A loaded upscaler model session
///
/// # Returns
///
/// A vector of upscaled images.
pub fn upscale_batch(
    images: &[DynamicImage],
    session: &mut UpscalerSession,
) -> Result<Vec<DynamicImage>, TransformError> {
    images.iter().map(|img| upscale(img, session)).collect()
}

/// Quick upscale function that loads the model and processes in one call.
///
/// For processing multiple images, prefer `load_upscaler` + `upscale` to avoid
/// reloading the model each time.
///
/// # Example
///
/// ```rust,no_run
/// use xeno_lib::upscale::{upscale_quick, UpscaleModel};
///
/// let input = image::open("small.jpg")?;
/// let output = upscale_quick(&input, UpscaleModel::RealEsrganX4Plus)?;
/// output.save("large.png")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn upscale_quick(
    image: &DynamicImage,
    model: UpscaleModel,
) -> Result<DynamicImage, TransformError> {
    let config = UpscaleConfig::new(model);
    let mut session = load_upscaler(&config)?;
    upscale(image, &mut session)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = UpscaleConfig::default();
        assert_eq!(config.model, UpscaleModel::RealEsrganX4Plus);
        assert_eq!(config.scale_factor(), 4);
        assert!(config.use_gpu);
    }

    #[test]
    fn test_model_scale_factors() {
        assert_eq!(UpscaleModel::RealEsrganX2.scale_factor(), 2);
        assert_eq!(UpscaleModel::RealEsrganX4Plus.scale_factor(), 4);
        assert_eq!(UpscaleModel::RealEsrganX4Anime.scale_factor(), 4);
        assert_eq!(UpscaleModel::RealEsrganX8.scale_factor(), 8);
    }

    #[test]
    fn test_config_builder() {
        let config = UpscaleConfig::new(UpscaleModel::RealEsrganX2)
            .with_tile_size(128)
            .with_tile_overlap(8)
            .with_gpu(false);

        assert_eq!(config.model, UpscaleModel::RealEsrganX2);
        assert_eq!(config.tile_size, 128);
        assert_eq!(config.tile_overlap, 8);
        assert!(!config.use_gpu);
    }
}
