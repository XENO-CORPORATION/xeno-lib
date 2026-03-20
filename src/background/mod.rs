// TODO: MIGRATE TO XENO-RT — this module belongs in the inference runtime, not the processing library.
// All ONNX Runtime inference (model loading, preprocessing, inference, postprocessing) should move
// to xeno-rt. The postprocess::apply_mask function contains pure image processing that could stay
// in xeno-lib as a general-purpose mask application utility.
//!
//! AI-powered background removal using ONNX Runtime.
//!
//! This module provides high-quality background removal powered by a BiRefNet-style
//! deep learning model. It supports both CUDA (GPU) and CPU execution, with
//! automatic fallback when CUDA is unavailable.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::background::{remove_background, load_model, BackgroundRemovalConfig};
//!
//! // Load the model once (reuse for multiple images)
//! let config = BackgroundRemovalConfig::default();
//! let mut session = load_model(&config)?;
//!
//! // Process an image
//! let input = image::open("photo.jpg")?;
//! let output = remove_background(&input, &mut session)?;
//! output.save("photo_nobg.png")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Model Download
//!
//! The model must be downloaded separately:
//!
//! - **Default path**: `~/.xeno-lib/models/birefnet-general.onnx`
//!
//! # Features
//!
//! - `background-removal`: CPU-only support
//! - `background-removal-cuda`: GPU acceleration via CUDA

mod model;
mod postprocess;
mod preprocess;

use image::DynamicImage;

pub use model::{load_model, BackgroundRemovalConfig, ModelSession};

use crate::error::TransformError;

/// Removes the background from an image, returning an RGBA image with
/// transparent background.
///
/// # Arguments
///
/// * `image` - The input image to process
/// * `session` - A loaded ONNX model session
///
/// # Returns
///
/// A new `DynamicImage` in RGBA format with the background made transparent.
///
/// # Example
///
/// ```rust,no_run
/// use xeno_lib::background::{remove_background, load_model, BackgroundRemovalConfig};
///
/// let mut session = load_model(&BackgroundRemovalConfig::default())?;
/// let input = image::open("portrait.jpg")?;
/// let result = remove_background(&input, &mut session)?;
/// result.save("portrait_nobg.png")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn remove_background(
    image: &DynamicImage,
    session: &mut ModelSession,
) -> Result<DynamicImage, TransformError> {
    // Store original dimensions for later
    let original_width = image.width();
    let original_height = image.height();

    // Preprocess: convert image to tensor
    let input_tensor = preprocess::image_to_tensor(image, session.input_size())?;

    // Run inference
    let mask_tensor = session.run(&input_tensor)?;

    // Postprocess: apply mask to original image
    postprocess::apply_mask(image, &mask_tensor, original_width, original_height, session.config().confidence_threshold)
}

/// Removes backgrounds from multiple images sequentially.
///
/// This function processes images one at a time using the same session.
/// ONNX Runtime handles internal parallelism for each inference call.
///
/// # Arguments
///
/// * `images` - A slice of input images to process
/// * `session` - A loaded ONNX model session
///
/// # Returns
///
/// A vector of processed images with transparent backgrounds.
///
/// # Example
///
/// ```rust,no_run
/// use xeno_lib::background::{remove_background_batch, load_model, BackgroundRemovalConfig};
///
/// let mut session = load_model(&BackgroundRemovalConfig::default())?;
/// let images: Vec<_> = vec!["a.jpg", "b.jpg", "c.jpg"]
///     .into_iter()
///     .filter_map(|p| image::open(p).ok())
///     .collect();
///
/// let results = remove_background_batch(&images, &mut session)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn remove_background_batch(
    images: &[DynamicImage],
    session: &mut ModelSession,
) -> Result<Vec<DynamicImage>, TransformError> {
    images
        .iter()
        .map(|img| remove_background(img, session))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    fn create_test_image(width: u32, height: u32) -> DynamicImage {
        let mut img = RgbaImage::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            // Create a simple gradient pattern
            *pixel = Rgba([
                (x * 255 / width) as u8,
                (y * 255 / height) as u8,
                128,
                255,
            ]);
        }
        DynamicImage::ImageRgba8(img)
    }

    #[test]
    fn test_config_default() {
        let config = BackgroundRemovalConfig::default();
        assert!(config.use_gpu);
        assert_eq!(config.gpu_device_id, 0);
        assert!((config.confidence_threshold - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_preprocess_dimensions() {
        let img = create_test_image(640, 480);
        let tensor = preprocess::image_to_tensor(&img, (1024, 1024)).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 1024, 1024]);
    }
}
