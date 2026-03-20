// TODO: MIGRATE TO XENO-RT — this module belongs in the inference runtime, not the processing library.
// All ONNX Runtime inference (model loading, colorization inference, pre/postprocessing) should move
// to xeno-rt. No pure processing code remains in this module.
//!
//! AI-powered image and video colorization using DDColor.
//!
//! This module provides automatic colorization of black & white images
//! using deep learning models.
//!
//! # Features
//!
//! - Colorize B&W photos with realistic colors
//! - Restore color to faded/damaged photos
//! - Batch processing for multiple images
//! - GPU acceleration via CUDA
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::colorize::{colorize, load_colorizer, ColorizeConfig};
//!
//! let config = ColorizeConfig::default();
//! let mut colorizer = load_colorizer(&config)?;
//!
//! let bw_image = image::open("old_bw_photo.jpg")?;
//! let colorized = colorize(&bw_image, &mut colorizer)?;
//! colorized.save("colorized.png")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Model Download
//!
//! Download DDColor ONNX model:
//! - DDColor: [HuggingFace](https://huggingface.co/piddnad/DDColor)
//!
//! Default path: `~/.xeno-lib/models/ddcolor.onnx`

mod config;
mod model;
mod processor;

use image::DynamicImage;

pub use config::{ColorizeConfig, ColorizeModel};
pub use model::{load_colorizer, ColorizerSession};

use crate::error::TransformError;

/// Colorizes a black & white image using AI.
///
/// # Arguments
///
/// * `image` - The input B&W or grayscale image
/// * `session` - A loaded colorizer model session
///
/// # Returns
///
/// A new colorized RGB image.
pub fn colorize(
    image: &DynamicImage,
    session: &mut ColorizerSession,
) -> Result<DynamicImage, TransformError> {
    processor::colorize_impl(image, session)
}

/// Colorizes multiple images.
pub fn colorize_batch(
    images: &[DynamicImage],
    session: &mut ColorizerSession,
) -> Result<Vec<DynamicImage>, TransformError> {
    images
        .iter()
        .map(|img| colorize(img, session))
        .collect()
}

/// Quick colorize function that loads the model and processes in one call.
pub fn colorize_quick(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    let config = ColorizeConfig::default();
    let mut session = load_colorizer(&config)?;
    colorize(image, &mut session)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ColorizeConfig::default();
        assert_eq!(config.model, ColorizeModel::DDColor);
        assert!(config.use_gpu);
    }
}
