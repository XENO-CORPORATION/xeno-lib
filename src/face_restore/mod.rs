// TODO: MIGRATE TO XENO-RT — this module belongs in the inference runtime, not the processing library.
// All ONNX Runtime inference (model loading, face detection, restoration inference) should move
// to xeno-rt. The FaceRegion/FaceLandmarks structs are shared data types that may stay as common
// types if needed by pure processing code.
//!
//! AI-powered face restoration using GFPGAN/CodeFormer.
//!
//! This module provides high-quality face restoration for blurry, low-resolution,
//! or damaged faces using deep learning models.
//!
//! # Features
//!
//! - Restore blurry/pixelated faces to high quality
//! - Fix old/damaged photos
//! - Enhance low-resolution faces
//! - Works with Real-ESRGAN upscaling for best results
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::face_restore::{restore_faces, load_restorer, FaceRestoreConfig};
//!
//! let config = FaceRestoreConfig::default();
//! let mut restorer = load_restorer(&config)?;
//!
//! let input = image::open("old_photo.jpg")?;
//! let restored = restore_faces(&input, &mut restorer)?;
//! restored.save("restored.png")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Model Download
//!
//! Download GFPGAN or CodeFormer ONNX models:
//! - GFPGAN: [GitHub](https://github.com/TencentARC/GFPGAN)
//! - CodeFormer: [GitHub](https://github.com/sczhou/CodeFormer)
//!
//! Default path: `~/.xeno-lib/models/gfpgan.onnx`

mod config;
mod model;
mod processor;

use image::DynamicImage;

pub use config::{FaceRestoreConfig, FaceRestoreModel};
pub use model::{load_restorer, FaceRestorerSession};

use crate::error::TransformError;

/// Restores faces in an image using AI.
///
/// This function detects faces in the image and applies AI-based restoration
/// to enhance their quality.
///
/// # Arguments
///
/// * `image` - The input image containing faces to restore
/// * `session` - A loaded face restorer model session
///
/// # Returns
///
/// A new image with restored faces.
pub fn restore_faces(
    image: &DynamicImage,
    session: &mut FaceRestorerSession,
) -> Result<DynamicImage, TransformError> {
    processor::restore_faces_impl(image, session)
}

/// Restores faces in multiple images.
pub fn restore_faces_batch(
    images: &[DynamicImage],
    session: &mut FaceRestorerSession,
) -> Result<Vec<DynamicImage>, TransformError> {
    images
        .iter()
        .map(|img| restore_faces(img, session))
        .collect()
}

/// Face region detected in an image.
#[derive(Debug, Clone)]
pub struct FaceRegion {
    /// Bounding box x coordinate
    pub x: u32,
    /// Bounding box y coordinate
    pub y: u32,
    /// Bounding box width
    pub width: u32,
    /// Bounding box height
    pub height: u32,
    /// Detection confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Facial landmarks (if available)
    pub landmarks: Option<FaceLandmarks>,
}

/// Facial landmarks for alignment.
#[derive(Debug, Clone)]
pub struct FaceLandmarks {
    /// Left eye center
    pub left_eye: (f32, f32),
    /// Right eye center
    pub right_eye: (f32, f32),
    /// Nose tip
    pub nose: (f32, f32),
    /// Left mouth corner
    pub left_mouth: (f32, f32),
    /// Right mouth corner
    pub right_mouth: (f32, f32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = FaceRestoreConfig::default();
        assert_eq!(config.model, FaceRestoreModel::GFPGAN);
        assert!(config.use_gpu);
    }

    #[test]
    fn test_face_region() {
        let region = FaceRegion {
            x: 100,
            y: 100,
            width: 200,
            height: 200,
            confidence: 0.95,
            landmarks: None,
        };
        assert_eq!(region.width, 200);
        assert!(region.confidence > 0.9);
    }
}
