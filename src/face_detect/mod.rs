//! AI-powered face detection using SCRFD.
//!
//! This module provides high-performance face detection with
//! optional facial landmark detection.
//!
//! # Features
//!
//! - Detect multiple faces in images
//! - Get bounding boxes with confidence scores
//! - Extract 5-point facial landmarks
//! - GPU acceleration via CUDA
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::face_detect::{detect_faces, load_detector, FaceDetectConfig};
//!
//! let config = FaceDetectConfig::default();
//! let mut detector = load_detector(&config)?;
//!
//! let image = image::open("photo.jpg")?;
//! let faces = detect_faces(&image, &mut detector)?;
//!
//! for face in &faces {
//!     println!("Face at {:?} with confidence {:.2}", face.bbox, face.confidence);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Model Download
//!
//! Download SCRFD ONNX model:
//! - SCRFD: [InsightFace](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
//!
//! Default path: `~/.xeno-lib/models/scrfd_10g.onnx`

mod config;
mod model;
mod processor;

use image::DynamicImage;

pub use config::{FaceDetectConfig, FaceDetectModel};
pub use model::{load_detector, FaceDetectorSession};
pub use processor::{DetectedFace, FaceLandmarks};

use crate::error::TransformError;

/// Detects faces in an image.
///
/// # Arguments
///
/// * `image` - The input image
/// * `session` - A loaded face detector model session
///
/// # Returns
///
/// A vector of detected faces with bounding boxes and optional landmarks.
pub fn detect_faces(
    image: &DynamicImage,
    session: &mut FaceDetectorSession,
) -> Result<Vec<DetectedFace>, TransformError> {
    processor::detect_faces_impl(image, session)
}

/// Draws detection boxes on an image for visualization.
pub fn visualize_detections(
    image: &DynamicImage,
    faces: &[DetectedFace],
) -> DynamicImage {
    processor::draw_detections(image, faces)
}

/// Quick detection that loads model and processes in one call.
pub fn detect_faces_quick(image: &DynamicImage) -> Result<Vec<DetectedFace>, TransformError> {
    let config = FaceDetectConfig::default();
    let mut session = load_detector(&config)?;
    detect_faces(image, &mut session)
}

/// Crops detected faces from an image.
pub fn crop_faces(
    image: &DynamicImage,
    faces: &[DetectedFace],
    padding: f32,
) -> Vec<DynamicImage> {
    faces
        .iter()
        .map(|face| {
            let (x, y, w, h) = face.bbox;

            // Add padding
            let pad_w = (w as f32 * padding) as u32;
            let pad_h = (h as f32 * padding) as u32;

            let x1 = x.saturating_sub(pad_w);
            let y1 = y.saturating_sub(pad_h);
            let x2 = (x + w + pad_w).min(image.width());
            let y2 = (y + h + pad_h).min(image.height());

            image.crop_imm(x1, y1, x2 - x1, y2 - y1)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = FaceDetectConfig::default();
        assert_eq!(config.model, FaceDetectModel::Scrfd);
        assert!(config.use_gpu);
        assert_eq!(config.confidence_threshold, 0.5);
    }

    #[test]
    fn test_config_builder() {
        let config = FaceDetectConfig::new(FaceDetectModel::RetinaFace)
            .with_gpu(false)
            .with_confidence(0.7);

        assert_eq!(config.model, FaceDetectModel::RetinaFace);
        assert!(!config.use_gpu);
        assert_eq!(config.confidence_threshold, 0.7);
    }
}
