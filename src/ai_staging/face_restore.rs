// Staged for migration to xeno-rt
//
// Extracted from: src/face_restore/mod.rs, src/ai_deprecated/face_restore/model.rs,
//                 src/ai_deprecated/face_restore/processor.rs, src/face_restore/config.rs
//
// Contains: ONNX model loading, session management, face image preprocessing,
//           inference execution, and tensor-to-image postprocessing for GFPGAN/CodeFormer.
//
// What STAYS in xeno-lib:
//   - FaceRegion, FaceLandmarks structs (shared data types)
//
// Output contract (preserve in xeno-rt):
//   - Input: RGBA image containing faces (DynamicImage)
//   - Output: RGBA image with restored faces (u8, 4 bytes/pixel)

// ---------------------------------------------------------------------------
// Configuration (from src/face_restore/config.rs)
// ---------------------------------------------------------------------------

//! Configuration for face restoration.

use std::path::PathBuf;

/// Available face restoration models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FaceRestoreModel {
    /// GFPGAN - Practical face restoration algorithm.
    /// Good balance of speed and quality.
    #[default]
    GFPGAN,

    /// CodeFormer - State-of-the-art face restoration.
    /// Better quality but slower.
    CodeFormer,

    /// RestoreFormer - Alternative restoration model.
    RestoreFormer,
}

impl FaceRestoreModel {
    /// Returns the default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            FaceRestoreModel::GFPGAN => "gfpgan.onnx",
            FaceRestoreModel::CodeFormer => "codeformer.onnx",
            FaceRestoreModel::RestoreFormer => "restoreformer.onnx",
        }
    }

    /// Returns a human-readable name.
    pub fn display_name(&self) -> &'static str {
        match self {
            FaceRestoreModel::GFPGAN => "GFPGAN",
            FaceRestoreModel::CodeFormer => "CodeFormer",
            FaceRestoreModel::RestoreFormer => "RestoreFormer",
        }
    }

    /// Returns expected input size for the model.
    pub fn input_size(&self) -> (u32, u32) {
        match self {
            FaceRestoreModel::GFPGAN => (512, 512),
            FaceRestoreModel::CodeFormer => (512, 512),
            FaceRestoreModel::RestoreFormer => (512, 512),
        }
    }
}

/// Configuration for face restoration.
#[derive(Debug, Clone)]
pub struct FaceRestoreConfig {
    /// The face restoration model to use.
    pub model: FaceRestoreModel,

    /// Path to the ONNX model file.
    pub model_path: Option<PathBuf>,

    /// Whether to use GPU acceleration.
    pub use_gpu: bool,

    /// CUDA device ID.
    pub gpu_device_id: i32,

    /// Face detection confidence threshold.
    pub detection_threshold: f32,

    /// Restoration strength (0.0 - 1.0).
    /// Higher values produce stronger restoration but may look less natural.
    pub strength: f32,

    /// Whether to only restore faces (keeping background unchanged).
    pub face_only: bool,

    /// Upscale factor to apply after restoration.
    pub upscale_factor: u32,
}

impl Default for FaceRestoreConfig {
    fn default() -> Self {
        Self {
            model: FaceRestoreModel::default(),
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            detection_threshold: 0.5,
            strength: 0.8,
            face_only: false,
            upscale_factor: 1,
        }
    }
}

impl FaceRestoreConfig {
    /// Create a new configuration with the specified model.
    pub fn new(model: FaceRestoreModel) -> Self {
        Self {
            model,
            ..Default::default()
        }
    }

    /// Set the model path explicitly.
    pub fn with_model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Enable or disable GPU acceleration.
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set restoration strength.
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set face detection threshold.
    pub fn with_detection_threshold(mut self, threshold: f32) -> Self {
        self.detection_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set whether to only restore faces.
    pub fn with_face_only(mut self, face_only: bool) -> Self {
        self.face_only = face_only;
        self
    }

    /// Get the effective model path.
    pub fn effective_model_path(&self) -> PathBuf {
        if let Some(ref path) = self.model_path {
            path.clone()
        } else {
            default_model_path(self.model.default_filename())
        }
    }
}

fn default_model_path(filename: &str) -> PathBuf {
    let home = dirs_next().unwrap_or_else(|| PathBuf::from("."));
    home.join(".xeno-lib").join("models").join(filename)
}

fn dirs_next() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE").ok().map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FaceRestoreConfig::default();
        assert_eq!(config.model, FaceRestoreModel::GFPGAN);
        assert!(config.use_gpu);
        assert!((config.strength - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_model_input_sizes() {
        assert_eq!(FaceRestoreModel::GFPGAN.input_size(), (512, 512));
        assert_eq!(FaceRestoreModel::CodeFormer.input_size(), (512, 512));
    }
}


// ---------------------------------------------------------------------------
// Model session and loading (from src/ai_deprecated/face_restore/model.rs)
// ---------------------------------------------------------------------------

//! ONNX model session management for face restoration.

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;
// NOTE: FaceRestoreConfig is defined above in this staging file (was use super::config::FaceRestoreConfig)

/// A loaded face restoration model session.
pub struct FaceRestorerSession {
    session: Session,
    config: FaceRestoreConfig,
}

impl FaceRestorerSession {
    /// Returns a reference to the configuration.
    pub fn config(&self) -> &FaceRestoreConfig {
        &self.config
    }

    /// Returns the expected input dimensions.
    pub fn input_size(&self) -> (u32, u32) {
        self.config.model.input_size()
    }

    /// Runs face restoration inference.
    ///
    /// # Arguments
    /// * `input` - Face image tensor of shape [1, 3, H, W] normalized to [-1, 1]
    ///
    /// # Returns
    /// Restored face tensor of shape [1, 3, H, W] in range [-1, 1]
    pub fn run(&mut self, input: &Array4<f32>) -> Result<Array4<f32>, TransformError> {
        let tensor_ref = TensorRef::from_array_view(input.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to create tensor ref: {e}"),
            }
        })?;

        let outputs = self
            .session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("face restoration inference failed: {e}"),
            })?;

        let output_tensor = outputs.iter().next().ok_or_else(|| {
            TransformError::InferenceFailed {
                message: "no output tensor found".to_string(),
            }
        })?;

        let (shape, data) = output_tensor
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("failed to extract output tensor: {e}"),
            })?;

        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if dims.len() != 4 {
            return Err(TransformError::InferenceFailed {
                message: format!("unexpected output shape: {:?}", dims),
            });
        }

        let flat_data: Vec<f32> = data.iter().copied().collect();
        Array4::from_shape_vec((dims[0], dims[1], dims[2], dims[3]), flat_data).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to reshape output: {e}"),
            }
        })
    }
}

/// Loads a face restoration model.
pub fn load_restorer(config: &FaceRestoreConfig) -> Result<FaceRestorerSession, TransformError> {
    let model_path = config.effective_model_path();

    if !model_path.exists() {
        return Err(TransformError::ModelNotFound {
            path: model_path.display().to_string(),
        });
    }

    let mut builder = Session::builder()
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to create session builder: {e}"),
        })?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to set optimization level: {e}"),
        })?;

    if config.use_gpu {
        builder = builder
            .with_execution_providers([
                CUDAExecutionProvider::default()
                    .with_device_id(config.gpu_device_id)
                    .build(),
                CPUExecutionProvider::default().build(),
            ])
            .map_err(|e| TransformError::ModelLoadFailed {
                message: format!("failed to configure execution providers: {e}"),
            })?;
    } else {
        builder = builder
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| TransformError::ModelLoadFailed {
                message: format!("failed to configure CPU execution provider: {e}"),
            })?;
    }

    let session = builder
        .commit_from_file(&model_path)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to load face restoration model: {e}"),
        })?;

    Ok(FaceRestorerSession {
        session,
        config: config.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_model_not_found() {
        let config = FaceRestoreConfig::default()
            .with_model_path(PathBuf::from("/nonexistent/model.onnx"));
        let result = load_restorer(&config);
        assert!(matches!(result, Err(TransformError::ModelNotFound { .. })));
    }
}


// ---------------------------------------------------------------------------
// Processor: preprocessing, inference, postprocessing
// (from src/ai_deprecated/face_restore/processor.rs)
// ---------------------------------------------------------------------------

//! Face restoration processing logic.

use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};
use ndarray::Array4;

use crate::error::TransformError;
// NOTE: FaceRestorerSession is defined above in this staging file (was use super::model::FaceRestorerSession)

/// Restores faces in an image.
pub fn restore_faces_impl(
    image: &DynamicImage,
    session: &mut FaceRestorerSession,
) -> Result<DynamicImage, TransformError> {
    let (input_w, input_h) = session.input_size();
    let original_width = image.width();
    let original_height = image.height();

    // For now, process the entire image as a face
    // In a full implementation, we'd detect faces first and process each one

    // Resize to model input size
    let resized = image.resize_exact(input_w, input_h, FilterType::Lanczos3);

    // Convert to tensor (normalized to [-1, 1] for GFPGAN)
    let input_tensor = image_to_tensor(&resized)?;

    // Run restoration
    let output_tensor = session.run(&input_tensor)?;

    // Convert back to image
    let restored = tensor_to_image(&output_tensor)?;

    // Resize back to original dimensions
    let final_image = restored.resize_exact(original_width, original_height, FilterType::Lanczos3);

    Ok(final_image)
}

/// Converts image to tensor normalized to [-1, 1].
fn image_to_tensor(image: &DynamicImage) -> Result<Array4<f32>, TransformError> {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                // Normalize to [-1, 1] (GFPGAN expects this range)
                tensor[[0, c, y, x]] = (pixel[c] as f32 / 127.5) - 1.0;
            }
        }
    }

    Ok(tensor)
}

/// Converts tensor from [-1, 1] range to image.
fn tensor_to_image(tensor: &Array4<f32>) -> Result<DynamicImage, TransformError> {
    let shape = tensor.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
        return Err(TransformError::InferenceFailed {
            message: format!("unexpected tensor shape: {:?}", shape),
        });
    }

    let height = shape[2];
    let width = shape[3];
    let mut image = RgbImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            // Convert from [-1, 1] to [0, 255]
            let r = ((tensor[[0, 0, y, x]] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            let g = ((tensor[[0, 1, y, x]] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            let b = ((tensor[[0, 2, y, x]] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
            image.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    Ok(DynamicImage::ImageRgb8(image))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(width: u32, height: u32) -> DynamicImage {
        let mut img = RgbImage::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Rgb([
                ((x * 255) / width.max(1)) as u8,
                ((y * 255) / height.max(1)) as u8,
                128,
            ]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_tensor_roundtrip() {
        let img = create_test_image(64, 64);
        let tensor = image_to_tensor(&img).unwrap();
        let recovered = tensor_to_image(&tensor).unwrap();

        // Check dimensions
        assert_eq!(recovered.width(), 64);
        assert_eq!(recovered.height(), 64);
    }

    #[test]
    fn test_tensor_range() {
        let img = create_test_image(32, 32);
        let tensor = image_to_tensor(&img).unwrap();

        let min_val = tensor.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = tensor.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Should be in [-1, 1] range
        assert!(min_val >= -1.01, "min {} < -1", min_val);
        assert!(max_val <= 1.01, "max {} > 1", max_val);
    }
}
