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
