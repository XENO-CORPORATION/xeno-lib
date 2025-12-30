//! Pose estimation configuration.

use std::path::PathBuf;

/// Pose estimation model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoseModel {
    /// MoveNet Lightning - fast, single person.
    #[default]
    MoveNetLightning,
    /// MoveNet Thunder - accurate, single person.
    MoveNetThunder,
    /// MoveNet MultiPose - multiple people.
    MoveNetMultiPose,
    /// MediaPipe Pose - 33 landmarks.
    MediaPipePose,
}

impl PoseModel {
    /// Get model filename.
    pub fn model_filename(&self) -> &'static str {
        match self {
            Self::MoveNetLightning => "movenet_lightning.onnx",
            Self::MoveNetThunder => "movenet_thunder.onnx",
            Self::MoveNetMultiPose => "movenet_multipose.onnx",
            Self::MediaPipePose => "mediapipe_pose.onnx",
        }
    }

    /// Get input size.
    pub fn input_size(&self) -> (u32, u32) {
        match self {
            Self::MoveNetLightning => (192, 192),
            Self::MoveNetThunder => (256, 256),
            Self::MoveNetMultiPose => (256, 256),
            Self::MediaPipePose => (256, 256),
        }
    }

    /// Number of keypoints.
    pub fn num_keypoints(&self) -> usize {
        match self {
            Self::MoveNetLightning | Self::MoveNetThunder | Self::MoveNetMultiPose => 17,
            Self::MediaPipePose => 33,
        }
    }

    /// Supports multiple people.
    pub fn multi_person(&self) -> bool {
        matches!(self, Self::MoveNetMultiPose)
    }
}

/// Pose estimation configuration.
#[derive(Debug, Clone)]
pub struct PoseConfig {
    /// Model to use.
    pub model: PoseModel,
    /// Custom model path.
    pub model_path: Option<PathBuf>,
    /// Keypoint confidence threshold.
    pub keypoint_threshold: f32,
    /// Person detection threshold (for multi-pose).
    pub person_threshold: f32,
    /// Maximum number of people to detect.
    pub max_persons: usize,
    /// Use GPU acceleration.
    pub use_gpu: bool,
    /// GPU device ID.
    pub gpu_device_id: i32,
}

impl Default for PoseConfig {
    fn default() -> Self {
        Self {
            model: PoseModel::default(),
            model_path: None,
            keypoint_threshold: 0.3,
            person_threshold: 0.3,
            max_persons: 6,
            use_gpu: true,
            gpu_device_id: 0,
        }
    }
}

impl PoseConfig {
    /// Create a new pose configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model.
    pub fn with_model(mut self, model: PoseModel) -> Self {
        self.model = model;
        self
    }

    /// Set keypoint threshold.
    pub fn with_keypoint_threshold(mut self, threshold: f32) -> Self {
        self.keypoint_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set maximum persons.
    pub fn with_max_persons(mut self, max: usize) -> Self {
        self.max_persons = max.max(1);
        self
    }

    /// Use CPU only.
    pub fn cpu_only(mut self) -> Self {
        self.use_gpu = false;
        self
    }

    /// Get effective model path.
    pub fn effective_model_path(&self) -> PathBuf {
        if let Some(ref path) = self.model_path {
            path.clone()
        } else {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."));
            home.join(".xeno-lib")
                .join("models")
                .join(self.model.model_filename())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PoseConfig::default();
        assert_eq!(config.model, PoseModel::MoveNetLightning);
        assert!(config.use_gpu);
    }

    #[test]
    fn test_model_properties() {
        let lightning = PoseModel::MoveNetLightning;
        assert_eq!(lightning.input_size(), (192, 192));
        assert_eq!(lightning.num_keypoints(), 17);
        assert!(!lightning.multi_person());

        let multipose = PoseModel::MoveNetMultiPose;
        assert!(multipose.multi_person());
    }

    #[test]
    fn test_config_builder() {
        let config = PoseConfig::new()
            .with_model(PoseModel::MoveNetThunder)
            .with_keypoint_threshold(0.5)
            .cpu_only();

        assert_eq!(config.model, PoseModel::MoveNetThunder);
        assert!((config.keypoint_threshold - 0.5).abs() < 0.001);
        assert!(!config.use_gpu);
    }
}
