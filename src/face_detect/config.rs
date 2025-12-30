//! Configuration for face detection.

use std::path::PathBuf;

/// Available face detection models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FaceDetectModel {
    /// SCRFD - Sample and Computation Redistribution for Face Detection.
    #[default]
    Scrfd,

    /// RetinaFace - High accuracy face detection.
    RetinaFace,

    /// YuNet - Lightweight face detection.
    YuNet,
}

impl FaceDetectModel {
    /// Returns the default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            FaceDetectModel::Scrfd => "scrfd_10g.onnx",
            FaceDetectModel::RetinaFace => "retinaface.onnx",
            FaceDetectModel::YuNet => "yunet.onnx",
        }
    }

    /// Returns expected input size.
    pub fn input_size(&self) -> (u32, u32) {
        match self {
            FaceDetectModel::Scrfd => (640, 640),
            FaceDetectModel::RetinaFace => (640, 640),
            FaceDetectModel::YuNet => (320, 320),
        }
    }
}

/// Configuration for face detection.
#[derive(Debug, Clone)]
pub struct FaceDetectConfig {
    /// The detection model to use.
    pub model: FaceDetectModel,

    /// Path to the ONNX model file.
    pub model_path: Option<PathBuf>,

    /// Whether to use GPU acceleration.
    pub use_gpu: bool,

    /// CUDA device ID.
    pub gpu_device_id: i32,

    /// Detection confidence threshold (0.0 - 1.0).
    pub confidence_threshold: f32,

    /// NMS IoU threshold.
    pub nms_threshold: f32,

    /// Maximum number of faces to detect.
    pub max_faces: usize,

    /// Whether to detect facial landmarks.
    pub detect_landmarks: bool,
}

impl Default for FaceDetectConfig {
    fn default() -> Self {
        Self {
            model: FaceDetectModel::default(),
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            confidence_threshold: 0.5,
            nms_threshold: 0.4,
            max_faces: 100,
            detect_landmarks: true,
        }
    }
}

impl FaceDetectConfig {
    /// Create a new configuration with the specified model.
    pub fn new(model: FaceDetectModel) -> Self {
        Self {
            model,
            ..Default::default()
        }
    }

    /// Set the model path.
    pub fn with_model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Enable or disable GPU.
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set confidence threshold.
    pub fn with_confidence(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Get effective model path.
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
    { std::env::var("USERPROFILE").ok().map(PathBuf::from) }
    #[cfg(not(windows))]
    { std::env::var("HOME").ok().map(PathBuf::from) }
}
