//! Configuration for depth estimation.

use std::path::PathBuf;

/// Available depth estimation models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DepthModel {
    /// MiDaS v3.1 Large - High quality depth estimation.
    #[default]
    MidasLarge,

    /// MiDaS v3.1 Small - Fast depth estimation.
    MidasSmall,

    /// DPT-Large - Dense Prediction Transformer.
    DptLarge,

    /// Depth Anything - State of the art depth.
    DepthAnything,
}

impl DepthModel {
    /// Returns the default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            DepthModel::MidasLarge => "midas_v31_large.onnx",
            DepthModel::MidasSmall => "midas_v31_small.onnx",
            DepthModel::DptLarge => "dpt_large.onnx",
            DepthModel::DepthAnything => "depth_anything.onnx",
        }
    }

    /// Returns expected input size.
    pub fn input_size(&self) -> (u32, u32) {
        match self {
            DepthModel::MidasLarge => (384, 384),
            DepthModel::MidasSmall => (256, 256),
            DepthModel::DptLarge => (384, 384),
            DepthModel::DepthAnything => (518, 518),
        }
    }
}

/// Configuration for depth estimation.
#[derive(Debug, Clone)]
pub struct DepthConfig {
    /// The depth model to use.
    pub model: DepthModel,

    /// Path to the ONNX model file.
    pub model_path: Option<PathBuf>,

    /// Whether to use GPU acceleration.
    pub use_gpu: bool,

    /// CUDA device ID.
    pub gpu_device_id: i32,

    /// Whether to normalize output to 0-1 range.
    pub normalize_output: bool,

    /// Whether to invert depth (closer = darker).
    pub invert_depth: bool,
}

impl Default for DepthConfig {
    fn default() -> Self {
        Self {
            model: DepthModel::default(),
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            normalize_output: true,
            invert_depth: false,
        }
    }
}

impl DepthConfig {
    /// Create a new configuration with the specified model.
    pub fn new(model: DepthModel) -> Self {
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

    /// Set whether to invert depth.
    pub fn with_inverted(mut self, invert: bool) -> Self {
        self.invert_depth = invert;
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
