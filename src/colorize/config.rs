//! Configuration for image colorization.

use std::path::PathBuf;

/// Available colorization models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorizeModel {
    /// DDColor - State-of-the-art dual decoder colorization.
    #[default]
    DDColor,

    /// DeOldify - Classic colorization model.
    DeOldify,
}

impl ColorizeModel {
    /// Returns the default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            ColorizeModel::DDColor => "ddcolor.onnx",
            ColorizeModel::DeOldify => "deoldify.onnx",
        }
    }

    /// Returns expected input size.
    pub fn input_size(&self) -> (u32, u32) {
        match self {
            ColorizeModel::DDColor => (512, 512),
            ColorizeModel::DeOldify => (256, 256),
        }
    }
}

/// Configuration for colorization.
#[derive(Debug, Clone)]
pub struct ColorizeConfig {
    /// The colorization model to use.
    pub model: ColorizeModel,

    /// Path to the ONNX model file.
    pub model_path: Option<PathBuf>,

    /// Whether to use GPU acceleration.
    pub use_gpu: bool,

    /// CUDA device ID.
    pub gpu_device_id: i32,

    /// Color saturation adjustment (0.0 - 2.0, 1.0 = no change).
    pub saturation: f32,

    /// Whether to preserve original luminance.
    pub preserve_luminance: bool,
}

impl Default for ColorizeConfig {
    fn default() -> Self {
        Self {
            model: ColorizeModel::default(),
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            saturation: 1.0,
            preserve_luminance: true,
        }
    }
}

impl ColorizeConfig {
    /// Create a new configuration with the specified model.
    pub fn new(model: ColorizeModel) -> Self {
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

    /// Set saturation adjustment.
    pub fn with_saturation(mut self, saturation: f32) -> Self {
        self.saturation = saturation.clamp(0.0, 2.0);
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
