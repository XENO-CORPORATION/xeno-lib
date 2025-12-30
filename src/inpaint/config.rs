//! Configuration for image inpainting/object removal.

use std::path::PathBuf;

/// Available inpainting models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InpaintModel {
    /// LaMa - Large Mask Inpainting with Fourier Convolutions.
    #[default]
    LaMa,

    /// MAT - Mask-Aware Transformer for Large Hole Inpainting.
    Mat,
}

impl InpaintModel {
    /// Returns the default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            InpaintModel::LaMa => "lama.onnx",
            InpaintModel::Mat => "mat.onnx",
        }
    }

    /// Returns expected input size (0 = dynamic).
    pub fn input_size(&self) -> (u32, u32) {
        match self {
            InpaintModel::LaMa => (512, 512),
            InpaintModel::Mat => (512, 512),
        }
    }
}

/// Configuration for inpainting.
#[derive(Debug, Clone)]
pub struct InpaintConfig {
    /// The inpainting model to use.
    pub model: InpaintModel,

    /// Path to the ONNX model file.
    pub model_path: Option<PathBuf>,

    /// Whether to use GPU acceleration.
    pub use_gpu: bool,

    /// CUDA device ID.
    pub gpu_device_id: i32,

    /// Whether to dilate the mask slightly for better results.
    pub dilate_mask: bool,

    /// Mask dilation radius in pixels.
    pub dilation_radius: u32,
}

impl Default for InpaintConfig {
    fn default() -> Self {
        Self {
            model: InpaintModel::default(),
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            dilate_mask: true,
            dilation_radius: 3,
        }
    }
}

impl InpaintConfig {
    /// Create a new configuration with the specified model.
    pub fn new(model: InpaintModel) -> Self {
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

    /// Enable or disable mask dilation.
    pub fn with_mask_dilation(mut self, dilate: bool, radius: u32) -> Self {
        self.dilate_mask = dilate;
        self.dilation_radius = radius;
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
