//! Configuration for frame interpolation.

use std::path::PathBuf;

/// Available frame interpolation models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationModel {
    /// RIFE v4.6 - Real-time Intermediate Flow Estimation.
    #[default]
    RifeV4,

    /// RIFE v4 HD - Higher quality for large frames.
    RifeV4HD,

    /// FILM - Frame Interpolation for Large Motion.
    Film,
}

impl InterpolationModel {
    /// Returns the default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            InterpolationModel::RifeV4 => "rife-v4.6.onnx",
            InterpolationModel::RifeV4HD => "rife-v4-hd.onnx",
            InterpolationModel::Film => "film.onnx",
        }
    }

    /// Returns expected input size (0 = dynamic).
    pub fn input_size(&self) -> (u32, u32) {
        match self {
            InterpolationModel::RifeV4 => (0, 0),    // Dynamic
            InterpolationModel::RifeV4HD => (0, 0),  // Dynamic
            InterpolationModel::Film => (0, 0),      // Dynamic
        }
    }

    /// Whether the model supports arbitrary timesteps.
    pub fn supports_arbitrary_timestep(&self) -> bool {
        match self {
            InterpolationModel::RifeV4 => true,
            InterpolationModel::RifeV4HD => true,
            InterpolationModel::Film => true,
        }
    }
}

/// Configuration for frame interpolation.
#[derive(Debug, Clone)]
pub struct InterpolationConfig {
    /// The interpolation model to use.
    pub model: InterpolationModel,

    /// Path to the ONNX model file.
    pub model_path: Option<PathBuf>,

    /// Whether to use GPU acceleration.
    pub use_gpu: bool,

    /// CUDA device ID.
    pub gpu_device_id: i32,

    /// Multiplier for frame rate (2 = double, 4 = quadruple).
    pub multiplier: u32,

    /// Whether to use scene detection to avoid interpolating across cuts.
    pub scene_detection: bool,

    /// Scene detection threshold (0.0 - 1.0).
    pub scene_threshold: f32,

    /// Whether to use ensemble mode for higher quality.
    pub ensemble: bool,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            model: InterpolationModel::default(),
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            multiplier: 2,
            scene_detection: true,
            scene_threshold: 0.3,
            ensemble: false,
        }
    }
}

impl InterpolationConfig {
    /// Create a new configuration with the specified model.
    pub fn new(model: InterpolationModel) -> Self {
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

    /// Set frame rate multiplier.
    pub fn with_multiplier(mut self, multiplier: u32) -> Self {
        self.multiplier = multiplier.max(2);
        self
    }

    /// Enable or disable scene detection.
    pub fn with_scene_detection(mut self, enabled: bool) -> Self {
        self.scene_detection = enabled;
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
