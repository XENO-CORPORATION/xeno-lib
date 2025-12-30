//! ONNX model session management for background removal.

use std::path::PathBuf;

use ndarray::{Array2, Array4};
use ort::{
    execution_providers::{CUDAExecutionProvider, CPUExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;

/// Configuration for background removal model loading and inference.
#[derive(Debug, Clone)]
pub struct BackgroundRemovalConfig {
    /// Path to the ONNX model file.
    /// Default: `~/.xeno-lib/models/rmbg-1.4.onnx`
    pub model_path: PathBuf,

    /// Whether to attempt GPU (CUDA) acceleration.
    /// Falls back to CPU if CUDA is unavailable.
    pub use_gpu: bool,

    /// CUDA device ID when using GPU acceleration.
    pub gpu_device_id: i32,

    /// Confidence threshold for foreground detection (0.0 - 1.0).
    /// Pixels with confidence above this threshold are kept as foreground.
    pub confidence_threshold: f32,
}

impl Default for BackgroundRemovalConfig {
    fn default() -> Self {
        Self {
            model_path: default_model_path(),
            use_gpu: true,
            gpu_device_id: 0,
            confidence_threshold: 0.5,
        }
    }
}

/// Returns the default model path based on the user's home directory.
fn default_model_path() -> PathBuf {
    let home = dirs_next().unwrap_or_else(|| PathBuf::from("."));
    home.join(".xeno-lib").join("models").join("rmbg-1.4.onnx")
}

/// Cross-platform home directory detection.
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

/// A loaded ONNX model session ready for inference.
///
/// This struct wraps an `ort::Session` and provides a convenient interface
/// for running background removal inference.
pub struct ModelSession {
    session: Session,
    input_size: (u32, u32),
    config: BackgroundRemovalConfig,
}

impl ModelSession {
    /// Returns the expected input dimensions (width, height).
    pub fn input_size(&self) -> (u32, u32) {
        self.input_size
    }

    /// Returns a reference to the configuration used to create this session.
    pub fn config(&self) -> &BackgroundRemovalConfig {
        &self.config
    }

    /// Runs inference on the provided input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - A 4D tensor of shape `[1, 3, H, W]` containing the
    ///   preprocessed image data.
    ///
    /// # Returns
    ///
    /// A 2D array of shape `[H, W]` containing the predicted alpha mask.
    pub fn run(&mut self, input: &Array4<f32>) -> Result<Array2<f32>, TransformError> {
        // Create tensor reference from input array
        let tensor_ref = TensorRef::from_array_view(input.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to create tensor ref: {e}"),
            }
        })?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("inference failed: {e}"),
            })?;

        // Extract output mask
        // RMBG-1.4 output shape: [1, 1, H, W]
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

        // Convert from [1, 1, H, W] to [H, W]
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let (h, w) = if dims.len() == 4 {
            (dims[2], dims[3])
        } else if dims.len() == 3 {
            (dims[1], dims[2])
        } else {
            (dims[0], dims[1])
        };

        let flat_data: Vec<f32> = data.iter().copied().collect();
        Array2::from_shape_vec((h, w), flat_data).map_err(|e| TransformError::InferenceFailed {
            message: format!("failed to reshape output: {e}"),
        })
    }
}

/// Loads an ONNX model for background removal.
///
/// This function initializes an ONNX Runtime session with the specified
/// configuration. It attempts to use CUDA execution if `use_gpu` is true,
/// falling back to CPU execution if CUDA is unavailable.
///
/// # Arguments
///
/// * `config` - Configuration specifying model path and execution options.
///
/// # Returns
///
/// A `ModelSession` ready for inference.
///
/// # Errors
///
/// Returns an error if:
/// - The model file does not exist
/// - The model cannot be loaded (invalid format, incompatible version, etc.)
///
/// # Example
///
/// ```rust,no_run
/// use xeno_lib::background::{load_model, BackgroundRemovalConfig};
///
/// // Use default configuration (CUDA with CPU fallback)
/// let session = load_model(&BackgroundRemovalConfig::default())?;
///
/// // Custom configuration
/// let config = BackgroundRemovalConfig {
///     model_path: "/path/to/custom/model.onnx".into(),
///     use_gpu: false,  // CPU only
///     ..Default::default()
/// };
/// let session = load_model(&config)?;
/// # Ok::<(), xeno_lib::TransformError>(())
/// ```
pub fn load_model(config: &BackgroundRemovalConfig) -> Result<ModelSession, TransformError> {
    // Verify model file exists
    if !config.model_path.exists() {
        return Err(TransformError::ModelNotFound {
            path: config.model_path.display().to_string(),
        });
    }

    // Build session with execution providers
    let mut builder = Session::builder()
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to create session builder: {e}"),
        })?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to set optimization level: {e}"),
        })?;

    // Configure execution providers
    if config.use_gpu {
        // Try CUDA first, fall back to CPU
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
        // CPU only
        builder = builder
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| TransformError::ModelLoadFailed {
                message: format!("failed to configure CPU execution provider: {e}"),
            })?;
    }

    // Load the model
    let session = builder
        .commit_from_file(&config.model_path)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to load model: {e}"),
        })?;

    // RMBG-1.4 uses 1024x1024 input
    let input_size = (1024, 1024);

    Ok(ModelSession {
        session,
        input_size,
        config: config.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BackgroundRemovalConfig::default();
        assert!(config.use_gpu);
        assert_eq!(config.gpu_device_id, 0);
        assert!((config.confidence_threshold - 0.5).abs() < f32::EPSILON);
        assert!(config.model_path.to_string_lossy().contains("rmbg-1.4.onnx"));
    }

    #[test]
    fn test_model_not_found() {
        let config = BackgroundRemovalConfig {
            model_path: PathBuf::from("/nonexistent/model.onnx"),
            ..Default::default()
        };
        let result = load_model(&config);
        assert!(matches!(result, Err(TransformError::ModelNotFound { .. })));
    }
}
