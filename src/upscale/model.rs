//! ONNX model session management for Real-ESRGAN upscaling.

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;

use super::config::UpscaleConfig;

/// A loaded ONNX model session ready for upscaling inference.
///
/// This struct wraps an `ort::Session` and provides a convenient interface
/// for running Real-ESRGAN inference.
pub struct UpscalerSession {
    session: Session,
    config: UpscaleConfig,
}

impl UpscalerSession {
    /// Returns the scale factor for this upscaler.
    pub fn scale_factor(&self) -> u32 {
        self.config.scale_factor()
    }

    /// Returns a reference to the configuration used to create this session.
    pub fn config(&self) -> &UpscaleConfig {
        &self.config
    }

    /// Runs inference on the provided input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - A 4D tensor of shape `[1, 3, H, W]` containing the
    ///   preprocessed image data in range [0, 1].
    ///
    /// # Returns
    ///
    /// A 4D tensor of shape `[1, 3, H*scale, W*scale]` containing the
    /// upscaled image data in range [0, 1].
    pub fn run(&mut self, input: &Array4<f32>) -> Result<Array4<f32>, TransformError> {
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
                message: format!("upscale inference failed: {e}"),
            })?;

        // Extract output tensor
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

        // Real-ESRGAN output shape: [1, 3, H*scale, W*scale]
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

/// Loads an ONNX model for upscaling.
///
/// This function initializes an ONNX Runtime session with the specified
/// configuration. It attempts to use CUDA execution if `use_gpu` is true,
/// falling back to CPU execution if CUDA is unavailable.
///
/// # Arguments
///
/// * `config` - Configuration specifying model and execution options.
///
/// # Returns
///
/// An `UpscalerSession` ready for inference.
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
/// use xeno_lib::upscale::{load_upscaler, UpscaleConfig, UpscaleModel};
///
/// // Use default configuration (4x upscale, CUDA with CPU fallback)
/// let session = load_upscaler(&UpscaleConfig::default())?;
///
/// // Custom configuration for anime upscaling
/// let config = UpscaleConfig::new(UpscaleModel::RealEsrganX4Anime)
///     .with_gpu(true)
///     .with_tile_size(128);
/// let session = load_upscaler(&config)?;
/// # Ok::<(), xeno_lib::TransformError>(())
/// ```
pub fn load_upscaler(config: &UpscaleConfig) -> Result<UpscalerSession, TransformError> {
    let model_path = config.effective_model_path();

    // Verify model file exists
    if !model_path.exists() {
        return Err(TransformError::ModelNotFound {
            path: model_path.display().to_string(),
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
        .commit_from_file(&model_path)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to load upscale model from {}: {e}", model_path.display()),
        })?;

    Ok(UpscalerSession {
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
        let config = UpscaleConfig::default()
            .with_model_path(PathBuf::from("/nonexistent/model.onnx"));
        let result = load_upscaler(&config);
        assert!(matches!(result, Err(TransformError::ModelNotFound { .. })));
    }
}
