//! ONNX model session management for 3D mesh generation.

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;
use super::config::Mesh3DConfig;

/// A loaded ONNX model session for 3D mesh generation.
///
/// # Thread Safety
///
/// Requires `&mut self` for `run()`, enforcing single-threaded access.
/// For multi-threaded workloads, create one session per thread or wrap in `Mutex`.
pub struct Mesh3DSession {
    session: Session,
    config: Mesh3DConfig,
}

impl Mesh3DSession {
    /// Returns a reference to the configuration.
    pub fn config(&self) -> &Mesh3DConfig {
        &self.config
    }

    /// Runs inference on the input image tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - A 4D tensor [1, 3, H, W] with pixel values in [0, 1].
    ///
    /// # Returns
    ///
    /// A 4D tensor containing the predicted 3D volume/triplane features.
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
                message: format!("3D mesh inference failed: {e}"),
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

/// Loads an ONNX model for 3D mesh generation.
///
/// # Arguments
///
/// * `config` - Configuration specifying model and execution options.
///
/// # Returns
///
/// A `Mesh3DSession` ready for inference.
///
/// # Errors
///
/// Returns error if model file doesn't exist or can't be loaded.
pub fn load_mesh_model(config: &Mesh3DConfig) -> Result<Mesh3DSession, TransformError> {
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
            message: format!("failed to load 3D model from {}: {e}", model_path.display()),
        })?;

    Ok(Mesh3DSession {
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
        let config = Mesh3DConfig::default()
            .with_model_path(PathBuf::from("/nonexistent/model.onnx"));
        let result = load_mesh_model(&config);
        assert!(matches!(result, Err(TransformError::ModelNotFound { .. })));
    }
}
