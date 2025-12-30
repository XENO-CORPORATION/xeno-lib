//! ONNX model session for depth estimation.

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;
use super::config::DepthConfig;

/// A loaded depth estimation model session.
pub struct DepthSession {
    session: Session,
    config: DepthConfig,
}

impl DepthSession {
    pub fn config(&self) -> &DepthConfig {
        &self.config
    }

    pub fn input_size(&self) -> (u32, u32) {
        self.config.model.input_size()
    }

    /// Runs depth estimation inference.
    pub fn run(&mut self, input: &Array4<f32>) -> Result<ndarray::Array2<f32>, TransformError> {
        let tensor_ref = TensorRef::from_array_view(input.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to create tensor ref: {e}"),
            }
        })?;

        let outputs = self
            .session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("depth estimation inference failed: {e}"),
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
        let flat_data: Vec<f32> = data.iter().copied().collect();

        // Depth output is typically [1, H, W] or [1, 1, H, W]
        let (height, width) = if dims.len() == 4 {
            (dims[2], dims[3])
        } else if dims.len() == 3 {
            (dims[1], dims[2])
        } else {
            return Err(TransformError::InferenceFailed {
                message: format!("unexpected depth output shape: {:?}", dims),
            });
        };

        ndarray::Array2::from_shape_vec((height, width), flat_data).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to reshape depth output: {e}"),
            }
        })
    }
}

/// Loads a depth estimation model.
pub fn load_depth_estimator(config: &DepthConfig) -> Result<DepthSession, TransformError> {
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
                message: format!("failed to configure CPU: {e}"),
            })?;
    }

    let session = builder
        .commit_from_file(&model_path)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to load depth model: {e}"),
        })?;

    Ok(DepthSession {
        session,
        config: config.clone(),
    })
}
