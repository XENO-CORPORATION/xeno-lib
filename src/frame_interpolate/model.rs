//! ONNX model session for frame interpolation.

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;
use super::config::InterpolationConfig;

/// A loaded frame interpolation model session.
pub struct InterpolatorSession {
    session: Session,
    config: InterpolationConfig,
}

impl InterpolatorSession {
    pub fn config(&self) -> &InterpolationConfig {
        &self.config
    }

    /// Runs frame interpolation inference.
    /// Takes two frames and a timestep (0.0 to 1.0) and returns the interpolated frame.
    pub fn run(
        &mut self,
        frame0: &Array4<f32>,
        frame1: &Array4<f32>,
        timestep: f32,
    ) -> Result<Array4<f32>, TransformError> {
        let tensor0 = TensorRef::from_array_view(frame0.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to create tensor ref for frame0: {e}"),
            }
        })?;

        let tensor1 = TensorRef::from_array_view(frame1.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to create tensor ref for frame1: {e}"),
            }
        })?;

        // Create timestep tensor
        let timestep_arr = ndarray::arr1(&[timestep]);
        let timestep_tensor = TensorRef::from_array_view(timestep_arr.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to create timestep tensor: {e}"),
            }
        })?;

        let outputs = self
            .session
            .run(ort::inputs![tensor0, tensor1, timestep_tensor])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("frame interpolation inference failed: {e}"),
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

        Array4::from_shape_vec((dims[0], dims[1], dims[2], dims[3]), flat_data).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to reshape output: {e}"),
            }
        })
    }

    /// Runs interpolation for a simple midpoint (timestep = 0.5).
    pub fn run_midpoint(
        &mut self,
        frame0: &Array4<f32>,
        frame1: &Array4<f32>,
    ) -> Result<Array4<f32>, TransformError> {
        self.run(frame0, frame1, 0.5)
    }
}

/// Loads a frame interpolation model.
pub fn load_interpolator(config: &InterpolationConfig) -> Result<InterpolatorSession, TransformError> {
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
            message: format!("failed to load interpolation model: {e}"),
        })?;

    Ok(InterpolatorSession {
        session,
        config: config.clone(),
    })
}
