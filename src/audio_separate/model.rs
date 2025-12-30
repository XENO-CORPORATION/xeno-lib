//! ONNX model session for audio source separation.

use ndarray::Array3;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;
use super::config::SeparationConfig;

/// A loaded audio separation model session.
pub struct SeparatorSession {
    session: Session,
    config: SeparationConfig,
}

impl SeparatorSession {
    pub fn config(&self) -> &SeparationConfig {
        &self.config
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.model.sample_rate()
    }

    /// Runs separation inference on a chunk of audio.
    /// Input shape: [batch, channels, samples]
    /// Output shape: [batch, stems, channels, samples]
    pub fn run(&mut self, input: &Array3<f32>) -> Result<ndarray::Array4<f32>, TransformError> {
        let tensor_ref = TensorRef::from_array_view(input.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to create tensor ref: {e}"),
            }
        })?;

        let outputs = self
            .session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("audio separation inference failed: {e}"),
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

        if dims.len() == 4 {
            ndarray::Array4::from_shape_vec(
                (dims[0], dims[1], dims[2], dims[3]),
                flat_data,
            ).map_err(|e| TransformError::InferenceFailed {
                message: format!("failed to reshape output: {e}"),
            })
        } else {
            Err(TransformError::InferenceFailed {
                message: format!("unexpected output shape: {:?}", dims),
            })
        }
    }
}

/// Loads an audio separation model.
pub fn load_separator(config: &SeparationConfig) -> Result<SeparatorSession, TransformError> {
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
            message: format!("failed to load separation model: {e}"),
        })?;

    Ok(SeparatorSession {
        session,
        config: config.clone(),
    })
}
