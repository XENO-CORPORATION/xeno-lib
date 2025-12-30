//! ONNX model session for speech transcription.

use ndarray::Array3;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;
use super::config::TranscribeConfig;

/// A loaded transcription model session.
pub struct TranscriberSession {
    session: Session,
    config: TranscribeConfig,
}

impl TranscriberSession {
    pub fn config(&self) -> &TranscribeConfig {
        &self.config
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.model.sample_rate()
    }

    /// Runs transcription inference on mel spectrogram input.
    pub fn run(&mut self, mel_spectrogram: &Array3<f32>) -> Result<Vec<i64>, TransformError> {
        let tensor_ref = TensorRef::from_array_view(mel_spectrogram.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to create tensor ref: {e}"),
            }
        })?;

        let outputs = self
            .session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("transcription inference failed: {e}"),
            })?;

        let output_tensor = outputs.iter().next().ok_or_else(|| {
            TransformError::InferenceFailed {
                message: "no output tensor found".to_string(),
            }
        })?;

        let (_shape, data) = output_tensor
            .1
            .try_extract_tensor::<i64>()
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("failed to extract output tensor: {e}"),
            })?;

        Ok(data.iter().copied().collect())
    }
}

/// Loads a transcription model.
pub fn load_transcriber(config: &TranscribeConfig) -> Result<TranscriberSession, TransformError> {
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
            message: format!("failed to load transcription model: {e}"),
        })?;

    Ok(TranscriberSession {
        session,
        config: config.clone(),
    })
}
