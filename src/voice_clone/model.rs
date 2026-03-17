//! ONNX model session for voice cloning / TTS inference.

use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
};

use crate::error::TransformError;
use super::config::VoiceCloneConfig;

/// A loaded ONNX model session for voice synthesis.
///
/// # Thread Safety
///
/// Requires `&mut self` for inference. Wrap in `Mutex` for shared access.
pub struct VoiceCloneSession {
    #[allow(dead_code)]
    session: Session,
    config: VoiceCloneConfig,
}

impl VoiceCloneSession {
    /// Returns the configuration.
    pub fn config(&self) -> &VoiceCloneConfig {
        &self.config
    }

    /// Returns the output sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

/// Loads a voice cloning / TTS model.
///
/// # Arguments
///
/// * `config` - Configuration specifying model and execution options.
///
/// # Returns
///
/// A `VoiceCloneSession` ready for synthesis.
pub fn load_voice_model(config: &VoiceCloneConfig) -> Result<VoiceCloneSession, TransformError> {
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
            message: format!("failed to load voice model from {}: {e}", model_path.display()),
        })?;

    Ok(VoiceCloneSession {
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
        let config = VoiceCloneConfig::default()
            .with_model_path(PathBuf::from("/nonexistent/model.onnx"));
        let result = load_voice_model(&config);
        assert!(matches!(result, Err(TransformError::ModelNotFound { .. })));
    }
}
