//! ONNX model session for video generation.

use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
};

use crate::error::TransformError;
use super::config::VideoGenConfig;

/// A loaded ONNX session for video generation.
pub struct VideoGenSession {
    #[allow(dead_code)]
    session: Session,
    config: VideoGenConfig,
}

impl VideoGenSession {
    /// Returns the configuration.
    pub fn config(&self) -> &VideoGenConfig {
        &self.config
    }
}

/// Loads a video generation model.
pub fn load_video_gen_model(config: &VideoGenConfig) -> Result<VideoGenSession, TransformError> {
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
            message: format!("failed to load video gen model from {}: {e}", model_path.display()),
        })?;

    Ok(VideoGenSession {
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
        let config = VideoGenConfig::default()
            .with_model_path(PathBuf::from("/nonexistent/model.onnx"));
        let result = load_video_gen_model(&config);
        assert!(matches!(result, Err(TransformError::ModelNotFound { .. })));
    }
}
