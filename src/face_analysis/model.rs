//! Face analysis ONNX model sessions.

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;
use super::config::FaceAnalysisConfig;

/// Face analysis model session.
pub struct FaceAnalyzerSession {
    age_session: Option<Session>,
    gender_session: Option<Session>,
    emotion_session: Option<Session>,
    config: FaceAnalysisConfig,
}

impl FaceAnalyzerSession {
    /// Get configuration.
    pub fn config(&self) -> &FaceAnalysisConfig {
        &self.config
    }

    /// Run age estimation.
    pub fn estimate_age(&mut self, input: &Array4<f32>) -> Result<(f32, f32), TransformError> {
        let session = self.age_session.as_ref().ok_or_else(|| {
            TransformError::InferenceFailed {
                message: "Age model not loaded".to_string(),
            }
        })?;

        let tensor_ref = TensorRef::from_array_view(input.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("Failed to create tensor: {e}"),
            }
        })?;

        let outputs = session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("Age estimation failed: {e}"),
            })?;

        let output = outputs.iter().next().ok_or_else(|| {
            TransformError::InferenceFailed {
                message: "No output tensor".to_string(),
            }
        })?;

        let (_, data) = output
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("Failed to extract output: {e}"),
            })?;

        let data: Vec<f32> = data.iter().copied().collect();

        // Assuming output is [age, confidence] or just [age]
        let age = data.first().copied().unwrap_or(25.0);
        let confidence = data.get(1).copied().unwrap_or(1.0);

        Ok((age.clamp(0.0, 100.0), confidence))
    }

    /// Run gender classification.
    pub fn classify_gender(&mut self, input: &Array4<f32>) -> Result<(f32, f32), TransformError> {
        let session = self.gender_session.as_ref().ok_or_else(|| {
            TransformError::InferenceFailed {
                message: "Gender model not loaded".to_string(),
            }
        })?;

        let tensor_ref = TensorRef::from_array_view(input.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("Failed to create tensor: {e}"),
            }
        })?;

        let outputs = session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("Gender classification failed: {e}"),
            })?;

        let output = outputs.iter().next().ok_or_else(|| {
            TransformError::InferenceFailed {
                message: "No output tensor".to_string(),
            }
        })?;

        let (_, data) = output
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("Failed to extract output: {e}"),
            })?;

        let data: Vec<f32> = data.iter().copied().collect();

        // Binary classification: male probability
        let male_score = data.first().copied().unwrap_or(0.5);
        let confidence = (male_score - 0.5).abs() * 2.0; // 0 at 0.5, 1 at extremes

        Ok((male_score, confidence))
    }

    /// Run emotion recognition.
    pub fn recognize_emotion(&mut self, input: &Array4<f32>) -> Result<Vec<f32>, TransformError> {
        let session = self.emotion_session.as_ref().ok_or_else(|| {
            TransformError::InferenceFailed {
                message: "Emotion model not loaded".to_string(),
            }
        })?;

        let tensor_ref = TensorRef::from_array_view(input.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("Failed to create tensor: {e}"),
            }
        })?;

        let outputs = session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("Emotion recognition failed: {e}"),
            })?;

        let output = outputs.iter().next().ok_or_else(|| {
            TransformError::InferenceFailed {
                message: "No output tensor".to_string(),
            }
        })?;

        let (_, data) = output
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("Failed to extract output: {e}"),
            })?;

        let scores: Vec<f32> = data.iter().copied().collect();

        // Apply softmax if not already applied
        let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scores.iter().map(|&x| (x - max_val).exp()).sum();
        let softmax: Vec<f32> = scores.iter().map(|&x| (x - max_val).exp() / exp_sum).collect();

        Ok(softmax)
    }
}

/// Load face analysis models.
pub fn load_analyzer(config: &FaceAnalysisConfig) -> Result<FaceAnalyzerSession, TransformError> {
    let build_session = |path: &std::path::Path| -> Result<Session, TransformError> {
        if !path.exists() {
            return Err(TransformError::ModelNotFound {
                path: path.display().to_string(),
            });
        }

        let mut builder = Session::builder()
            .map_err(|e| TransformError::ModelLoadFailed {
                message: format!("Failed to create session builder: {e}"),
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| TransformError::ModelLoadFailed {
                message: format!("Failed to set optimization: {e}"),
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
                    message: format!("Failed to configure GPU: {e}"),
                })?;
        } else {
            builder = builder
                .with_execution_providers([CPUExecutionProvider::default().build()])
                .map_err(|e| TransformError::ModelLoadFailed {
                    message: format!("Failed to configure CPU: {e}"),
                })?;
        }

        builder.commit_from_file(path).map_err(|e| TransformError::ModelLoadFailed {
            message: format!("Failed to load model: {e}"),
        })
    };

    let age_session = if config.estimate_age {
        Some(build_session(&config.age_model_path())?)
    } else {
        None
    };

    let gender_session = if config.classify_gender {
        Some(build_session(&config.gender_model_path())?)
    } else {
        None
    };

    let emotion_session = if config.recognize_emotion {
        Some(build_session(&config.emotion_model_path())?)
    } else {
        None
    };

    Ok(FaceAnalyzerSession {
        age_session,
        gender_session,
        emotion_session,
        config: config.clone(),
    })
}
