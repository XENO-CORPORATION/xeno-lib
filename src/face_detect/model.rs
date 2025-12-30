//! ONNX model session for face detection.

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;
use super::config::FaceDetectConfig;

/// A loaded face detection model session.
pub struct FaceDetectorSession {
    session: Session,
    config: FaceDetectConfig,
}

impl FaceDetectorSession {
    pub fn config(&self) -> &FaceDetectConfig {
        &self.config
    }

    pub fn input_size(&self) -> (u32, u32) {
        self.config.model.input_size()
    }

    /// Runs face detection inference.
    pub fn run(&mut self, input: &Array4<f32>) -> Result<Vec<RawDetection>, TransformError> {
        let tensor_ref = TensorRef::from_array_view(input.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to create tensor ref: {e}"),
            }
        })?;

        let confidence_threshold = self.config.confidence_threshold;
        let detect_landmarks = self.config.detect_landmarks;

        let outputs = self
            .session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("face detection inference failed: {e}"),
            })?;

        // Parse outputs - structure varies by model
        parse_outputs(&outputs, confidence_threshold, detect_landmarks)
    }
}

fn parse_outputs(
    outputs: &ort::session::SessionOutputs,
    confidence_threshold: f32,
    detect_landmarks: bool,
) -> Result<Vec<RawDetection>, TransformError> {
    let mut detections = Vec::new();

    // Get first output which typically contains bounding boxes
    if let Some(output) = outputs.iter().next() {
        let (shape, data) = output
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("failed to extract output tensor: {e}"),
            })?;

        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let data: Vec<f32> = data.iter().copied().collect();

        // Parse based on shape - typical format is [batch, num_detections, 5+]
        // where each detection has [x1, y1, x2, y2, score, ...]
        if shape.len() >= 2 {
            let num_detections = if shape.len() == 3 { shape[1] } else { shape[0] };
            let stride = if shape.len() == 3 { shape[2] } else { shape[1] };

            for i in 0..num_detections {
                if stride >= 5 {
                    let offset = i * stride;
                    let score = data.get(offset + 4).copied().unwrap_or(0.0);

                    if score >= confidence_threshold {
                        detections.push(RawDetection {
                            x1: data.get(offset).copied().unwrap_or(0.0),
                            y1: data.get(offset + 1).copied().unwrap_or(0.0),
                            x2: data.get(offset + 2).copied().unwrap_or(0.0),
                            y2: data.get(offset + 3).copied().unwrap_or(0.0),
                            score,
                            landmarks: if stride > 5 && detect_landmarks {
                                // Extract 5-point landmarks if available
                                let mut lm = Vec::new();
                                for j in 0..5 {
                                    let lx = data.get(offset + 5 + j * 2).copied();
                                    let ly = data.get(offset + 6 + j * 2).copied();
                                    if let (Some(x), Some(y)) = (lx, ly) {
                                        lm.push((x, y));
                                    }
                                }
                                if lm.len() == 5 { Some(lm) } else { None }
                            } else {
                                None
                            },
                        });
                    }
                }
            }
        }
    }

    Ok(detections)
}

/// Raw detection result before post-processing.
#[derive(Debug, Clone)]
pub struct RawDetection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub landmarks: Option<Vec<(f32, f32)>>,
}

/// Loads a face detection model.
pub fn load_detector(config: &FaceDetectConfig) -> Result<FaceDetectorSession, TransformError> {
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
            message: format!("failed to load face detection model: {e}"),
        })?;

    Ok(FaceDetectorSession {
        session,
        config: config.clone(),
    })
}
