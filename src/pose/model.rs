//! Pose estimation ONNX model session.

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;
use super::config::PoseConfig;

/// A detected keypoint.
#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    /// X coordinate (normalized 0-1).
    pub x: f32,
    /// Y coordinate (normalized 0-1).
    pub y: f32,
    /// Confidence score.
    pub confidence: f32,
}

impl Keypoint {
    /// Convert to pixel coordinates.
    pub fn to_pixel(&self, width: u32, height: u32) -> (u32, u32) {
        (
            (self.x * width as f32).round() as u32,
            (self.y * height as f32).round() as u32,
        )
    }

    /// Check if keypoint is visible (above threshold).
    pub fn is_visible(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

/// A detected person's pose.
#[derive(Debug, Clone)]
pub struct DetectedPose {
    /// Keypoints (17 for COCO, 33 for MediaPipe).
    pub keypoints: Vec<Keypoint>,
    /// Overall pose confidence.
    pub confidence: f32,
    /// Bounding box (x, y, width, height) normalized.
    pub bbox: Option<(f32, f32, f32, f32)>,
}

impl DetectedPose {
    /// Get a specific keypoint by index.
    pub fn get_keypoint(&self, index: usize) -> Option<&Keypoint> {
        self.keypoints.get(index)
    }

    /// Get keypoint by body part.
    pub fn get(&self, part: super::BodyKeypoint) -> Option<&Keypoint> {
        self.keypoints.get(part as usize)
    }

    /// Calculate bounding box from keypoints.
    pub fn calculate_bbox(&self, threshold: f32) -> Option<(f32, f32, f32, f32)> {
        let visible: Vec<_> = self
            .keypoints
            .iter()
            .filter(|k| k.confidence >= threshold)
            .collect();

        if visible.is_empty() {
            return None;
        }

        let min_x = visible.iter().map(|k| k.x).fold(f32::MAX, f32::min);
        let max_x = visible.iter().map(|k| k.x).fold(f32::MIN, f32::max);
        let min_y = visible.iter().map(|k| k.y).fold(f32::MAX, f32::min);
        let max_y = visible.iter().map(|k| k.y).fold(f32::MIN, f32::max);

        Some((min_x, min_y, max_x - min_x, max_y - min_y))
    }
}

/// Pose estimation model session.
pub struct PoseSession {
    session: Session,
    config: PoseConfig,
}

impl PoseSession {
    /// Get configuration.
    pub fn config(&self) -> &PoseConfig {
        &self.config
    }

    /// Get input size.
    pub fn input_size(&self) -> (u32, u32) {
        self.config.model.input_size()
    }

    /// Run pose estimation inference.
    pub fn run(&mut self, input: &Array4<f32>) -> Result<Vec<DetectedPose>, TransformError> {
        let tensor_ref = TensorRef::from_array_view(input.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("Failed to create tensor ref: {e}"),
            }
        })?;

        let (shape, data) = {
            let outputs = self
                .session
                .run(ort::inputs![tensor_ref])
                .map_err(|e| TransformError::InferenceFailed {
                    message: format!("Pose estimation failed: {e}"),
                })?;

            let output_tensor = outputs.iter().next().ok_or_else(|| {
                TransformError::InferenceFailed {
                    message: "No output tensor found".to_string(),
                }
            })?;

            let (shape, data) = output_tensor
                .1
                .try_extract_tensor::<f32>()
                .map_err(|e| TransformError::InferenceFailed {
                    message: format!("Failed to extract output: {e}"),
                })?;

            let shape: Vec<i64> = shape.iter().copied().collect();
            let data: Vec<f32> = data.iter().copied().collect();
            (shape, data)
        };

        self.parse_output(&data, &shape)
    }

    fn parse_output(&self, data: &[f32], shape: &[i64]) -> Result<Vec<DetectedPose>, TransformError> {
        let mut poses = Vec::new();
        let num_keypoints = self.config.model.num_keypoints();

        if self.config.model.multi_person() {
            // Multi-pose output: [1, N, 56] where N is number of people
            // 56 = 17 keypoints * 3 (y, x, score) + 5 (bbox)
            if shape.len() >= 2 {
                let num_people = shape[1] as usize;
                let stride = if shape.len() > 2 { shape[2] as usize } else { 56 };

                for i in 0..num_people.min(self.config.max_persons) {
                    let offset = i * stride;
                    if offset + stride <= data.len() {
                        let mut keypoints = Vec::with_capacity(num_keypoints);

                        for k in 0..num_keypoints {
                            let kp_offset = offset + k * 3;
                            if kp_offset + 2 < data.len() {
                                keypoints.push(Keypoint {
                                    y: data[kp_offset],
                                    x: data[kp_offset + 1],
                                    confidence: data[kp_offset + 2],
                                });
                            }
                        }

                        // Get person confidence (last values are bbox)
                        let confidence = keypoints
                            .iter()
                            .map(|k| k.confidence)
                            .sum::<f32>() / num_keypoints as f32;

                        if confidence >= self.config.person_threshold {
                            poses.push(DetectedPose {
                                keypoints,
                                confidence,
                                bbox: None, // Calculate from keypoints if needed
                            });
                        }
                    }
                }
            }
        } else {
            // Single pose output: [1, 1, 17, 3]
            let mut keypoints = Vec::with_capacity(num_keypoints);

            for k in 0..num_keypoints {
                let offset = k * 3;
                if offset + 2 < data.len() {
                    keypoints.push(Keypoint {
                        y: data[offset],
                        x: data[offset + 1],
                        confidence: data[offset + 2],
                    });
                }
            }

            let confidence = keypoints
                .iter()
                .map(|k| k.confidence)
                .sum::<f32>() / num_keypoints as f32;

            if confidence >= self.config.person_threshold {
                poses.push(DetectedPose {
                    keypoints,
                    confidence,
                    bbox: None,
                });
            }
        }

        Ok(poses)
    }
}

/// Load pose estimation model.
pub fn load_pose_model(config: &PoseConfig) -> Result<PoseSession, TransformError> {
    let model_path = config.effective_model_path();

    if !model_path.exists() {
        return Err(TransformError::ModelNotFound {
            path: model_path.display().to_string(),
        });
    }

    let mut builder = Session::builder()
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("Failed to create session builder: {e}"),
        })?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("Failed to set optimization level: {e}"),
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
                message: format!("Failed to configure execution providers: {e}"),
            })?;
    } else {
        builder = builder
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| TransformError::ModelLoadFailed {
                message: format!("Failed to configure CPU: {e}"),
            })?;
    }

    let session = builder
        .commit_from_file(&model_path)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("Failed to load pose model: {e}"),
        })?;

    Ok(PoseSession {
        session,
        config: config.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypoint_to_pixel() {
        let kp = Keypoint {
            x: 0.5,
            y: 0.5,
            confidence: 0.9,
        };
        let (px, py) = kp.to_pixel(640, 480);
        assert_eq!(px, 320);
        assert_eq!(py, 240);
    }

    #[test]
    fn test_keypoint_visibility() {
        let kp = Keypoint {
            x: 0.5,
            y: 0.5,
            confidence: 0.8,
        };
        assert!(kp.is_visible(0.5));
        assert!(!kp.is_visible(0.9));
    }

    #[test]
    fn test_detected_pose_bbox() {
        let pose = DetectedPose {
            keypoints: vec![
                Keypoint { x: 0.2, y: 0.3, confidence: 0.9 },
                Keypoint { x: 0.8, y: 0.7, confidence: 0.9 },
            ],
            confidence: 0.9,
            bbox: None,
        };

        let bbox = pose.calculate_bbox(0.5).unwrap();
        assert!((bbox.0 - 0.2).abs() < 0.001);
        assert!((bbox.1 - 0.3).abs() < 0.001);
        assert!((bbox.2 - 0.6).abs() < 0.001);
        assert!((bbox.3 - 0.4).abs() < 0.001);
    }
}
