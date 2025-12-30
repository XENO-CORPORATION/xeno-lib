//! OCR ONNX model session.

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;
use super::config::OcrConfig;

/// Text detection output.
#[derive(Debug, Clone)]
pub struct TextBox {
    /// Bounding box vertices (4 corners).
    pub vertices: [(f32, f32); 4],
    /// Detection confidence.
    pub confidence: f32,
}

impl TextBox {
    /// Get axis-aligned bounding box (x, y, width, height).
    pub fn bbox(&self) -> (u32, u32, u32, u32) {
        let min_x = self.vertices.iter().map(|v| v.0).fold(f32::MAX, f32::min);
        let max_x = self.vertices.iter().map(|v| v.0).fold(f32::MIN, f32::max);
        let min_y = self.vertices.iter().map(|v| v.1).fold(f32::MAX, f32::min);
        let max_y = self.vertices.iter().map(|v| v.1).fold(f32::MIN, f32::max);

        (
            min_x as u32,
            min_y as u32,
            (max_x - min_x) as u32,
            (max_y - min_y) as u32,
        )
    }

    /// Get rotation angle in degrees.
    pub fn angle(&self) -> f32 {
        let dx = self.vertices[1].0 - self.vertices[0].0;
        let dy = self.vertices[1].1 - self.vertices[0].1;
        dy.atan2(dx).to_degrees()
    }
}

/// OCR model session (detection + recognition).
pub struct OcrSession {
    det_session: Session,
    rec_session: Session,
    config: OcrConfig,
}

impl OcrSession {
    /// Get configuration.
    pub fn config(&self) -> &OcrConfig {
        &self.config
    }

    /// Run text detection.
    pub fn detect(&mut self, input: &Array4<f32>) -> Result<Vec<TextBox>, TransformError> {
        let tensor_ref = TensorRef::from_array_view(input.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("Failed to create tensor ref: {e}"),
            }
        })?;

        let outputs = self
            .det_session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("Text detection failed: {e}"),
            })?;

        // Parse detection output (implementation depends on model format)
        let mut boxes = Vec::new();

        if let Some((_, output)) = outputs.iter().next() {
            let (shape, data) = output
                .try_extract_tensor::<f32>()
                .map_err(|e| TransformError::InferenceFailed {
                    message: format!("Failed to extract detection output: {e}"),
                })?;

            // Parse boxes from output (simplified - actual format depends on model)
            let data: Vec<f32> = data.iter().copied().collect();
            let threshold = self.config.det_threshold;

            // Assuming output format: [N, 5] where each row is [x1, y1, x2, y2, score]
            // or [N, 9] for rotated boxes [x1, y1, x2, y2, x3, y3, x4, y4, score]
            if shape.len() >= 2 {
                let num_boxes = shape[0] as usize;
                let box_size = if shape.len() > 1 { shape[1] as usize } else { 5 };

                for i in 0..num_boxes {
                    let offset = i * box_size;
                    if offset + box_size <= data.len() {
                        let score = data[offset + box_size - 1];
                        if score >= threshold {
                            let vertices = if box_size >= 9 {
                                [
                                    (data[offset], data[offset + 1]),
                                    (data[offset + 2], data[offset + 3]),
                                    (data[offset + 4], data[offset + 5]),
                                    (data[offset + 6], data[offset + 7]),
                                ]
                            } else {
                                let x1 = data[offset];
                                let y1 = data[offset + 1];
                                let x2 = data[offset + 2];
                                let y2 = data[offset + 3];
                                [
                                    (x1, y1),
                                    (x2, y1),
                                    (x2, y2),
                                    (x1, y2),
                                ]
                            };
                            boxes.push(TextBox {
                                vertices,
                                confidence: score,
                            });
                        }
                    }
                }
            }
        }

        Ok(boxes)
    }

    /// Run text recognition on a cropped text region.
    pub fn recognize(&mut self, input: &Array4<f32>) -> Result<(String, f32), TransformError> {
        let tensor_ref = TensorRef::from_array_view(input.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("Failed to create tensor ref: {e}"),
            }
        })?;

        let outputs = self
            .rec_session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("Text recognition failed: {e}"),
            })?;

        // Parse recognition output
        let mut text = String::new();
        let mut confidence = 0.0f32;

        if let Some((_, output)) = outputs.iter().next() {
            let (shape, data) = output
                .try_extract_tensor::<f32>()
                .map_err(|e| TransformError::InferenceFailed {
                    message: format!("Failed to extract recognition output: {e}"),
                })?;

            // Assuming CTC output format: [1, T, num_classes]
            // Decode using greedy decoding
            let data: Vec<f32> = data.iter().copied().collect();

            if shape.len() >= 3 {
                let seq_len = shape[1] as usize;
                let num_classes = shape[2] as usize;
                let mut prev_idx = num_classes; // Blank token

                for t in 0..seq_len {
                    let offset = t * num_classes;
                    if offset + num_classes <= data.len() {
                        // Find argmax
                        let mut max_idx = 0;
                        let mut max_val = data[offset];
                        for c in 1..num_classes {
                            if data[offset + c] > max_val {
                                max_val = data[offset + c];
                                max_idx = c;
                            }
                        }

                        // CTC decoding: skip blanks and repeated characters
                        if max_idx != num_classes - 1 && max_idx != prev_idx {
                            // Convert index to character (simplified)
                            if let Some(c) = idx_to_char(max_idx) {
                                text.push(c);
                                confidence += max_val;
                            }
                        }
                        prev_idx = max_idx;
                    }
                }

                if !text.is_empty() {
                    confidence /= text.len() as f32;
                }
            }
        }

        Ok((text, confidence))
    }
}

/// Convert index to character (simplified ASCII + common characters).
fn idx_to_char(idx: usize) -> Option<char> {
    // Standard character set (adjust based on actual model)
    const CHARS: &str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";
    CHARS.chars().nth(idx)
}

/// Load OCR model (detection + recognition).
pub fn load_ocr_model(config: &OcrConfig) -> Result<OcrSession, TransformError> {
    let det_path = config.det_model_path();
    let rec_path = config.rec_model_path();

    if !det_path.exists() {
        return Err(TransformError::ModelNotFound {
            path: det_path.display().to_string(),
        });
    }
    if !rec_path.exists() {
        return Err(TransformError::ModelNotFound {
            path: rec_path.display().to_string(),
        });
    }

    let build_session = |path: &std::path::Path| -> Result<Session, TransformError> {
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

        builder.commit_from_file(path).map_err(|e| TransformError::ModelLoadFailed {
            message: format!("Failed to load model {}: {e}", path.display()),
        })
    };

    let det_session = build_session(&det_path)?;
    let rec_session = build_session(&rec_path)?;

    Ok(OcrSession {
        det_session,
        rec_session,
        config: config.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_box_bbox() {
        let text_box = TextBox {
            vertices: [
                (10.0, 20.0),
                (110.0, 20.0),
                (110.0, 50.0),
                (10.0, 50.0),
            ],
            confidence: 0.9,
        };

        let (x, y, w, h) = text_box.bbox();
        assert_eq!(x, 10);
        assert_eq!(y, 20);
        assert_eq!(w, 100);
        assert_eq!(h, 30);
    }

    #[test]
    fn test_text_box_angle() {
        let text_box = TextBox {
            vertices: [
                (0.0, 0.0),
                (100.0, 0.0),
                (100.0, 30.0),
                (0.0, 30.0),
            ],
            confidence: 0.9,
        };

        let angle = text_box.angle();
        assert!(angle.abs() < 0.001); // Horizontal text
    }

    #[test]
    fn test_idx_to_char() {
        assert_eq!(idx_to_char(0), Some('0'));
        assert_eq!(idx_to_char(10), Some('a'));
        assert_eq!(idx_to_char(36), Some('A'));
    }
}
