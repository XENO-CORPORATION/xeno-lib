//! Face detection processing logic.

use image::{DynamicImage, Rgb, imageops::FilterType};
use ndarray::Array4;

use crate::error::TransformError;
use super::model::{FaceDetectorSession, RawDetection};

/// A detected face with bounding box and optional landmarks.
#[derive(Debug, Clone)]
pub struct DetectedFace {
    /// Bounding box (x, y, width, height) in original image coordinates.
    pub bbox: (u32, u32, u32, u32),

    /// Detection confidence score.
    pub confidence: f32,

    /// 5-point facial landmarks (eyes, nose, mouth corners).
    pub landmarks: Option<FaceLandmarks>,
}

/// 5-point facial landmarks.
#[derive(Debug, Clone)]
pub struct FaceLandmarks {
    pub left_eye: (f32, f32),
    pub right_eye: (f32, f32),
    pub nose: (f32, f32),
    pub left_mouth: (f32, f32),
    pub right_mouth: (f32, f32),
}

/// Detects faces in an image.
pub fn detect_faces_impl(
    image: &DynamicImage,
    session: &mut FaceDetectorSession,
) -> Result<Vec<DetectedFace>, TransformError> {
    let (input_w, input_h) = session.input_size();
    let original_width = image.width();
    let original_height = image.height();

    // Resize to model input size
    let resized = image.resize_exact(input_w, input_h, FilterType::Lanczos3);

    // Convert to tensor
    let input_tensor = image_to_tensor(&resized)?;

    // Run detection
    let raw_detections = session.run(&input_tensor)?;

    // Apply NMS
    let nms_threshold = session.config().nms_threshold;
    let filtered = non_maximum_suppression(raw_detections, nms_threshold);

    // Scale back to original coordinates
    let scale_x = original_width as f32 / input_w as f32;
    let scale_y = original_height as f32 / input_h as f32;

    let mut faces: Vec<DetectedFace> = filtered
        .into_iter()
        .take(session.config().max_faces)
        .map(|det| {
            let x1 = (det.x1 * scale_x).max(0.0) as u32;
            let y1 = (det.y1 * scale_y).max(0.0) as u32;
            let x2 = (det.x2 * scale_x).min(original_width as f32) as u32;
            let y2 = (det.y2 * scale_y).min(original_height as f32) as u32;

            let landmarks = det.landmarks.map(|lm| {
                FaceLandmarks {
                    left_eye: (lm[0].0 * scale_x, lm[0].1 * scale_y),
                    right_eye: (lm[1].0 * scale_x, lm[1].1 * scale_y),
                    nose: (lm[2].0 * scale_x, lm[2].1 * scale_y),
                    left_mouth: (lm[3].0 * scale_x, lm[3].1 * scale_y),
                    right_mouth: (lm[4].0 * scale_x, lm[4].1 * scale_y),
                }
            });

            DetectedFace {
                bbox: (x1, y1, x2.saturating_sub(x1), y2.saturating_sub(y1)),
                confidence: det.score,
                landmarks,
            }
        })
        .collect();

    // Sort by confidence
    faces.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    Ok(faces)
}

/// Non-maximum suppression to filter overlapping detections.
fn non_maximum_suppression(mut detections: Vec<RawDetection>, iou_threshold: f32) -> Vec<RawDetection> {
    // Sort by score descending
    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = Vec::new();

    while !detections.is_empty() {
        let best = detections.remove(0);
        keep.push(best.clone());

        detections.retain(|det| {
            iou(&best, det) < iou_threshold
        });
    }

    keep
}

/// Calculate Intersection over Union between two boxes.
fn iou(a: &RawDetection, b: &RawDetection) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) * (y2 - y1);
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let union = area_a + area_b - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

/// Converts image to tensor normalized to [0, 1] in RGB order.
fn image_to_tensor(image: &DynamicImage) -> Result<Array4<f32>, TransformError> {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                tensor[[0, c, y, x]] = pixel[c] as f32 / 255.0;
            }
        }
    }

    Ok(tensor)
}

/// Draws detection boxes on an image (for visualization).
pub fn draw_detections(image: &DynamicImage, faces: &[DetectedFace]) -> DynamicImage {
    let mut rgb = image.to_rgb8();

    for face in faces {
        let (x, y, w, h) = face.bbox;
        let color = Rgb([0, 255, 0]); // Green

        // Draw rectangle
        for px in x..x.saturating_add(w).min(rgb.width()) {
            if y < rgb.height() {
                rgb.put_pixel(px, y, color);
            }
            let bottom = y.saturating_add(h).min(rgb.height() - 1);
            rgb.put_pixel(px, bottom, color);
        }
        for py in y..y.saturating_add(h).min(rgb.height()) {
            if x < rgb.width() {
                rgb.put_pixel(x, py, color);
            }
            let right = x.saturating_add(w).min(rgb.width() - 1);
            rgb.put_pixel(right, py, color);
        }

        // Draw landmarks if available
        if let Some(ref lm) = face.landmarks {
            let points = [lm.left_eye, lm.right_eye, lm.nose, lm.left_mouth, lm.right_mouth];
            let lm_color = Rgb([255, 0, 0]); // Red

            for (lx, ly) in points {
                let lx = lx as u32;
                let ly = ly as u32;
                // Draw small cross
                for dx in 0..3 {
                    let px = lx.saturating_add(dx).saturating_sub(1);
                    if px < rgb.width() && ly < rgb.height() {
                        rgb.put_pixel(px, ly, lm_color);
                    }
                }
                for dy in 0..3 {
                    let py = ly.saturating_add(dy).saturating_sub(1);
                    if lx < rgb.width() && py < rgb.height() {
                        rgb.put_pixel(lx, py, lm_color);
                    }
                }
            }
        }
    }

    DynamicImage::ImageRgb8(rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_no_overlap() {
        let a = RawDetection {
            x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0, score: 1.0, landmarks: None
        };
        let b = RawDetection {
            x1: 20.0, y1: 20.0, x2: 30.0, y2: 30.0, score: 1.0, landmarks: None
        };
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn test_iou_full_overlap() {
        let a = RawDetection {
            x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0, score: 1.0, landmarks: None
        };
        assert!((iou(&a, &a) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let a = RawDetection {
            x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0, score: 1.0, landmarks: None
        };
        let b = RawDetection {
            x1: 5.0, y1: 5.0, x2: 15.0, y2: 15.0, score: 1.0, landmarks: None
        };
        // Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175
        let expected = 25.0 / 175.0;
        assert!((iou(&a, &b) - expected).abs() < 0.001);
    }

    #[test]
    fn test_nms_removes_duplicates() {
        let detections = vec![
            RawDetection { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0, score: 0.9, landmarks: None },
            RawDetection { x1: 1.0, y1: 1.0, x2: 11.0, y2: 11.0, score: 0.8, landmarks: None },
            RawDetection { x1: 50.0, y1: 50.0, x2: 60.0, y2: 60.0, score: 0.85, landmarks: None },
        ];

        let filtered = non_maximum_suppression(detections, 0.3);
        assert_eq!(filtered.len(), 2); // Should keep best overlapping + non-overlapping
    }
}
