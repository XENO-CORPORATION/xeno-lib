//! OCR processing logic.

use image::{DynamicImage, GrayImage, Rgba, RgbaImage, imageops::FilterType};
use ndarray::Array4;

use crate::error::TransformError;
use super::model::{OcrSession, TextBox};

/// OCR result containing extracted text and layout information.
#[derive(Debug, Clone)]
pub struct OcrResult {
    /// Full extracted text.
    pub text: String,
    /// Individual text blocks with position information.
    pub blocks: Vec<TextBlock>,
    /// Overall confidence score.
    pub confidence: f32,
}

/// A detected text block.
#[derive(Debug, Clone)]
pub struct TextBlock {
    /// Recognized text.
    pub text: String,
    /// Bounding box (x, y, width, height).
    pub bbox: (u32, u32, u32, u32),
    /// Recognition confidence.
    pub confidence: f32,
    /// Text angle in degrees.
    pub angle: f32,
    /// Original detection box with rotation.
    pub vertices: [(f32, f32); 4],
}

/// Extract text from an image.
pub fn extract_text(
    image: &DynamicImage,
    session: &mut OcrSession,
) -> Result<OcrResult, TransformError> {
    let config = session.config();
    let (orig_w, orig_h) = (image.width(), image.height());

    // Resize if needed
    let max_dim = config.max_dimension;
    let (work_w, work_h) = if orig_w > max_dim || orig_h > max_dim {
        let scale = max_dim as f32 / orig_w.max(orig_h) as f32;
        ((orig_w as f32 * scale) as u32, (orig_h as f32 * scale) as u32)
    } else {
        (orig_w, orig_h)
    };

    let resized = if work_w != orig_w || work_h != orig_h {
        image.resize_exact(work_w, work_h, FilterType::Lanczos3)
    } else {
        image.clone()
    };

    // Preprocess for detection
    let det_input = preprocess_detection(&resized)?;

    // Detect text regions
    let text_boxes = session.detect(&det_input)?;

    // Scale factor for mapping back to original coordinates
    let scale_x = orig_w as f32 / work_w as f32;
    let scale_y = orig_h as f32 / work_h as f32;

    // Recognize text in each region
    let mut blocks = Vec::new();
    let mut full_text = String::new();
    let mut total_confidence = 0.0f32;

    for text_box in text_boxes {
        // Scale vertices back to original image coordinates
        let scaled_vertices: [(f32, f32); 4] = [
            (text_box.vertices[0].0 * scale_x, text_box.vertices[0].1 * scale_y),
            (text_box.vertices[1].0 * scale_x, text_box.vertices[1].1 * scale_y),
            (text_box.vertices[2].0 * scale_x, text_box.vertices[2].1 * scale_y),
            (text_box.vertices[3].0 * scale_x, text_box.vertices[3].1 * scale_y),
        ];

        // Crop and preprocess region for recognition
        let cropped = crop_text_region(image, &scaled_vertices)?;
        let rec_input = preprocess_recognition(&cropped)?;

        // Recognize
        let (text, confidence) = session.recognize(&rec_input)?;

        if confidence >= config.rec_threshold && !text.is_empty() {
            let scaled_box = TextBox {
                vertices: scaled_vertices,
                confidence: text_box.confidence,
            };

            blocks.push(TextBlock {
                text: text.clone(),
                bbox: scaled_box.bbox(),
                confidence,
                angle: scaled_box.angle(),
                vertices: scaled_vertices,
            });

            if !full_text.is_empty() {
                full_text.push('\n');
            }
            full_text.push_str(&text);
            total_confidence += confidence;
        }
    }

    let avg_confidence = if blocks.is_empty() {
        0.0
    } else {
        total_confidence / blocks.len() as f32
    };

    // Sort blocks by reading order (top-to-bottom, left-to-right)
    blocks.sort_by(|a, b| {
        let y_diff = a.bbox.1 as i32 - b.bbox.1 as i32;
        if y_diff.abs() < 20 {
            // Same line
            a.bbox.0.cmp(&b.bbox.0)
        } else {
            a.bbox.1.cmp(&b.bbox.1)
        }
    });

    Ok(OcrResult {
        text: full_text,
        blocks,
        confidence: avg_confidence,
    })
}

/// Extract text with simple interface (returns just the text).
pub fn extract_text_simple(
    image: &DynamicImage,
    session: &mut OcrSession,
) -> Result<String, TransformError> {
    let result = extract_text(image, session)?;
    Ok(result.text)
}

/// Visualize OCR results by drawing bounding boxes.
pub fn visualize_ocr(
    image: &DynamicImage,
    result: &OcrResult,
) -> DynamicImage {
    let mut rgba = image.to_rgba8();

    for block in &result.blocks {
        draw_box(&mut rgba, &block.vertices, Rgba([0, 255, 0, 255]));
    }

    DynamicImage::ImageRgba8(rgba)
}

/// Preprocess image for text detection model.
fn preprocess_detection(image: &DynamicImage) -> Result<Array4<f32>, TransformError> {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    // Pad to multiple of 32 for model compatibility
    let padded_w = ((width + 31) / 32) * 32;
    let padded_h = ((height + 31) / 32) * 32;

    let mut tensor = Array4::<f32>::zeros((1, 3, padded_h, padded_w));

    // Normalize to [0, 1] and apply ImageNet normalization
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                tensor[[0, c, y, x]] = (val - mean[c]) / std[c];
            }
        }
    }

    Ok(tensor)
}

/// Preprocess cropped text region for recognition model.
fn preprocess_recognition(image: &DynamicImage) -> Result<Array4<f32>, TransformError> {
    // Standard text recognition input: 32 height, variable width
    let target_height = 32u32;
    let aspect = image.width() as f32 / image.height() as f32;
    let target_width = (target_height as f32 * aspect).round() as u32;
    let target_width = target_width.clamp(32, 320); // Max width for memory

    let resized = image.resize_exact(target_width, target_height, FilterType::Lanczos3);
    let gray = resized.to_luma8();

    let mut tensor = Array4::<f32>::zeros((1, 1, target_height as usize, target_width as usize));

    for y in 0..target_height as usize {
        for x in 0..target_width as usize {
            let pixel = gray.get_pixel(x as u32, y as u32);
            // Normalize to [-1, 1]
            tensor[[0, 0, y, x]] = (pixel[0] as f32 / 255.0) * 2.0 - 1.0;
        }
    }

    Ok(tensor)
}

/// Crop text region from image using quadrilateral vertices.
fn crop_text_region(
    image: &DynamicImage,
    vertices: &[(f32, f32); 4],
) -> Result<DynamicImage, TransformError> {
    // Get axis-aligned bounding box
    let min_x = vertices.iter().map(|v| v.0).fold(f32::MAX, f32::min).max(0.0) as u32;
    let max_x = vertices.iter().map(|v| v.0).fold(f32::MIN, f32::max).min(image.width() as f32) as u32;
    let min_y = vertices.iter().map(|v| v.1).fold(f32::MAX, f32::min).max(0.0) as u32;
    let max_y = vertices.iter().map(|v| v.1).fold(f32::MIN, f32::max).min(image.height() as f32) as u32;

    let width = max_x.saturating_sub(min_x).max(1);
    let height = max_y.saturating_sub(min_y).max(1);

    let cropped = image.crop_imm(min_x, min_y, width, height);

    // TODO: Apply perspective transform for rotated text

    Ok(cropped)
}

/// Draw quadrilateral box on image.
fn draw_box(img: &mut RgbaImage, vertices: &[(f32, f32); 4], color: Rgba<u8>) {
    for i in 0..4 {
        let (x1, y1) = vertices[i];
        let (x2, y2) = vertices[(i + 1) % 4];
        draw_line(img, x1 as i32, y1 as i32, x2 as i32, y2 as i32, color);
    }
}

/// Draw a line using Bresenham's algorithm.
fn draw_line(img: &mut RgbaImage, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: Rgba<u8>) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let (w, h) = (img.width() as i32, img.height() as i32);

    loop {
        if x0 >= 0 && x0 < w && y0 >= 0 && y0 < h {
            img.put_pixel(x0 as u32, y0 as u32, color);
        }

        if x0 == x1 && y0 == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_detection() {
        let img = DynamicImage::new_rgb8(100, 100);
        let tensor = preprocess_detection(&img).unwrap();

        // Should be padded to multiple of 32
        assert_eq!(tensor.shape()[2], 128);
        assert_eq!(tensor.shape()[3], 128);
    }

    #[test]
    fn test_preprocess_recognition() {
        let img = DynamicImage::new_rgb8(200, 50);
        let tensor = preprocess_recognition(&img).unwrap();

        // Height should be 32
        assert_eq!(tensor.shape()[2], 32);
    }

    #[test]
    fn test_ocr_result_structure() {
        let result = OcrResult {
            text: "Hello World".to_string(),
            blocks: vec![
                TextBlock {
                    text: "Hello".to_string(),
                    bbox: (10, 10, 50, 20),
                    confidence: 0.95,
                    angle: 0.0,
                    vertices: [(10.0, 10.0), (60.0, 10.0), (60.0, 30.0), (10.0, 30.0)],
                },
            ],
            confidence: 0.95,
        };

        assert_eq!(result.blocks.len(), 1);
        assert_eq!(result.blocks[0].text, "Hello");
    }
}
