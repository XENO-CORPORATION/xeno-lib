//! OCR processing logic.

use image::{DynamicImage, Rgba, RgbaImage, imageops::FilterType};
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
    let source = image.to_rgba8();
    if source.width() == 0 || source.height() == 0 {
        return Err(TransformError::InvalidDimensions {
            width: source.width(),
            height: source.height(),
        });
    }

    // Clamp vertices to image bounds and normalize ordering to:
    // top-left, top-right, bottom-right, bottom-left.
    let max_x = source.width().saturating_sub(1) as f32;
    let max_y = source.height().saturating_sub(1) as f32;
    let mut clamped = [(0.0f32, 0.0f32); 4];
    for (i, (x, y)) in vertices.iter().enumerate() {
        clamped[i] = (x.clamp(0.0, max_x), y.clamp(0.0, max_y));
    }
    let ordered = order_quad_points(&clamped);

    // Estimate output size from opposing edge lengths.
    let top_w = point_distance(ordered[0], ordered[1]);
    let bottom_w = point_distance(ordered[3], ordered[2]);
    let left_h = point_distance(ordered[0], ordered[3]);
    let right_h = point_distance(ordered[1], ordered[2]);

    let out_width = top_w.max(bottom_w).round().max(1.0) as u32;
    let out_height = left_h.max(right_h).round().max(1.0) as u32;

    let mut output = RgbaImage::new(out_width, out_height);
    let width_denom = out_width.saturating_sub(1).max(1) as f32;
    let height_denom = out_height.saturating_sub(1).max(1) as f32;

    // Warp quadrilateral to an axis-aligned rectangle using bilinear interpolation
    // between quad edges. This handles rotated/tilted text regions for recognition.
    for y in 0..out_height {
        let t = y as f32 / height_denom;
        let left = lerp_point(ordered[0], ordered[3], t);
        let right = lerp_point(ordered[1], ordered[2], t);

        for x in 0..out_width {
            let s = x as f32 / width_denom;
            let src = lerp_point(left, right, s);
            let px = bilinear_sample_rgba(&source, src.0, src.1);
            output.put_pixel(x, y, px);
        }
    }

    Ok(DynamicImage::ImageRgba8(output))
}

/// Order quadrilateral points to [top-left, top-right, bottom-right, bottom-left].
fn order_quad_points(vertices: &[(f32, f32); 4]) -> [(f32, f32); 4] {
    let mut top_left = vertices[0];
    let mut top_right = vertices[0];
    let mut bottom_right = vertices[0];
    let mut bottom_left = vertices[0];

    let mut min_sum = vertices[0].0 + vertices[0].1;
    let mut max_sum = min_sum;
    let mut max_diff = vertices[0].0 - vertices[0].1;
    let mut min_diff = max_diff;

    for &(x, y) in vertices.iter().skip(1) {
        let sum = x + y;
        let diff = x - y;

        if sum < min_sum {
            min_sum = sum;
            top_left = (x, y);
        }
        if sum > max_sum {
            max_sum = sum;
            bottom_right = (x, y);
        }
        if diff > max_diff {
            max_diff = diff;
            top_right = (x, y);
        }
        if diff < min_diff {
            min_diff = diff;
            bottom_left = (x, y);
        }
    }

    [top_left, top_right, bottom_right, bottom_left]
}

#[inline]
fn point_distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

#[inline]
fn lerp_point(a: (f32, f32), b: (f32, f32), t: f32) -> (f32, f32) {
    let t = t.clamp(0.0, 1.0);
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}

fn bilinear_sample_rgba(image: &RgbaImage, x: f32, y: f32) -> Rgba<u8> {
    let max_x = image.width().saturating_sub(1) as f32;
    let max_y = image.height().saturating_sub(1) as f32;
    let x = x.clamp(0.0, max_x);
    let y = y.clamp(0.0, max_y);

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width().saturating_sub(1));
    let y1 = (y0 + 1).min(image.height().saturating_sub(1));

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let p00 = image.get_pixel(x0, y0);
    let p10 = image.get_pixel(x1, y0);
    let p01 = image.get_pixel(x0, y1);
    let p11 = image.get_pixel(x1, y1);

    let mut out = [0u8; 4];
    for c in 0..4 {
        let v00 = p00[c] as f32;
        let v10 = p10[c] as f32;
        let v01 = p01[c] as f32;
        let v11 = p11[c] as f32;

        let top = v00 * (1.0 - fx) + v10 * fx;
        let bottom = v01 * (1.0 - fx) + v11 * fx;
        let value = top * (1.0 - fy) + bottom * fy;
        out[c] = value.round().clamp(0.0, 255.0) as u8;
    }

    Rgba(out)
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

    #[test]
    fn test_crop_text_region_axis_aligned() {
        let mut img = RgbaImage::from_pixel(120, 80, Rgba([0, 0, 0, 255]));
        for y in 20..50 {
            for x in 30..90 {
                img.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        let image = DynamicImage::ImageRgba8(img);
        let vertices = [(30.0, 20.0), (90.0, 20.0), (90.0, 50.0), (30.0, 50.0)];

        let cropped = crop_text_region(&image, &vertices).unwrap();
        assert!(cropped.width() >= 59);
        assert!(cropped.height() >= 29);

        let center = cropped
            .to_rgba8()
            .get_pixel(cropped.width() / 2, cropped.height() / 2)
            .0;
        assert!(center[0] > 200);
        assert!(center[1] > 200);
        assert!(center[2] > 200);
    }

    #[test]
    fn test_crop_text_region_rotated_quad() {
        let mut img = RgbaImage::from_pixel(120, 120, Rgba([0, 0, 0, 255]));
        for y in 35..85 {
            for x in 40..80 {
                img.put_pixel(x, y, Rgba([220, 220, 220, 255]));
            }
        }
        let image = DynamicImage::ImageRgba8(img);

        // Roughly rotated text box.
        let vertices = [(35.0, 45.0), (75.0, 30.0), (88.0, 75.0), (48.0, 90.0)];
        let cropped = crop_text_region(&image, &vertices).unwrap();

        assert!(cropped.width() > 20);
        assert!(cropped.height() > 20);
    }
}
