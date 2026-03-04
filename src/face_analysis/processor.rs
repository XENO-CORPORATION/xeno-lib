//! Face analysis processing logic.

use image::{imageops::FilterType, DynamicImage};
use ndarray::Array4;

use super::model::FaceAnalyzerSession;
use super::{Emotion, FaceAnalysisResult, Gender};
use crate::error::TransformError;

/// Analyze faces in an image.
///
/// This function expects faces to already be detected. Use face_detect module first.
pub fn analyze_face(
    image: &DynamicImage,
    session: &mut FaceAnalyzerSession,
) -> Result<FaceAnalysisResult, TransformError> {
    let config = session.config().clone();
    let (w, h) = (image.width(), image.height());

    // Resize to model input size (typically 224x224 for these models)
    let resized = image.resize_exact(224, 224, FilterType::Lanczos3);
    let tensor = image_to_tensor(&resized)?;

    // Run each analysis
    let (age, age_confidence) = if config.estimate_age {
        session.estimate_age(&tensor)?
    } else {
        (0.0, 0.0)
    };

    let (male_score, gender_confidence) = if config.classify_gender {
        session.classify_gender(&tensor)?
    } else {
        (0.5, 0.0)
    };

    let emotion_scores = if config.recognize_emotion {
        let scores = session.recognize_emotion(&tensor)?;
        Emotion::all()
            .iter()
            .zip(scores.iter())
            .map(|(&e, &s)| (e, s))
            .collect()
    } else {
        Vec::new()
    };

    // Find primary emotion
    let (emotion, emotion_confidence) = emotion_scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|&(e, c)| (e, c))
        .unwrap_or((Emotion::Unknown, 0.0));

    Ok(FaceAnalysisResult {
        age,
        age_confidence,
        gender: Gender::from_score(male_score),
        gender_confidence,
        emotion,
        emotion_confidence,
        emotion_scores,
        bbox: (0, 0, w, h), // Full image as bbox since face was pre-cropped
    })
}

/// Analyze multiple face regions.
pub fn analyze_faces(
    image: &DynamicImage,
    face_regions: &[(u32, u32, u32, u32)], // (x, y, width, height)
    session: &mut FaceAnalyzerSession,
) -> Result<Vec<FaceAnalysisResult>, TransformError> {
    let config = session.config().clone();
    let mut results = Vec::new();

    for &(x, y, w, h) in face_regions {
        // Skip faces that are too small
        if w < config.min_face_size || h < config.min_face_size {
            continue;
        }

        // Crop face region
        let face_crop = image.crop_imm(x, y, w, h);

        // Analyze
        let mut result = analyze_face(&face_crop, session)?;
        result.bbox = (x, y, w, h);
        results.push(result);
    }

    Ok(results)
}

/// Analyze face with visualization.
pub fn analyze_and_annotate(
    image: &DynamicImage,
    face_regions: &[(u32, u32, u32, u32)],
    session: &mut FaceAnalyzerSession,
) -> Result<(DynamicImage, Vec<FaceAnalysisResult>), TransformError> {
    let results = analyze_faces(image, face_regions, session)?;
    let annotated = visualize_analysis(image, &results);
    Ok((annotated, results))
}

/// Visualize analysis results on image.
pub fn visualize_analysis(image: &DynamicImage, results: &[FaceAnalysisResult]) -> DynamicImage {
    use image::Rgba;

    let mut rgba = image.to_rgba8();

    for result in results {
        let (x, y, w, h) = result.bbox;
        let color = Rgba([0, 255, 0, 255]); // Green

        // Draw bounding box
        draw_rect(&mut rgba, x, y, w, h, color);

        // Draw label background
        let label = format!(
            "{}y {:?} {:?}",
            result.age as u32, result.gender, result.emotion
        );
        let label_height = 20u32;
        let label_y = if y >= label_height {
            y - label_height
        } else {
            y + h
        };

        // Simple label background
        for dy in 0..label_height {
            for dx in 0..w.min(label.len() as u32 * 8) {
                let px = x + dx;
                let py = label_y + dy;
                if px < rgba.width() && py < rgba.height() {
                    rgba.put_pixel(px, py, Rgba([0, 0, 0, 180]));
                }
            }
        }
    }

    DynamicImage::ImageRgba8(rgba)
}

/// Convert image to tensor for face analysis models.
fn image_to_tensor(image: &DynamicImage) -> Result<Array4<f32>, TransformError> {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    // ImageNet normalization
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

/// Draw rectangle outline.
fn draw_rect(img: &mut image::RgbaImage, x: u32, y: u32, w: u32, h: u32, color: image::Rgba<u8>) {
    let (img_w, img_h) = (img.width(), img.height());

    // Top and bottom edges
    for dx in 0..w {
        let px = x + dx;
        if px < img_w {
            if y < img_h {
                img.put_pixel(px, y, color);
            }
            let bottom = y + h - 1;
            if bottom < img_h {
                img.put_pixel(px, bottom, color);
            }
        }
    }

    // Left and right edges
    for dy in 0..h {
        let py = y + dy;
        if py < img_h {
            if x < img_w {
                img.put_pixel(x, py, color);
            }
            let right = x + w - 1;
            if right < img_w {
                img.put_pixel(right, py, color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_to_tensor() {
        let img = DynamicImage::new_rgb8(224, 224);
        let tensor = image_to_tensor(&img).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
    }

    #[test]
    fn test_face_analysis_result() {
        let result = FaceAnalysisResult {
            age: 32.5,
            age_confidence: 0.85,
            gender: Gender::Male,
            gender_confidence: 0.92,
            emotion: Emotion::Happy,
            emotion_confidence: 0.78,
            emotion_scores: vec![
                (Emotion::Happy, 0.78),
                (Emotion::Neutral, 0.15),
                (Emotion::Surprised, 0.07),
            ],
            bbox: (10, 20, 100, 120),
        };

        assert_eq!(result.age_range(), "30-35");
        let top = result.top_emotions(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, Emotion::Happy);
    }

    #[test]
    fn test_gender_from_score() {
        assert_eq!(Gender::from_score(0.8), Gender::Male);
        assert_eq!(Gender::from_score(0.2), Gender::Female);
        assert_eq!(Gender::from_score(0.5), Gender::Unknown);
    }
}
