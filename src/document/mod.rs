//! Document processing module.
//!
//! Provides tools for processing scanned documents:
//! - Deskewing (rotation correction)
//! - Perspective correction
//! - Binarization (adaptive thresholding)
//! - Border detection and cropping
//! - Enhancement for OCR

use image::{DynamicImage, GrayImage, Rgba, RgbaImage};

/// Document processing result.
#[derive(Debug, Clone)]
pub struct DocumentResult {
    /// Processed image.
    pub image: DynamicImage,
    /// Detected skew angle in degrees.
    pub skew_angle: f64,
    /// Detected document corners (if found).
    pub corners: Option<[(u32, u32); 4]>,
    /// Processing statistics.
    pub stats: ProcessingStats,
}

/// Processing statistics.
#[derive(Debug, Clone, Default)]
pub struct ProcessingStats {
    /// Confidence of skew detection (0-1).
    pub skew_confidence: f64,
    /// Whether document edges were detected.
    pub edges_detected: bool,
    /// Binarization threshold used.
    pub threshold: u8,
}

/// Document processing configuration.
#[derive(Debug, Clone)]
pub struct DocumentConfig {
    /// Enable deskewing.
    pub deskew: bool,
    /// Enable perspective correction.
    pub perspective_correct: bool,
    /// Enable binarization.
    pub binarize: bool,
    /// Enable border removal.
    pub remove_border: bool,
    /// Enable enhancement for OCR.
    pub enhance_for_ocr: bool,
    /// Maximum skew angle to consider (degrees).
    pub max_skew_angle: f64,
    /// Binarization block size for adaptive threshold.
    pub binarize_block_size: u32,
    /// Border detection margin.
    pub border_margin: u32,
}

impl Default for DocumentConfig {
    fn default() -> Self {
        Self {
            deskew: true,
            perspective_correct: true,
            binarize: false,
            remove_border: true,
            enhance_for_ocr: false,
            max_skew_angle: 15.0,
            binarize_block_size: 31,
            border_margin: 10,
        }
    }
}

impl DocumentConfig {
    /// Create config optimized for OCR.
    pub fn for_ocr() -> Self {
        Self {
            deskew: true,
            perspective_correct: true,
            binarize: true,
            remove_border: true,
            enhance_for_ocr: true,
            max_skew_angle: 15.0,
            binarize_block_size: 31,
            border_margin: 10,
        }
    }

    /// Create config for photo scanning.
    pub fn for_photo() -> Self {
        Self {
            deskew: true,
            perspective_correct: true,
            binarize: false,
            remove_border: true,
            enhance_for_ocr: false,
            max_skew_angle: 10.0,
            binarize_block_size: 31,
            border_margin: 20,
        }
    }

    /// Deskew only.
    pub fn deskew_only() -> Self {
        Self {
            deskew: true,
            perspective_correct: false,
            binarize: false,
            remove_border: false,
            enhance_for_ocr: false,
            max_skew_angle: 15.0,
            binarize_block_size: 31,
            border_margin: 10,
        }
    }
}

/// Process a document image.
pub fn process_document(image: &DynamicImage, config: &DocumentConfig) -> DocumentResult {
    let mut result = image.clone();
    let mut stats = ProcessingStats::default();
    let mut skew_angle = 0.0;
    let mut corners = None;

    // Step 1: Detect skew angle
    if config.deskew {
        let (angle, confidence) = detect_skew(&result);
        skew_angle = angle;
        stats.skew_confidence = confidence;

        if angle.abs() > 0.1 && angle.abs() <= config.max_skew_angle {
            result = rotate_image(&result, -angle);
        }
    }

    // Step 2: Detect document edges
    if config.perspective_correct || config.remove_border {
        if let Some(detected) = detect_document_corners(&result) {
            corners = Some(detected);
            stats.edges_detected = true;

            if config.perspective_correct {
                result = correct_perspective(&result, &detected);
            }
        }
    }

    // Step 3: Remove borders
    if config.remove_border {
        result = remove_borders(&result, config.border_margin);
    }

    // Step 4: Binarization
    if config.binarize {
        let (binary, threshold) = adaptive_binarize(&result, config.binarize_block_size);
        result = DynamicImage::ImageLuma8(binary);
        stats.threshold = threshold;
    }

    // Step 5: OCR enhancement
    if config.enhance_for_ocr {
        result = enhance_for_ocr(&result);
    }

    DocumentResult {
        image: result,
        skew_angle,
        corners,
        stats,
    }
}

/// Detect skew angle of a document.
pub fn detect_skew(image: &DynamicImage) -> (f64, f64) {
    let gray = image.to_luma8();

    // Edge detection first
    let edges = detect_edges(&gray);

    // Use Hough transform for line detection
    let angles = hough_transform(&edges, 180);

    // Find dominant angle (most common line orientation)
    let mut best_angle = 0.0;
    let mut max_votes = 0;

    for (angle, votes) in angles.iter().enumerate() {
        if *votes > max_votes {
            max_votes = *votes;
            best_angle = angle as f64 - 90.0;
        }
    }

    // Normalize to -45 to 45 degree range
    while best_angle > 45.0 {
        best_angle -= 90.0;
    }
    while best_angle < -45.0 {
        best_angle += 90.0;
    }

    // Confidence based on vote strength
    let total_votes: u32 = angles.iter().sum();
    let confidence = if total_votes > 0 {
        max_votes as f64 / total_votes as f64 * 3.0 // Scale up
    } else {
        0.0
    };

    (best_angle, confidence.min(1.0))
}

/// Deskew an image by a specific angle.
pub fn deskew(image: &DynamicImage, angle: Option<f64>) -> DynamicImage {
    let angle = angle.unwrap_or_else(|| detect_skew(image).0);
    if angle.abs() < 0.1 {
        return image.clone();
    }
    rotate_image(image, -angle)
}

/// Simple rotation using bilinear interpolation.
fn rotate_image(image: &DynamicImage, angle_deg: f64) -> DynamicImage {
    let rgba = image.to_rgba8();
    let (width, height) = (rgba.width(), rgba.height());
    let angle_rad = angle_deg.to_radians();

    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;

    // Calculate new dimensions
    let new_width = ((width as f64 * cos_a.abs()) + (height as f64 * sin_a.abs())).ceil() as u32;
    let new_height = ((width as f64 * sin_a.abs()) + (height as f64 * cos_a.abs())).ceil() as u32;

    let ncx = new_width as f64 / 2.0;
    let ncy = new_height as f64 / 2.0;

    let mut result = RgbaImage::from_pixel(new_width, new_height, Rgba([255, 255, 255, 255]));

    for y in 0..new_height {
        for x in 0..new_width {
            // Translate to center
            let dx = x as f64 - ncx;
            let dy = y as f64 - ncy;

            // Rotate back
            let sx = dx * cos_a + dy * sin_a + cx;
            let sy = -dx * sin_a + dy * cos_a + cy;

            if sx >= 0.0 && sx < (width - 1) as f64 && sy >= 0.0 && sy < (height - 1) as f64 {
                // Bilinear interpolation
                let x0 = sx.floor() as u32;
                let y0 = sy.floor() as u32;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);

                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;

                let p00 = rgba.get_pixel(x0, y0);
                let p01 = rgba.get_pixel(x0, y1);
                let p10 = rgba.get_pixel(x1, y0);
                let p11 = rgba.get_pixel(x1, y1);

                let mut pixel = [0u8; 4];
                for c in 0..4 {
                    let v = (1.0 - fx) * (1.0 - fy) * p00.0[c] as f64
                        + fx * (1.0 - fy) * p10.0[c] as f64
                        + (1.0 - fx) * fy * p01.0[c] as f64
                        + fx * fy * p11.0[c] as f64;
                    pixel[c] = v.round().clamp(0.0, 255.0) as u8;
                }

                result.put_pixel(x, y, Rgba(pixel));
            }
        }
    }

    DynamicImage::ImageRgba8(result)
}

/// Detect edges using Sobel operator.
fn detect_edges(gray: &GrayImage) -> GrayImage {
    let (width, height) = (gray.width() as i32, gray.height() as i32);
    let mut edges = GrayImage::new(width as u32, height as u32);

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            // Sobel kernels
            let gx = -1 * gray.get_pixel((x - 1) as u32, (y - 1) as u32).0[0] as i32
                + 1 * gray.get_pixel((x + 1) as u32, (y - 1) as u32).0[0] as i32
                + -2 * gray.get_pixel((x - 1) as u32, y as u32).0[0] as i32
                + 2 * gray.get_pixel((x + 1) as u32, y as u32).0[0] as i32
                + -1 * gray.get_pixel((x - 1) as u32, (y + 1) as u32).0[0] as i32
                + 1 * gray.get_pixel((x + 1) as u32, (y + 1) as u32).0[0] as i32;

            let gy = -1 * gray.get_pixel((x - 1) as u32, (y - 1) as u32).0[0] as i32
                + -2 * gray.get_pixel(x as u32, (y - 1) as u32).0[0] as i32
                + -1 * gray.get_pixel((x + 1) as u32, (y - 1) as u32).0[0] as i32
                + 1 * gray.get_pixel((x - 1) as u32, (y + 1) as u32).0[0] as i32
                + 2 * gray.get_pixel(x as u32, (y + 1) as u32).0[0] as i32
                + 1 * gray.get_pixel((x + 1) as u32, (y + 1) as u32).0[0] as i32;

            let magnitude = ((gx * gx + gy * gy) as f64).sqrt();
            let edge_val = (magnitude / 4.0).min(255.0) as u8;

            edges.put_pixel(x as u32, y as u32, image::Luma([edge_val]));
        }
    }

    // Threshold edges
    let threshold = 50u8;
    for pixel in edges.pixels_mut() {
        pixel.0[0] = if pixel.0[0] > threshold { 255 } else { 0 };
    }

    edges
}

/// Simple Hough transform for line angle detection.
fn hough_transform(edges: &GrayImage, num_angles: usize) -> Vec<u32> {
    let (width, height) = (edges.width() as i32, edges.height() as i32);
    let mut angle_votes = vec![0u32; num_angles];

    let diagonal = ((width * width + height * height) as f64).sqrt() as i32;
    let num_rhos = 2 * diagonal;

    // Vote for each edge pixel
    for y in 0..height {
        for x in 0..width {
            if edges.get_pixel(x as u32, y as u32).0[0] > 128 {
                for angle_idx in 0..num_angles {
                    let theta = (angle_idx as f64) * std::f64::consts::PI / num_angles as f64;
                    let rho = x as f64 * theta.cos() + y as f64 * theta.sin();
                    let _rho_idx = ((rho + diagonal as f64) as i32).max(0).min(num_rhos - 1);

                    angle_votes[angle_idx] += 1;
                }
            }
        }
    }

    // Smooth the votes
    let mut smoothed = vec![0u32; num_angles];
    for i in 0..num_angles {
        let mut sum = angle_votes[i];
        let mut count = 1;
        if i > 0 {
            sum += angle_votes[i - 1];
            count += 1;
        }
        if i < num_angles - 1 {
            sum += angle_votes[i + 1];
            count += 1;
        }
        smoothed[i] = sum / count;
    }

    smoothed
}

/// Detect document corners for perspective correction.
fn detect_document_corners(image: &DynamicImage) -> Option<[(u32, u32); 4]> {
    let gray = image.to_luma8();
    let (width, height) = (gray.width(), gray.height());
    let edges = detect_edges(&gray);

    // Find corner regions using Harris-like detection
    let region_size = width.min(height) / 4;

    let corners = [
        find_corner_in_region(&edges, 0, 0, region_size, region_size, true, true),
        find_corner_in_region(
            &edges,
            width - region_size,
            0,
            region_size,
            region_size,
            false,
            true,
        ),
        find_corner_in_region(
            &edges,
            0,
            height - region_size,
            region_size,
            region_size,
            true,
            false,
        ),
        find_corner_in_region(
            &edges,
            width - region_size,
            height - region_size,
            region_size,
            region_size,
            false,
            false,
        ),
    ];

    // Check if all corners were found
    if corners.iter().all(|c| c.is_some()) {
        Some([
            corners[0].unwrap(),
            corners[1].unwrap(),
            corners[2].unwrap(),
            corners[3].unwrap(),
        ])
    } else {
        None
    }
}

fn find_corner_in_region(
    edges: &GrayImage,
    rx: u32,
    ry: u32,
    rw: u32,
    rh: u32,
    prefer_left: bool,
    prefer_top: bool,
) -> Option<(u32, u32)> {
    let (img_w, img_h) = (edges.width(), edges.height());
    let mut best_x = if prefer_left { img_w } else { 0 };
    let mut best_y = if prefer_top { img_h } else { 0 };
    let mut found = false;

    for y in ry..(ry + rh).min(img_h) {
        for x in rx..(rx + rw).min(img_w) {
            if edges.get_pixel(x, y).0[0] > 128 {
                let is_better = if prefer_left && prefer_top {
                    x < best_x || y < best_y
                } else if prefer_left && !prefer_top {
                    x < best_x || y > best_y
                } else if !prefer_left && prefer_top {
                    x > best_x || y < best_y
                } else {
                    x > best_x || y > best_y
                };

                if is_better || !found {
                    best_x = x;
                    best_y = y;
                    found = true;
                }
            }
        }
    }

    if found {
        Some((best_x, best_y))
    } else {
        None
    }
}

/// Correct perspective distortion.
fn correct_perspective(image: &DynamicImage, corners: &[(u32, u32); 4]) -> DynamicImage {
    // Simple perspective correction - for now just crop to bounding box
    // Full perspective transform would require more complex math
    let min_x = corners.iter().map(|c| c.0).min().unwrap();
    let max_x = corners.iter().map(|c| c.0).max().unwrap();
    let min_y = corners.iter().map(|c| c.1).min().unwrap();
    let max_y = corners.iter().map(|c| c.1).max().unwrap();

    let width = max_x - min_x;
    let height = max_y - min_y;

    if width > 10 && height > 10 {
        image.crop_imm(min_x, min_y, width, height)
    } else {
        image.clone()
    }
}

/// Remove borders from document image.
fn remove_borders(image: &DynamicImage, margin: u32) -> DynamicImage {
    let gray = image.to_luma8();
    let (width, height) = (gray.width(), gray.height());

    // Find content bounds
    let threshold = 250u8; // Near-white threshold

    let mut left = 0u32;
    let mut right = width;
    let mut top = 0u32;
    let mut bottom = height;

    // Find left edge
    'outer: for x in 0..width {
        for y in 0..height {
            if gray.get_pixel(x, y).0[0] < threshold {
                left = x.saturating_sub(margin);
                break 'outer;
            }
        }
    }

    // Find right edge
    'outer: for x in (0..width).rev() {
        for y in 0..height {
            if gray.get_pixel(x, y).0[0] < threshold {
                right = (x + 1 + margin).min(width);
                break 'outer;
            }
        }
    }

    // Find top edge
    'outer: for y in 0..height {
        for x in 0..width {
            if gray.get_pixel(x, y).0[0] < threshold {
                top = y.saturating_sub(margin);
                break 'outer;
            }
        }
    }

    // Find bottom edge
    'outer: for y in (0..height).rev() {
        for x in 0..width {
            if gray.get_pixel(x, y).0[0] < threshold {
                bottom = (y + 1 + margin).min(height);
                break 'outer;
            }
        }
    }

    let crop_width = right.saturating_sub(left);
    let crop_height = bottom.saturating_sub(top);

    if crop_width > 10 && crop_height > 10 {
        image.crop_imm(left, top, crop_width, crop_height)
    } else {
        image.clone()
    }
}

/// Adaptive binarization (Otsu's method + local adaptive).
fn adaptive_binarize(image: &DynamicImage, block_size: u32) -> (GrayImage, u8) {
    let gray = image.to_luma8();
    let (width, height) = (gray.width(), gray.height());

    // Global Otsu threshold
    let global_threshold = otsu_threshold(&gray);

    // Adaptive thresholding
    let mut result = GrayImage::new(width, height);
    let half_block = block_size / 2;

    for y in 0..height {
        for x in 0..width {
            // Calculate local mean
            let x_start = x.saturating_sub(half_block);
            let x_end = (x + half_block + 1).min(width);
            let y_start = y.saturating_sub(half_block);
            let y_end = (y + half_block + 1).min(height);

            let mut sum = 0u64;
            let mut count = 0u64;

            for ly in y_start..y_end {
                for lx in x_start..x_end {
                    sum += gray.get_pixel(lx, ly).0[0] as u64;
                    count += 1;
                }
            }

            let local_mean = (sum / count) as u8;
            let pixel_val = gray.get_pixel(x, y).0[0];

            // Use combination of global and local threshold
            let threshold = ((local_mean as u16 + global_threshold as u16) / 2) as u8;
            let binary = if pixel_val > threshold.saturating_sub(5) {
                255
            } else {
                0
            };

            result.put_pixel(x, y, image::Luma([binary]));
        }
    }

    (result, global_threshold)
}

/// Otsu's thresholding method.
fn otsu_threshold(gray: &GrayImage) -> u8 {
    let mut histogram = [0u64; 256];
    let total = (gray.width() * gray.height()) as f64;

    for pixel in gray.pixels() {
        histogram[pixel.0[0] as usize] += 1;
    }

    let mut sum = 0.0f64;
    for (i, &count) in histogram.iter().enumerate() {
        sum += i as f64 * count as f64;
    }

    let mut sum_b = 0.0f64;
    let mut w_b = 0.0f64;
    let mut max_variance = 0.0f64;
    let mut threshold = 0u8;

    for (i, &count) in histogram.iter().enumerate() {
        w_b += count as f64;
        if w_b == 0.0 {
            continue;
        }

        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }

        sum_b += i as f64 * count as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;

        let variance = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if variance > max_variance {
            max_variance = variance;
            threshold = i as u8;
        }
    }

    threshold
}

/// Enhance image for OCR.
fn enhance_for_ocr(image: &DynamicImage) -> DynamicImage {
    let gray = image.to_luma8();
    let (width, height) = (gray.width(), gray.height());
    let mut enhanced = gray.clone();

    // Increase contrast
    let min_val = gray.pixels().map(|p| p.0[0]).min().unwrap_or(0);
    let max_val = gray.pixels().map(|p| p.0[0]).max().unwrap_or(255);

    if max_val > min_val {
        let range = (max_val - min_val) as f32;
        for y in 0..height {
            for x in 0..width {
                let val = gray.get_pixel(x, y).0[0];
                let stretched = ((val - min_val) as f32 / range * 255.0) as u8;
                enhanced.put_pixel(x, y, image::Luma([stretched]));
            }
        }
    }

    // Sharpen
    let mut sharpened = enhanced.clone();
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let center = enhanced.get_pixel(x, y).0[0] as i32;
            let top = enhanced.get_pixel(x, y - 1).0[0] as i32;
            let bottom = enhanced.get_pixel(x, y + 1).0[0] as i32;
            let left = enhanced.get_pixel(x - 1, y).0[0] as i32;
            let right = enhanced.get_pixel(x + 1, y).0[0] as i32;

            // Unsharp mask
            let sharpened_val = center * 5 - top - bottom - left - right;
            let val = sharpened_val.clamp(0, 255) as u8;
            sharpened.put_pixel(x, y, image::Luma([val]));
        }
    }

    DynamicImage::ImageLuma8(sharpened)
}

/// Quick deskew function - detects and corrects skew automatically.
pub fn quick_deskew(image: &DynamicImage) -> DynamicImage {
    let (angle, confidence) = detect_skew(image);
    if confidence > 0.3 && angle.abs() > 0.5 && angle.abs() <= 15.0 {
        rotate_image(image, -angle)
    } else {
        image.clone()
    }
}

/// Scan enhancement - full document processing pipeline.
pub fn scan_enhance(image: &DynamicImage) -> DocumentResult {
    let config = DocumentConfig::for_ocr();
    process_document(image, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = DocumentConfig::default();
        assert!(config.deskew);
        assert!(config.perspective_correct);
        assert!(!config.binarize);
    }

    #[test]
    fn test_config_presets() {
        let ocr = DocumentConfig::for_ocr();
        assert!(ocr.binarize);
        assert!(ocr.enhance_for_ocr);

        let photo = DocumentConfig::for_photo();
        assert!(!photo.binarize);
        assert!(!photo.enhance_for_ocr);
    }

    #[test]
    fn test_detect_skew_uniform() {
        let img = DynamicImage::new_rgb8(100, 100);
        let (_angle, confidence) = detect_skew(&img);
        // Uniform image should have low confidence
        assert!(confidence < 0.5);
    }

    #[test]
    fn test_otsu_threshold() {
        let gray = GrayImage::new(10, 10);
        let threshold = otsu_threshold(&gray);
        // Uniform black image
        assert_eq!(threshold, 0);
    }

    #[test]
    fn test_process_document() {
        let img = DynamicImage::new_rgb8(100, 100);
        let config = DocumentConfig::default();
        let result = process_document(&img, &config);

        assert!(result.skew_angle.abs() < 45.0);
    }

    #[test]
    fn test_quick_deskew() {
        let img = DynamicImage::new_rgb8(100, 100);
        let result = quick_deskew(&img);
        // Should return image of valid dimensions
        assert!(result.width() > 0 && result.height() > 0);
    }

    #[test]
    fn test_adaptive_binarize() {
        let img = DynamicImage::new_rgb8(50, 50);
        let (binary, _threshold) = adaptive_binarize(&img, 15);
        assert_eq!(binary.width(), 50);
        assert_eq!(binary.height(), 50);
    }

    #[test]
    fn test_remove_borders() {
        let img = DynamicImage::new_rgb8(100, 100);
        let result = remove_borders(&img, 5);
        // Should return valid image
        assert!(result.width() > 0);
    }
}
