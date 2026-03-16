//! Image Quality Assessment module.
//!
//! Provides comprehensive image quality analysis including:
//! - Sharpness/blur detection
//! - Noise level estimation
//! - Exposure analysis
//! - Contrast measurement
//! - Color distribution analysis
//! - Overall quality scoring

use image::DynamicImage;

/// Quality assessment result.
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Overall quality score (0.0 - 100.0).
    pub overall_score: f32,
    /// Quality grade.
    pub grade: QualityGrade,
    /// Sharpness score (0.0 - 100.0). Higher = sharper.
    pub sharpness: f32,
    /// Noise level (0.0 - 100.0). Lower = cleaner.
    pub noise_level: f32,
    /// Exposure score (0.0 - 100.0). 50 = ideal, extremes = over/under.
    pub exposure: f32,
    /// Contrast score (0.0 - 100.0).
    pub contrast: f32,
    /// Saturation score (0.0 - 100.0).
    pub saturation: f32,
    /// Color distribution score (0.0 - 100.0).
    pub color_distribution: f32,
    /// Brightness level (0-255).
    pub brightness: f32,
    /// Dynamic range (0.0 - 1.0).
    pub dynamic_range: f32,
    /// Issues detected.
    pub issues: Vec<QualityIssue>,
    /// Detailed metrics.
    pub metrics: QualityMetrics,
}

impl QualityReport {
    /// Check if image has acceptable quality.
    pub fn is_acceptable(&self) -> bool {
        self.overall_score >= 50.0
    }

    /// Check if image has good quality.
    pub fn is_good(&self) -> bool {
        self.overall_score >= 70.0
    }

    /// Check if image has excellent quality.
    pub fn is_excellent(&self) -> bool {
        self.overall_score >= 85.0
    }

    /// Get primary issue if any.
    pub fn primary_issue(&self) -> Option<&QualityIssue> {
        self.issues.first()
    }

    /// Get recommendations for improvement.
    pub fn recommendations(&self) -> Vec<String> {
        let mut recs = Vec::new();

        for issue in &self.issues {
            match issue {
                QualityIssue::Blurry => {
                    recs.push("Apply sharpening or use a higher resolution source".into());
                }
                QualityIssue::Noisy => {
                    recs.push("Apply noise reduction filter".into());
                }
                QualityIssue::Overexposed => {
                    recs.push("Reduce brightness or apply exposure correction".into());
                }
                QualityIssue::Underexposed => {
                    recs.push("Increase brightness or use histogram equalization".into());
                }
                QualityIssue::LowContrast => {
                    recs.push("Apply contrast enhancement".into());
                }
                QualityIssue::Oversaturated => {
                    recs.push("Reduce saturation for more natural colors".into());
                }
                QualityIssue::Desaturated => {
                    recs.push("Increase saturation or check if grayscale is intended".into());
                }
                QualityIssue::ColorCast(color) => {
                    recs.push(format!(
                        "Apply white balance correction ({} cast detected)",
                        color
                    ));
                }
                QualityIssue::LowResolution => {
                    recs.push("Use a higher resolution source or apply AI upscaling".into());
                }
                QualityIssue::CompressionArtifacts => {
                    recs.push("Use a higher quality source or apply artifact removal".into());
                }
            }
        }

        recs
    }
}

/// Quality grade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityGrade {
    /// Excellent quality (85-100).
    Excellent,
    /// Good quality (70-85).
    Good,
    /// Acceptable quality (50-70).
    Acceptable,
    /// Poor quality (25-50).
    Poor,
    /// Very poor quality (0-25).
    VeryPoor,
}

impl QualityGrade {
    /// Get grade from score.
    pub fn from_score(score: f32) -> Self {
        if score >= 85.0 {
            Self::Excellent
        } else if score >= 70.0 {
            Self::Good
        } else if score >= 50.0 {
            Self::Acceptable
        } else if score >= 25.0 {
            Self::Poor
        } else {
            Self::VeryPoor
        }
    }

    /// Get letter grade.
    pub fn letter(&self) -> &'static str {
        match self {
            Self::Excellent => "A",
            Self::Good => "B",
            Self::Acceptable => "C",
            Self::Poor => "D",
            Self::VeryPoor => "F",
        }
    }
}

/// Quality issues that can be detected.
#[derive(Debug, Clone, PartialEq)]
pub enum QualityIssue {
    /// Image is blurry/out of focus.
    Blurry,
    /// High noise level.
    Noisy,
    /// Overexposed (too bright).
    Overexposed,
    /// Underexposed (too dark).
    Underexposed,
    /// Low contrast.
    LowContrast,
    /// Oversaturated colors.
    Oversaturated,
    /// Desaturated/washed out.
    Desaturated,
    /// Color cast detected.
    ColorCast(String),
    /// Resolution too low.
    LowResolution,
    /// Visible compression artifacts.
    CompressionArtifacts,
}

/// Detailed quality metrics.
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Laplacian variance (sharpness indicator).
    pub laplacian_variance: f64,
    /// Standard deviation of pixel values.
    pub std_deviation: f64,
    /// Histogram entropy.
    pub entropy: f64,
    /// Peak signal-to-noise ratio estimate.
    pub psnr_estimate: f64,
    /// Mean pixel value.
    pub mean_brightness: f64,
    /// Median pixel value.
    pub median_brightness: u8,
    /// Histogram data (256 bins).
    pub histogram: [u32; 256],
    /// Color histogram (RGB).
    pub color_histogram: ColorHistogram,
    /// Edge density.
    pub edge_density: f64,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            laplacian_variance: 0.0,
            std_deviation: 0.0,
            entropy: 0.0,
            psnr_estimate: 0.0,
            mean_brightness: 0.0,
            median_brightness: 0,
            histogram: [0u32; 256],
            color_histogram: ColorHistogram::default(),
            edge_density: 0.0,
        }
    }
}

/// RGB color histogram.
#[derive(Debug, Clone)]
pub struct ColorHistogram {
    pub red: [u32; 256],
    pub green: [u32; 256],
    pub blue: [u32; 256],
}

impl Default for ColorHistogram {
    fn default() -> Self {
        Self {
            red: [0u32; 256],
            green: [0u32; 256],
            blue: [0u32; 256],
        }
    }
}

/// Quality assessment configuration.
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// Weight for sharpness in overall score.
    pub sharpness_weight: f32,
    /// Weight for noise in overall score.
    pub noise_weight: f32,
    /// Weight for exposure in overall score.
    pub exposure_weight: f32,
    /// Weight for contrast in overall score.
    pub contrast_weight: f32,
    /// Weight for color in overall score.
    pub color_weight: f32,
    /// Minimum resolution threshold.
    pub min_resolution: (u32, u32),
    /// Enable compression artifact detection.
    pub detect_artifacts: bool,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            sharpness_weight: 0.30,
            noise_weight: 0.20,
            exposure_weight: 0.20,
            contrast_weight: 0.15,
            color_weight: 0.15,
            min_resolution: (640, 480),
            detect_artifacts: true,
        }
    }
}

/// Assess image quality.
pub fn assess_quality(image: &DynamicImage, config: &QualityConfig) -> QualityReport {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    let gray = image.to_luma8();

    // Calculate basic statistics
    let metrics = calculate_metrics(&gray, &rgb);

    // Calculate individual scores
    let sharpness = calculate_sharpness(&metrics);
    let noise_level = calculate_noise(&gray);
    let exposure = calculate_exposure(&metrics);
    let contrast = calculate_contrast(&metrics);
    let (saturation, color_distribution) = calculate_color_metrics(&rgb);

    // Detect issues
    let mut issues = Vec::new();

    if sharpness < 40.0 {
        issues.push(QualityIssue::Blurry);
    }
    if noise_level > 60.0 {
        issues.push(QualityIssue::Noisy);
    }
    if exposure > 80.0 {
        issues.push(QualityIssue::Overexposed);
    } else if exposure < 20.0 {
        issues.push(QualityIssue::Underexposed);
    }
    if contrast < 30.0 {
        issues.push(QualityIssue::LowContrast);
    }
    if saturation > 85.0 {
        issues.push(QualityIssue::Oversaturated);
    } else if saturation < 15.0 {
        issues.push(QualityIssue::Desaturated);
    }

    // Check color cast
    if let Some(cast) = detect_color_cast(&metrics.color_histogram) {
        issues.push(QualityIssue::ColorCast(cast));
    }

    // Check resolution
    if width < config.min_resolution.0 || height < config.min_resolution.1 {
        issues.push(QualityIssue::LowResolution);
    }

    // Check for compression artifacts
    if config.detect_artifacts && detect_compression_artifacts(&gray) {
        issues.push(QualityIssue::CompressionArtifacts);
    }

    // Calculate overall score
    let exposure_score = 100.0 - (exposure - 50.0).abs() * 2.0;
    let noise_score = 100.0 - noise_level;

    let overall_score = (sharpness * config.sharpness_weight
        + noise_score * config.noise_weight
        + exposure_score.max(0.0) * config.exposure_weight
        + contrast * config.contrast_weight
        + color_distribution * config.color_weight)
        .clamp(0.0, 100.0);

    let grade = QualityGrade::from_score(overall_score);

    // Calculate dynamic range
    let min_val = metrics.histogram.iter().position(|&x| x > 0).unwrap_or(0) as f32;
    let max_val = metrics
        .histogram
        .iter()
        .rposition(|&x| x > 0)
        .unwrap_or(255) as f32;
    let dynamic_range = (max_val - min_val) / 255.0;

    QualityReport {
        overall_score,
        grade,
        sharpness,
        noise_level,
        exposure,
        contrast,
        saturation,
        color_distribution,
        brightness: metrics.mean_brightness as f32,
        dynamic_range,
        issues,
        metrics,
    }
}

/// Assess multiple images and rank them.
pub fn rank_images(images: &[DynamicImage], config: &QualityConfig) -> Vec<(usize, QualityReport)> {
    let mut results: Vec<_> = images
        .iter()
        .enumerate()
        .map(|(i, img)| (i, assess_quality(img, config)))
        .collect();

    results.sort_by(|a, b| b.1.overall_score.partial_cmp(&a.1.overall_score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Find the best quality image from a set.
pub fn find_best_image(
    images: &[DynamicImage],
    config: &QualityConfig,
) -> Option<(usize, QualityReport)> {
    rank_images(images, config).into_iter().next()
}

/// Quick quality check - returns true if acceptable.
pub fn is_acceptable_quality(image: &DynamicImage) -> bool {
    let config = QualityConfig::default();
    let report = assess_quality(image, &config);
    report.is_acceptable()
}

// Internal helper functions

fn calculate_metrics(gray: &image::GrayImage, rgb: &image::RgbImage) -> QualityMetrics {
    let (width, height) = (gray.width() as usize, gray.height() as usize);
    let total_pixels = (width * height) as f64;

    // Calculate histogram
    let mut histogram = [0u32; 256];
    let mut color_histogram = ColorHistogram::default();
    let mut sum = 0u64;

    for (gp, cp) in gray.pixels().zip(rgb.pixels()) {
        let v = gp.0[0];
        histogram[v as usize] += 1;
        sum += v as u64;

        color_histogram.red[cp.0[0] as usize] += 1;
        color_histogram.green[cp.0[1] as usize] += 1;
        color_histogram.blue[cp.0[2] as usize] += 1;
    }

    let mean_brightness = sum as f64 / total_pixels;

    // Calculate standard deviation
    let variance: f64 = gray
        .pixels()
        .map(|p| {
            let diff = p.0[0] as f64 - mean_brightness;
            diff * diff
        })
        .sum::<f64>()
        / total_pixels;
    let std_deviation = variance.sqrt();

    // Calculate entropy
    let entropy: f64 = histogram
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total_pixels;
            -p * p.log2()
        })
        .sum();

    // Find median
    let mut cumulative = 0u64;
    let half = (total_pixels / 2.0) as u64;
    let mut median_brightness = 128u8;
    for (i, &count) in histogram.iter().enumerate() {
        cumulative += count as u64;
        if cumulative >= half {
            median_brightness = i as u8;
            break;
        }
    }

    // Calculate Laplacian variance for sharpness
    let laplacian_variance = calculate_laplacian_variance(gray);

    // Calculate edge density
    let edge_density = calculate_edge_density(gray);

    // Estimate PSNR (rough estimate based on noise)
    let psnr_estimate = if std_deviation > 0.0 {
        20.0 * (255.0 / std_deviation).log10()
    } else {
        100.0
    };

    QualityMetrics {
        laplacian_variance,
        std_deviation,
        entropy,
        psnr_estimate,
        mean_brightness,
        median_brightness,
        histogram,
        color_histogram,
        edge_density,
    }
}

fn calculate_laplacian_variance(gray: &image::GrayImage) -> f64 {
    let (width, height) = (gray.width() as i32, gray.height() as i32);
    let mut sum = 0.0f64;
    let mut count = 0u64;

    // Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let center = gray.get_pixel(x as u32, y as u32).0[0] as f64;
            let top = gray.get_pixel(x as u32, (y - 1) as u32).0[0] as f64;
            let bottom = gray.get_pixel(x as u32, (y + 1) as u32).0[0] as f64;
            let left = gray.get_pixel((x - 1) as u32, y as u32).0[0] as f64;
            let right = gray.get_pixel((x + 1) as u32, y as u32).0[0] as f64;

            let laplacian = top + bottom + left + right - 4.0 * center;
            sum += laplacian * laplacian;
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

fn calculate_edge_density(gray: &image::GrayImage) -> f64 {
    let (width, height) = (gray.width() as i32, gray.height() as i32);
    let mut edge_count = 0u64;
    let threshold = 30;

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let center = gray.get_pixel(x as u32, y as u32).0[0] as i32;
            let right = gray.get_pixel((x + 1) as u32, y as u32).0[0] as i32;
            let bottom = gray.get_pixel(x as u32, (y + 1) as u32).0[0] as i32;

            let gx = (right - center).abs();
            let gy = (bottom - center).abs();

            if gx > threshold || gy > threshold {
                edge_count += 1;
            }
        }
    }

    let total = ((width - 2) * (height - 2)) as f64;
    if total > 0.0 {
        edge_count as f64 / total
    } else {
        0.0
    }
}

fn calculate_sharpness(metrics: &QualityMetrics) -> f32 {
    // Normalize Laplacian variance to 0-100 score
    // Higher variance = sharper image
    let variance = metrics.laplacian_variance;

    // Typical sharp image has variance > 500, blurry < 100
    let score = if variance < 100.0 {
        (variance / 100.0) * 30.0
    } else if variance < 500.0 {
        30.0 + ((variance - 100.0) / 400.0) * 40.0
    } else if variance < 2000.0 {
        70.0 + ((variance - 500.0) / 1500.0) * 30.0
    } else {
        100.0
    };

    (score as f32).clamp(0.0, 100.0)
}

fn calculate_noise(gray: &image::GrayImage) -> f32 {
    // Estimate noise using high-frequency content analysis
    let (width, height) = (gray.width() as i32, gray.height() as i32);

    // Calculate local variance in smooth regions
    let block_size = 8;
    let mut min_variance = f64::MAX;

    for by in (0..height - block_size).step_by(block_size as usize) {
        for bx in (0..width - block_size).step_by(block_size as usize) {
            let mut sum = 0u64;
            let mut sq_sum = 0u64;

            for dy in 0..block_size {
                for dx in 0..block_size {
                    let v = gray.get_pixel((bx + dx) as u32, (by + dy) as u32).0[0] as u64;
                    sum += v;
                    sq_sum += v * v;
                }
            }

            let n = (block_size * block_size) as f64;
            let mean = sum as f64 / n;
            let variance = (sq_sum as f64 / n) - (mean * mean);

            if variance < min_variance {
                min_variance = variance;
            }
        }
    }

    // If no blocks were analyzed (image too small), assume no noise
    if min_variance == f64::MAX {
        return 0.0;
    }

    // Noise level based on variance in smooth regions
    let noise_estimate = min_variance.sqrt();

    // Convert to 0-100 scale (higher = more noise)
    let noise_level = (noise_estimate / 10.0 * 100.0).min(100.0);

    noise_level as f32
}

fn calculate_exposure(metrics: &QualityMetrics) -> f32 {
    // Score based on mean brightness
    // Ideal is around 127-128
    let mean = metrics.mean_brightness;

    // Convert to 0-100 scale where 50 is ideal
    ((mean / 255.0) * 100.0) as f32
}

fn calculate_contrast(metrics: &QualityMetrics) -> f32 {
    // Use standard deviation as contrast measure
    // Also consider histogram spread

    let std_score = (metrics.std_deviation / 80.0 * 100.0).min(100.0);

    // Find histogram range
    let min_val = metrics.histogram.iter().position(|&c| c > 0).unwrap_or(0);
    let max_val = metrics.histogram.iter().rposition(|&c| c > 0).unwrap_or(255);

    let range = max_val.saturating_sub(min_val) as f64;
    let range_score = (range / 255.0) * 100.0;

    // Combine both metrics
    ((std_score + range_score) / 2.0) as f32
}

fn calculate_color_metrics(rgb: &image::RgbImage) -> (f32, f32) {
    let mut total_saturation = 0.0f64;
    let mut hue_histogram = [0u32; 360];
    let pixel_count = (rgb.width() * rgb.height()) as f64;

    for pixel in rgb.pixels() {
        let r = pixel.0[0] as f64 / 255.0;
        let g = pixel.0[1] as f64 / 255.0;
        let b = pixel.0[2] as f64 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        // Saturation
        let saturation = if max > 0.0 { delta / max } else { 0.0 };
        total_saturation += saturation;

        // Hue for color distribution
        if delta > 0.01 {
            let hue = if max == r {
                60.0 * (((g - b) / delta) % 6.0)
            } else if max == g {
                60.0 * (((b - r) / delta) + 2.0)
            } else {
                60.0 * (((r - g) / delta) + 4.0)
            };

            let hue_idx = ((hue + 360.0) % 360.0) as usize;
            hue_histogram[hue_idx.min(359)] += 1;
        }
    }

    let avg_saturation = (total_saturation / pixel_count) * 100.0;

    // Color distribution - entropy of hue histogram
    let hue_total: u32 = hue_histogram.iter().sum();
    let color_entropy: f64 = if hue_total > 0 {
        hue_histogram
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / hue_total as f64;
                -p * p.log2()
            })
            .sum()
    } else {
        0.0
    };

    // Normalize entropy (max is log2(360) ≈ 8.5)
    let color_distribution = ((color_entropy / 8.5) * 100.0).min(100.0);

    (avg_saturation as f32, color_distribution as f32)
}

fn detect_color_cast(color_hist: &ColorHistogram) -> Option<String> {
    let r_mean: f64 = color_hist
        .red
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum::<f64>()
        / color_hist.red.iter().sum::<u32>().max(1) as f64;

    let g_mean: f64 = color_hist
        .green
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum::<f64>()
        / color_hist.green.iter().sum::<u32>().max(1) as f64;

    let b_mean: f64 = color_hist
        .blue
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum::<f64>()
        / color_hist.blue.iter().sum::<u32>().max(1) as f64;

    let avg = (r_mean + g_mean + b_mean) / 3.0;
    let threshold = 15.0;

    if r_mean - avg > threshold && r_mean - g_mean > threshold && r_mean - b_mean > threshold {
        Some("red".into())
    } else if g_mean - avg > threshold && g_mean - r_mean > threshold && g_mean - b_mean > threshold
    {
        Some("green".into())
    } else if b_mean - avg > threshold && b_mean - r_mean > threshold && b_mean - g_mean > threshold
    {
        Some("blue".into())
    } else if r_mean - b_mean > threshold && g_mean - b_mean > threshold {
        Some("yellow".into())
    } else if b_mean - r_mean > threshold && b_mean - g_mean > threshold {
        Some("cool/blue".into())
    } else {
        None
    }
}

fn detect_compression_artifacts(gray: &image::GrayImage) -> bool {
    // Detect 8x8 block artifacts typical of JPEG compression
    let (width, height) = (gray.width() as i32, gray.height() as i32);

    if width < 32 || height < 32 {
        return false;
    }

    let mut block_edge_sum = 0.0f64;
    let mut non_block_edge_sum = 0.0f64;
    let mut block_count = 0u64;
    let mut non_block_count = 0u64;

    // Compare edges at 8-pixel boundaries vs. non-boundaries
    for y in 8..(height - 8) {
        for x in 8..(width - 8) {
            let current = gray.get_pixel(x as u32, y as u32).0[0] as f64;
            let right = gray.get_pixel((x + 1) as u32, y as u32).0[0] as f64;
            let below = gray.get_pixel(x as u32, (y + 1) as u32).0[0] as f64;

            let h_diff = (current - right).abs();
            let v_diff = (current - below).abs();

            if x % 8 == 7 {
                block_edge_sum += h_diff;
                block_count += 1;
            } else {
                non_block_edge_sum += h_diff;
                non_block_count += 1;
            }

            if y % 8 == 7 {
                block_edge_sum += v_diff;
                block_count += 1;
            } else {
                non_block_edge_sum += v_diff;
                non_block_count += 1;
            }
        }
    }

    if block_count == 0 || non_block_count == 0 {
        return false;
    }

    let block_avg = block_edge_sum / block_count as f64;
    let non_block_avg = non_block_edge_sum / non_block_count as f64;

    // If block edges are significantly stronger, artifacts are present
    block_avg > non_block_avg * 1.3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_grade() {
        assert_eq!(QualityGrade::from_score(90.0), QualityGrade::Excellent);
        assert_eq!(QualityGrade::from_score(75.0), QualityGrade::Good);
        assert_eq!(QualityGrade::from_score(60.0), QualityGrade::Acceptable);
        assert_eq!(QualityGrade::from_score(30.0), QualityGrade::Poor);
        assert_eq!(QualityGrade::from_score(10.0), QualityGrade::VeryPoor);
    }

    #[test]
    fn test_grade_letters() {
        assert_eq!(QualityGrade::Excellent.letter(), "A");
        assert_eq!(QualityGrade::Good.letter(), "B");
        assert_eq!(QualityGrade::Acceptable.letter(), "C");
        assert_eq!(QualityGrade::Poor.letter(), "D");
        assert_eq!(QualityGrade::VeryPoor.letter(), "F");
    }

    #[test]
    fn test_assess_uniform_image() {
        let img = DynamicImage::new_rgb8(100, 100);
        let config = QualityConfig::default();
        let report = assess_quality(&img, &config);

        // Uniform image should be low quality (low contrast, low sharpness)
        assert!(report.contrast < 30.0);
    }

    #[test]
    fn test_quality_report_methods() {
        let report = QualityReport {
            overall_score: 75.0,
            grade: QualityGrade::Good,
            sharpness: 80.0,
            noise_level: 20.0,
            exposure: 50.0,
            contrast: 70.0,
            saturation: 60.0,
            color_distribution: 50.0,
            brightness: 127.0,
            dynamic_range: 0.9,
            issues: vec![QualityIssue::Noisy],
            metrics: QualityMetrics::default(),
        };

        assert!(report.is_acceptable());
        assert!(report.is_good());
        assert!(!report.is_excellent());
        assert_eq!(report.primary_issue(), Some(&QualityIssue::Noisy));
    }

    #[test]
    fn test_recommendations() {
        let report = QualityReport {
            overall_score: 40.0,
            grade: QualityGrade::Poor,
            sharpness: 30.0,
            noise_level: 70.0,
            exposure: 85.0,
            contrast: 25.0,
            saturation: 10.0,
            color_distribution: 30.0,
            brightness: 200.0,
            dynamic_range: 0.4,
            issues: vec![
                QualityIssue::Blurry,
                QualityIssue::Overexposed,
                QualityIssue::LowContrast,
            ],
            metrics: QualityMetrics::default(),
        };

        let recs = report.recommendations();
        assert_eq!(recs.len(), 3);
    }

    #[test]
    fn test_config_defaults() {
        let config = QualityConfig::default();
        assert!((config.sharpness_weight - 0.30).abs() < 0.01);
        assert!((config.noise_weight - 0.20).abs() < 0.01);
        assert_eq!(config.min_resolution, (640, 480));
    }

    #[test]
    fn test_assess_small_image_no_panic() {
        // Very small image (< block_size) should not cause calculate_noise to return MAX
        let img = DynamicImage::new_rgb8(4, 4);
        let config = QualityConfig::default();
        let report = assess_quality(&img, &config);
        // noise_level should be 0 for tiny images, not some huge value
        assert!(report.noise_level < 50.0, "Small uniform image noise should be low, got {}", report.noise_level);
    }

    #[test]
    fn test_contrast_all_zeros() {
        // Image where all pixels are value 0 - min_val should correctly find bin 0
        let img = DynamicImage::new_rgb8(10, 10);
        let config = QualityConfig::default();
        let report = assess_quality(&img, &config);
        // Contrast should be very low for uniform image
        assert!(report.contrast < 10.0);
    }

    #[test]
    fn test_quick_quality_check() {
        let img = DynamicImage::new_rgb8(100, 100);
        // Very small uniform image likely won't pass
        let result = is_acceptable_quality(&img);
        // Result depends on implementation but should not panic
        assert!(result == true || result == false);
    }
}
