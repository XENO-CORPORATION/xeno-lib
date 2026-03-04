//! Raster-to-vector conversion helpers (PNG/JPG/etc -> SVG).
//!
//! Backed by `vtracer` with a typed config suitable for xeno-lib pipelines.

use std::path::Path;

use image::DynamicImage;
use thiserror::Error;

/// High-level preset tuned for common image categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorizePreset {
    /// Binary black/white tracing (logos, line-art).
    Bw,
    /// Balanced color tracing for flat illustrations.
    Poster,
    /// Photo-oriented tracing with higher detail.
    Photo,
}

impl Default for VectorizePreset {
    fn default() -> Self {
        Self::Photo
    }
}

impl VectorizePreset {
    fn to_vtracer(self) -> vtracer::Preset {
        match self {
            Self::Bw => vtracer::Preset::Bw,
            Self::Poster => vtracer::Preset::Poster,
            Self::Photo => vtracer::Preset::Photo,
        }
    }
}

/// Color mode for vectorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorizeColorMode {
    /// Preserve and cluster colors.
    Color,
    /// Convert to black/white tracing.
    Binary,
}

impl VectorizeColorMode {
    fn to_vtracer(self) -> vtracer::ColorMode {
        match self {
            Self::Color => vtracer::ColorMode::Color,
            Self::Binary => vtracer::ColorMode::Binary,
        }
    }
}

/// Layering strategy for path generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorizeHierarchy {
    /// Draw layers in stack order.
    Stacked,
    /// Cut lower layers by upper layers.
    Cutout,
}

impl VectorizeHierarchy {
    fn to_vtracer(self) -> vtracer::Hierarchical {
        match self {
            Self::Stacked => vtracer::Hierarchical::Stacked,
            Self::Cutout => vtracer::Hierarchical::Cutout,
        }
    }
}

/// Vectorization options. Use preset defaults and override as needed.
#[derive(Debug, Clone)]
pub struct VectorizeConfig {
    /// Base preset.
    pub preset: VectorizePreset,
    /// Optional color mode override.
    pub color_mode: Option<VectorizeColorMode>,
    /// Optional hierarchy override.
    pub hierarchical: Option<VectorizeHierarchy>,
    /// Remove small blobs (in pixels). Lower keeps more detail.
    pub filter_speckle: Option<u32>,
    /// Color precision [1..=8], higher preserves more colors.
    pub color_precision: Option<u8>,
    /// Difference threshold between layers [0..=255].
    pub layer_difference: Option<u8>,
    /// Corner threshold in degrees [0..=180].
    pub corner_threshold: Option<u16>,
    /// Minimum segment length (> 0.0).
    pub length_threshold: Option<f64>,
    /// Curve fitting iterations (> 0).
    pub max_iterations: Option<u32>,
    /// Splice threshold in degrees [0..=180].
    pub splice_threshold: Option<u16>,
    /// Coordinate precision in output paths.
    pub path_precision: Option<u32>,
}

impl Default for VectorizeConfig {
    fn default() -> Self {
        Self::from_preset(VectorizePreset::Photo)
    }
}

impl VectorizeConfig {
    /// Create options from preset.
    pub fn from_preset(preset: VectorizePreset) -> Self {
        Self {
            preset,
            color_mode: None,
            hierarchical: None,
            filter_speckle: None,
            color_precision: None,
            layer_difference: None,
            corner_threshold: None,
            length_threshold: None,
            max_iterations: None,
            splice_threshold: None,
            path_precision: Some(2),
        }
    }
}

/// Vectorization errors.
#[derive(Debug, Error)]
pub enum VectorizeError {
    #[error("image has zero width or height")]
    EmptyImage,
    #[error("invalid vectorize config: {0}")]
    InvalidConfig(String),
    #[error("vectorization failed: {0}")]
    Vectorization(String),
    #[error("failed to write svg output: {0}")]
    Io(#[from] std::io::Error),
}

/// Vectorize an in-memory image and return SVG content.
pub fn vectorize_image_to_svg_string(
    image: &DynamicImage,
    config: &VectorizeConfig,
) -> Result<String, VectorizeError> {
    let rgba = image.to_rgba8();
    let width = rgba.width() as usize;
    let height = rgba.height() as usize;
    if width == 0 || height == 0 {
        return Err(VectorizeError::EmptyImage);
    }

    let cfg = build_vtracer_config(config)?;
    let color_image = vtracer::ColorImage {
        pixels: rgba.into_raw(),
        width,
        height,
    };
    let svg = vtracer::convert(color_image, cfg).map_err(VectorizeError::Vectorization)?;
    Ok(svg.to_string())
}

/// Vectorize an in-memory image and write SVG file.
pub fn vectorize_image_to_svg_file(
    image: &DynamicImage,
    output_path: &Path,
    config: &VectorizeConfig,
) -> Result<(), VectorizeError> {
    let svg = vectorize_image_to_svg_string(image, config)?;
    std::fs::write(output_path, svg)?;
    Ok(())
}

/// Vectorize an input image file and write SVG output.
pub fn vectorize_file_to_svg(
    input_path: &Path,
    output_path: &Path,
    config: &VectorizeConfig,
) -> Result<(), VectorizeError> {
    let cfg = build_vtracer_config(config)?;
    vtracer::convert_image_to_svg(input_path, output_path, cfg)
        .map_err(VectorizeError::Vectorization)
}

fn build_vtracer_config(config: &VectorizeConfig) -> Result<vtracer::Config, VectorizeError> {
    validate_config(config)?;

    let mut cfg = vtracer::Config::from_preset(config.preset.to_vtracer());

    if let Some(color_mode) = config.color_mode {
        cfg.color_mode = color_mode.to_vtracer();
    }
    if let Some(hierarchical) = config.hierarchical {
        cfg.hierarchical = hierarchical.to_vtracer();
    }
    if let Some(filter_speckle) = config.filter_speckle {
        cfg.filter_speckle = filter_speckle as usize;
    }
    if let Some(color_precision) = config.color_precision {
        cfg.color_precision = color_precision as i32;
    }
    if let Some(layer_difference) = config.layer_difference {
        cfg.layer_difference = layer_difference as i32;
    }
    if let Some(corner_threshold) = config.corner_threshold {
        cfg.corner_threshold = corner_threshold as i32;
    }
    if let Some(length_threshold) = config.length_threshold {
        cfg.length_threshold = length_threshold;
    }
    if let Some(max_iterations) = config.max_iterations {
        cfg.max_iterations = max_iterations as usize;
    }
    if let Some(splice_threshold) = config.splice_threshold {
        cfg.splice_threshold = splice_threshold as i32;
    }
    cfg.path_precision = config.path_precision;

    Ok(cfg)
}

fn validate_config(config: &VectorizeConfig) -> Result<(), VectorizeError> {
    if let Some(color_precision) = config.color_precision {
        if !(1..=8).contains(&color_precision) {
            return Err(VectorizeError::InvalidConfig(
                "color_precision must be in range 1..=8".to_string(),
            ));
        }
    }
    if let Some(corner_threshold) = config.corner_threshold {
        if corner_threshold > 180 {
            return Err(VectorizeError::InvalidConfig(
                "corner_threshold must be in range 0..=180".to_string(),
            ));
        }
    }
    if let Some(splice_threshold) = config.splice_threshold {
        if splice_threshold > 180 {
            return Err(VectorizeError::InvalidConfig(
                "splice_threshold must be in range 0..=180".to_string(),
            ));
        }
    }
    if let Some(length_threshold) = config.length_threshold {
        if !(length_threshold.is_finite() && length_threshold > 0.0) {
            return Err(VectorizeError::InvalidConfig(
                "length_threshold must be a finite number greater than 0".to_string(),
            ));
        }
    }
    if let Some(max_iterations) = config.max_iterations {
        if max_iterations == 0 {
            return Err(VectorizeError::InvalidConfig(
                "max_iterations must be greater than 0".to_string(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, Rgba, RgbaImage};

    #[test]
    fn vectorize_returns_svg_document() {
        let mut img = RgbaImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                let px = if x < 8 { [255, 0, 0, 255] } else { [0, 0, 255, 255] };
                img.put_pixel(x, y, Rgba(px));
            }
        }

        let svg = vectorize_image_to_svg_string(
            &DynamicImage::ImageRgba8(img),
            &VectorizeConfig::default(),
        )
        .expect("vectorize svg");
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn invalid_config_is_rejected() {
        let cfg = VectorizeConfig {
            length_threshold: Some(0.0),
            ..VectorizeConfig::default()
        };
        let err = build_vtracer_config(&cfg).expect_err("invalid config");
        assert!(matches!(err, VectorizeError::InvalidConfig(_)));
    }
}
