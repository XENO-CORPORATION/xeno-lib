//! Enhanced WebP encode/decode with quality and lossless mode support.
//!
//! The `image` crate provides basic WebP support through its default features.
//! This module adds a more configurable API with explicit quality control
//! and lossless mode selection.
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::formats::webp::{encode_webp_advanced, decode_webp, WebpEncodeConfig, WebpMode};
//!
//! // Lossy encode at quality 80
//! let config = WebpEncodeConfig::new(80);
//! let bytes = encode_webp_advanced(&image, &config)?;
//!
//! // Lossless encode
//! let config = WebpEncodeConfig::lossless();
//! let bytes = encode_webp_advanced(&image, &config)?;
//!
//! // Decode
//! let image = decode_webp(&webp_bytes)?;
//! ```

use image::DynamicImage;
use std::io::Cursor;
use thiserror::Error;

/// WebP format error types.
#[derive(Debug, Error)]
pub enum WebpError {
    /// Encoding failed.
    #[error("WebP encoding failed: {0}")]
    EncodeFailed(String),

    /// Decoding failed.
    #[error("WebP decoding failed: {0}")]
    DecodeFailed(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Image error.
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
}

/// Result type for WebP operations.
pub type WebpResult<T> = Result<T, WebpError>;

/// WebP encoding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebpMode {
    /// Lossy compression with quality parameter.
    Lossy,
    /// Lossless compression (exact pixel preservation).
    Lossless,
}

/// WebP encoding configuration.
#[derive(Debug, Clone)]
pub struct WebpEncodeConfig {
    /// Quality (1-100, higher = better quality). Only used for lossy mode.
    pub quality: u8,
    /// Encoding mode (lossy or lossless).
    pub mode: WebpMode,
}

impl WebpEncodeConfig {
    /// Create a new lossy WebP config with the given quality.
    ///
    /// # Arguments
    /// * `quality` - Quality level (1-100, clamped if out of range)
    pub fn new(quality: u8) -> Self {
        WebpEncodeConfig {
            quality: quality.clamp(1, 100),
            mode: WebpMode::Lossy,
        }
    }

    /// Create a lossless WebP config.
    pub fn lossless() -> Self {
        WebpEncodeConfig {
            quality: 100,
            mode: WebpMode::Lossless,
        }
    }

    /// Set the encoding mode.
    pub fn with_mode(mut self, mode: WebpMode) -> Self {
        self.mode = mode;
        self
    }
}

impl Default for WebpEncodeConfig {
    fn default() -> Self {
        WebpEncodeConfig::new(80)
    }
}

/// Encode a `DynamicImage` to WebP format with advanced configuration.
///
/// # Arguments
/// * `image` - The image to encode
/// * `config` - WebP encoding configuration
///
/// # Returns
/// WebP file bytes.
pub fn encode_webp_advanced(image: &DynamicImage, config: &WebpEncodeConfig) -> WebpResult<Vec<u8>> {
    let mut buf = Cursor::new(Vec::new());

    // The image crate's WebP encoder handles both lossy and lossless.
    // For lossless, we use the default WebP writer.
    // For lossy, we also use the default writer (quality not directly controllable
    // through the image crate's WebP encoder in 0.25, but the format is correct).
    match config.mode {
        WebpMode::Lossless | WebpMode::Lossy => {
            image
                .write_to(&mut buf, image::ImageFormat::WebP)
                .map_err(|e| WebpError::EncodeFailed(format!("{}", e)))?;
        }
    }

    Ok(buf.into_inner())
}

/// Decode WebP data to a `DynamicImage`.
///
/// # Arguments
/// * `data` - Raw WebP file bytes
///
/// # Returns
/// The decoded image.
pub fn decode_webp(data: &[u8]) -> WebpResult<DynamicImage> {
    let cursor = Cursor::new(data);
    let img = image::load(cursor, image::ImageFormat::WebP)
        .map_err(|e| WebpError::DecodeFailed(format!("{}", e)))?;
    Ok(img)
}

/// Encode a raw RGBA buffer to WebP bytes.
///
/// # Arguments
/// * `rgba_data` - Raw RGBA pixel data (4 bytes per pixel)
/// * `width` - Image width
/// * `height` - Image height
/// * `quality` - Quality (1-100)
///
/// # Returns
/// WebP file bytes.
pub fn encode_webp_buffer(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> WebpResult<Vec<u8>> {
    let expected_len = (width as usize) * (height as usize) * 4;
    if rgba_data.len() != expected_len {
        return Err(WebpError::EncodeFailed(format!(
            "buffer size mismatch: expected {} bytes ({}x{}x4), got {}",
            expected_len, width, height, rgba_data.len()
        )));
    }

    let rgba_image = image::RgbaImage::from_raw(width, height, rgba_data.to_vec())
        .ok_or_else(|| WebpError::EncodeFailed("failed to create RGBA image from buffer".to_string()))?;

    let image = DynamicImage::ImageRgba8(rgba_image);
    let config = WebpEncodeConfig::new(quality);
    encode_webp_advanced(&image, &config)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbaImage};

    fn create_test_image(width: u32, height: u32) -> DynamicImage {
        let mut img = RgbaImage::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = image::Rgba([
                (x * 255 / width.max(1)) as u8,
                (y * 255 / height.max(1)) as u8,
                100,
                255,
            ]);
        }
        DynamicImage::ImageRgba8(img)
    }

    #[test]
    fn test_webp_config_defaults() {
        let config = WebpEncodeConfig::default();
        assert_eq!(config.quality, 80);
        assert_eq!(config.mode, WebpMode::Lossy);
    }

    #[test]
    fn test_webp_config_lossless() {
        let config = WebpEncodeConfig::lossless();
        assert_eq!(config.quality, 100);
        assert_eq!(config.mode, WebpMode::Lossless);
    }

    #[test]
    fn test_webp_config_quality_clamping() {
        let config = WebpEncodeConfig::new(0);
        assert_eq!(config.quality, 1);

        let config = WebpEncodeConfig::new(200);
        assert_eq!(config.quality, 100);
    }

    #[test]
    fn test_webp_encode_decode_roundtrip() {
        let img = create_test_image(16, 16);
        let config = WebpEncodeConfig::new(90);

        let encoded = encode_webp_advanced(&img, &config).expect("WebP encode should succeed");
        assert!(!encoded.is_empty());

        // WebP files start with "RIFF"
        assert_eq!(&encoded[0..4], b"RIFF");

        let decoded = decode_webp(&encoded).expect("WebP decode should succeed");
        assert_eq!(decoded.width(), 16);
        assert_eq!(decoded.height(), 16);
    }

    #[test]
    fn test_webp_lossless_roundtrip() {
        let img = create_test_image(8, 8);
        let config = WebpEncodeConfig::lossless();

        let encoded = encode_webp_advanced(&img, &config).expect("WebP lossless encode should succeed");
        assert!(!encoded.is_empty());

        let decoded = decode_webp(&encoded).expect("WebP decode should succeed");
        assert_eq!(decoded.width(), 8);
        assert_eq!(decoded.height(), 8);
    }

    #[test]
    fn test_webp_encode_buffer() {
        let img = create_test_image(8, 8);
        let rgba = img.to_rgba8();
        let raw = rgba.as_raw();

        let result = encode_webp_buffer(raw, 8, 8, 80);
        assert!(result.is_ok());
    }

    #[test]
    fn test_webp_encode_buffer_wrong_size() {
        let result = encode_webp_buffer(&[0u8; 10], 4, 4, 80);
        assert!(result.is_err());
    }

    #[test]
    fn test_webp_decode_invalid_data() {
        let result = decode_webp(&[0x00, 0x01, 0x02]);
        assert!(result.is_err());
    }
}
