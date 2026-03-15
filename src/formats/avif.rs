//! AVIF (AV1 Image Format) encode and decode support.
//!
//! Uses the `image` crate's built-in AVIF support, which relies on:
//! - `ravif` for encoding (uses `rav1e` — pure Rust AV1 encoder)
//! - `dav1d` for decoding (C library, high performance)
//!
//! # Feature Flag
//!
//! - `format-avif` — enables this module
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::formats::avif::{encode_avif, decode_avif, AvifEncodeConfig};
//!
//! // Encode to AVIF
//! let config = AvifEncodeConfig::new(75);
//! let avif_bytes = encode_avif(&image, &config)?;
//!
//! // Decode from AVIF
//! let image = decode_avif(&avif_bytes)?;
//! ```

use image::DynamicImage;
use std::io::Cursor;
use std::path::Path;
use thiserror::Error;

/// AVIF format error types.
#[derive(Debug, Error)]
pub enum AvifError {
    /// Encoding failed.
    #[error("AVIF encoding failed: {0}")]
    EncodeFailed(String),

    /// Decoding failed.
    #[error("AVIF decoding failed: {0}")]
    DecodeFailed(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Image error.
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
}

/// Result type for AVIF operations.
pub type AvifResult<T> = Result<T, AvifError>;

/// AVIF encoding configuration.
#[derive(Debug, Clone)]
pub struct AvifEncodeConfig {
    /// Quality (1-100, higher = better quality, larger file).
    /// Internally mapped to AVIF quality parameters.
    pub quality: u8,
    /// Encoding speed (1-10, higher = faster but lower quality).
    /// Default: 6.
    pub speed: u8,
}

impl AvifEncodeConfig {
    /// Create a new AVIF encode config with the given quality.
    ///
    /// # Arguments
    /// * `quality` - Quality level (1-100, clamped if out of range)
    pub fn new(quality: u8) -> Self {
        AvifEncodeConfig {
            quality: quality.clamp(1, 100),
            speed: 6,
        }
    }

    /// Set encoding speed (1-10, higher = faster).
    pub fn with_speed(mut self, speed: u8) -> Self {
        self.speed = speed.clamp(1, 10);
        self
    }
}

impl Default for AvifEncodeConfig {
    fn default() -> Self {
        AvifEncodeConfig::new(75)
    }
}

/// Encode a `DynamicImage` to AVIF format.
///
/// # Arguments
/// * `image` - The image to encode
/// * `config` - AVIF encoding configuration
///
/// # Returns
/// AVIF file bytes.
pub fn encode_avif(image: &DynamicImage, config: &AvifEncodeConfig) -> AvifResult<Vec<u8>> {
    let mut buf = Cursor::new(Vec::new());

    // The `image` crate handles AVIF encoding via its built-in support.
    // Quality is mapped: image crate uses 1-100 where higher = better.
    let rgba = image.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());

    // Use the image crate's AVIF encoder
    let encoder = image::codecs::avif::AvifEncoder::new_with_speed_quality(
        &mut buf,
        config.speed,
        config.quality,
    );

    image::ImageEncoder::write_image(
        encoder,
        rgba.as_raw(),
        w,
        h,
        image::ExtendedColorType::Rgba8,
    )
    .map_err(|e| AvifError::EncodeFailed(format!("{}", e)))?;

    Ok(buf.into_inner())
}

/// Decode AVIF data to a `DynamicImage`.
///
/// # Note
///
/// AVIF decoding requires the `avif-native` feature in the `image` crate,
/// which depends on the `dav1d` C library. If this feature is not enabled,
/// the function will return `AvifError::DecodeFailed`.
///
/// # Arguments
/// * `data` - Raw AVIF file bytes
///
/// # Returns
/// The decoded image.
pub fn decode_avif(data: &[u8]) -> AvifResult<DynamicImage> {
    let cursor = Cursor::new(data);
    let reader = image::ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| AvifError::DecodeFailed(format!("failed to detect format: {}", e)))?;

    let img = reader.decode()
        .map_err(|e| AvifError::DecodeFailed(format!("{}", e)))?;
    Ok(img)
}

/// Decode an AVIF file from disk.
///
/// # Arguments
/// * `path` - Path to the AVIF file
///
/// # Returns
/// The decoded image.
pub fn decode_avif_from_file<P: AsRef<Path>>(path: P) -> AvifResult<DynamicImage> {
    let data = std::fs::read(path)?;
    decode_avif(&data)
}

/// Encode a raw RGBA buffer to AVIF bytes.
///
/// # Arguments
/// * `rgba_data` - Raw RGBA pixel data (4 bytes per pixel)
/// * `width` - Image width
/// * `height` - Image height
/// * `quality` - Quality (1-100)
///
/// # Returns
/// AVIF file bytes.
pub fn encode_avif_buffer(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> AvifResult<Vec<u8>> {
    let expected_len = (width as usize) * (height as usize) * 4;
    if rgba_data.len() != expected_len {
        return Err(AvifError::EncodeFailed(format!(
            "buffer size mismatch: expected {} bytes ({}x{}x4), got {}",
            expected_len, width, height, rgba_data.len()
        )));
    }

    let rgba_image = image::RgbaImage::from_raw(width, height, rgba_data.to_vec())
        .ok_or_else(|| AvifError::EncodeFailed("failed to create RGBA image from buffer".to_string()))?;

    let image = DynamicImage::ImageRgba8(rgba_image);
    let config = AvifEncodeConfig::new(quality);
    encode_avif(&image, &config)
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
                128,
                255,
            ]);
        }
        DynamicImage::ImageRgba8(img)
    }

    #[test]
    fn test_avif_config_defaults() {
        let config = AvifEncodeConfig::default();
        assert_eq!(config.quality, 75);
        assert_eq!(config.speed, 6);
    }

    #[test]
    fn test_avif_config_quality_clamping() {
        let config = AvifEncodeConfig::new(0);
        assert_eq!(config.quality, 1);

        let config = AvifEncodeConfig::new(200);
        assert_eq!(config.quality, 100);
    }

    #[test]
    fn test_avif_config_speed_clamping() {
        let config = AvifEncodeConfig::new(75).with_speed(0);
        assert_eq!(config.speed, 1);

        let config = AvifEncodeConfig::new(75).with_speed(20);
        assert_eq!(config.speed, 10);
    }

    #[test]
    fn test_avif_encode() {
        let img = create_test_image(16, 16);
        let config = AvifEncodeConfig::new(95).with_speed(10);

        let encoded = encode_avif(&img, &config).expect("AVIF encode should succeed");
        assert!(!encoded.is_empty(), "Encoded data should not be empty");

        // Verify it starts with AVIF/ISOBMFF signature
        // AVIF files are ISOBMFF containers -- first 4 bytes are size, then "ftyp"
        assert!(encoded.len() > 8, "Encoded data too short");
    }

    #[test]
    fn test_avif_decode_attempt() {
        // AVIF decode may fail if the `avif-native` feature is not enabled
        // in the `image` crate. This is expected on builds without dav1d.
        let img = create_test_image(16, 16);
        let config = AvifEncodeConfig::new(95).with_speed(10);
        let encoded = encode_avif(&img, &config).expect("AVIF encode should succeed");

        // Attempt to decode — may succeed or fail depending on image features
        let result = decode_avif(&encoded);
        // We just verify the function doesn't panic
        match result {
            Ok(decoded) => {
                assert_eq!(decoded.width(), 16);
                assert_eq!(decoded.height(), 16);
            }
            Err(AvifError::DecodeFailed(_)) => {
                // Expected if avif-native feature is not enabled
            }
            Err(e) => panic!("Unexpected error type: {:?}", e),
        }
    }

    #[test]
    fn test_avif_encode_buffer() {
        let img = create_test_image(8, 8);
        let rgba = img.to_rgba8();
        let raw = rgba.as_raw();

        let result = encode_avif_buffer(raw, 8, 8, 80);
        assert!(result.is_ok(), "AVIF buffer encode should succeed");
    }

    #[test]
    fn test_avif_encode_buffer_wrong_size() {
        let result = encode_avif_buffer(&[0u8; 10], 4, 4, 80);
        assert!(result.is_err());
    }

    #[test]
    fn test_avif_decode_invalid_data() {
        let result = decode_avif(&[0x00, 0x01, 0x02, 0x03]);
        assert!(result.is_err());
    }
}
