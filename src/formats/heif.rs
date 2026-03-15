//! HEIF/HEIC (High Efficiency Image Format) support.
//!
//! # Status
//!
//! **Stub implementation** — HEIF support requires `libheif-rs` (C library bindings
//! to libheif). No pure Rust HEIF codec exists as of 2026.
//!
//! # Architecture
//!
//! When `libheif-rs` is integrated, this module will:
//! 1. Use `libheif_rs::HeifContext` for decoding HEIF/HEIC files
//! 2. Use `libheif_rs::Encoder` for encoding to HEIF format
//! 3. Support both HEVC and AV1 payloads within the HEIF container
//!
//! # Feature Flag
//!
//! - `format-heif` — enables this module
//!
//! # Future Integration
//!
//! To activate HEIF support:
//! 1. Add `libheif-rs = { version = "2.5", optional = true, features = ["embedded-libheif"] }` to Cargo.toml
//! 2. Change `format-heif` feature to depend on `dep:libheif-rs`
//! 3. Implement encode/decode using `libheif_rs` API
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::formats::heif::{decode_heif, encode_heif, HeifEncodeConfig};
//!
//! // Decode HEIF/HEIC
//! let image = decode_heif(&heic_bytes)?;
//!
//! // Encode to HEIF
//! let config = HeifEncodeConfig::new(75);
//! let heif_bytes = encode_heif(&image, &config)?;
//! ```

use image::DynamicImage;
use std::path::Path;
use thiserror::Error;

/// HEIF format error types.
#[derive(Debug, Error)]
pub enum HeifError {
    /// HEIF support is not available (stub implementation).
    #[error("HEIF support not available: requires libheif-rs C bindings. Enable format-heif with libheif-rs when available.")]
    NotAvailable,

    /// Encoding failed.
    #[error("HEIF encoding failed: {0}")]
    EncodeFailed(String),

    /// Decoding failed.
    #[error("HEIF decoding failed: {0}")]
    DecodeFailed(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for HEIF operations.
pub type HeifResult<T> = Result<T, HeifError>;

/// HEIF encoding configuration.
#[derive(Debug, Clone)]
pub struct HeifEncodeConfig {
    /// Quality (1-100, higher = better quality, larger file).
    pub quality: u8,
    /// Use lossless encoding.
    pub lossless: bool,
}

impl HeifEncodeConfig {
    /// Create a new HEIF encode config with the given quality.
    pub fn new(quality: u8) -> Self {
        HeifEncodeConfig {
            quality: quality.clamp(1, 100),
            lossless: false,
        }
    }

    /// Enable lossless encoding.
    pub fn with_lossless(mut self) -> Self {
        self.lossless = true;
        self
    }
}

impl Default for HeifEncodeConfig {
    fn default() -> Self {
        HeifEncodeConfig::new(75)
    }
}

/// Check if HEIF support is available in the current build.
///
/// Returns `false` until libheif-rs is integrated.
pub fn is_heif_available() -> bool {
    false
}

/// Encode a `DynamicImage` to HEIF format.
///
/// # Status
///
/// **Currently a stub** — returns `HeifError::NotAvailable`.
pub fn encode_heif(_image: &DynamicImage, _config: &HeifEncodeConfig) -> HeifResult<Vec<u8>> {
    Err(HeifError::NotAvailable)
}

/// Decode HEIF/HEIC data to a `DynamicImage`.
///
/// # Status
///
/// **Currently a stub** — returns `HeifError::NotAvailable`.
pub fn decode_heif(_data: &[u8]) -> HeifResult<DynamicImage> {
    Err(HeifError::NotAvailable)
}

/// Decode a HEIF/HEIC file from disk.
///
/// # Status
///
/// **Currently a stub** — returns `HeifError::NotAvailable`.
pub fn decode_heif_from_file<P: AsRef<Path>>(_path: P) -> HeifResult<DynamicImage> {
    Err(HeifError::NotAvailable)
}

/// Encode a raw RGBA buffer to HEIF bytes.
///
/// # Status
///
/// **Currently a stub** — returns `HeifError::NotAvailable`.
pub fn encode_heif_buffer(
    _rgba_data: &[u8],
    _width: u32,
    _height: u32,
    _quality: u8,
) -> HeifResult<Vec<u8>> {
    Err(HeifError::NotAvailable)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heif_not_available() {
        assert!(!is_heif_available());
    }

    #[test]
    fn test_heif_encode_returns_not_available() {
        let img = DynamicImage::new_rgba8(4, 4);
        let config = HeifEncodeConfig::default();
        let result = encode_heif(&img, &config);
        assert!(result.is_err());
        match result.unwrap_err() {
            HeifError::NotAvailable => {}
            e => panic!("Expected NotAvailable, got {:?}", e),
        }
    }

    #[test]
    fn test_heif_decode_returns_not_available() {
        let result = decode_heif(&[0x00, 0x01, 0x02]);
        assert!(result.is_err());
        match result.unwrap_err() {
            HeifError::NotAvailable => {}
            e => panic!("Expected NotAvailable, got {:?}", e),
        }
    }

    #[test]
    fn test_heif_config_defaults() {
        let config = HeifEncodeConfig::default();
        assert_eq!(config.quality, 75);
        assert!(!config.lossless);
    }

    #[test]
    fn test_heif_config_quality_clamping() {
        let config = HeifEncodeConfig::new(0);
        assert_eq!(config.quality, 1);

        let config = HeifEncodeConfig::new(200);
        assert_eq!(config.quality, 100);
    }

    #[test]
    fn test_heif_config_lossless() {
        let config = HeifEncodeConfig::new(75).with_lossless();
        assert!(config.lossless);
    }

    #[test]
    fn test_heif_encode_buffer_returns_not_available() {
        let result = encode_heif_buffer(&[0u8; 64], 4, 4, 75);
        assert!(result.is_err());
    }

    #[test]
    fn test_heif_decode_file_returns_not_available() {
        let result = decode_heif_from_file("/tmp/test.heic");
        assert!(result.is_err());
    }
}
