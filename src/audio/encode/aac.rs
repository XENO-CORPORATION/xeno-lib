//! AAC audio encoding for video export audio tracks.
//!
//! # Status
//!
//! **Stub implementation** — no pure Rust AAC encoder exists as of 2026.
//! The recommended path forward is to integrate `fdk-aac` (C library bindings
//! via the `fdk-aac` crate) behind the `audio-encode-aac` feature flag.
//!
//! # Architecture
//!
//! When a real AAC encoder is linked, the `encode_aac` function will:
//! 1. Accept interleaved f32 PCM samples
//! 2. Convert to the encoder's expected format (16-bit interleaved)
//! 3. Encode frames into raw AAC (ADTS or raw bitstream)
//! 4. Return the encoded bytes
//!
//! # Feature Flag
//!
//! - `audio-encode-aac` — enables this module
//!
//! # Future Integration
//!
//! To activate AAC encoding:
//! 1. Add `fdk-aac = { version = "0.8", optional = true }` to Cargo.toml
//! 2. Change `audio-encode-aac` feature to depend on `dep:fdk-aac`
//! 3. Implement the encoding using `fdk_aac::enc::Encoder`
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::audio::encode::aac::{encode_aac, encode_aac_to_bytes, AacEncoderConfig, AacProfile};
//!
//! let config = AacEncoderConfig {
//!     sample_rate: 44100,
//!     channels: 2,
//!     bitrate: 128000,
//!     profile: AacProfile::Lc,
//! };
//! let bytes = encode_aac_to_bytes(&samples, &config)?;
//! ```

use std::path::{Path, PathBuf};
use thiserror::Error;

/// AAC encoding error types.
#[derive(Debug, Error)]
pub enum AacEncodeError {
    /// AAC encoder is not available (stub implementation).
    #[error("AAC encoder not available: no pure Rust AAC encoder exists. Enable fdk-aac C bindings when available.")]
    EncoderNotAvailable,

    /// Invalid configuration.
    #[error("invalid AAC configuration: {0}")]
    InvalidConfig(String),

    /// Encoding failed.
    #[error("AAC encoding failed: {0}")]
    EncodeFailed(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to create output file.
    #[error("failed to create output file: {path}")]
    OpenFailed {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Result type for AAC encoding operations.
pub type AacEncodeResult<T> = Result<T, AacEncodeError>;

/// AAC profile (encoding mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AacProfile {
    /// AAC-LC (Low Complexity) — most common, good quality at 128+ kbps.
    Lc,
    /// HE-AAC (High Efficiency) — better at low bitrates (48-96 kbps).
    He,
    /// HE-AACv2 — best at very low bitrates (24-48 kbps), stereo only.
    HeV2,
}

impl AacProfile {
    /// Get the human-readable name for this profile.
    pub fn name(&self) -> &'static str {
        match self {
            AacProfile::Lc => "AAC-LC",
            AacProfile::He => "HE-AAC",
            AacProfile::HeV2 => "HE-AACv2",
        }
    }
}

impl Default for AacProfile {
    fn default() -> Self {
        AacProfile::Lc
    }
}

/// AAC encoder configuration.
#[derive(Debug, Clone)]
pub struct AacEncoderConfig {
    /// Sample rate in Hz (e.g., 44100, 48000).
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u32,
    /// Target bitrate in bits per second (e.g., 128000, 192000, 256000, 320000).
    pub bitrate: u32,
    /// AAC profile to use.
    pub profile: AacProfile,
}

impl AacEncoderConfig {
    /// Create a new AAC encoder configuration with default settings.
    ///
    /// Defaults: AAC-LC, 128 kbps.
    pub fn new(sample_rate: u32, channels: u32) -> Self {
        AacEncoderConfig {
            sample_rate,
            channels,
            bitrate: 128_000,
            profile: AacProfile::Lc,
        }
    }

    /// Set the target bitrate.
    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.bitrate = bitrate;
        self
    }

    /// Set the AAC profile.
    pub fn with_profile(mut self, profile: AacProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> AacEncodeResult<()> {
        if self.sample_rate == 0 || self.sample_rate > 96_000 {
            return Err(AacEncodeError::InvalidConfig(format!(
                "sample rate must be between 1 and 96000, got {}",
                self.sample_rate
            )));
        }
        if self.channels == 0 || self.channels > 8 {
            return Err(AacEncodeError::InvalidConfig(format!(
                "channel count must be between 1 and 8, got {}",
                self.channels
            )));
        }
        if self.bitrate < 8_000 || self.bitrate > 576_000 {
            return Err(AacEncodeError::InvalidConfig(format!(
                "bitrate must be between 8000 and 576000, got {}",
                self.bitrate
            )));
        }
        if self.profile == AacProfile::HeV2 && self.channels != 2 {
            return Err(AacEncodeError::InvalidConfig(
                "HE-AACv2 requires exactly 2 channels (stereo)".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for AacEncoderConfig {
    fn default() -> Self {
        AacEncoderConfig::new(44100, 2)
    }
}

/// Encode interleaved f32 PCM samples to raw AAC bytes.
///
/// # Status
///
/// **Currently a stub** — returns `AacEncodeError::EncoderNotAvailable`.
/// Will be implemented when fdk-aac C bindings are integrated.
///
/// # Arguments
/// * `samples` - Interleaved audio samples (f32, normalized -1.0 to 1.0)
/// * `config` - AAC encoding configuration
///
/// # Returns
/// Raw AAC encoded bytes (ADTS format)
pub fn encode_aac(samples: &[f32], config: &AacEncoderConfig) -> AacEncodeResult<Vec<u8>> {
    config.validate()?;

    if samples.is_empty() {
        return Err(AacEncodeError::InvalidConfig(
            "sample buffer must not be empty".to_string(),
        ));
    }

    // Stub: no AAC encoder is available yet
    Err(AacEncodeError::EncoderNotAvailable)
}

/// Encode interleaved f32 PCM samples to AAC and write to a file.
///
/// # Status
///
/// **Currently a stub** — returns `AacEncodeError::EncoderNotAvailable`.
///
/// # Arguments
/// * `samples` - Interleaved audio samples (f32, normalized -1.0 to 1.0)
/// * `output` - Output file path
/// * `config` - AAC encoding configuration
pub fn encode_aac_to_file<P: AsRef<Path>>(
    samples: &[f32],
    _output: P,
    config: &AacEncoderConfig,
) -> AacEncodeResult<u64> {
    config.validate()?;

    if samples.is_empty() {
        return Err(AacEncodeError::InvalidConfig(
            "sample buffer must not be empty".to_string(),
        ));
    }

    // Stub: no AAC encoder is available yet
    Err(AacEncodeError::EncoderNotAvailable)
}

/// Encode interleaved f32 PCM samples to AAC bytes (in memory).
///
/// Alias for `encode_aac` for API consistency with other encoders.
pub fn encode_aac_to_bytes(samples: &[f32], config: &AacEncoderConfig) -> AacEncodeResult<Vec<u8>> {
    encode_aac(samples, config)
}

/// Check if AAC encoding is available in the current build.
///
/// Returns `false` until fdk-aac C bindings are integrated.
pub fn is_aac_available() -> bool {
    false
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aac_not_available() {
        assert!(!is_aac_available());
    }

    #[test]
    fn test_aac_encode_returns_not_available() {
        let config = AacEncoderConfig::default();
        let samples = vec![0.0f32; 1024];
        let result = encode_aac(&samples, &config);
        assert!(result.is_err());
        match result.unwrap_err() {
            AacEncodeError::EncoderNotAvailable => {}
            e => panic!("Expected EncoderNotAvailable, got {:?}", e),
        }
    }

    #[test]
    fn test_aac_config_validation() {
        // Valid config
        let config = AacEncoderConfig::new(44100, 2).with_bitrate(128_000);
        assert!(config.validate().is_ok());

        // Invalid sample rate
        let config = AacEncoderConfig::new(0, 2);
        assert!(config.validate().is_err());

        // Invalid channels
        let config = AacEncoderConfig::new(44100, 0);
        assert!(config.validate().is_err());

        // Invalid bitrate
        let config = AacEncoderConfig::new(44100, 2).with_bitrate(0);
        assert!(config.validate().is_err());

        // HE-AACv2 requires stereo
        let config = AacEncoderConfig::new(44100, 1).with_profile(AacProfile::HeV2);
        assert!(config.validate().is_err());

        // HE-AACv2 with stereo is fine
        let config = AacEncoderConfig::new(44100, 2).with_profile(AacProfile::HeV2);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_aac_profile_names() {
        assert_eq!(AacProfile::Lc.name(), "AAC-LC");
        assert_eq!(AacProfile::He.name(), "HE-AAC");
        assert_eq!(AacProfile::HeV2.name(), "HE-AACv2");
    }

    #[test]
    fn test_empty_samples_rejected() {
        let config = AacEncoderConfig::default();
        let result = encode_aac(&[], &config);
        assert!(result.is_err());
        match result.unwrap_err() {
            AacEncodeError::InvalidConfig(msg) => {
                assert!(msg.contains("empty"));
            }
            e => panic!("Expected InvalidConfig, got {:?}", e),
        }
    }

    #[test]
    fn test_encode_to_bytes_alias() {
        let config = AacEncoderConfig::default();
        let samples = vec![0.0f32; 1024];
        let result = encode_aac_to_bytes(&samples, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_to_file_stub() {
        let config = AacEncoderConfig::default();
        let samples = vec![0.0f32; 1024];
        let result = encode_aac_to_file(&samples, "/tmp/test.aac", &config);
        assert!(result.is_err());
    }
}
