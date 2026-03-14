//! Audio encoding using pure Rust libraries.
//!
//! Supports:
//! - WAV (via hound) - Lossless, widely compatible
//! - FLAC (via flacenc) - Lossless compression (~60% size reduction)
//! - Ogg Opus (via audiopus) - High-quality lossy compression
//!
//! # Example: Encode audio to WAV
//!
//! ```ignore
//! use xeno_lib::audio::encode::{encode_wav, WavConfig};
//!
//! let samples: Vec<f32> = load_samples();
//! let config = WavConfig::new(44100, 2).with_bits(24);
//! encode_wav(&samples, "output.wav", config)?;
//! ```
//!
//! # Example: Encode audio to FLAC
//!
//! ```ignore
//! use xeno_lib::audio::encode::{encode_flac, FlacConfig};
//!
//! let samples: Vec<f32> = load_samples();
//! let config = FlacConfig::new(44100, 2);
//! encode_flac(&samples, "output.flac", config)?;
//! ```
//!
//! # Example: Encode audio to Opus
//!
//! ```ignore
//! use xeno_lib::audio::encode::opus::{OpusEncoder, OpusEncoderConfig};
//!
//! let config = OpusEncoderConfig::new(48000, 2).with_bitrate(128000);
//! let mut encoder = OpusEncoder::new(config)?;
//! let packet = encoder.encode(&pcm_samples)?;
//! ```

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::path::PathBuf;
use thiserror::Error;

// Opus encoding submodule
#[cfg(feature = "audio-encode-opus")]
pub mod opus;

#[cfg(feature = "audio-encode-opus")]
pub use opus::{
    encode_opus, encode_opus_ogg, encode_opus_ogg_to_bytes, OpusApplication, OpusEncoder,
    OpusEncoderConfig, OpusError, OpusResult,
};

/// Audio encoding error types.
#[derive(Debug, Error)]
pub enum AudioEncodeError {
    /// Failed to create output file.
    #[error("failed to create output file: {path}")]
    OpenFailed {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Encoding failed.
    #[error("encoding failed: {0}")]
    EncodeFailed(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for audio encoding operations.
pub type AudioEncodeResult<T> = Result<T, AudioEncodeError>;

/// WAV encoding configuration.
#[derive(Debug, Clone)]
pub struct WavConfig {
    /// Sample rate in Hz (e.g., 44100, 48000)
    pub sample_rate: u32,
    /// Number of channels (1=mono, 2=stereo)
    pub channels: u16,
    /// Bits per sample (8, 16, 24, or 32)
    pub bits_per_sample: u16,
    /// Use float format (32-bit only)
    pub float_format: bool,
}

impl WavConfig {
    /// Create a new WAV config with default settings.
    ///
    /// Defaults: 16-bit integer PCM
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        WavConfig {
            sample_rate,
            channels,
            bits_per_sample: 16,
            float_format: false,
        }
    }

    /// Set bits per sample (8, 16, 24, or 32).
    pub fn with_bits(mut self, bits: u16) -> Self {
        self.bits_per_sample = bits;
        self
    }

    /// Use 32-bit float format instead of integer.
    pub fn with_float(mut self) -> Self {
        self.bits_per_sample = 32;
        self.float_format = true;
        self
    }
}

impl Default for WavConfig {
    fn default() -> Self {
        WavConfig::new(44100, 2)
    }
}

/// Encode audio samples to WAV file.
///
/// # Arguments
/// * `samples` - Interleaved audio samples (f32, normalized -1.0 to 1.0)
/// * `output` - Output file path
/// * `config` - WAV encoding configuration
///
/// # Returns
/// Number of samples written
pub fn encode_wav<P: AsRef<Path>>(
    samples: &[f32],
    output: P,
    config: WavConfig,
) -> AudioEncodeResult<u64> {
    let spec = hound::WavSpec {
        channels: config.channels,
        sample_rate: config.sample_rate,
        bits_per_sample: config.bits_per_sample,
        sample_format: if config.float_format {
            hound::SampleFormat::Float
        } else {
            hound::SampleFormat::Int
        },
    };

    let file = File::create(output.as_ref()).map_err(|e| AudioEncodeError::OpenFailed {
        path: output.as_ref().to_path_buf(),
        source: e,
    })?;
    let writer = BufWriter::new(file);
    let mut wav_writer = hound::WavWriter::new(writer, spec)
        .map_err(|e| AudioEncodeError::EncodeFailed(format!("Failed to create WAV writer: {}", e)))?;

    let mut count = 0u64;

    match (config.bits_per_sample, config.float_format) {
        (32, true) => {
            // 32-bit float
            for &sample in samples {
                wav_writer
                    .write_sample(sample)
                    .map_err(|e| AudioEncodeError::EncodeFailed(format!("Write failed: {}", e)))?;
                count += 1;
            }
        }
        (8, false) => {
            // 8-bit unsigned
            for &sample in samples {
                let s = ((sample * 127.0) + 128.0).clamp(0.0, 255.0) as u8;
                wav_writer
                    .write_sample(s as i8)
                    .map_err(|e| AudioEncodeError::EncodeFailed(format!("Write failed: {}", e)))?;
                count += 1;
            }
        }
        (16, false) => {
            // 16-bit signed
            for &sample in samples {
                let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                wav_writer
                    .write_sample(s)
                    .map_err(|e| AudioEncodeError::EncodeFailed(format!("Write failed: {}", e)))?;
                count += 1;
            }
        }
        (24, false) => {
            // 24-bit signed (written as i32)
            for &sample in samples {
                let s = (sample * 8388607.0).clamp(-8388608.0, 8388607.0) as i32;
                wav_writer
                    .write_sample(s)
                    .map_err(|e| AudioEncodeError::EncodeFailed(format!("Write failed: {}", e)))?;
                count += 1;
            }
        }
        (32, false) => {
            // 32-bit signed
            for &sample in samples {
                let s = (sample * 2147483647.0).clamp(-2147483648.0, 2147483647.0) as i32;
                wav_writer
                    .write_sample(s)
                    .map_err(|e| AudioEncodeError::EncodeFailed(format!("Write failed: {}", e)))?;
                count += 1;
            }
        }
        _ => {
            return Err(AudioEncodeError::EncodeFailed(format!(
                "Unsupported bit depth: {} (float={})",
                config.bits_per_sample, config.float_format
            )));
        }
    }

    wav_writer
        .finalize()
        .map_err(|e| AudioEncodeError::EncodeFailed(format!("Failed to finalize WAV: {}", e)))?;

    Ok(count)
}

/// Encode audio samples to WAV bytes (in memory).
pub fn encode_wav_to_bytes(samples: &[f32], config: WavConfig) -> AudioEncodeResult<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: config.channels,
        sample_rate: config.sample_rate,
        bits_per_sample: config.bits_per_sample,
        sample_format: if config.float_format {
            hound::SampleFormat::Float
        } else {
            hound::SampleFormat::Int
        },
    };

    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut wav_writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| AudioEncodeError::EncodeFailed(format!("Failed to create WAV writer: {}", e)))?;

        match (config.bits_per_sample, config.float_format) {
            (32, true) => {
                for &sample in samples {
                    wav_writer
                        .write_sample(sample)
                        .map_err(|e| AudioEncodeError::EncodeFailed(format!("Write failed: {}", e)))?;
                }
            }
            (16, false) => {
                for &sample in samples {
                    let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                    wav_writer
                        .write_sample(s)
                        .map_err(|e| AudioEncodeError::EncodeFailed(format!("Write failed: {}", e)))?;
                }
            }
            (24, false) => {
                for &sample in samples {
                    let s = (sample * 8388607.0).clamp(-8388608.0, 8388607.0) as i32;
                    wav_writer
                        .write_sample(s)
                        .map_err(|e| AudioEncodeError::EncodeFailed(format!("Write failed: {}", e)))?;
                }
            }
            _ => {
                return Err(AudioEncodeError::EncodeFailed(format!(
                    "Unsupported bit depth for in-memory encoding: {}",
                    config.bits_per_sample
                )));
            }
        }

        wav_writer
            .finalize()
            .map_err(|e| AudioEncodeError::EncodeFailed(format!("Failed to finalize WAV: {}", e)))?;
    }

    Ok(cursor.into_inner())
}

// ============================================================================
// FLAC Encoding (requires audio-encode-flac feature)
// ============================================================================

#[cfg(feature = "audio-encode-flac")]
mod flac {
    use super::*;
    use flacenc::component::BitRepr;
    use flacenc::config::Encoder as FlacEncoderConfig;
    use flacenc::error::Verify;
    use flacenc::source::MemSource;

    /// FLAC encoding configuration.
    #[derive(Debug, Clone)]
    pub struct FlacConfig {
        /// Sample rate in Hz
        pub sample_rate: u32,
        /// Number of channels
        pub channels: u16,
        /// Bits per sample (16 or 24)
        pub bits_per_sample: u16,
        /// Block size (number of samples per block, 0=auto)
        pub block_size: usize,
        /// Compression level (0-8, higher = more compression, slower)
        pub compression_level: u8,
    }

    impl FlacConfig {
        /// Create a new FLAC config with default settings.
        pub fn new(sample_rate: u32, channels: u16) -> Self {
            FlacConfig {
                sample_rate,
                channels,
                bits_per_sample: 16,
                block_size: 0, // auto
                compression_level: 5,
            }
        }

        /// Set bits per sample (16 or 24).
        pub fn with_bits(mut self, bits: u16) -> Self {
            self.bits_per_sample = bits;
            self
        }

        /// Set block size (0 for auto).
        pub fn with_block_size(mut self, size: usize) -> Self {
            self.block_size = size;
            self
        }

        /// Set compression level (0-8).
        pub fn with_compression(mut self, level: u8) -> Self {
            self.compression_level = level.min(8);
            self
        }
    }

    impl Default for FlacConfig {
        fn default() -> Self {
            FlacConfig::new(44100, 2)
        }
    }

    /// Encode audio samples to FLAC file.
    ///
    /// # Arguments
    /// * `samples` - Interleaved audio samples (f32, normalized -1.0 to 1.0)
    /// * `output` - Output file path
    /// * `config` - FLAC encoding configuration
    ///
    /// # Returns
    /// Number of frames written
    pub fn encode_flac<P: AsRef<Path>>(
        samples: &[f32],
        output: P,
        config: FlacConfig,
    ) -> AudioEncodeResult<u64> {
        // Convert f32 samples to i32 (FLAC native format)
        let scale = match config.bits_per_sample {
            16 => 32767.0,
            24 => 8388607.0,
            _ => {
                return Err(AudioEncodeError::EncodeFailed(format!(
                    "FLAC only supports 16 or 24 bits per sample, got {}",
                    config.bits_per_sample
                )));
            }
        };

        let int_samples: Vec<i32> = samples
            .iter()
            .map(|&s| (s * scale).clamp(-scale - 1.0, scale) as i32)
            .collect();

        let num_frames = int_samples.len() / config.channels as usize;

        // Determine block size
        let block_size = if config.block_size > 0 {
            config.block_size
        } else {
            // Auto block size based on sample rate
            match config.sample_rate {
                0..=8000 => 576,
                8001..=16000 => 1152,
                16001..=22050 => 2304,
                22051..=44100 => 4096,
                _ => 4608,
            }
        };

        // Create FLAC encoder config
        let enc_config = FlacEncoderConfig::default()
            .into_verified()
            .map_err(|e| AudioEncodeError::EncodeFailed(format!("Invalid FLAC config: {:?}", e)))?;

        // Create memory source from samples
        let source = MemSource::from_samples(
            &int_samples,
            config.channels as usize,
            config.bits_per_sample as usize,
            config.sample_rate as usize,
        );

        // Encode
        let stream = flacenc::encode_with_fixed_block_size(&enc_config, source, block_size)
            .map_err(|e| AudioEncodeError::EncodeFailed(format!("FLAC encoding failed: {:?}", e)))?;

        // Write to file
        let mut file = File::create(output.as_ref()).map_err(|e| AudioEncodeError::OpenFailed {
            path: output.as_ref().to_path_buf(),
            source: e,
        })?;

        // Serialize FLAC stream to bytes
        let mut sink = flacenc::bitsink::ByteSink::new();
        stream.write(&mut sink)
            .map_err(|e| AudioEncodeError::EncodeFailed(format!("Failed to serialize FLAC: {:?}", e)))?;

        std::io::Write::write_all(&mut file, sink.as_slice())
            .map_err(|e| AudioEncodeError::EncodeFailed(format!("Failed to write FLAC file: {}", e)))?;

        Ok(num_frames as u64)
    }

    /// Encode audio samples to FLAC bytes (in memory).
    pub fn encode_flac_to_bytes(samples: &[f32], config: FlacConfig) -> AudioEncodeResult<Vec<u8>> {
        // Convert f32 samples to i32
        let scale = match config.bits_per_sample {
            16 => 32767.0,
            24 => 8388607.0,
            _ => {
                return Err(AudioEncodeError::EncodeFailed(format!(
                    "FLAC only supports 16 or 24 bits per sample, got {}",
                    config.bits_per_sample
                )));
            }
        };

        let int_samples: Vec<i32> = samples
            .iter()
            .map(|&s| (s * scale).clamp(-scale - 1.0, scale) as i32)
            .collect();

        // Determine block size
        let block_size = if config.block_size > 0 {
            config.block_size
        } else {
            match config.sample_rate {
                0..=8000 => 576,
                8001..=16000 => 1152,
                16001..=22050 => 2304,
                22051..=44100 => 4096,
                _ => 4608,
            }
        };

        // Create FLAC encoder config
        let enc_config = FlacEncoderConfig::default()
            .into_verified()
            .map_err(|e| AudioEncodeError::EncodeFailed(format!("Invalid FLAC config: {:?}", e)))?;

        // Create memory source
        let source = MemSource::from_samples(
            &int_samples,
            config.channels as usize,
            config.bits_per_sample as usize,
            config.sample_rate as usize,
        );

        // Encode
        let stream = flacenc::encode_with_fixed_block_size(&enc_config, source, block_size)
            .map_err(|e| AudioEncodeError::EncodeFailed(format!("FLAC encoding failed: {:?}", e)))?;

        // Serialize to bytes
        let mut sink = flacenc::bitsink::ByteSink::new();
        stream.write(&mut sink)
            .map_err(|e| AudioEncodeError::EncodeFailed(format!("Failed to serialize FLAC: {:?}", e)))?;

        Ok(sink.as_slice().to_vec())
    }
}

#[cfg(feature = "audio-encode-flac")]
pub use flac::{encode_flac, encode_flac_to_bytes, FlacConfig};

// ============================================================================
// Audio transcoding helpers
// ============================================================================

/// Supported audio output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioOutputFormat {
    /// WAV - Lossless, uncompressed
    Wav,
    /// FLAC - Lossless, compressed (requires audio-encode-flac feature)
    Flac,
    /// Opus - High-quality lossy Ogg Opus output (requires audio-encode-opus feature)
    Opus,
}

impl AudioOutputFormat {
    /// Get file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            AudioOutputFormat::Wav => "wav",
            AudioOutputFormat::Flac => "flac",
            AudioOutputFormat::Opus => "opus",
        }
    }

    /// Detect format from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "wav" | "wave" => Some(AudioOutputFormat::Wav),
            "flac" => Some(AudioOutputFormat::Flac),
            "opus" | "ogg" => Some(AudioOutputFormat::Opus),
            _ => None,
        }
    }

    /// Check if this format is supported in the current build.
    pub fn is_supported(&self) -> bool {
        match self {
            AudioOutputFormat::Wav => true,
            #[cfg(feature = "audio-encode-flac")]
            AudioOutputFormat::Flac => true,
            #[cfg(not(feature = "audio-encode-flac"))]
            AudioOutputFormat::Flac => false,
            #[cfg(feature = "audio-encode-opus")]
            AudioOutputFormat::Opus => true,
            #[cfg(not(feature = "audio-encode-opus"))]
            AudioOutputFormat::Opus => false,
        }
    }
}
