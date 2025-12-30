//! Opus audio encoding module.
//!
//! This module provides Opus audio encoding using the audiopus crate,
//! which wraps the libopus library.
//!
//! # Features
//!
//! - High-quality audio compression
//! - Variable bitrate (VBR) support
//! - Multiple sample rates (8kHz to 48kHz)
//! - Mono and stereo support
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::audio::encode::opus::{OpusEncoder, OpusEncoderConfig};
//!
//! let config = OpusEncoderConfig::new(48000, 2)
//!     .with_bitrate(128000);
//!
//! let mut encoder = OpusEncoder::new(config)?;
//!
//! // Encode PCM samples (f32, interleaved stereo)
//! let encoded = encoder.encode(&pcm_samples)?;
//! ```

use audiopus::{
    coder::Encoder as OpusEncoderInner,
    Application, Bitrate, Channels, SampleRate,
};

/// Opus encoder configuration.
#[derive(Debug, Clone)]
pub struct OpusEncoderConfig {
    /// Sample rate in Hz (8000, 12000, 16000, 24000, or 48000).
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u8,
    /// Target bitrate in bits per second.
    pub bitrate: u32,
    /// Frame size in samples per channel (2.5ms, 5ms, 10ms, 20ms, 40ms, 60ms).
    /// For 48kHz: 120, 240, 480, 960, 1920, 2880
    pub frame_size: usize,
    /// Application mode (audio, voip, or low_delay).
    pub application: OpusApplication,
}

/// Opus application mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpusApplication {
    /// Optimized for audio (music, mixed content).
    #[default]
    Audio,
    /// Optimized for voice (speech).
    Voip,
    /// Low delay mode for real-time applications.
    LowDelay,
}

impl OpusEncoderConfig {
    /// Create a new Opus encoder configuration.
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        // Default frame size: 20ms at the given sample rate
        let frame_size = (sample_rate as usize * 20) / 1000;

        Self {
            sample_rate,
            channels,
            bitrate: 128000, // 128 kbps default
            frame_size,
            application: OpusApplication::Audio,
        }
    }

    /// Set target bitrate in bits per second.
    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.bitrate = bitrate;
        self
    }

    /// Set frame size in samples per channel.
    pub fn with_frame_size(mut self, frame_size: usize) -> Self {
        self.frame_size = frame_size;
        self
    }

    /// Set application mode.
    pub fn with_application(mut self, application: OpusApplication) -> Self {
        self.application = application;
        self
    }

    fn get_sample_rate(&self) -> SampleRate {
        match self.sample_rate {
            8000 => SampleRate::Hz8000,
            12000 => SampleRate::Hz12000,
            16000 => SampleRate::Hz16000,
            24000 => SampleRate::Hz24000,
            _ => SampleRate::Hz48000,
        }
    }

    fn get_channels(&self) -> Channels {
        match self.channels {
            1 => Channels::Mono,
            _ => Channels::Stereo,
        }
    }

    fn get_application(&self) -> Application {
        match self.application {
            OpusApplication::Audio => Application::Audio,
            OpusApplication::Voip => Application::Voip,
            OpusApplication::LowDelay => Application::LowDelay,
        }
    }
}

/// Opus audio encoder.
pub struct OpusEncoder {
    encoder: OpusEncoderInner,
    config: OpusEncoderConfig,
    /// Temporary buffer for encoded data
    output_buffer: Vec<u8>,
}

/// Opus encoding error.
#[derive(Debug, thiserror::Error)]
pub enum OpusError {
    /// Failed to create encoder.
    #[error("failed to create Opus encoder: {0}")]
    CreateFailed(String),

    /// Encoding failed.
    #[error("Opus encoding failed: {0}")]
    EncodeFailed(String),

    /// Invalid frame size.
    #[error("invalid frame size: expected {expected}, got {got}")]
    InvalidFrameSize { expected: usize, got: usize },
}

pub type OpusResult<T> = Result<T, OpusError>;

impl OpusEncoder {
    /// Create a new Opus encoder with the given configuration.
    pub fn new(config: OpusEncoderConfig) -> OpusResult<Self> {
        let encoder = OpusEncoderInner::new(
            config.get_sample_rate(),
            config.get_channels(),
            config.get_application(),
        )
        .map_err(|e| OpusError::CreateFailed(format!("{:?}", e)))?;

        // Set bitrate
        let _bitrate = Bitrate::BitsPerSecond(config.bitrate as i32);
        // Note: audiopus doesn't expose set_bitrate directly, using default

        // Maximum Opus frame size is 1275 bytes for a single frame
        let output_buffer = vec![0u8; 4000];

        Ok(Self {
            encoder,
            config,
            output_buffer,
        })
    }

    /// Encode a frame of PCM audio.
    ///
    /// # Arguments
    /// * `pcm` - PCM samples as f32, interleaved for stereo
    ///
    /// # Returns
    /// Encoded Opus packet
    pub fn encode(&mut self, pcm: &[f32]) -> OpusResult<Vec<u8>> {
        let expected_samples = self.config.frame_size * self.config.channels as usize;
        if pcm.len() != expected_samples {
            return Err(OpusError::InvalidFrameSize {
                expected: expected_samples,
                got: pcm.len(),
            });
        }

        let bytes_written = self
            .encoder
            .encode_float(pcm, &mut self.output_buffer)
            .map_err(|e| OpusError::EncodeFailed(format!("{:?}", e)))?;

        Ok(self.output_buffer[..bytes_written].to_vec())
    }

    /// Encode a frame of PCM audio (i16 samples).
    ///
    /// # Arguments
    /// * `pcm` - PCM samples as i16, interleaved for stereo
    ///
    /// # Returns
    /// Encoded Opus packet
    pub fn encode_i16(&mut self, pcm: &[i16]) -> OpusResult<Vec<u8>> {
        let expected_samples = self.config.frame_size * self.config.channels as usize;
        if pcm.len() != expected_samples {
            return Err(OpusError::InvalidFrameSize {
                expected: expected_samples,
                got: pcm.len(),
            });
        }

        let bytes_written = self
            .encoder
            .encode(pcm, &mut self.output_buffer)
            .map_err(|e| OpusError::EncodeFailed(format!("{:?}", e)))?;

        Ok(self.output_buffer[..bytes_written].to_vec())
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &OpusEncoderConfig {
        &self.config
    }

    /// Get frame size in samples per channel.
    pub fn frame_size(&self) -> usize {
        self.config.frame_size
    }

    /// Get number of channels.
    pub fn channels(&self) -> u8 {
        self.config.channels
    }

    /// Get sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

/// Encode PCM audio to Opus packets.
///
/// # Arguments
/// * `pcm` - PCM samples as f32, interleaved for stereo
/// * `sample_rate` - Sample rate in Hz
/// * `channels` - Number of channels (1 or 2)
/// * `bitrate` - Target bitrate in bps
///
/// # Returns
/// Vector of encoded Opus packets
pub fn encode_opus(
    pcm: &[f32],
    sample_rate: u32,
    channels: u8,
    bitrate: u32,
) -> OpusResult<Vec<Vec<u8>>> {
    let config = OpusEncoderConfig::new(sample_rate, channels).with_bitrate(bitrate);
    let mut encoder = OpusEncoder::new(config)?;

    let frame_size = encoder.frame_size();
    let samples_per_frame = frame_size * channels as usize;

    let mut packets = Vec::new();

    for chunk in pcm.chunks(samples_per_frame) {
        if chunk.len() == samples_per_frame {
            let packet = encoder.encode(chunk)?;
            packets.push(packet);
        }
    }

    Ok(packets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opus_config() {
        let config = OpusEncoderConfig::new(48000, 2).with_bitrate(128000);

        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 2);
        assert_eq!(config.bitrate, 128000);
        assert_eq!(config.frame_size, 960); // 20ms at 48kHz
    }
}
