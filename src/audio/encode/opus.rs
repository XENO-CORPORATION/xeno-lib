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

use std::path::Path;

use audiopus::{
    coder::Encoder as OpusEncoderInner,
    Application, Bitrate, Channels, SampleRate,
};

use super::{AudioEncodeError, AudioEncodeResult};

const OGG_CAPTURE_PATTERN: &[u8; 4] = b"OggS";
const OGG_VERSION: u8 = 0;
const OGG_BOS: u8 = 0x02;
const OGG_EOS: u8 = 0x04;
const OGG_CRC_POLY: u32 = 0x04C11DB7;
const OPUS_HEAD_MAGIC: &[u8; 8] = b"OpusHead";
const OPUS_TAGS_MAGIC: &[u8; 8] = b"OpusTags";
const OPUS_OUTPUT_RATE: u32 = 48_000;
const OPUS_VENDOR: &str = "xeno-lib";
const XENO_OPUS_STREAM_SERIAL: u32 = 0x5845_4E4F;

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

    pub fn is_supported_sample_rate(sample_rate: u32) -> bool {
        matches!(sample_rate, 8000 | 12000 | 16000 | 24000 | 48000)
    }

    fn try_sample_rate(&self) -> OpusResult<SampleRate> {
        match self.sample_rate {
            8000 => Ok(SampleRate::Hz8000),
            12000 => Ok(SampleRate::Hz12000),
            16000 => Ok(SampleRate::Hz16000),
            24000 => Ok(SampleRate::Hz24000),
            48000 => Ok(SampleRate::Hz48000),
            other => Err(OpusError::UnsupportedSampleRate(other)),
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

    /// Unsupported input sample rate.
    #[error("unsupported Opus sample rate: {0} (use 8000, 12000, 16000, 24000, or 48000)")]
    UnsupportedSampleRate(u32),

    /// Unsupported number of channels.
    #[error("unsupported Opus channel count: {0} (use mono or stereo)")]
    UnsupportedChannels(u8),
}

pub type OpusResult<T> = Result<T, OpusError>;

impl OpusEncoder {
    /// Create a new Opus encoder with the given configuration.
    pub fn new(config: OpusEncoderConfig) -> OpusResult<Self> {
        if config.channels == 0 || config.channels > 2 {
            return Err(OpusError::UnsupportedChannels(config.channels));
        }

        let mut encoder = OpusEncoderInner::new(
            config.try_sample_rate()?,
            config.get_channels(),
            config.get_application(),
        )
        .map_err(|e| OpusError::CreateFailed(format!("{:?}", e)))?;

        encoder
            .set_bitrate(Bitrate::BitsPerSecond(config.bitrate as i32))
            .map_err(|e| OpusError::CreateFailed(format!("{:?}", e)))?;

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

    /// Get encoder lookahead in samples at the encoder sample rate.
    pub fn lookahead(&self) -> OpusResult<u32> {
        self.encoder
            .lookahead()
            .map_err(|e| OpusError::EncodeFailed(format!("{:?}", e)))
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

fn opus_error_to_audio_error(error: OpusError) -> AudioEncodeError {
    AudioEncodeError::EncodeFailed(error.to_string())
}

fn build_opus_head_packet(channels: u8, pre_skip: u16, input_sample_rate: u32) -> Vec<u8> {
    let mut packet = Vec::with_capacity(19);
    packet.extend_from_slice(OPUS_HEAD_MAGIC);
    packet.push(1);
    packet.push(channels);
    packet.extend_from_slice(&pre_skip.to_le_bytes());
    packet.extend_from_slice(&input_sample_rate.to_le_bytes());
    packet.extend_from_slice(&0i16.to_le_bytes());
    packet.push(0);
    packet
}

fn build_opus_tags_packet() -> Vec<u8> {
    let vendor = OPUS_VENDOR.as_bytes();
    let mut packet = Vec::with_capacity(16 + vendor.len());
    packet.extend_from_slice(OPUS_TAGS_MAGIC);
    packet.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    packet.extend_from_slice(vendor);
    packet.extend_from_slice(&0u32.to_le_bytes());
    packet
}

fn ogg_lacing(packet_len: usize) -> Vec<u8> {
    let mut remaining = packet_len;
    let mut segments = Vec::new();

    loop {
        let segment = remaining.min(255);
        segments.push(segment as u8);
        remaining -= segment;
        if segment < 255 {
            break;
        }
    }

    segments
}

fn ogg_crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0u32;

    for &byte in bytes {
        crc ^= (byte as u32) << 24;
        for _ in 0..8 {
            crc = if (crc & 0x8000_0000) != 0 {
                (crc << 1) ^ OGG_CRC_POLY
            } else {
                crc << 1
            };
        }
    }

    crc
}

fn write_ogg_packet_page(
    output: &mut Vec<u8>,
    packet: &[u8],
    header_type: u8,
    granule_position: u64,
    serial: u32,
    sequence: u32,
) -> AudioEncodeResult<()> {
    let lacing = ogg_lacing(packet.len());
    if lacing.len() > 255 {
        return Err(AudioEncodeError::EncodeFailed(
            "Opus packet produced too many Ogg lacing segments".to_string(),
        ));
    }

    let start = output.len();
    output.extend_from_slice(OGG_CAPTURE_PATTERN);
    output.push(OGG_VERSION);
    output.push(header_type);
    output.extend_from_slice(&granule_position.to_le_bytes());
    output.extend_from_slice(&serial.to_le_bytes());
    output.extend_from_slice(&sequence.to_le_bytes());
    output.extend_from_slice(&0u32.to_le_bytes());
    output.push(lacing.len() as u8);
    output.extend_from_slice(&lacing);
    output.extend_from_slice(packet);

    let crc = ogg_crc32(&output[start..]);
    output[start + 22..start + 26].copy_from_slice(&crc.to_le_bytes());

    Ok(())
}

/// Encode PCM audio into an Ogg Opus file.
pub fn encode_opus_ogg<P: AsRef<Path>>(
    pcm: &[f32],
    output: P,
    config: OpusEncoderConfig,
) -> AudioEncodeResult<u64> {
    let encoded = encode_opus_ogg_to_bytes(pcm, config.clone())?;

    std::fs::write(output.as_ref(), encoded).map_err(|source| AudioEncodeError::OpenFailed {
        path: output.as_ref().to_path_buf(),
        source,
    })?;

    Ok((pcm.len() / config.channels as usize) as u64)
}

/// Encode PCM audio into an Ogg Opus byte buffer.
pub fn encode_opus_ogg_to_bytes(pcm: &[f32], config: OpusEncoderConfig) -> AudioEncodeResult<Vec<u8>> {
    if !OpusEncoderConfig::is_supported_sample_rate(config.sample_rate) {
        return Err(opus_error_to_audio_error(OpusError::UnsupportedSampleRate(
            config.sample_rate,
        )));
    }
    if config.channels == 0 || config.channels > 2 {
        return Err(opus_error_to_audio_error(OpusError::UnsupportedChannels(
            config.channels,
        )));
    }
    if pcm.len() % config.channels as usize != 0 {
        return Err(AudioEncodeError::EncodeFailed(format!(
            "PCM sample count {} is not divisible by channel count {}",
            pcm.len(),
            config.channels
        )));
    }

    let total_frames = pcm.len() / config.channels as usize;
    let granule_scale = OPUS_OUTPUT_RATE as u64 / config.sample_rate as u64;
    let mut encoder = OpusEncoder::new(config.clone()).map_err(opus_error_to_audio_error)?;
    let pre_skip = encoder.lookahead().map_err(opus_error_to_audio_error)? as u64 * granule_scale;
    let frame_size = encoder.frame_size();
    let samples_per_packet = frame_size * config.channels as usize;
    let packet_granule = frame_size as u64 * granule_scale;
    let total_granule = total_frames as u64 * granule_scale;
    let total_packets = total_frames.div_ceil(frame_size);

    let mut output = Vec::new();
    let mut sequence = 0u32;

    let pre_skip_u16 = u16::try_from(pre_skip).map_err(|_| {
        AudioEncodeError::EncodeFailed(format!("Opus pre-skip exceeds u16 range: {}", pre_skip))
    })?;

    let head_packet = build_opus_head_packet(config.channels, pre_skip_u16, config.sample_rate);
    write_ogg_packet_page(&mut output, &head_packet, OGG_BOS, 0, XENO_OPUS_STREAM_SERIAL, sequence)?;
    sequence += 1;

    let tags_packet = build_opus_tags_packet();
    write_ogg_packet_page(&mut output, &tags_packet, 0, 0, XENO_OPUS_STREAM_SERIAL, sequence)?;
    sequence += 1;

    let mut frame_buffer = vec![0.0f32; samples_per_packet];

    for packet_index in 0..total_packets {
        let frame_start = packet_index * frame_size;
        let frame_end = (frame_start + frame_size).min(total_frames);
        let actual_frames = frame_end - frame_start;
        let sample_start = frame_start * config.channels as usize;
        let sample_end = frame_end * config.channels as usize;

        frame_buffer.fill(0.0);
        frame_buffer[..sample_end - sample_start].copy_from_slice(&pcm[sample_start..sample_end]);

        let packet = encoder.encode(&frame_buffer).map_err(opus_error_to_audio_error)?;
        let is_last = packet_index + 1 == total_packets;
        let granule_position = if is_last {
            pre_skip + total_granule
        } else {
            pre_skip + ((packet_index as u64 + 1) * packet_granule)
        };
        let header_type = if is_last { OGG_EOS } else { 0 };

        write_ogg_packet_page(
            &mut output,
            &packet,
            header_type,
            granule_position,
            XENO_OPUS_STREAM_SERIAL,
            sequence,
        )?;
        sequence += 1;

        // Avoid silently accepting a packetized stream that lost all payload.
        if actual_frames == 0 {
            break;
        }
    }

    Ok(output)
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

    #[test]
    fn test_opus_encoder_applies_bitrate() {
        let encoder = OpusEncoder::new(OpusEncoderConfig::new(48_000, 2).with_bitrate(96_000))
            .expect("create opus encoder");
        let bitrate = encoder.encoder.bitrate().expect("read encoder bitrate");
        assert_eq!(i32::from(bitrate), 96_000);
    }

    #[cfg(feature = "audio")]
    #[test]
    fn test_encode_ogg_opus_round_trip() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let path = temp_dir.path().join("roundtrip.opus");
        let sample_rate = 48_000u32;
        let channels = 2u8;
        let frames = 4 * 960usize;

        let mut samples = Vec::with_capacity(frames * channels as usize);
        for index in 0..frames {
            let t = index as f32 / sample_rate as f32;
            let sample = (t * 440.0 * std::f32::consts::TAU).sin() * 0.25;
            samples.push(sample);
            samples.push(sample);
        }

        encode_opus_ogg(
            &samples,
            &path,
            OpusEncoderConfig::new(sample_rate, channels).with_bitrate(96_000),
        )
        .expect("encode ogg opus");

        let decoded = crate::audio::decode_file(&path).expect("decode opus");
        assert_eq!(decoded.sample_rate, sample_rate);
        assert_eq!(decoded.channels, channels as u32);
        assert!(!decoded.samples.is_empty());
        assert!((decoded.samples.len() as isize - samples.len() as isize).abs() < samples.len() as isize / 4);
    }
}
