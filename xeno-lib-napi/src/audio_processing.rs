//! Audio processing bindings (Priority 2 -- for xeno-sound).
//!
//! Audio decode is async (large files). Encode is sync (fast).
//! All audio samples are f32 PCM (-1.0 to 1.0).
//!
//! # Input Validation
//!
//! - File paths are checked for existence before decoding.
//! - Sample arrays are checked for emptiness and invalid values (NaN, Infinity).
//! - Sample rate and channel count are validated for sane ranges.
//! - WAV bit depth is validated to be one of: 8, 16, 24, 32.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::validation::{validate_audio_samples, validate_file_path, validate_wav_bit_depth};

// ---------------------------------------------------------------------------
// Result types exposed to TypeScript
// ---------------------------------------------------------------------------

/// Decoded audio data returned from `decode_audio`.
///
/// Samples are interleaved f32 PCM (-1.0 to 1.0), encoded as f64 for
/// JavaScript compatibility (napi-rs does not support f32 arrays in
/// object fields).
#[napi(object)]
pub struct AudioData {
    /// Interleaved f32 PCM samples (-1.0 to 1.0), encoded as f64.
    pub samples: Vec<f64>,
    /// Sample rate in Hz (e.g., 44100, 48000).
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u32,
    /// Total duration in milliseconds.
    pub duration_ms: u32,
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Decode an audio file to raw f32 PCM samples.
///
/// Supports MP3, AAC, FLAC, Vorbis, ALAC, WAV, AIFF, Ogg, and more
/// (via the Symphonia pure-Rust audio decoder).
///
/// # Arguments
/// * `file_path` - Path to the audio file (must exist and be non-empty)
///
/// # Returns
/// An `AudioData` object containing interleaved PCM samples, sample rate,
/// channel count, and duration.
///
/// # Errors
/// - If file path is empty
/// - If the file does not exist
/// - If the audio format is unsupported or the file is corrupt
/// - If decoding fails for any reason
#[napi(ts_return_type = "Promise<AudioData>")]
pub async fn decode_audio(file_path: String) -> Result<AudioData> {
    validate_file_path(&file_path)?;

    // Run the potentially slow decode on a blocking thread so we don't
    // block the libuv event loop.
    let decoded = tokio::task::spawn_blocking(move || {
        xeno_lib::decode_audio_file(&file_path).map_err(|e| {
            Error::new(Status::GenericFailure, format!("Audio decode failed: {e}"))
        })
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))??;

    let duration_ms =
        (decoded.duration_secs() * 1000.0).round() as u32;

    // Convert f32 samples to f64 for JavaScript (napi doesn't have f32 array in object fields)
    let samples_f64: Vec<f64> = decoded.samples.iter().map(|&s| s as f64).collect();

    Ok(AudioData {
        samples: samples_f64,
        sample_rate: decoded.sample_rate,
        channels: decoded.channels,
        duration_ms,
    })
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Encode f32 PCM samples to WAV format.
///
/// # Arguments
/// * `samples` - Interleaved PCM samples as f64 array (will be converted to f32 internally)
/// * `sample_rate` - Sample rate in Hz (1-384000)
/// * `channels` - Number of channels (1-32)
/// * `bit_depth` - Bits per sample (8, 16, 24, or 32)
///
/// # Returns
/// A `Buffer` containing the complete WAV file bytes (including header).
///
/// # Errors
/// - If samples array is empty
/// - If sample rate is zero or exceeds 384000
/// - If channel count is zero or exceeds 32
/// - If bit depth is not 8, 16, 24, or 32
/// - If any sample is NaN or Infinity
/// - If WAV encoding fails
#[napi]
pub fn encode_wav(
    samples: Float64Array,
    sample_rate: u32,
    channels: u32,
    bit_depth: u32,
) -> Result<Buffer> {
    validate_audio_samples(&samples, sample_rate, channels)?;
    validate_wav_bit_depth(bit_depth)?;

    let f32_samples: Vec<f32> = samples.iter().map(|&s| s as f32).collect();

    let config = xeno_lib::audio::encode::WavConfig {
        sample_rate,
        channels: channels as u16,
        bits_per_sample: bit_depth as u16,
        float_format: bit_depth == 32,
    };

    let bytes = xeno_lib::audio::encode::encode_wav_to_bytes(&f32_samples, config).map_err(|e| {
        Error::new(Status::GenericFailure, format!("WAV encode failed: {e}"))
    })?;

    Ok(bytes.into())
}

/// Encode f32 PCM samples to AAC format (lossy compression for video export).
///
/// # Status
///
/// **Currently a stub** — no pure Rust AAC encoder exists. Will return an error
/// until fdk-aac C bindings are integrated. Use Opus encoding as an alternative.
///
/// # Arguments
/// * `samples` - Interleaved PCM samples as f64 array (will be converted to f32 internally)
/// * `sample_rate` - Sample rate in Hz (1-96000)
/// * `channels` - Number of channels (1-8)
/// * `bitrate` - Target bitrate in bps (e.g., 128000, 192000, 256000)
///
/// # Returns
/// A `Buffer` containing the AAC encoded bytes.
///
/// # Errors
/// - If samples array is empty
/// - If parameters are out of range
/// - If AAC encoder is not available (current stub)
#[napi]
pub fn encode_aac(
    samples: Float64Array,
    sample_rate: u32,
    channels: u32,
    bitrate: u32,
) -> Result<Buffer> {
    validate_audio_samples(&samples, sample_rate, channels)?;

    let f32_samples: Vec<f32> = samples.iter().map(|&s| s as f32).collect();

    let config = xeno_lib::audio::encode::aac::AacEncoderConfig::new(sample_rate, channels)
        .with_bitrate(bitrate);

    let bytes = xeno_lib::audio::encode::aac::encode_aac(&f32_samples, &config).map_err(|e| {
        Error::new(Status::GenericFailure, format!("AAC encode failed: {e}"))
    })?;

    Ok(bytes.into())
}

/// Encode f32 PCM samples to FLAC format (lossless compression).
///
/// # Arguments
/// * `samples` - Interleaved PCM samples as f64 array (will be converted to f32 internally)
/// * `sample_rate` - Sample rate in Hz (1-384000)
/// * `channels` - Number of channels (1-32)
///
/// # Returns
/// A `Buffer` containing the complete FLAC file bytes.
///
/// # Errors
/// - If samples array is empty
/// - If sample rate is zero or exceeds 384000
/// - If channel count is zero or exceeds 32
/// - If any sample is NaN or Infinity
/// - If FLAC encoding fails
#[napi]
pub fn encode_flac(
    samples: Float64Array,
    sample_rate: u32,
    channels: u32,
) -> Result<Buffer> {
    validate_audio_samples(&samples, sample_rate, channels)?;

    let f32_samples: Vec<f32> = samples.iter().map(|&s| s as f32).collect();

    let config = xeno_lib::audio::encode::FlacConfig::new(sample_rate, channels as u16);

    let bytes =
        xeno_lib::audio::encode::encode_flac_to_bytes(&f32_samples, config).map_err(|e| {
            Error::new(Status::GenericFailure, format!("FLAC encode failed: {e}"))
        })?;

    Ok(bytes.into())
}

/// Encode f32 PCM samples to Ogg Opus format (high-quality lossy compression).
///
/// Opus is the recommended lossy audio format for xeno-sound export.
/// It provides excellent quality at low bitrates and is royalty-free.
/// The output is a complete Ogg Opus file (with Ogg container framing).
///
/// # Arguments
/// * `samples` - Interleaved PCM samples as f64 array (will be converted to f32 internally)
/// * `sample_rate` - Input sample rate in Hz. Opus internally operates at 48kHz;
///                    input will be resampled if needed. Valid: 8000, 12000, 16000, 24000, 48000.
/// * `channels` - Number of channels (1 = mono, 2 = stereo)
/// * `bitrate` - Target bitrate in bits per second (e.g., 64000, 96000, 128000, 192000)
///
/// # Returns
/// A `Buffer` containing the complete Ogg Opus file bytes.
///
/// # Errors
/// - If samples array is empty
/// - If sample rate is not a valid Opus sample rate
/// - If channel count is not 1 or 2
/// - If any sample is NaN or Infinity
/// - If Opus encoding fails
///
/// # Example (JavaScript)
///
/// ```js
/// const { encodeOpus } = require('@xeno/lib');
/// const opusBytes = encodeOpus(pcmSamples, 48000, 2, 128000);
/// fs.writeFileSync('output.opus', opusBytes);
/// ```
#[napi]
pub fn encode_opus(
    samples: Float64Array,
    sample_rate: u32,
    channels: u32,
    bitrate: u32,
) -> Result<Buffer> {
    validate_audio_samples(&samples, sample_rate, channels)?;

    // Validate Opus-specific constraints
    let valid_rates = [8000, 12000, 16000, 24000, 48000];
    if !valid_rates.contains(&sample_rate) {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Opus sample rate must be one of {:?}, got {}",
                valid_rates, sample_rate
            ),
        ));
    }

    if channels == 0 || channels > 2 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("Opus channels must be 1 (mono) or 2 (stereo), got {}", channels),
        ));
    }

    if bitrate == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "Opus bitrate must be > 0".to_string(),
        ));
    }

    let f32_samples: Vec<f32> = samples.iter().map(|&s| s as f32).collect();

    let config = xeno_lib::audio::encode::OpusEncoderConfig::new(sample_rate, channels as u8)
        .with_bitrate(bitrate);

    let bytes =
        xeno_lib::audio::encode::encode_opus_ogg_to_bytes(&f32_samples, config).map_err(|e| {
            Error::new(Status::GenericFailure, format!("Opus encode failed: {e}"))
        })?;

    Ok(bytes.into())
}

/// Get available audio encoder formats.
///
/// Returns an object describing which audio encoders are compiled in
/// and available at runtime. Use this to decide which format to offer
/// for audio export.
///
/// # Example (JavaScript)
///
/// ```js
/// const { getAudioEncoders } = require('@xeno/lib');
/// const encoders = getAudioEncoders();
/// console.log(`WAV: ${encoders.wav}`);      // always true
/// console.log(`FLAC: ${encoders.flac}`);    // true if audio-encode-flac
/// console.log(`Opus: ${encoders.opus}`);    // true if audio-encode-opus
/// console.log(`AAC: ${encoders.aac}`);      // true if fdk-aac linked
/// ```
#[napi(object)]
pub struct AudioEncoders {
    /// WAV encoding (always available).
    pub wav: bool,
    /// FLAC lossless encoding.
    pub flac: bool,
    /// Opus lossy encoding (Ogg Opus output).
    pub opus: bool,
    /// AAC lossy encoding (stub until fdk-aac is linked).
    pub aac: bool,
}

/// Get available audio encoder formats.
#[napi]
pub fn get_audio_encoders() -> AudioEncoders {
    // audio-encode, audio-encode-flac, and audio-encode-opus are all enabled
    // unconditionally in xeno-lib-napi's Cargo.toml dependency on xeno-lib.
    // AAC is a stub until fdk-aac C bindings are integrated.
    AudioEncoders {
        wav: true,  // hound — always available
        flac: true, // flacenc — pure Rust, always enabled
        opus: true, // audiopus — libopus bindings, always enabled
        aac: xeno_lib::audio::encode::AudioOutputFormat::Aac.is_supported(),
    }
}
