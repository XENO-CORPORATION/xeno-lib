//! Audio processing bindings (Priority 2 -- for xeno-sound).
//!
//! Audio decode is async (large files). Encode is sync (fast).
//! All audio samples are f32 PCM (-1.0 to 1.0).

use napi::bindgen_prelude::*;
use napi_derive::napi;

// ---------------------------------------------------------------------------
// Result types exposed to TypeScript
// ---------------------------------------------------------------------------

/// Decoded audio data returned from `decode_audio`.
#[napi(object)]
pub struct AudioData {
    /// Interleaved f32 PCM samples (-1.0 to 1.0).
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
/// Supports MP3, AAC, FLAC, Vorbis, ALAC, WAV, AIFF, Ogg, and more.
/// This is async because audio files can be large.
#[napi(ts_return_type = "Promise<AudioData>")]
pub async fn decode_audio(file_path: String) -> Result<AudioData> {
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
/// Returns a `Buffer` containing the WAV file bytes.
#[napi]
pub fn encode_wav(
    samples: Float64Array,
    sample_rate: u32,
    channels: u32,
    bit_depth: u32,
) -> Result<Buffer> {
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

/// Encode f32 PCM samples to FLAC format.
///
/// Returns a `Buffer` containing the FLAC file bytes.
#[napi]
pub fn encode_flac(
    samples: Float64Array,
    sample_rate: u32,
    channels: u32,
) -> Result<Buffer> {
    let f32_samples: Vec<f32> = samples.iter().map(|&s| s as f32).collect();

    let config = xeno_lib::audio::encode::FlacConfig::new(sample_rate, channels as u16);

    let bytes =
        xeno_lib::audio::encode::encode_flac_to_bytes(&f32_samples, config).map_err(|e| {
            Error::new(Status::GenericFailure, format!("FLAC encode failed: {e}"))
        })?;

    Ok(bytes.into())
}
