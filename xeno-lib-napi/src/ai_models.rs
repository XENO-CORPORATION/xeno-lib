//! AI model bindings (Priority 3 -- for all apps).
//!
//! All AI functions are async because inference is slow.
//! Model sessions are lazily loaded and cached across calls via `Mutex<Option<T>>`.
//!
//! # Error Handling
//!
//! - Input buffers are validated (dimensions, size) before inference.
//! - Missing model files produce descriptive errors (not panics).
//! - Model session initialization errors are surfaced to JavaScript.
//! - Poisoned mutex locks are caught and reported.
//!
//! # Model Caching
//!
//! Each model uses a `static Mutex<Option<Session>>` pattern. The first call
//! loads the ONNX model from `~/.xeno-lib/models/` and caches it. Subsequent
//! calls reuse the cached session, avoiding repeated disk I/O and model init.

use std::sync::Mutex;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::helpers::{buffer_to_image, image_to_vec, transform_err};
use crate::validation::{
    validate_audio_samples, validate_file_path, validate_image_buffer,
    validate_interpolation_factor, validate_upscale_factor,
};

// ---------------------------------------------------------------------------
// Lazy model caches (loaded on first call, reused thereafter)
// ---------------------------------------------------------------------------

static BG_SESSION: Mutex<Option<xeno_lib::ModelSession>> = Mutex::new(None);
static UPSCALER_SESSION: Mutex<Option<xeno_lib::UpscalerSession>> = Mutex::new(None);
static INTERPOLATOR_SESSION: Mutex<Option<xeno_lib::InterpolatorSession>> = Mutex::new(None);
static TRANSCRIBER_SESSION: Mutex<Option<xeno_lib::TranscriberSession>> = Mutex::new(None);
static SEPARATOR_SESSION: Mutex<Option<xeno_lib::SeparatorSession>> = Mutex::new(None);

/// Helper macro: lock the session mutex, lazily init if None, then run a closure with &mut Session.
///
/// If the mutex is poisoned (a previous holder panicked), returns a descriptive error.
/// If session initialization fails (e.g., model file missing), returns the init error.
macro_rules! with_session {
    ($static_mutex:expr, $init:expr, $body:expr) => {{
        let mut guard = $static_mutex.lock().map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Session lock poisoned: {e}"),
            )
        })?;
        if guard.is_none() {
            let session = $init.map_err(transform_err)?;
            *guard = Some(session);
        }
        // SAFETY rationale for expect: we just set guard to Some in the branch above,
        // so this will never panic. The expect message is for documentation only.
        let session = guard.as_mut().expect("session was just initialised");
        $body(session)
    }};
}

// ---------------------------------------------------------------------------
// TypeScript result types
// ---------------------------------------------------------------------------

/// A single transcription segment with start/end timestamps.
#[napi(object)]
pub struct TranscriptionSegment {
    /// Start time in milliseconds.
    pub start_ms: u32,
    /// End time in milliseconds.
    pub end_ms: u32,
    /// Transcribed text for this segment.
    pub text: String,
}

/// Full transcription result containing segments and detected language.
#[napi(object)]
pub struct TranscriptionResult {
    /// Individual segments with timestamps.
    pub segments: Vec<TranscriptionSegment>,
    /// Detected language (ISO 639-1 code, e.g., "en", "es"). Empty string if not detected.
    pub language: String,
}

/// Stem separation result. Each field contains interleaved stereo f32 PCM
/// samples (encoded as f64 for JavaScript compatibility).
#[napi(object)]
pub struct StemSeparationResult {
    /// Vocal track samples (interleaved stereo f64).
    pub vocals: Vec<f64>,
    /// Drum track samples (interleaved stereo f64).
    pub drums: Vec<f64>,
    /// Bass track samples (interleaved stereo f64).
    pub bass: Vec<f64>,
    /// Other instruments track samples (interleaved stereo f64).
    pub other: Vec<f64>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

// ---------------------------------------------------------------------------
// Image AI
// ---------------------------------------------------------------------------

/// Remove the background from an RGBA image using AI (BiRefNet).
///
/// The model is lazily loaded from `~/.xeno-lib/models/` on the first call
/// and cached for subsequent calls.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
///
/// # Returns
/// RGBA u8 buffer with the background made transparent.
///
/// # Errors
/// - If buffer size does not match `width * height * 4`
/// - If width or height is zero
/// - If the model file is not found in `~/.xeno-lib/models/`
/// - If ONNX inference fails
#[napi(ts_return_type = "Promise<Buffer>")]
pub async fn remove_background(buffer: Buffer, width: u32, height: u32) -> Result<Buffer> {
    // Validate before moving data to the blocking thread
    validate_image_buffer(&buffer, width, height)?;

    let data = buffer.to_vec();
    tokio::task::spawn_blocking(move || {
        let img = buffer_to_image(&data, width, height)?;
        with_session!(
            BG_SESSION,
            {
                let config = xeno_lib::BackgroundRemovalConfig::default();
                xeno_lib::load_model(&config)
            },
            |session: &mut xeno_lib::ModelSession| -> Result<Buffer> {
                let result =
                    xeno_lib::remove_background(&img, session).map_err(transform_err)?;
                Ok(image_to_vec(&result).into())
            }
        )
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

/// Upscale an RGBA image using AI (Real-ESRGAN).
///
/// The model is lazily loaded from `~/.xeno-lib/models/` on the first call
/// and cached for subsequent calls.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
/// * `scale` - Upscale factor (must be 2 or 4)
///
/// # Returns
/// Upscaled RGBA u8 buffer (`width*scale * height*scale * 4` bytes).
///
/// # Errors
/// - If buffer size does not match `width * height * 4`
/// - If width or height is zero
/// - If scale is not 2 or 4
/// - If the model file is not found in `~/.xeno-lib/models/`
/// - If ONNX inference fails
#[napi(ts_return_type = "Promise<Buffer>")]
pub async fn upscale_image(
    buffer: Buffer,
    width: u32,
    height: u32,
    scale: u32,
) -> Result<Buffer> {
    validate_image_buffer(&buffer, width, height)?;
    validate_upscale_factor(scale)?;

    let data = buffer.to_vec();
    tokio::task::spawn_blocking(move || {
        let img = buffer_to_image(&data, width, height)?;
        with_session!(
            UPSCALER_SESSION,
            {
                let config = xeno_lib::UpscaleConfig::default();
                xeno_lib::load_upscaler(&config)
            },
            |session: &mut xeno_lib::UpscalerSession| -> Result<Buffer> {
                let result = xeno_lib::ai_upscale(&img, session).map_err(transform_err)?;
                Ok(image_to_vec(&result).into())
            }
        )
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

/// Denoise an RGBA image using the built-in spatial denoise filter.
///
/// This uses a non-AI spatial filter with moderate strength for fast denoising.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
///
/// # Returns
/// Denoised RGBA u8 buffer (same dimensions as input).
///
/// # Errors
/// - If buffer size does not match `width * height * 4`
/// - If width or height is zero
/// - If the denoise filter fails
#[napi(ts_return_type = "Promise<Buffer>")]
pub async fn denoise_image(buffer: Buffer, width: u32, height: u32) -> Result<Buffer> {
    validate_image_buffer(&buffer, width, height)?;

    let data = buffer.to_vec();
    tokio::task::spawn_blocking(move || {
        let img = buffer_to_image(&data, width, height)?;
        // Use the spatial denoise filter with a moderate strength
        let result = xeno_lib::denoise(&img, 5).map_err(transform_err)?;
        Ok(image_to_vec(&result).into())
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

// ---------------------------------------------------------------------------
// Audio AI
// ---------------------------------------------------------------------------

/// Denoise audio using the built-in audio limiter and normalization pipeline.
///
/// This is a fast, non-AI denoise pipeline that applies a limiter followed
/// by peak normalization.
///
/// # Arguments
/// * `samples` - Interleaved PCM samples as f64 array
/// * `sample_rate` - Sample rate in Hz (1-384000)
///
/// # Returns
/// Denoised f32 PCM samples (as Float64Array for JavaScript compatibility).
///
/// # Errors
/// - If samples array is empty
/// - If sample rate is zero or exceeds 384000
/// - If any sample is NaN or Infinity
#[napi(ts_return_type = "Promise<Float64Array>")]
pub async fn denoise_audio(
    samples: Float64Array,
    sample_rate: u32,
) -> Result<Float64Array> {
    // Validate with channels=1 since denoise_audio doesn't use channel info
    validate_audio_samples(&samples, sample_rate, 1)?;

    let f32_samples: Vec<f32> = samples.iter().map(|&s| s as f32).collect();
    tokio::task::spawn_blocking(move || {
        // Use limiter + peak normalization for a simple denoise pipeline
        let gated = xeno_lib::audio::filters::limit(&f32_samples, -1.0);
        let normalized = xeno_lib::audio::filters::normalize_peak(&gated, -0.5);
        let result: Vec<f64> = normalized.iter().map(|&s| s as f64).collect();
        Ok(Float64Array::new(result))
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

/// Separate audio into stems (vocals, drums, bass, other) using AI (HTDemucs).
///
/// The model is lazily loaded from `~/.xeno-lib/models/` on the first call
/// and cached for subsequent calls.
///
/// # Arguments
/// * `file_path` - Path to an audio file (must exist). Supports all formats
///   handled by the Symphonia decoder.
///
/// # Returns
/// A `StemSeparationResult` with interleaved stereo f64 PCM samples for each
/// of the four stems (vocals, drums, bass, other) and the output sample rate.
///
/// # Errors
/// - If file path is empty or the file does not exist
/// - If the audio file cannot be decoded
/// - If the model file is not found in `~/.xeno-lib/models/`
/// - If ONNX inference fails
#[napi(ts_return_type = "Promise<StemSeparationResult>")]
pub async fn separate_stems(file_path: String) -> Result<StemSeparationResult> {
    validate_file_path(&file_path)?;

    tokio::task::spawn_blocking(move || {
        // First decode the audio
        let decoded = xeno_lib::decode_audio_file(&file_path).map_err(|e| {
            Error::new(Status::GenericFailure, format!("Audio decode failed: {e}"))
        })?;

        // Convert to stereo
        let stereo = if decoded.channels == 1 {
            xeno_lib::audio_separate::StereoAudio {
                left: decoded.samples.clone(),
                right: decoded.samples,
            }
        } else {
            // Deinterleave stereo
            let mut left = Vec::with_capacity(decoded.samples.len() / 2);
            let mut right = Vec::with_capacity(decoded.samples.len() / 2);
            for chunk in decoded.samples.chunks(2) {
                left.push(chunk[0]);
                right.push(if chunk.len() > 1 { chunk[1] } else { chunk[0] });
            }
            xeno_lib::audio_separate::StereoAudio { left, right }
        };

        with_session!(
            SEPARATOR_SESSION,
            {
                let config = xeno_lib::SeparationConfig::default();
                xeno_lib::load_separator(&config)
            },
            |session: &mut xeno_lib::SeparatorSession| -> Result<StemSeparationResult> {
                let separated =
                    xeno_lib::audio_separate::separate(&stereo, decoded.sample_rate, session)
                        .map_err(transform_err)?;

                let to_interleaved_f64 =
                    |stem: &xeno_lib::audio_separate::StereoAudio| -> Vec<f64> {
                        let mut out = Vec::with_capacity(stem.left.len() * 2);
                        for i in 0..stem.left.len() {
                            out.push(stem.left[i] as f64);
                            out.push(if i < stem.right.len() {
                                stem.right[i] as f64
                            } else {
                                stem.left[i] as f64
                            });
                        }
                        out
                    };

                let empty_stereo = xeno_lib::audio_separate::StereoAudio {
                    left: vec![],
                    right: vec![],
                };

                let vocals = separated
                    .stems
                    .get(&xeno_lib::audio_separate::AudioStem::Vocals)
                    .unwrap_or(&empty_stereo);
                let drums = separated
                    .stems
                    .get(&xeno_lib::audio_separate::AudioStem::Drums)
                    .unwrap_or(&empty_stereo);
                let bass = separated
                    .stems
                    .get(&xeno_lib::audio_separate::AudioStem::Bass)
                    .unwrap_or(&empty_stereo);
                let other = separated
                    .stems
                    .get(&xeno_lib::audio_separate::AudioStem::Other)
                    .unwrap_or(&empty_stereo);

                Ok(StemSeparationResult {
                    vocals: to_interleaved_f64(vocals),
                    drums: to_interleaved_f64(drums),
                    bass: to_interleaved_f64(bass),
                    other: to_interleaved_f64(other),
                    sample_rate: separated.sample_rate,
                })
            }
        )
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

/// Transcribe audio from a file using AI (Whisper).
///
/// The model is lazily loaded from `~/.xeno-lib/models/` on the first call
/// and cached for subsequent calls.
///
/// # Arguments
/// * `file_path` - Path to an audio file (must exist). Supports all formats
///   handled by the Symphonia decoder.
///
/// # Returns
/// A `TranscriptionResult` with timestamped segments and the detected language.
///
/// # Errors
/// - If file path is empty or the file does not exist
/// - If the audio file cannot be decoded
/// - If the model file is not found in `~/.xeno-lib/models/`
/// - If ONNX inference fails
#[napi(ts_return_type = "Promise<TranscriptionResult>")]
pub async fn transcribe_audio(file_path: String) -> Result<TranscriptionResult> {
    validate_file_path(&file_path)?;

    tokio::task::spawn_blocking(move || {
        // Decode audio first
        let decoded = xeno_lib::decode_audio_file(&file_path).map_err(|e| {
            Error::new(Status::GenericFailure, format!("Audio decode failed: {e}"))
        })?;

        // Convert to mono for transcription
        let mono = decoded.to_mono();

        with_session!(
            TRANSCRIBER_SESSION,
            {
                let config = xeno_lib::TranscribeConfig::default();
                xeno_lib::load_transcriber(&config)
            },
            |session: &mut xeno_lib::TranscriberSession| -> Result<TranscriptionResult> {
                let transcript =
                    xeno_lib::transcribe::transcribe(&mono.samples, mono.sample_rate, session)
                        .map_err(transform_err)?;

                let segments = transcript
                    .segments
                    .iter()
                    .map(|seg| TranscriptionSegment {
                        start_ms: (seg.start * 1000.0).round() as u32,
                        end_ms: (seg.end * 1000.0).round() as u32,
                        text: seg.text.clone(),
                    })
                    .collect();

                Ok(TranscriptionResult {
                    segments,
                    language: transcript.language.unwrap_or_default(),
                })
            }
        )
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

/// Interpolate a frame between two RGBA frames using AI (RIFE).
///
/// The model is lazily loaded from `~/.xeno-lib/models/` on the first call
/// and cached for subsequent calls.
///
/// # Arguments
/// * `frame1` - First RGBA u8 frame (4 bytes per pixel, row-major)
/// * `frame2` - Second RGBA u8 frame (same dimensions as frame1)
/// * `width` - Frame width in pixels (must be > 0)
/// * `height` - Frame height in pixels (must be > 0)
/// * `factor` - Interpolation position (0.0 = frame1, 1.0 = frame2, must be in [0.0, 1.0])
///
/// # Returns
/// Interpolated RGBA u8 buffer (same dimensions as input frames).
///
/// # Errors
/// - If either buffer size does not match `width * height * 4`
/// - If width or height is zero
/// - If factor is outside [0.0, 1.0], NaN, or Infinity
/// - If the model file is not found in `~/.xeno-lib/models/`
/// - If ONNX inference fails
#[napi(ts_return_type = "Promise<Buffer>")]
pub async fn interpolate_frames(
    frame1: Buffer,
    frame2: Buffer,
    width: u32,
    height: u32,
    factor: f64,
) -> Result<Buffer> {
    // Validate both frames and the interpolation factor before moving to blocking thread
    validate_image_buffer(&frame1, width, height)?;
    validate_image_buffer(&frame2, width, height)?;
    validate_interpolation_factor(factor)?;

    let data1 = frame1.to_vec();
    let data2 = frame2.to_vec();
    tokio::task::spawn_blocking(move || {
        let img1 = buffer_to_image(&data1, width, height)?;
        let img2 = buffer_to_image(&data2, width, height)?;
        with_session!(
            INTERPOLATOR_SESSION,
            {
                let config = xeno_lib::InterpolationConfig::default();
                xeno_lib::load_interpolator(&config)
            },
            |session: &mut xeno_lib::InterpolatorSession| -> Result<Buffer> {
                let result =
                    xeno_lib::interpolate_frame(&img1, &img2, factor as f32, session)
                        .map_err(transform_err)?;
                Ok(image_to_vec(&result).into())
            }
        )
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}
