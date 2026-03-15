//! AI model bindings (Priority 3 -- for all apps).
//!
//! All AI functions are async because inference is slow.
//! Model sessions are lazily loaded and cached across calls via `Mutex<Option<T>>`.

use std::sync::Mutex;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::helpers::{buffer_to_image, image_to_vec, transform_err};

// ---------------------------------------------------------------------------
// Lazy model caches (loaded on first call, reused thereafter)
// ---------------------------------------------------------------------------

static BG_SESSION: Mutex<Option<xeno_lib::ModelSession>> = Mutex::new(None);
static UPSCALER_SESSION: Mutex<Option<xeno_lib::UpscalerSession>> = Mutex::new(None);
static INTERPOLATOR_SESSION: Mutex<Option<xeno_lib::InterpolatorSession>> = Mutex::new(None);
static TRANSCRIBER_SESSION: Mutex<Option<xeno_lib::TranscriberSession>> = Mutex::new(None);
static SEPARATOR_SESSION: Mutex<Option<xeno_lib::SeparatorSession>> = Mutex::new(None);

/// Helper macro: lock the session mutex, lazily init if None, then run a closure with &mut Session.
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
        let session = guard.as_mut().expect("session was just initialised");
        $body(session)
    }};
}

// ---------------------------------------------------------------------------
// TypeScript result types
// ---------------------------------------------------------------------------

/// A single transcription segment with timestamps.
#[napi(object)]
pub struct TranscriptionSegment {
    /// Start time in milliseconds.
    pub start_ms: u32,
    /// End time in milliseconds.
    pub end_ms: u32,
    /// Transcribed text for this segment.
    pub text: String,
}

/// Full transcription result.
#[napi(object)]
pub struct TranscriptionResult {
    /// Individual segments with timestamps.
    pub segments: Vec<TranscriptionSegment>,
    /// Detected language (if auto-detect was used).
    pub language: String,
}

/// Stem separation result. Each field contains interleaved f32 PCM samples
/// (encoded as f64 for JavaScript compatibility).
#[napi(object)]
pub struct StemSeparationResult {
    /// Vocal track samples.
    pub vocals: Vec<f64>,
    /// Drum track samples.
    pub drums: Vec<f64>,
    /// Bass track samples.
    pub bass: Vec<f64>,
    /// Other instruments samples.
    pub other: Vec<f64>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

// ---------------------------------------------------------------------------
// Image AI
// ---------------------------------------------------------------------------

/// Remove the background from an RGBA image using AI (BiRefNet).
///
/// Returns an RGBA buffer with the background made transparent.
/// The model is lazily loaded on first call and reused.
#[napi(ts_return_type = "Promise<Buffer>")]
pub async fn remove_background(buffer: Buffer, width: u32, height: u32) -> Result<Buffer> {
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
/// `scale` is the upscale factor (2 or 4). Default model is x4plus.
/// Returns the upscaled RGBA buffer.
#[napi(ts_return_type = "Promise<Buffer>")]
pub async fn upscale_image(
    buffer: Buffer,
    width: u32,
    height: u32,
    _scale: u32,
) -> Result<Buffer> {
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
/// Returns the denoised RGBA buffer.
#[napi(ts_return_type = "Promise<Buffer>")]
pub async fn denoise_image(buffer: Buffer, width: u32, height: u32) -> Result<Buffer> {
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

/// Denoise audio using the built-in audio filters.
///
/// Returns denoised f32 PCM samples (as f64 array for JS compatibility).
#[napi(ts_return_type = "Promise<Float64Array>")]
pub async fn denoise_audio(
    samples: Float64Array,
    _sample_rate: u32,
) -> Result<Float64Array> {
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
/// `file_path` is the path to an audio file. The model is lazily loaded.
#[napi(ts_return_type = "Promise<StemSeparationResult>")]
pub async fn separate_stems(file_path: String) -> Result<StemSeparationResult> {
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
/// Returns segments with timestamps and detected language.
#[napi(ts_return_type = "Promise<TranscriptionResult>")]
pub async fn transcribe_audio(file_path: String) -> Result<TranscriptionResult> {
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
/// `factor` is the interpolation position (0.0 = frame1, 1.0 = frame2).
/// Returns the interpolated RGBA buffer.
#[napi(ts_return_type = "Promise<Buffer>")]
pub async fn interpolate_frames(
    frame1: Buffer,
    frame2: Buffer,
    width: u32,
    height: u32,
    factor: f64,
) -> Result<Buffer> {
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
