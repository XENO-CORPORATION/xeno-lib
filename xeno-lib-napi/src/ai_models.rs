// DEPRECATED: All AI model N-API bindings will be served by xeno-rt inference API instead.
// When xeno-rt absorbs these models, consuming apps (Pixel, Motion, Sound, Hub) will call
// xeno-rt's API (HTTP or IPC) for inference, not these N-API functions.
// The non-AI functions in this file (denoise_image using spatial filter, denoise_audio using
// limiter/normalization) are NOT deprecated and should move to the image_processing or
// audio_processing NAPI modules.
//!
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

use image::DynamicImage;

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
static FACE_RESTORER_SESSION: Mutex<Option<xeno_lib::FaceRestorerSession>> = Mutex::new(None);
static COLORIZER_SESSION: Mutex<Option<xeno_lib::ColorizerSession>> = Mutex::new(None);
static INPAINTER_SESSION: Mutex<Option<xeno_lib::InpainterSession>> = Mutex::new(None);
static FACE_DETECTOR_SESSION: Mutex<Option<xeno_lib::FaceDetectorSession>> = Mutex::new(None);
static DEPTH_SESSION: Mutex<Option<xeno_lib::DepthSession>> = Mutex::new(None);
static STYLE_SESSION: Mutex<Option<xeno_lib::StyleSession>> = Mutex::new(None);
static OCR_SESSION: Mutex<Option<xeno_lib::OcrSession>> = Mutex::new(None);
static POSE_SESSION: Mutex<Option<xeno_lib::PoseSession>> = Mutex::new(None);
static FACE_ANALYZER_SESSION: Mutex<Option<xeno_lib::FaceAnalyzerSession>> = Mutex::new(None);

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

/// A detected face bounding box with landmarks.
#[napi(object)]
pub struct DetectedFaceJs {
    /// Bounding box x coordinate (pixels).
    pub x: f64,
    /// Bounding box y coordinate (pixels).
    pub y: f64,
    /// Bounding box width (pixels).
    pub width: f64,
    /// Bounding box height (pixels).
    pub height: f64,
    /// Detection confidence (0.0 to 1.0).
    pub confidence: f64,
}

/// Depth estimation result.
#[napi(object)]
pub struct DepthMapResult {
    /// Depth values as f64 array (0.0=near, 1.0=far), row-major.
    pub data: Vec<f64>,
    /// Width of the depth map.
    pub width: u32,
    /// Height of the depth map.
    pub height: u32,
}

/// OCR text block result.
#[napi(object)]
pub struct OcrTextBlock {
    /// Recognized text.
    pub text: String,
    /// Bounding box x coordinate (pixels).
    pub x: f64,
    /// Bounding box y coordinate (pixels).
    pub y: f64,
    /// Bounding box width (pixels).
    pub width: f64,
    /// Bounding box height (pixels).
    pub height: f64,
    /// Recognition confidence (0.0 to 1.0).
    pub confidence: f64,
}

/// OCR result containing all detected text blocks.
#[napi(object)]
pub struct OcrResultJs {
    /// All recognized text blocks.
    pub blocks: Vec<OcrTextBlock>,
    /// Full text concatenated.
    pub full_text: String,
}

/// Detected body pose keypoint.
#[napi(object)]
pub struct PoseKeypointJs {
    /// Keypoint name (e.g., "nose", "left_shoulder").
    pub name: String,
    /// X coordinate (pixels).
    pub x: f64,
    /// Y coordinate (pixels).
    pub y: f64,
    /// Detection confidence (0.0 to 1.0).
    pub confidence: f64,
}

/// Face analysis result.
#[napi(object)]
pub struct FaceAnalysisResultJs {
    /// Estimated age.
    pub age: f64,
    /// Detected gender ("male" or "female").
    pub gender: String,
    /// Gender confidence (0.0 to 1.0).
    pub gender_confidence: f64,
    /// Dominant emotion.
    pub emotion: String,
    /// Emotion confidence (0.0 to 1.0).
    pub emotion_confidence: f64,
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

/// DEPRECATED: will be served by xeno-rt inference API instead.
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

/// DEPRECATED: will be served by xeno-rt inference API instead.
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
        // Select the correct model based on the requested scale factor
        let model = match scale {
            2 => xeno_lib::UpscaleModel::RealEsrganX2,
            4 => xeno_lib::UpscaleModel::RealEsrganX4Plus,
            8 => xeno_lib::UpscaleModel::RealEsrganX8,
            _ => xeno_lib::UpscaleModel::RealEsrganX4Plus, // fallback (validated above)
        };
        with_session!(
            UPSCALER_SESSION,
            {
                let config = xeno_lib::UpscaleConfig::new(model);
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

/// DEPRECATED: will be served by xeno-rt inference API instead.
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

/// DEPRECATED: will be served by xeno-rt inference API instead.
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

/// DEPRECATED: will be served by xeno-rt inference API instead.
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

// ---------------------------------------------------------------------------
// Face Restore (GFPGAN)
// ---------------------------------------------------------------------------

/// DEPRECATED: will be served by xeno-rt inference API instead.
/// Restore faces in an RGBA image using AI (GFPGAN).
///
/// Detects faces and enhances/restores their quality.
/// Model is lazily loaded on first call and cached.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
///
/// # Returns
/// RGBA u8 buffer with restored faces (same dimensions).
#[napi(ts_return_type = "Promise<Buffer>")]
pub async fn restore_faces(buffer: Buffer, width: u32, height: u32) -> Result<Buffer> {
    validate_image_buffer(&buffer, width, height)?;

    let data = buffer.to_vec();
    tokio::task::spawn_blocking(move || {
        let img = buffer_to_image(&data, width, height)?;
        with_session!(
            FACE_RESTORER_SESSION,
            {
                let config = xeno_lib::FaceRestoreConfig::default();
                xeno_lib::load_restorer(&config)
            },
            |session: &mut xeno_lib::FaceRestorerSession| -> Result<Buffer> {
                let result =
                    xeno_lib::restore_faces(&img, session).map_err(transform_err)?;
                Ok(image_to_vec(&result).into())
            }
        )
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

// ---------------------------------------------------------------------------
// Colorize (DDColor)
// ---------------------------------------------------------------------------

/// DEPRECATED: will be served by xeno-rt inference API instead.
/// Colorize a grayscale RGBA image using AI (DDColor).
///
/// Model is lazily loaded on first call and cached.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
///
/// # Returns
/// Colorized RGBA u8 buffer (same dimensions).
#[napi(ts_return_type = "Promise<Buffer>")]
pub async fn colorize(buffer: Buffer, width: u32, height: u32) -> Result<Buffer> {
    validate_image_buffer(&buffer, width, height)?;

    let data = buffer.to_vec();
    tokio::task::spawn_blocking(move || {
        let img = buffer_to_image(&data, width, height)?;
        with_session!(
            COLORIZER_SESSION,
            {
                let config = xeno_lib::ColorizeConfig::default();
                xeno_lib::load_colorizer(&config)
            },
            |session: &mut xeno_lib::ColorizerSession| -> Result<Buffer> {
                let result =
                    xeno_lib::colorize(&img, session).map_err(transform_err)?;
                Ok(image_to_vec(&result).into())
            }
        )
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

// ---------------------------------------------------------------------------
// Inpaint (LaMa)
// ---------------------------------------------------------------------------

/// DEPRECATED: will be served by xeno-rt inference API instead.
/// Inpaint (fill) masked regions of an RGBA image using AI (LaMa).
///
/// Model is lazily loaded on first call and cached.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `mask` - Single-channel u8 mask (width * height bytes; 255=inpaint, 0=keep)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
///
/// # Returns
/// Inpainted RGBA u8 buffer (same dimensions).
#[napi(ts_return_type = "Promise<Buffer>")]
pub async fn inpaint(
    buffer: Buffer,
    mask: Buffer,
    width: u32,
    height: u32,
) -> Result<Buffer> {
    validate_image_buffer(&buffer, width, height)?;

    // Validate mask dimensions
    let expected_mask_len = (width as usize) * (height as usize);
    if mask.len() != expected_mask_len {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Mask size mismatch: expected {} bytes ({}x{}), got {}",
                expected_mask_len, width, height, mask.len()
            ),
        ));
    }

    let img_data = buffer.to_vec();
    let mask_data = mask.to_vec();
    tokio::task::spawn_blocking(move || {
        let img = buffer_to_image(&img_data, width, height)?;
        // Convert mask bytes to a grayscale image for xeno-lib
        let mask_img = image::GrayImage::from_raw(width, height, mask_data).ok_or_else(|| {
            Error::new(
                Status::GenericFailure,
                "Failed to construct mask image from buffer".to_string(),
            )
        })?;
        let mask_dynamic = DynamicImage::ImageLuma8(mask_img);
        with_session!(
            INPAINTER_SESSION,
            {
                let config = xeno_lib::InpaintConfig::default();
                xeno_lib::load_inpainter(&config)
            },
            |session: &mut xeno_lib::InpainterSession| -> Result<Buffer> {
                let result =
                    xeno_lib::inpaint(&img, &mask_dynamic, session).map_err(transform_err)?;
                Ok(image_to_vec(&result).into())
            }
        )
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

// ---------------------------------------------------------------------------
// Face Detection (SCRFD)
// ---------------------------------------------------------------------------

/// DEPRECATED: will be served by xeno-rt inference API instead.
/// Detect faces in an RGBA image using AI (SCRFD).
///
/// Model is lazily loaded on first call and cached.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
///
/// # Returns
/// Array of detected face bounding boxes with confidence scores.
#[napi(ts_return_type = "Promise<Array>")]
pub async fn detect_faces(
    buffer: Buffer,
    width: u32,
    height: u32,
) -> Result<Vec<DetectedFaceJs>> {
    validate_image_buffer(&buffer, width, height)?;

    let data = buffer.to_vec();
    tokio::task::spawn_blocking(move || {
        let img = buffer_to_image(&data, width, height)?;
        with_session!(
            FACE_DETECTOR_SESSION,
            {
                let config = xeno_lib::FaceDetectConfig::default();
                xeno_lib::load_detector(&config)
            },
            |session: &mut xeno_lib::FaceDetectorSession| -> Result<Vec<DetectedFaceJs>> {
                let faces =
                    xeno_lib::detect_faces(&img, session).map_err(transform_err)?;
                Ok(faces
                    .iter()
                    .map(|f| DetectedFaceJs {
                        x: f.bbox.0 as f64,
                        y: f.bbox.1 as f64,
                        width: f.bbox.2 as f64,
                        height: f.bbox.3 as f64,
                        confidence: f.confidence as f64,
                    })
                    .collect())
            }
        )
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

// ---------------------------------------------------------------------------
// Depth Estimation (Depth Anything / MiDaS)
// ---------------------------------------------------------------------------

/// DEPRECATED: will be served by xeno-rt inference API instead.
/// Estimate depth map from an RGBA image using AI (Depth Anything).
///
/// Model is lazily loaded on first call and cached.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
///
/// # Returns
/// DepthMapResult with f32 depth values (0.0=near, 1.0=far).
#[napi(ts_return_type = "Promise<DepthMapResult>")]
pub async fn estimate_depth(
    buffer: Buffer,
    width: u32,
    height: u32,
) -> Result<DepthMapResult> {
    validate_image_buffer(&buffer, width, height)?;

    let data = buffer.to_vec();
    tokio::task::spawn_blocking(move || {
        let img = buffer_to_image(&data, width, height)?;
        with_session!(
            DEPTH_SESSION,
            {
                let config = xeno_lib::DepthConfig::default();
                xeno_lib::load_depth_estimator(&config)
            },
            |session: &mut xeno_lib::DepthSession| -> Result<DepthMapResult> {
                let depth_map =
                    xeno_lib::estimate_depth(&img, session).map_err(transform_err)?;
                let (h, w) = depth_map.values.dim();
                Ok(DepthMapResult {
                    data: depth_map.values.iter().map(|&v| v as f64).collect(),
                    width: w as u32,
                    height: h as u32,
                })
            }
        )
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

// ---------------------------------------------------------------------------
// Style Transfer (Fast NST)
// ---------------------------------------------------------------------------

/// DEPRECATED: will be served by xeno-rt inference API instead.
/// Apply neural style transfer to an RGBA image using AI.
///
/// Model is lazily loaded on first call and cached.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
/// * `style` - Style name (e.g., "mosaic", "candy", "rain_princess", "udnie", "pointilism")
///
/// # Returns
/// Stylized RGBA u8 buffer (same dimensions).
#[napi(ts_return_type = "Promise<Buffer>")]
pub async fn style_transfer(
    buffer: Buffer,
    width: u32,
    height: u32,
    style: String,
) -> Result<Buffer> {
    validate_image_buffer(&buffer, width, height)?;

    let data = buffer.to_vec();
    tokio::task::spawn_blocking(move || {
        let img = buffer_to_image(&data, width, height)?;
        // Parse the style string into the PretrainedStyle enum
        let pretrained_style = match style.to_lowercase().as_str() {
            "candy" => xeno_lib::PretrainedStyle::Candy,
            "mosaic" => xeno_lib::PretrainedStyle::Mosaic,
            "rain_princess" | "rain-princess" => xeno_lib::PretrainedStyle::RainPrincess,
            "udnie" => xeno_lib::PretrainedStyle::Udnie,
            "pointilism" | "pointillism" => xeno_lib::PretrainedStyle::Pointillism,
            _ => xeno_lib::PretrainedStyle::Mosaic, // default fallback
        };
        let config = xeno_lib::StyleConfig {
            pretrained_style: Some(pretrained_style),
            ..Default::default()
        };
        with_session!(
            STYLE_SESSION,
            { xeno_lib::load_style_model(&config) },
            |session: &mut xeno_lib::StyleSession| -> Result<Buffer> {
                let result =
                    xeno_lib::stylize(&img, session).map_err(transform_err)?;
                Ok(image_to_vec(&result).into())
            }
        )
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

// ---------------------------------------------------------------------------
// OCR (PaddleOCR)
// ---------------------------------------------------------------------------

/// DEPRECATED: will be served by xeno-rt inference API instead.
/// Extract text from an RGBA image using AI (PaddleOCR).
///
/// Model is lazily loaded on first call and cached.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
///
/// # Returns
/// OcrResultJs with detected text blocks and full text.
#[napi(ts_return_type = "Promise<OcrResultJs>")]
pub async fn extract_text(
    buffer: Buffer,
    width: u32,
    height: u32,
) -> Result<OcrResultJs> {
    validate_image_buffer(&buffer, width, height)?;

    let data = buffer.to_vec();
    tokio::task::spawn_blocking(move || {
        let img = buffer_to_image(&data, width, height)?;
        with_session!(
            OCR_SESSION,
            {
                let config = xeno_lib::OcrConfig::default();
                xeno_lib::load_ocr_model(&config)
            },
            |session: &mut xeno_lib::OcrSession| -> Result<OcrResultJs> {
                let result =
                    xeno_lib::extract_text(&img, session).map_err(transform_err)?;
                let blocks: Vec<OcrTextBlock> = result
                    .blocks
                    .iter()
                    .map(|b| OcrTextBlock {
                        text: b.text.clone(),
                        x: b.bbox.0 as f64,
                        y: b.bbox.1 as f64,
                        width: b.bbox.2 as f64,
                        height: b.bbox.3 as f64,
                        confidence: b.confidence as f64,
                    })
                    .collect();
                Ok(OcrResultJs { blocks, full_text: result.text.clone() })
            }
        )
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

// ---------------------------------------------------------------------------
// Pose Estimation (MoveNet)
// ---------------------------------------------------------------------------

/// DEPRECATED: will be served by xeno-rt inference API instead.
/// Detect body poses in an RGBA image using AI (MoveNet).
///
/// Model is lazily loaded on first call and cached.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
///
/// # Returns
/// Array of arrays of keypoints (one array per detected person).
#[napi(ts_return_type = "Promise<Array>")]
pub async fn detect_poses(
    buffer: Buffer,
    width: u32,
    height: u32,
) -> Result<Vec<Vec<PoseKeypointJs>>> {
    validate_image_buffer(&buffer, width, height)?;

    let data = buffer.to_vec();
    tokio::task::spawn_blocking(move || {
        let img = buffer_to_image(&data, width, height)?;
        with_session!(
            POSE_SESSION,
            {
                let config = xeno_lib::PoseConfig::default();
                xeno_lib::load_pose_model(&config)
            },
            |session: &mut xeno_lib::PoseSession| -> Result<Vec<Vec<PoseKeypointJs>>> {
                let poses =
                    xeno_lib::detect_poses(&img, session).map_err(transform_err)?;
                // Map keypoint index to COCO body part name
                let body_parts = [
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle",
                ];
                Ok(poses
                    .iter()
                    .map(|pose| {
                        pose.keypoints
                            .iter()
                            .enumerate()
                            .map(|(i, kp)| PoseKeypointJs {
                                name: body_parts.get(i).unwrap_or(&"unknown").to_string(),
                                x: kp.x as f64,
                                y: kp.y as f64,
                                confidence: kp.confidence as f64,
                            })
                            .collect()
                    })
                    .collect())
            }
        )
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

// ---------------------------------------------------------------------------
// Face Analysis (Multi-task CNN)
// ---------------------------------------------------------------------------

/// DEPRECATED: will be served by xeno-rt inference API instead.
/// Analyze faces in an RGBA image (age, gender, emotion) using AI.
///
/// Model is lazily loaded on first call and cached.
///
/// # Arguments
/// * `buffer` - RGBA u8 pixel data (4 bytes per pixel, row-major)
/// * `width` - Image width in pixels (must be > 0)
/// * `height` - Image height in pixels (must be > 0)
///
/// # Returns
/// Array of analysis results (one per detected face).
#[napi(ts_return_type = "Promise<Array>")]
pub async fn analyze_faces(
    buffer: Buffer,
    width: u32,
    height: u32,
) -> Result<Vec<FaceAnalysisResultJs>> {
    validate_image_buffer(&buffer, width, height)?;

    let data = buffer.to_vec();
    tokio::task::spawn_blocking(move || {
        let img = buffer_to_image(&data, width, height)?;
        with_session!(
            FACE_ANALYZER_SESSION,
            {
                let config = xeno_lib::FaceAnalysisConfig::default();
                xeno_lib::load_analyzer(&config)
            },
            |session: &mut xeno_lib::FaceAnalyzerSession| -> Result<Vec<FaceAnalysisResultJs>> {
                // First detect faces to get bounding boxes
                let faces = xeno_lib::detect_faces_quick(&img).map_err(transform_err)?;
                let regions: Vec<(u32, u32, u32, u32)> = faces
                    .iter()
                    .map(|f| f.bbox)
                    .collect();

                if regions.is_empty() {
                    return Ok(vec![]);
                }

                let results =
                    xeno_lib::analyze_faces(&img, &regions, session).map_err(transform_err)?;
                Ok(results
                    .iter()
                    .map(|r| FaceAnalysisResultJs {
                        age: r.age as f64,
                        gender: format!("{:?}", r.gender),
                        gender_confidence: r.gender_confidence as f64,
                        emotion: format!("{:?}", r.emotion),
                        emotion_confidence: r.emotion_confidence as f64,
                    })
                    .collect())
            }
        )
    })
    .await
    .map_err(|e| Error::new(Status::GenericFailure, format!("Task join error: {e}")))?
}

// ---------------------------------------------------------------------------
// Model Management
// ---------------------------------------------------------------------------

/// DEPRECATED: will be served by xeno-rt inference API instead.
/// Get the path to the model directory (~/.xeno-lib/models/).
/// Creates the directory if it doesn't exist.
#[napi]
pub fn get_model_dir() -> Result<String> {
    let home = dirs_next::home_dir().ok_or_else(|| {
        Error::new(Status::GenericFailure, "Cannot determine home directory".to_string())
    })?;
    let model_dir = home.join(".xeno-lib").join("models");
    if !model_dir.exists() {
        std::fs::create_dir_all(&model_dir).map_err(|e| {
            Error::new(Status::GenericFailure, format!("Failed to create model dir: {e}"))
        })?;
    }
    Ok(model_dir.to_string_lossy().to_string())
}

/// DEPRECATED: will be served by xeno-rt inference API instead.
/// Check if a specific model file exists in the model directory.
///
/// # Arguments
/// * `model_name` - Name of the model file (e.g., "birefnet.onnx")
///
/// # Returns
/// true if the model file exists.
#[napi]
pub fn is_model_available(model_name: String) -> Result<bool> {
    let home = dirs_next::home_dir().ok_or_else(|| {
        Error::new(Status::GenericFailure, "Cannot determine home directory".to_string())
    })?;
    let model_path = home.join(".xeno-lib").join("models").join(&model_name);
    Ok(model_path.exists())
}

/// DEPRECATED: will be served by xeno-rt inference API instead.
/// List all model files in the model directory.
///
/// # Returns
/// Array of model file names present in ~/.xeno-lib/models/.
#[napi]
pub fn list_available_models() -> Result<Vec<String>> {
    let home = dirs_next::home_dir().ok_or_else(|| {
        Error::new(Status::GenericFailure, "Cannot determine home directory".to_string())
    })?;
    let model_dir = home.join(".xeno-lib").join("models");
    if !model_dir.exists() {
        return Ok(vec![]);
    }
    let entries = std::fs::read_dir(&model_dir).map_err(|e| {
        Error::new(Status::GenericFailure, format!("Failed to read model dir: {e}"))
    })?;
    let mut names = Vec::new();
    for entry in entries {
        if let Ok(entry) = entry {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".onnx") {
                    names.push(name.to_string());
                }
            }
        }
    }
    Ok(names)
}
