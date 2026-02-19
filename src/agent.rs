//! Agent-friendly API for AI integration.
//!
//! This module provides structured output types and utilities designed for
//! AI agent consumption. All operations return JSON-serializable results
//! with consistent schemas for easy parsing.
//!
//! # Design Principles
//!
//! 1. **Structured Output**: All results are serializable to JSON
//! 2. **Consistent Errors**: Errors include codes, messages, and context
//! 3. **Progress Tracking**: Long operations support progress callbacks
//! 4. **Batch Operations**: Efficient processing of multiple items
//! 5. **Metadata Rich**: Results include timing, file sizes, and diagnostics
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::agent::{AgentResult, VideoAnalysis, analyze_video};
//!
//! let result = analyze_video("input.mp4")?;
//! println!("{}", serde_json::to_string_pretty(&result)?);
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Result type for agent operations with structured error information.
#[derive(Debug, Clone)]
pub struct AgentResult<T> {
    /// Whether the operation succeeded
    pub success: bool,
    /// The result data (if successful)
    pub data: Option<T>,
    /// Error information (if failed)
    pub error: Option<AgentError>,
    /// Operation timing information
    pub timing: OperationTiming,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl<T> AgentResult<T> {
    /// Create a successful result.
    pub fn ok(data: T, timing: OperationTiming) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timing,
            metadata: HashMap::new(),
        }
    }

    /// Create a failed result.
    pub fn err(error: AgentError, timing: OperationTiming) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            timing,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the result.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Structured error information for agents.
#[derive(Debug, Clone)]
pub struct AgentError {
    /// Error code (e.g., "IO_ERROR", "DECODE_ERROR", "INVALID_INPUT")
    pub code: String,
    /// Human-readable error message
    pub message: String,
    /// Additional context about the error
    pub context: HashMap<String, String>,
    /// Whether the error is recoverable
    pub recoverable: bool,
    /// Suggested action for the agent
    pub suggestion: Option<String>,
}

impl AgentError {
    /// Create a new error.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            context: HashMap::new(),
            recoverable: false,
            suggestion: None,
        }
    }

    /// Add context to the error.
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Mark as recoverable with suggestion.
    pub fn recoverable(mut self, suggestion: impl Into<String>) -> Self {
        self.recoverable = true;
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Common error: file not found.
    pub fn file_not_found(path: impl AsRef<Path>) -> Self {
        Self::new("FILE_NOT_FOUND", format!("File not found: {}", path.as_ref().display()))
            .with_context("path", path.as_ref().display().to_string())
            .recoverable("Check file path and ensure the file exists")
    }

    /// Common error: unsupported format.
    pub fn unsupported_format(format: &str, supported: &[&str]) -> Self {
        Self::new("UNSUPPORTED_FORMAT", format!("Unsupported format: {}", format))
            .with_context("format", format.to_string())
            .with_context("supported", supported.join(", "))
            .recoverable("Convert to a supported format first")
    }

    /// Common error: decode failure.
    pub fn decode_error(message: impl Into<String>) -> Self {
        Self::new("DECODE_ERROR", message)
    }

    /// Common error: encode failure.
    pub fn encode_error(message: impl Into<String>) -> Self {
        Self::new("ENCODE_ERROR", message)
    }

    /// Common error: invalid parameters.
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self::new("INVALID_PARAMS", message)
            .recoverable("Check input parameters")
    }
}

/// Timing information for an operation.
#[derive(Debug, Clone)]
pub struct OperationTiming {
    /// Total operation duration
    pub total_ms: u64,
    /// Breakdown of phases (optional)
    pub phases: HashMap<String, u64>,
    /// Timestamp when operation started (Unix ms)
    pub started_at: u64,
    /// Timestamp when operation completed (Unix ms)
    pub completed_at: u64,
}

impl OperationTiming {
    /// Create timing from a start instant.
    pub fn from_start(start: Instant) -> Self {
        let elapsed = start.elapsed();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            total_ms: elapsed.as_millis() as u64,
            phases: HashMap::new(),
            started_at: now - elapsed.as_millis() as u64,
            completed_at: now,
        }
    }

    /// Add a phase duration.
    pub fn with_phase(mut self, name: impl Into<String>, duration_ms: u64) -> Self {
        self.phases.insert(name.into(), duration_ms);
        self
    }
}

/// Video analysis result for agents.
#[derive(Debug, Clone)]
pub struct VideoAnalysis {
    /// File path
    pub path: String,
    /// Container format (MP4, MKV, IVF, etc.)
    pub container: String,
    /// Video codec
    pub video_codec: Option<String>,
    /// Audio codec
    pub audio_codec: Option<String>,
    /// Video width in pixels
    pub width: u32,
    /// Video height in pixels
    pub height: u32,
    /// Duration in seconds
    pub duration_secs: f64,
    /// Frame rate (fps)
    pub frame_rate: f64,
    /// Total frame count
    pub frame_count: u32,
    /// Video bitrate in kbps
    pub bitrate_kbps: u32,
    /// File size in bytes
    pub file_size: u64,
    /// Whether NVDEC can decode this
    pub nvdec_compatible: bool,
    /// Whether dav1d can decode this (AV1 only)
    pub dav1d_compatible: bool,
    /// Audio sample rate (if audio present)
    pub audio_sample_rate: Option<u32>,
    /// Audio channels (if audio present)
    pub audio_channels: Option<u8>,
}

/// Image analysis result for agents.
#[derive(Debug, Clone)]
pub struct ImageAnalysis {
    /// File path
    pub path: String,
    /// Image format (PNG, JPEG, WebP, etc.)
    pub format: String,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Color type (RGB, RGBA, Grayscale, etc.)
    pub color_type: String,
    /// Bit depth
    pub bit_depth: u8,
    /// File size in bytes
    pub file_size: u64,
    /// Has alpha channel
    pub has_alpha: bool,
    /// Is animated (GIF, APNG)
    pub is_animated: bool,
}

/// Encode result for agents.
#[derive(Debug, Clone)]
pub struct EncodeResult {
    /// Output file path
    pub output_path: String,
    /// Number of frames encoded
    pub frame_count: u32,
    /// Output file size in bytes
    pub file_size: u64,
    /// Achieved bitrate in kbps
    pub bitrate_kbps: u32,
    /// Encoding speed in fps
    pub encode_fps: f64,
    /// Codec used
    pub codec: String,
    /// Quality setting used
    pub quality: u8,
    /// Speed preset used
    pub speed_preset: String,
}

/// Decode result for agents.
#[derive(Debug, Clone)]
pub struct DecodeResult {
    /// Input file path
    pub input_path: String,
    /// Number of frames decoded
    pub frame_count: u32,
    /// Decoder backend used
    pub backend: String,
    /// Decoding speed in fps
    pub decode_fps: f64,
    /// Frame dimensions
    pub width: u32,
    pub height: u32,
}

/// Transcode result for agents.
#[derive(Debug, Clone)]
pub struct TranscodeResult {
    /// Input file path
    pub input_path: String,
    /// Output file path
    pub output_path: String,
    /// Number of frames processed
    pub frame_count: u32,
    /// Input file size
    pub input_size: u64,
    /// Output file size
    pub output_size: u64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Decode backend
    pub decode_backend: String,
    /// Encode codec
    pub encode_codec: String,
    /// Transforms applied
    pub transforms: Vec<String>,
}

/// Progress callback type for long operations.
pub type ProgressCallback = Box<dyn Fn(ProgressInfo) + Send + Sync>;

/// Progress information for long operations.
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// Current step/frame
    pub current: u64,
    /// Total steps/frames (0 if unknown)
    pub total: u64,
    /// Progress percentage (0-100)
    pub percent: f32,
    /// Current phase name
    pub phase: String,
    /// Estimated time remaining in seconds
    pub eta_secs: Option<f64>,
    /// Processing speed (items/sec)
    pub speed: Option<f64>,
}

impl ProgressInfo {
    /// Create progress info.
    pub fn new(current: u64, total: u64, phase: impl Into<String>) -> Self {
        let percent = if total > 0 {
            (current as f32 / total as f32) * 100.0
        } else {
            0.0
        };

        Self {
            current,
            total,
            percent,
            phase: phase.into(),
            eta_secs: None,
            speed: None,
        }
    }

    /// Add ETA and speed estimates.
    pub fn with_speed(mut self, speed: f64, _elapsed_secs: f64) -> Self {
        self.speed = Some(speed);
        if self.total > 0 && speed > 0.0 {
            let remaining = self.total - self.current;
            self.eta_secs = Some(remaining as f64 / speed);
        }
        self
    }
}

/// Batch operation result for agents.
#[derive(Debug, Clone)]
pub struct BatchResult<T> {
    /// Total items processed
    pub total: usize,
    /// Successful items
    pub succeeded: usize,
    /// Failed items
    pub failed: usize,
    /// Individual results
    pub results: Vec<BatchItemResult<T>>,
    /// Total timing
    pub timing: OperationTiming,
}

/// Individual item result in a batch.
#[derive(Debug, Clone)]
pub struct BatchItemResult<T> {
    /// Input identifier (path, index, etc.)
    pub input: String,
    /// Whether this item succeeded
    pub success: bool,
    /// Result data (if successful)
    pub data: Option<T>,
    /// Error (if failed)
    pub error: Option<AgentError>,
}

impl<T> BatchResult<T> {
    /// Create a new batch result.
    pub fn new(timing: OperationTiming) -> Self {
        Self {
            total: 0,
            succeeded: 0,
            failed: 0,
            results: Vec::new(),
            timing,
        }
    }

    /// Add a successful result.
    pub fn add_success(&mut self, input: impl Into<String>, data: T) {
        self.total += 1;
        self.succeeded += 1;
        self.results.push(BatchItemResult {
            input: input.into(),
            success: true,
            data: Some(data),
            error: None,
        });
    }

    /// Add a failed result.
    pub fn add_failure(&mut self, input: impl Into<String>, error: AgentError) {
        self.total += 1;
        self.failed += 1;
        self.results.push(BatchItemResult {
            input: input.into(),
            success: false,
            data: None,
            error: Some(error),
        });
    }
}

/// Capabilities report for agents to understand available features.
#[derive(Debug, Clone)]
pub struct Capabilities {
    /// Library version
    pub version: String,
    /// Available video codecs for encoding
    pub encode_codecs: Vec<String>,
    /// Available video codecs for decoding
    pub decode_codecs: Vec<String>,
    /// Available container formats for reading
    pub demux_formats: Vec<String>,
    /// Available container formats for writing
    pub mux_formats: Vec<String>,
    /// Available audio codecs
    pub audio_codecs: Vec<String>,
    /// Available image transforms
    pub transforms: Vec<String>,
    /// GPU decoder available (NVDEC)
    pub gpu_decode: bool,
    /// GPU encoder available (NVENC)
    pub gpu_encode: bool,
    /// Software AV1 decoder available (dav1d)
    pub sw_av1_decode: bool,
    /// Background removal available
    pub background_removal: bool,
}

impl Capabilities {
    /// Query current capabilities.
    pub fn query() -> Self {
        // Mutated conditionally based on compile-time features.
        #[allow(unused_mut)]
        let mut caps = Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            encode_codecs: Vec::new(),
            decode_codecs: Vec::new(),
            demux_formats: Vec::new(),
            mux_formats: Vec::new(),
            audio_codecs: Vec::new(),
            transforms: vec![
                "resize".to_string(),
                "rotate".to_string(),
                "flip".to_string(),
                "crop".to_string(),
                "brightness".to_string(),
                "contrast".to_string(),
                "saturation".to_string(),
                "blur".to_string(),
                "sharpen".to_string(),
            ],
            gpu_decode: false,
            gpu_encode: false,
            sw_av1_decode: false,
            background_removal: false,
        };

        // Video encoding - AV1
        #[cfg(feature = "video-encode")]
        {
            caps.encode_codecs.push("av1".to_string());
            caps.mux_formats.push("ivf".to_string());
        }

        // Video encoding - H.264
        #[cfg(feature = "video-encode-h264")]
        {
            caps.encode_codecs.push("h264".to_string());
            caps.mux_formats.push("mp4".to_string());
            caps.mux_formats.push("h264".to_string());
        }

        // Video decoding
        #[cfg(feature = "video-decode")]
        {
            caps.decode_codecs.extend(vec![
                "av1".to_string(),
                "h264".to_string(),
                "h265".to_string(),
                "vp8".to_string(),
                "vp9".to_string(),
            ]);

            // Check NVDEC
            #[cfg(feature = "video-decode")]
            {
                use crate::video::decode::NvDecoder;
                caps.gpu_decode = NvDecoder::is_available();
            }
        }

        // Software AV1 decoder
        #[cfg(feature = "video-decode-sw")]
        {
            caps.sw_av1_decode = true;
        }

        // Container demuxing
        #[cfg(feature = "video")]
        {
            caps.demux_formats.extend(vec![
                "mp4".to_string(),
                "mkv".to_string(),
                "webm".to_string(),
                "ivf".to_string(),
            ]);
        }

        // Audio
        #[cfg(feature = "audio")]
        {
            caps.audio_codecs.extend(vec![
                "mp3".to_string(),
                "aac".to_string(),
                "flac".to_string(),
                "vorbis".to_string(),
                "opus".to_string(),
                "wav".to_string(),
            ]);
        }

        // Background removal
        #[cfg(feature = "background-removal")]
        {
            caps.background_removal = true;
        }

        caps
    }
}

/// Convert to JSON-like format (for display/logging).
pub trait ToAgentJson {
    /// Convert to a JSON string representation.
    fn to_agent_json(&self) -> String;
}

impl ToAgentJson for Capabilities {
    fn to_agent_json(&self) -> String {
        format!(
            r#"{{
  "version": "{}",
  "encode_codecs": {:?},
  "decode_codecs": {:?},
  "demux_formats": {:?},
  "mux_formats": {:?},
  "audio_codecs": {:?},
  "transforms": {:?},
  "gpu_decode": {},
  "gpu_encode": {},
  "sw_av1_decode": {},
  "background_removal": {}
}}"#,
            self.version,
            self.encode_codecs,
            self.decode_codecs,
            self.demux_formats,
            self.mux_formats,
            self.audio_codecs,
            self.transforms,
            self.gpu_decode,
            self.gpu_encode,
            self.sw_av1_decode,
            self.background_removal,
        )
    }
}

impl ToAgentJson for VideoAnalysis {
    fn to_agent_json(&self) -> String {
        format!(
            r#"{{
  "path": "{}",
  "container": "{}",
  "video_codec": {},
  "audio_codec": {},
  "width": {},
  "height": {},
  "duration_secs": {:.3},
  "frame_rate": {:.2},
  "frame_count": {},
  "bitrate_kbps": {},
  "file_size": {},
  "nvdec_compatible": {},
  "dav1d_compatible": {},
  "audio_sample_rate": {},
  "audio_channels": {}
}}"#,
            self.path,
            self.container,
            self.video_codec.as_ref().map_or("null".to_string(), |s| format!("\"{}\"", s)),
            self.audio_codec.as_ref().map_or("null".to_string(), |s| format!("\"{}\"", s)),
            self.width,
            self.height,
            self.duration_secs,
            self.frame_rate,
            self.frame_count,
            self.bitrate_kbps,
            self.file_size,
            self.nvdec_compatible,
            self.dav1d_compatible,
            self.audio_sample_rate.map_or("null".to_string(), |v| v.to_string()),
            self.audio_channels.map_or("null".to_string(), |v| v.to_string()),
        )
    }
}

impl ToAgentJson for EncodeResult {
    fn to_agent_json(&self) -> String {
        format!(
            r#"{{
  "output_path": "{}",
  "frame_count": {},
  "file_size": {},
  "bitrate_kbps": {},
  "encode_fps": {:.2},
  "codec": "{}",
  "quality": {},
  "speed_preset": "{}"
}}"#,
            self.output_path,
            self.frame_count,
            self.file_size,
            self.bitrate_kbps,
            self.encode_fps,
            self.codec,
            self.quality,
            self.speed_preset,
        )
    }
}

impl<T: ToAgentJson> ToAgentJson for AgentResult<T> {
    fn to_agent_json(&self) -> String {
        let data_json = self.data.as_ref().map_or("null".to_string(), |d| d.to_agent_json());
        let error_json = self.error.as_ref().map_or("null".to_string(), |e| {
            format!(
                r#"{{"code": "{}", "message": "{}", "recoverable": {}}}"#,
                e.code, e.message, e.recoverable
            )
        });

        format!(
            r#"{{
  "success": {},
  "data": {},
  "error": {},
  "timing": {{
    "total_ms": {},
    "started_at": {},
    "completed_at": {}
  }}
}}"#,
            self.success,
            data_json,
            error_json,
            self.timing.total_ms,
            self.timing.started_at,
            self.timing.completed_at,
        )
    }
}

// ============================================================================
// Unified Transcoding API for AI Agents
// ============================================================================

/// Unified transcoding configuration for AI agents.
///
/// This structure provides a complete specification for video/image operations
/// with sensible defaults. AI agents can construct this programmatically and
/// pass it to the CLI via JSON.
///
/// # Example JSON
///
/// ```json
/// {
///   "input": "input.ivf",
///   "output": "output.ivf",
///   "operation": "transcode",
///   "video": {
///     "codec": "av1",
///     "width": 1920,
///     "height": 1080,
///     "fps": 30.0,
///     "quality": 80,
///     "speed": 6
///   },
///   "transforms": ["rotate:90", "flip:h"],
///   "trim": { "start": 0.0, "end": 10.0 }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct TranscodeConfig {
    /// Input file path
    pub input: String,
    /// Output file path
    pub output: String,
    /// Operation type: "transcode", "trim", "concat", "extract-frames", "encode"
    pub operation: String,
    /// Video encoding settings (optional)
    pub video: Option<VideoSettings>,
    /// Audio settings (optional)
    pub audio: Option<AudioSettings>,
    /// List of transforms to apply (e.g., "rotate:90", "flip:h", "resize:1920x1080")
    pub transforms: Vec<String>,
    /// Trim settings (optional)
    pub trim: Option<TrimSettings>,
    /// Frame extraction settings (optional)
    pub frame_extraction: Option<FrameExtractionSettings>,
    /// Whether to suppress output (quiet mode)
    pub quiet: bool,
    /// Output format override (e.g., "json" for structured output)
    pub output_format: Option<String>,
}

/// Video encoding settings
#[derive(Debug, Clone)]
pub struct VideoSettings {
    /// Codec to use (e.g., "av1", "h264")
    pub codec: String,
    /// Output width (0 = preserve source)
    pub width: u32,
    /// Output height (0 = preserve source)
    pub height: u32,
    /// Frame rate (0 = preserve source)
    pub fps: f64,
    /// Quality (0-255 for AV1, lower = better)
    pub quality: u8,
    /// Speed preset (0-10, higher = faster)
    pub speed: u8,
    /// Target bitrate in kbps (0 = use quality mode)
    pub bitrate: u32,
    /// Number of threads (0 = auto)
    pub threads: usize,
}

impl Default for VideoSettings {
    fn default() -> Self {
        Self {
            codec: "av1".to_string(),
            width: 0,
            height: 0,
            fps: 0.0,
            quality: 100,
            speed: 6,
            bitrate: 0,
            threads: 0,
        }
    }
}

/// Audio settings
#[derive(Debug, Clone)]
pub struct AudioSettings {
    /// Audio codec (e.g., "aac", "opus")
    pub codec: String,
    /// Sample rate (0 = preserve source)
    pub sample_rate: u32,
    /// Number of channels (0 = preserve source)
    pub channels: u8,
    /// Audio bitrate in kbps
    pub bitrate: u32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            codec: "aac".to_string(),
            sample_rate: 0,
            channels: 0,
            bitrate: 128,
        }
    }
}

/// Trim settings for cutting video
#[derive(Debug, Clone)]
pub struct TrimSettings {
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds (0 = to end)
    pub end: f64,
}

/// Frame extraction settings
#[derive(Debug, Clone)]
pub struct FrameExtractionSettings {
    /// Output directory for frames
    pub output_dir: String,
    /// Output format (png, jpg, webp)
    pub format: String,
    /// Extract every Nth frame (1 = all frames)
    pub every: u32,
    /// Maximum frames to extract (0 = all)
    pub max_frames: u32,
}

impl Default for TranscodeConfig {
    fn default() -> Self {
        Self {
            input: String::new(),
            output: String::new(),
            operation: "transcode".to_string(),
            video: Some(VideoSettings::default()),
            audio: None,
            transforms: Vec::new(),
            trim: None,
            frame_extraction: None,
            quiet: false,
            output_format: None,
        }
    }
}

impl TranscodeConfig {
    /// Create a new transcoding config with input and output paths.
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            ..Default::default()
        }
    }

    /// Set operation type.
    pub fn operation(mut self, op: impl Into<String>) -> Self {
        self.operation = op.into();
        self
    }

    /// Set video settings.
    pub fn with_video(mut self, settings: VideoSettings) -> Self {
        self.video = Some(settings);
        self
    }

    /// Set video quality (0-255, lower = better).
    pub fn quality(mut self, quality: u8) -> Self {
        if let Some(ref mut v) = self.video {
            v.quality = quality;
        }
        self
    }

    /// Set encoding speed preset (0-10).
    pub fn speed(mut self, speed: u8) -> Self {
        if let Some(ref mut v) = self.video {
            v.speed = speed;
        }
        self
    }

    /// Set output resolution.
    pub fn resize(mut self, width: u32, height: u32) -> Self {
        if let Some(ref mut v) = self.video {
            v.width = width;
            v.height = height;
        }
        self
    }

    /// Add a transform.
    pub fn transform(mut self, t: impl Into<String>) -> Self {
        self.transforms.push(t.into());
        self
    }

    /// Set trim range.
    pub fn trim(mut self, start: f64, end: f64) -> Self {
        self.trim = Some(TrimSettings { start, end });
        self
    }

    /// Enable quiet mode.
    pub fn quiet(mut self) -> Self {
        self.quiet = true;
        self
    }

    /// Set JSON output format.
    pub fn json_output(mut self) -> Self {
        self.output_format = Some("json".to_string());
        self
    }
}

impl ToAgentJson for TranscodeConfig {
    fn to_agent_json(&self) -> String {
        let video_json = self.video.as_ref().map_or("null".to_string(), |v| {
            format!(
                r#"{{
      "codec": "{}",
      "width": {},
      "height": {},
      "fps": {},
      "quality": {},
      "speed": {},
      "bitrate": {},
      "threads": {}
    }}"#,
                v.codec, v.width, v.height, v.fps, v.quality, v.speed, v.bitrate, v.threads
            )
        });

        let trim_json = self.trim.as_ref().map_or("null".to_string(), |t| {
            format!(r#"{{"start": {}, "end": {}}}"#, t.start, t.end)
        });

        format!(
            r#"{{
  "input": "{}",
  "output": "{}",
  "operation": "{}",
  "video": {},
  "transforms": {:?},
  "trim": {},
  "quiet": {}
}}"#,
            self.input,
            self.output,
            self.operation,
            video_json,
            self.transforms,
            trim_json,
            self.quiet,
        )
    }
}

impl ToAgentJson for TranscodeResult {
    fn to_agent_json(&self) -> String {
        format!(
            r#"{{
  "input_path": "{}",
  "output_path": "{}",
  "frame_count": {},
  "input_size": {},
  "output_size": {},
  "compression_ratio": {:.2},
  "decode_backend": "{}",
  "encode_codec": "{}",
  "transforms": {:?}
}}"#,
            self.input_path,
            self.output_path,
            self.frame_count,
            self.input_size,
            self.output_size,
            self.compression_ratio,
            self.decode_backend,
            self.encode_codec,
            self.transforms,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capabilities_query() {
        let caps = Capabilities::query();
        assert!(!caps.version.is_empty());
    }

    #[test]
    fn test_agent_result_success() {
        let timing = OperationTiming::from_start(Instant::now());
        let result: AgentResult<String> = AgentResult::ok("test".to_string(), timing);
        assert!(result.success);
        assert!(result.data.is_some());
        assert!(result.error.is_none());
    }

    #[test]
    fn test_agent_error() {
        let err = AgentError::file_not_found("/path/to/file.mp4");
        assert_eq!(err.code, "FILE_NOT_FOUND");
        assert!(err.recoverable);
    }

    #[test]
    fn test_progress_info() {
        let progress = ProgressInfo::new(50, 100, "encoding");
        assert_eq!(progress.percent, 50.0);
    }
}
