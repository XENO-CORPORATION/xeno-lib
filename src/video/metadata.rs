//! Video metadata types.

use std::fmt;
use std::time::Duration;

/// Video container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFormat {
    /// MP4 / ISO Base Media File Format (.mp4, .m4v)
    Mp4,
    /// Matroska (.mkv)
    Mkv,
    /// WebM (.webm) - subset of Matroska
    WebM,
    /// QuickTime Movie (.mov)
    Mov,
    /// Audio Video Interleave (.avi)
    Avi,
    /// Unknown container format
    Unknown,
}

impl ContainerFormat {
    /// Get the typical file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            ContainerFormat::Mp4 => "mp4",
            ContainerFormat::Mkv => "mkv",
            ContainerFormat::WebM => "webm",
            ContainerFormat::Mov => "mov",
            ContainerFormat::Avi => "avi",
            ContainerFormat::Unknown => "",
        }
    }

    /// Get the MIME type for this format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            ContainerFormat::Mp4 => "video/mp4",
            ContainerFormat::Mkv => "video/x-matroska",
            ContainerFormat::WebM => "video/webm",
            ContainerFormat::Mov => "video/quicktime",
            ContainerFormat::Avi => "video/x-msvideo",
            ContainerFormat::Unknown => "application/octet-stream",
        }
    }
}

impl fmt::Display for ContainerFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContainerFormat::Mp4 => write!(f, "MP4"),
            ContainerFormat::Mkv => write!(f, "MKV"),
            ContainerFormat::WebM => write!(f, "WebM"),
            ContainerFormat::Mov => write!(f, "MOV"),
            ContainerFormat::Avi => write!(f, "AVI"),
            ContainerFormat::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Video codec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoCodec {
    /// AV1 (AOMedia Video 1)
    Av1,
    /// VP9
    Vp9,
    /// VP8
    Vp8,
    /// H.264 / AVC
    H264,
    /// H.265 / HEVC
    H265,
    /// MPEG-4 Part 2
    Mpeg4,
    /// Unknown codec
    Unknown(String),
}

impl VideoCodec {
    /// Get the codec name.
    pub fn name(&self) -> &str {
        match self {
            VideoCodec::Av1 => "AV1",
            VideoCodec::Vp9 => "VP9",
            VideoCodec::Vp8 => "VP8",
            VideoCodec::H264 => "H.264",
            VideoCodec::H265 => "H.265",
            VideoCodec::Mpeg4 => "MPEG-4",
            VideoCodec::Unknown(s) => s.as_str(),
        }
    }

    /// Check if this codec is royalty-free.
    pub fn is_royalty_free(&self) -> bool {
        matches!(self, VideoCodec::Av1 | VideoCodec::Vp9 | VideoCodec::Vp8)
    }
}

impl fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Audio codec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioCodec {
    /// AAC (Advanced Audio Coding)
    Aac,
    /// Opus
    Opus,
    /// Vorbis
    Vorbis,
    /// MP3 (MPEG Layer 3)
    Mp3,
    /// FLAC
    Flac,
    /// ALAC (Apple Lossless Audio Codec)
    Alac,
    /// PCM (uncompressed)
    Pcm,
    /// No audio
    None,
    /// Unknown codec
    Unknown(String),
}

impl fmt::Display for AudioCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioCodec::Aac => write!(f, "AAC"),
            AudioCodec::Opus => write!(f, "Opus"),
            AudioCodec::Vorbis => write!(f, "Vorbis"),
            AudioCodec::Mp3 => write!(f, "MP3"),
            AudioCodec::Flac => write!(f, "FLAC"),
            AudioCodec::Alac => write!(f, "ALAC"),
            AudioCodec::Pcm => write!(f, "PCM"),
            AudioCodec::None => write!(f, "None"),
            AudioCodec::Unknown(s) => write!(f, "{}", s),
        }
    }
}

/// Comprehensive video file metadata.
#[derive(Debug, Clone)]
pub struct VideoMetadata {
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Frame rate (frames per second).
    pub frame_rate: f64,
    /// Total duration in milliseconds.
    pub duration_ms: i64,
    /// Estimated total frame count.
    pub frame_count: u64,
    /// Video codec.
    pub video_codec: VideoCodec,
    /// Audio codec (if present).
    pub audio_codec: AudioCodec,
    /// Container format.
    pub container: ContainerFormat,
    /// Video bitrate in bits per second (if known).
    pub video_bitrate: Option<u32>,
    /// Audio bitrate in bits per second (if known).
    pub audio_bitrate: Option<u32>,
    /// Audio sample rate in Hz (if audio present).
    pub audio_sample_rate: Option<u32>,
    /// Number of audio channels (if audio present).
    pub audio_channels: Option<u8>,
}

impl VideoMetadata {
    /// Create metadata with minimal required fields.
    pub fn new(width: u32, height: u32, frame_rate: f64, duration_ms: i64) -> Self {
        let frame_count = if frame_rate > 0.0 {
            ((duration_ms as f64 / 1000.0) * frame_rate) as u64
        } else {
            0
        };

        Self {
            width,
            height,
            frame_rate,
            duration_ms,
            frame_count,
            video_codec: VideoCodec::Unknown(String::new()),
            audio_codec: AudioCodec::None,
            container: ContainerFormat::Unknown,
            video_bitrate: None,
            audio_bitrate: None,
            audio_sample_rate: None,
            audio_channels: None,
        }
    }

    /// Get duration as a `Duration` type.
    pub fn duration(&self) -> Duration {
        Duration::from_millis(self.duration_ms as u64)
    }

    /// Get duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.duration_ms as f64 / 1000.0
    }

    /// Check if video has audio track.
    pub fn has_audio(&self) -> bool {
        !matches!(self.audio_codec, AudioCodec::None)
    }

    /// Get aspect ratio.
    pub fn aspect_ratio(&self) -> f64 {
        if self.height > 0 {
            self.width as f64 / self.height as f64
        } else {
            0.0
        }
    }

    /// Get resolution as a string (e.g., "1920x1080").
    pub fn resolution(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }

    /// Get a human-readable duration string (e.g., "1:23:45").
    pub fn duration_string(&self) -> String {
        let total_secs = self.duration_ms / 1000;
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;

        if hours > 0 {
            format!("{}:{:02}:{:02}", hours, minutes, seconds)
        } else {
            format!("{}:{:02}", minutes, seconds)
        }
    }
}

impl fmt::Display for VideoMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{} @ {:.2} fps, {} ({}), {}",
            self.width,
            self.height,
            self.frame_rate,
            self.video_codec,
            self.container,
            self.duration_string()
        )
    }
}
