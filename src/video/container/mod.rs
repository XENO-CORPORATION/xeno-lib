//! Container format demuxing for MP4, MKV, WebM, AVI, and IVF.
//!
//! This module provides demuxers that extract raw video/audio packets from
//! container formats. The packets can then be fed to hardware decoders (NVDEC)
//! or software decoders for processing.
//!
//! # Supported Containers
//!
//! - **MP4** (.mp4, .m4v, .mov) - via the `mp4` crate
//! - **MKV/WebM** (.mkv, .webm) - metadata + indexed packet demuxing
//! - **AVI** (.avi) - indexed packet demuxing (built-in)
//! - **IVF** (.ivf) - raw VP8/VP9/AV1 bitstream container (built-in)
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::video::container::{Demuxer, open_container};
//!
//! let mut demuxer = open_container("video.mp4")?;
//! println!("Video: {:?}, {}x{}",
//!     demuxer.video_info().map(|v| v.codec),
//!     demuxer.video_info().map(|v| v.width).unwrap_or(0),
//!     demuxer.video_info().map(|v| v.height).unwrap_or(0));
//!
//! while let Some(packet) = demuxer.next_video_packet()? {
//!     // Feed packet to decoder
//!     decoder.decode_packet(&packet.data, packet.pts)?;
//! }
//! ```

#[cfg(feature = "mp4")]
mod mp4_demux;

#[cfg(feature = "matroska")]
mod mkv_demux;

mod avi_demux;
mod ivf_demux;

use crate::video::{AudioCodec, ContainerFormat, VideoCodec, VideoError};
use std::path::Path;

/// Supported container file extensions.
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "mp4", "m4v", "mov", // MP4 family
    "mkv", "webm", // Matroska family
    "ivf",  // IVF (raw bitstream)
    "avi",  // AVI (limited support)
];

/// Check if a file extension indicates a supported video format.
pub fn is_supported_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Detect container format from file extension.
pub fn detect_format_from_extension(path: &Path) -> ContainerFormat {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .as_deref()
    {
        Some("mp4") | Some("m4v") => ContainerFormat::Mp4,
        Some("mkv") => ContainerFormat::Mkv,
        Some("webm") => ContainerFormat::WebM,
        Some("mov") => ContainerFormat::Mov,
        Some("avi") => ContainerFormat::Avi,
        Some("ivf") => ContainerFormat::Unknown, // IVF is raw bitstream, not a "container" per se
        _ => ContainerFormat::Unknown,
    }
}

/// Container format type (for demuxer selection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerType {
    /// MP4/M4V/MOV container
    Mp4,
    /// Matroska container
    Mkv,
    /// WebM container (Matroska subset)
    WebM,
    /// IVF raw bitstream container
    Ivf,
    /// AVI container
    Avi,
}

impl From<ContainerFormat> for Option<ContainerType> {
    fn from(format: ContainerFormat) -> Self {
        match format {
            ContainerFormat::Mp4 | ContainerFormat::Mov => Some(ContainerType::Mp4),
            ContainerFormat::Mkv => Some(ContainerType::Mkv),
            ContainerFormat::WebM => Some(ContainerType::WebM),
            ContainerFormat::Avi => Some(ContainerType::Avi),
            ContainerFormat::Unknown => None,
        }
    }
}

/// A raw packet from the container.
#[derive(Debug, Clone)]
pub struct Packet {
    /// Raw packet data (codec-specific bitstream)
    pub data: Vec<u8>,
    /// Presentation timestamp (in timebase units)
    pub pts: i64,
    /// Decode timestamp (may differ from PTS for B-frames)
    pub dts: i64,
    /// Packet duration (in timebase units)
    pub duration: i64,
    /// True if this is a keyframe
    pub is_keyframe: bool,
    /// Stream index
    pub stream_index: u32,
}

/// Video stream information.
#[derive(Debug, Clone)]
pub struct VideoStreamInfo {
    /// Video codec
    pub codec: VideoCodec,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Frame rate (if known)
    pub frame_rate: Option<f64>,
    /// Duration in seconds (if known)
    pub duration: Option<f64>,
    /// Total frame count (if known)
    pub frame_count: Option<u64>,
    /// Timebase numerator
    pub timebase_num: u32,
    /// Timebase denominator
    pub timebase_den: u32,
    /// Codec-specific configuration data (SPS/PPS for H.264, etc.)
    pub extra_data: Vec<u8>,
}

/// Audio stream information.
#[derive(Debug, Clone)]
pub struct AudioStreamInfo {
    /// Audio codec
    pub codec: AudioCodec,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Bits per sample (if applicable)
    pub bits_per_sample: Option<u16>,
    /// Duration in seconds (if known)
    pub duration: Option<f64>,
    /// Codec-specific configuration data
    pub extra_data: Vec<u8>,
}

/// Container demuxer trait.
pub trait Demuxer: Send {
    /// Get the container type.
    fn container_type(&self) -> ContainerType;

    /// Get video stream info (if present).
    fn video_info(&self) -> Option<&VideoStreamInfo>;

    /// Get audio stream info (if present).
    fn audio_info(&self) -> Option<&AudioStreamInfo>;

    /// Get the next video packet.
    fn next_video_packet(&mut self) -> Result<Option<Packet>, VideoError>;

    /// Get the next audio packet.
    fn next_audio_packet(&mut self) -> Result<Option<Packet>, VideoError>;

    /// Seek to a timestamp (in seconds).
    fn seek(&mut self, timestamp: f64) -> Result<(), VideoError>;

    /// Reset to the beginning.
    fn reset(&mut self) -> Result<(), VideoError>;
}

/// Detect container type from file path.
pub fn detect_container_type(path: &Path) -> Option<ContainerType> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    match ext.as_str() {
        "mp4" | "m4v" | "mov" => Some(ContainerType::Mp4),
        "mkv" => Some(ContainerType::Mkv),
        "webm" => Some(ContainerType::WebM),
        "ivf" => Some(ContainerType::Ivf),
        "avi" => Some(ContainerType::Avi),
        _ => None,
    }
}

/// Open a container file and return the appropriate demuxer.
pub fn open_container<P: AsRef<Path>>(path: P) -> Result<Box<dyn Demuxer>, VideoError> {
    let path = path.as_ref();

    let container_type = detect_container_type(path).ok_or_else(|| VideoError::Container {
        message: format!("Unsupported container format: {:?}", path.extension()),
    })?;

    match container_type {
        ContainerType::Ivf => {
            let demuxer = ivf_demux::IvfDemuxer::open(path)?;
            Ok(Box::new(demuxer))
        }
        #[cfg(feature = "mp4")]
        ContainerType::Mp4 => {
            let demuxer = mp4_demux::Mp4Demuxer::open(path)?;
            Ok(Box::new(demuxer))
        }
        #[cfg(feature = "matroska")]
        ContainerType::Mkv | ContainerType::WebM => {
            let demuxer = mkv_demux::MkvDemuxer::open(path)?;
            Ok(Box::new(demuxer))
        }
        #[cfg(not(feature = "mp4"))]
        ContainerType::Mp4 => Err(VideoError::Container {
            message: "MP4 support not enabled. Enable 'containers' or 'mp4' feature.".to_string(),
        }),
        #[cfg(not(feature = "matroska"))]
        ContainerType::Mkv | ContainerType::WebM => Err(VideoError::Container {
            message: "MKV/WebM support not enabled. Enable 'containers' or 'matroska' feature."
                .to_string(),
        }),
        ContainerType::Avi => {
            let demuxer = avi_demux::AviDemuxer::open(path)?;
            Ok(Box::new(demuxer))
        }
    }
}

// Re-exports
pub use avi_demux::AviDemuxer;
pub use ivf_demux::IvfDemuxer;

#[cfg(feature = "mp4")]
pub use mp4_demux::Mp4Demuxer;

#[cfg(feature = "matroska")]
pub use mkv_demux::MkvDemuxer;
