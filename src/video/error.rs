//! Video processing error types.

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during video processing.
#[derive(Debug, Error)]
pub enum VideoError {
    /// Failed to open video file.
    #[error("failed to open video file: {path}")]
    OpenFailed {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Unsupported container format.
    #[error("unsupported container format: {format}")]
    UnsupportedContainer { format: String },

    /// Unsupported video codec.
    #[error("unsupported video codec: {codec}")]
    UnsupportedCodec { codec: String },

    /// No video track found in container.
    #[error("no video track found in file")]
    NoVideoTrack,

    /// Failed to decode video frame.
    #[error("failed to decode frame {frame_number}: {message}")]
    DecodeFailed { frame_number: u64, message: String },

    /// Failed to seek to position.
    #[error("failed to seek to frame {frame_number}")]
    SeekFailed { frame_number: u64 },

    /// Demuxer error.
    #[error("demuxer error: {0}")]
    DemuxError(String),

    /// Encoder error.
    #[error("encoder error: {0}")]
    EncodeError(String),

    /// Encoding error with message.
    #[error("encoding failed: {message}")]
    Encoding { message: String },

    /// Muxer error.
    #[error("muxer error: {0}")]
    MuxError(String),

    /// End of stream reached.
    #[error("end of video stream")]
    EndOfStream,

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// IO error with message.
    #[error("IO error: {message}")]
    Io { message: String },

    /// Configuration error.
    #[error("invalid configuration: {message}")]
    Config { message: String },

    /// Invalid user input or arguments.
    #[error("invalid input: {message}")]
    InvalidInput { message: String },

    /// Decoding error.
    #[error("decoding failed: {message}")]
    Decoding { message: String },

    /// GPU/CUDA error.
    #[error("GPU error: {message}")]
    Gpu { message: String },

    /// Container parsing/demuxing error.
    #[error("container error: {message}")]
    Container { message: String },
}

/// Result type for video operations.
pub type VideoResult<T> = Result<T, VideoError>;
