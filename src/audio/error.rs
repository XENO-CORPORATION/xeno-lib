//! Audio processing error types.

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during audio processing.
#[derive(Debug, Error)]
pub enum AudioError {
    /// Failed to open audio file.
    #[error("failed to open audio file: {path}")]
    OpenFailed {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Unsupported audio format.
    #[error("unsupported audio format: {format}")]
    UnsupportedFormat { format: String },

    /// Unsupported audio codec.
    #[error("unsupported audio codec: {codec}")]
    UnsupportedCodec { codec: String },

    /// No audio track found.
    #[error("no audio track found in file")]
    NoAudioTrack,

    /// Failed to decode audio.
    #[error("failed to decode audio: {message}")]
    DecodeFailed { message: String },

    /// Failed to seek to position.
    #[error("failed to seek to position: {message}")]
    SeekFailed { message: String },

    /// Symphonia error.
    #[error("audio decoder error: {0}")]
    Symphonia(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Resampling error.
    #[error("resampling error: {message}")]
    Resample { message: String },

    /// Invalid sample rate.
    #[error("invalid sample rate: {rate}")]
    InvalidSampleRate { rate: u32 },

    /// Invalid channel count.
    #[error("invalid channel count: {count}")]
    InvalidChannels { count: u32 },

    /// Failed to encode audio.
    #[error("failed to encode audio: {0}")]
    EncodeFailed(String),
}

/// Result type for audio operations.
pub type AudioResult<T> = Result<T, AudioError>;
