//! Video encoding module.
//!
//! Provides video encoding using:
//! - AV1 via rav1e (pure Rust + ASM) - modern, high efficiency
//! - H.264 via OpenH264 (Cisco's codec) - universal compatibility
//!
//! # Feature Flags
//!
//! - `video-encode` - Enables AV1 encoding via rav1e
//! - `video-encode-h264` - Enables H.264 encoding via OpenH264
//! - `video-encode-full` - Enables all encoders
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::video::encode::*;
//!
//! // AV1 encoding (best compression)
//! let config = Av1EncoderConfig::new(1920, 1080)
//!     .with_frame_rate(30.0)
//!     .with_speed(EncodingSpeed::Medium);
//! let frames = encode_to_ivf(images.iter(), "output.ivf", config)?;
//!
//! // H.264 encoding (universal playback)
//! let config = H264EncoderConfig::new(1920, 1080)
//!     .with_frame_rate(30.0)
//!     .with_bitrate(5000);
//! let frames = encode_h264_to_mp4(images.iter(), "output.mp4", config)?;
//! ```

#[cfg(feature = "video-encode")]
mod av1;

#[cfg(feature = "video-encode-h264")]
mod h264;

#[cfg(feature = "video-encode-nvenc")]
pub mod nvenc;

#[cfg(feature = "video-encode")]
pub use av1::{
    encode_to_ivf, encode_to_mp4, Av1Encoder, Av1EncoderConfig, EncodingSpeed,
};

#[cfg(feature = "video-encode-h264")]
pub use h264::{
    encode_h264_to_mp4, encode_to_h264, EncodedFrame, H264Encoder, H264EncoderConfig, H264Profile,
};

use crate::video::VideoError;
use image::DynamicImage;

/// Video encoder configuration trait.
pub trait VideoEncoderConfig {
    /// Get the output width.
    fn width(&self) -> u32;
    /// Get the output height.
    fn height(&self) -> u32;
    /// Get the frame rate.
    fn frame_rate(&self) -> f64;
}

/// Video encoder trait for encoding image sequences to video.
pub trait VideoEncoder {
    /// The configuration type.
    type Config: VideoEncoderConfig;
    /// The output packet type.
    type Packet;

    /// Create a new encoder with the given configuration.
    fn new(config: Self::Config) -> Result<Self, VideoError>
    where
        Self: Sized;

    /// Send a frame to the encoder.
    fn send_frame(&mut self, frame: &DynamicImage) -> Result<(), VideoError>;

    /// Signal end of input (flush).
    fn flush(&mut self);

    /// Receive an encoded packet.
    fn receive_packet(&mut self) -> Result<Option<Self::Packet>, VideoError>;

    /// Get encoder configuration.
    fn config(&self) -> &Self::Config;
}
