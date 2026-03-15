//! Native video processing module for xeno-lib.
//!
//! This module provides pure Rust video types, container format detection,
//! AV1 video encoding, and GPU-accelerated video decoding.
//!
//! # Features
//!
//! - `video` - Container format detection and metadata types
//! - `video-encode` - AV1 encoding via rav1e
//! - `video-decode` - GPU decoding via NVIDIA NVDEC
//! - `video-full` - All video features
//!
//! # Current Capabilities
//!
//! - Container format detection from file extension
//! - Video metadata types (VideoFrame, VideoMetadata)
//! - AV1 video encoding from image sequences
//! - GPU-accelerated decoding (AV1, H.264, H.265, VP9)
//! - Error types for video operations
//!
//! # Example: Encode images to AV1 video
//!
//! ```ignore
//! use xeno_lib::video::encode::{encode_to_ivf, Av1EncoderConfig, EncodingSpeed};
//!
//! let images: Vec<image::DynamicImage> = load_your_images()?;
//! let config = Av1EncoderConfig::new(1920, 1080)
//!     .with_frame_rate(30.0)
//!     .with_speed(EncodingSpeed::Fast);
//!
//! let frames = encode_to_ivf(images.iter(), "output.ivf", config)?;
//! println!("Encoded {} frames", frames);
//! ```
//!
//! # Example: Decode video with NVDEC
//!
//! ```ignore
//! use xeno_lib::video::decode::{NvDecoder, DecoderConfig, DecodeCodec};
//!
//! let config = DecoderConfig::new(DecodeCodec::Av1);
//! let mut decoder = NvDecoder::new(config)?;
//!
//! let frames = decoder.decode_ivf_file("input.ivf")?;
//! for frame in frames {
//!     let image = frame.to_rgba_image()?;
//!     image.save(format!("frame_{}.png", frame.decode_index))?;
//! }
//! ```

mod error;
mod frame;
mod metadata;

#[cfg(feature = "video")]
pub mod container;

#[cfg(any(feature = "video-encode", feature = "video-encode-h264", feature = "video-encode-nvenc"))]
pub mod encode;

#[cfg(any(feature = "video-decode", feature = "video-decode-sw", feature = "video-decode-hevc", feature = "video-decode-vp9"))]
pub mod decode;

#[cfg(feature = "av-mux")]
pub mod mux;

// Video editing operations (Phase 3 - trim, cut, concat, speed change)
#[cfg(feature = "video")]
pub mod edit;

pub use error::{VideoError, VideoResult};
pub use frame::VideoFrame;
pub use metadata::{AudioCodec, ContainerFormat, VideoCodec, VideoMetadata};

#[cfg(feature = "video")]
pub use container::{
    detect_format_from_extension, is_supported_extension, SUPPORTED_EXTENSIONS,
    // Container demuxing types
    open_container, ContainerType, Demuxer, Packet, VideoStreamInfo, AudioStreamInfo,
    IvfDemuxer,
};

#[cfg(feature = "video-encode")]
pub use encode::{encode_to_ivf, encode_to_mp4, Av1Encoder, Av1EncoderConfig, EncodingSpeed};

#[cfg(feature = "video-encode-h264")]
pub use encode::{
    encode_h264_to_mp4, encode_to_h264, EncodedFrame, H264Encoder, H264EncoderConfig, H264Profile,
};

#[cfg(feature = "video-decode")]
pub use decode::{
    DecodeCodec, DecodedFrame, DecoderCapabilities, DecoderConfig, NvDecoder, OutputFormat,
    VideoDecoder,
};

#[cfg(all(feature = "video-decode", feature = "video-decode-sw"))]
pub use decode::OpenH264Decoder;

#[cfg(feature = "av-mux")]
pub use mux::{
    annex_b_to_avcc, extract_sps_pps, AudioCodecType, AudioConfig, AvMuxConfig, AvMuxer,
    VideoConfig,
};
