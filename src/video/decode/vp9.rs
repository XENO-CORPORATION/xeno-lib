//! VP9 software video decoder.
//!
//! # Status
//!
//! **Stub implementation** — no pure Rust VP9 decoder exists. The recommended
//! path is to integrate `vpx-rs` (libvpx C bindings) behind the
//! `video-decode-vp9` feature flag.
//!
//! Note: VP9 hardware decoding via NVDEC is already supported when the
//! `video-decode` feature is enabled and an NVIDIA GPU is available.
//! This module provides the **software fallback** path.
//!
//! # Architecture
//!
//! When libvpx bindings are linked, the `Vp9Decoder` will:
//! 1. Initialize a `vpx_codec_ctx` with `vpx_codec_vp9_dx()`
//! 2. Feed raw VP9 bitstream packets via `decode_packet()`
//! 3. Retrieve decoded YUV frames and convert to RGBA
//! 4. Return `DecodedFrame` instances compatible with the decode pipeline
//!
//! # Feature Flag
//!
//! - `video-decode-vp9` — enables this module
//!
//! # Future Integration
//!
//! To activate VP9 software decoding:
//! 1. Add `vpx-rs = { version = "0.4", optional = true }` to Cargo.toml
//! 2. Change `video-decode-vp9` feature to depend on `dep:vpx-rs`
//! 3. Implement the decoder using `vpx::Decoder`
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::video::decode::vp9::Vp9Decoder;
//!
//! let mut decoder = Vp9Decoder::new()?;
//! decoder.decode_packet(vp9_data, 0)?;
//! while let Some(frame) = decoder.next_frame()? {
//!     let image = frame.to_rgba_image()?;
//! }
//! ```

use std::collections::VecDeque;

use crate::video::VideoError;
use super::{DecodeCodec, DecodedFrame, DecoderCapabilities, VideoDecoder};

/// VP9 software decoder.
///
/// Currently a stub that documents the interface. The decoder lifecycle is
/// fully defined and ready to activate once libvpx is linked.
pub struct Vp9Decoder {
    /// Decoded frames waiting to be consumed.
    frames: VecDeque<DecodedFrame>,
}

impl Vp9Decoder {
    /// Create a new VP9 software decoder.
    ///
    /// # Returns
    /// A new `Vp9Decoder` instance.
    ///
    /// # Errors
    /// Returns an error if the VP9 decoder library is not available.
    pub fn new() -> Result<Self, VideoError> {
        Ok(Vp9Decoder {
            frames: VecDeque::new(),
        })
    }

    /// Check if VP9 software decoding is available.
    ///
    /// Returns `false` until libvpx bindings are integrated.
    pub fn is_available() -> bool {
        false
    }

    /// Get the decoder capabilities for VP9.
    pub fn capabilities() -> DecoderCapabilities {
        DecoderCapabilities {
            supported: false,
            max_width: 8192,
            max_height: 4352,
            max_bit_depth: 12,
            num_engines: 0,
        }
    }
}

impl VideoDecoder for Vp9Decoder {
    fn decode_file(&mut self, _path: &str) -> Result<(), VideoError> {
        Err(VideoError::Decoding {
            message: "VP9 software decoder not available. Use NVDEC for VP9 decoding, or add libvpx bindings.".to_string(),
        })
    }

    fn decode_packet(&mut self, _data: &[u8], _pts: i64) -> Result<(), VideoError> {
        Err(VideoError::Decoding {
            message: "VP9 software decoder not available. Use NVDEC for VP9 decoding, or add libvpx bindings.".to_string(),
        })
    }

    fn flush(&mut self) -> Result<(), VideoError> {
        self.frames.clear();
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<DecodedFrame>, VideoError> {
        Ok(self.frames.pop_front())
    }

    fn get_capabilities(&self, codec: DecodeCodec) -> Result<DecoderCapabilities, VideoError> {
        if codec != DecodeCodec::Vp9 {
            return Err(VideoError::UnsupportedCodec {
                codec: format!("{:?} (VP9 decoder only supports VP9)", codec),
            });
        }
        Ok(Vp9Decoder::capabilities())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vp9_not_available() {
        assert!(!Vp9Decoder::is_available());
    }

    #[test]
    fn test_vp9_capabilities() {
        let caps = Vp9Decoder::capabilities();
        assert!(!caps.supported);
        assert_eq!(caps.max_width, 8192);
        assert_eq!(caps.max_height, 4352);
    }

    #[test]
    fn test_vp9_decoder_create() {
        let decoder = Vp9Decoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_vp9_decode_packet_returns_stub_error() {
        let mut decoder = Vp9Decoder::new().expect("should create decoder");
        let result = decoder.decode_packet(&[0x00, 0x01, 0x02], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_vp9_decode_file_returns_stub_error() {
        let mut decoder = Vp9Decoder::new().expect("should create decoder");
        let result = decoder.decode_file("test.webm");
        assert!(result.is_err());
    }

    #[test]
    fn test_vp9_flush_succeeds() {
        let mut decoder = Vp9Decoder::new().expect("should create decoder");
        assert!(decoder.flush().is_ok());
    }

    #[test]
    fn test_vp9_next_frame_returns_none() {
        let mut decoder = Vp9Decoder::new().expect("should create decoder");
        let frame = decoder.next_frame().expect("should not error");
        assert!(frame.is_none());
    }

    #[test]
    fn test_vp9_wrong_codec_capabilities() {
        let decoder = Vp9Decoder::new().expect("should create decoder");
        let result = decoder.get_capabilities(DecodeCodec::H264);
        assert!(result.is_err());
    }
}
