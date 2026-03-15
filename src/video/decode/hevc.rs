//! HEVC/H.265 software decoder stub.
//!
//! # Current Status: Stub Module
//!
//! As of 2026-03, there is **no mature pure-Rust H.265/HEVC full-frame decoder**.
//! The landscape:
//!
//! - **scuffle-h265**: Pure Rust, but only parses SPS/PPS NAL headers — no pixel decoding.
//! - **hevc_parser**: Pure Rust HEVC bitstream parser — NAL units only, no reconstruction.
//! - **cros-codecs**: ChromeOS project with H.265 parser, but targets VAAPI hardware backends on Linux only.
//! - **libde265**: Standalone C HEVC decoder (not FFmpeg). Has Rust bindings via `libde265-sys2`
//!   with optional static compilation from embedded sources. This is the most viable path
//!   for a future full implementation.
//!
//! ## Recommended Implementation Path
//!
//! 1. **Short term**: Use NVDEC for GPU-accelerated H.265 decode (already supported in `nvdec.rs`).
//! 2. **Medium term**: Integrate `libde265` via `libde265-sys2` crate with `embedded-libde265`
//!    feature for static compilation from source (similar to how we use OpenH264).
//!    - libde265 is LGPL-3.0 licensed — evaluate license compatibility.
//!    - Alternative: wrap the BSD-licensed portions or negotiate a commercial license.
//! 3. **Long term**: Monitor pure-Rust HEVC decoder efforts. If `cros-codecs` adds a
//!    software backend, or a new crate emerges, migrate to pure Rust.
//!
//! ## What This Module Provides Now
//!
//! - `HevcDecoder` struct with the `VideoDecoder` trait implemented as stubs
//! - `decode_hevc_frame()` convenience function that returns a clear error
//! - `is_available()` returns false (no decoder linked yet)
//! - Feature flag: `video-decode-hevc`
//!
//! When a real decoder is integrated, this module will provide full H.265
//! decode with the same API as `OpenH264Decoder`.

use std::collections::VecDeque;

use crate::video::VideoError;

use super::{DecodeCodec, DecodedFrame, DecoderCapabilities, VideoDecoder};

/// Decode a single HEVC/H.265 frame from raw NAL unit data to RGBA pixels.
///
/// # Current Status
///
/// This function is a **stub** — it always returns an error explaining that
/// no H.265 software decoder is currently linked. Use NVDEC for hardware-
/// accelerated H.265 decoding, or wait for libde265 integration.
///
/// # Arguments
///
/// * `data` - Raw HEVC NAL unit data (Annex B or HVCC format)
///
/// # Returns
///
/// An RGBA frame on success (once a decoder is integrated), or an error.
///
/// # Example
///
/// ```ignore
/// use xeno_lib::video::decode::hevc::decode_hevc_frame;
///
/// let hevc_data = std::fs::read("frame.h265")?;
/// let frame = decode_hevc_frame(&hevc_data)?;
/// println!("Decoded {}x{} frame", frame.width, frame.height);
/// ```
pub fn decode_hevc_frame(data: &[u8]) -> Result<DecodedFrame, VideoError> {
    if data.is_empty() {
        return Err(VideoError::InvalidInput {
            message: "HEVC data is empty".to_string(),
        });
    }

    Err(VideoError::UnsupportedCodec {
        codec: "H.265/HEVC software decode is not yet available. \
                Use NVDEC for hardware-accelerated H.265 decoding (feature: video-decode), \
                or await libde265 integration in a future release."
            .to_string(),
    })
}

/// Software HEVC/H.265 decoder.
///
/// # Current Status
///
/// This is a **stub implementation**. The decoder can be constructed but will
/// return errors on all decode operations. It exists to:
///
/// 1. Reserve the API surface for future integration
/// 2. Allow `best_decoder_for(DecodeCodec::H265)` to fall through gracefully
/// 3. Provide clear error messages directing users to NVDEC
///
/// # Future Integration
///
/// When libde265 (or a pure-Rust alternative) is integrated, this struct will
/// hold the decoder state and produce decoded frames identically to `OpenH264Decoder`.
pub struct HevcDecoder {
    frames: VecDeque<DecodedFrame>,
    #[allow(dead_code)] // Reserved for future use when a real decoder is integrated
    frame_count: u64,
}

impl HevcDecoder {
    /// Create a new HEVC decoder.
    ///
    /// Currently succeeds but all decode operations will return errors
    /// until a real decoder backend is integrated.
    pub fn new() -> Result<Self, VideoError> {
        Ok(Self {
            frames: VecDeque::new(),
            frame_count: 0,
        })
    }

    /// Check if HEVC software decoding is available.
    ///
    /// Returns `false` until a real decoder (libde265 or pure Rust) is integrated.
    pub fn is_available() -> bool {
        false
    }

    /// Get capabilities for software HEVC decoding.
    ///
    /// Returns unsupported capabilities until a real decoder is integrated.
    pub fn capabilities() -> DecoderCapabilities {
        DecoderCapabilities {
            supported: false,
            max_width: 0,
            max_height: 0,
            max_bit_depth: 0,
            num_engines: 0,
        }
    }
}

impl VideoDecoder for HevcDecoder {
    fn decode_file(&mut self, path: &str) -> Result<(), VideoError> {
        let _ = path;
        Err(VideoError::UnsupportedCodec {
            codec: "H.265/HEVC software decode not yet available. Use NVDEC (feature: video-decode)."
                .to_string(),
        })
    }

    fn decode_packet(&mut self, data: &[u8], pts: i64) -> Result<(), VideoError> {
        let _ = (data, pts);
        Err(VideoError::UnsupportedCodec {
            codec: "H.265/HEVC software decode not yet available. Use NVDEC (feature: video-decode)."
                .to_string(),
        })
    }

    fn flush(&mut self) -> Result<(), VideoError> {
        // Flush is a no-op on a stub decoder
        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<DecodedFrame>, VideoError> {
        Ok(self.frames.pop_front())
    }

    fn get_capabilities(&self, codec: DecodeCodec) -> Result<DecoderCapabilities, VideoError> {
        Ok(if codec == DecodeCodec::H265 {
            Self::capabilities()
        } else {
            DecoderCapabilities {
                supported: false,
                max_width: 0,
                max_height: 0,
                max_bit_depth: 0,
                num_engines: 0,
            }
        })
    }
}

impl Default for HevcDecoder {
    fn default() -> Self {
        Self::new().expect("Failed to create HEVC decoder stub")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hevc_decoder_reports_unavailable() {
        assert!(!HevcDecoder::is_available());
    }

    #[test]
    fn hevc_decoder_capabilities_unsupported() {
        let caps = HevcDecoder::capabilities();
        assert!(!caps.supported);
        assert_eq!(caps.max_width, 0);
    }

    #[test]
    fn hevc_decode_frame_returns_error() {
        let result = decode_hevc_frame(&[0x00, 0x00, 0x00, 0x01, 0x40]);
        assert!(result.is_err());
    }

    #[test]
    fn hevc_decode_frame_empty_data_returns_error() {
        let result = decode_hevc_frame(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn hevc_decoder_flush_succeeds() {
        let mut decoder = HevcDecoder::new().unwrap();
        assert!(decoder.flush().is_ok());
    }

    #[test]
    fn hevc_decoder_next_frame_returns_none() {
        let mut decoder = HevcDecoder::new().unwrap();
        assert!(decoder.next_frame().unwrap().is_none());
    }

    #[test]
    fn hevc_decoder_decode_packet_returns_error() {
        let mut decoder = HevcDecoder::new().unwrap();
        let result = decoder.decode_packet(&[0x00, 0x00, 0x00, 0x01], 0);
        assert!(result.is_err());
    }
}
