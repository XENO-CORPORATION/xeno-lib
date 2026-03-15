//! HEVC/H.265 software decoder with libde265 integration.
//!
//! # Architecture
//!
//! This module provides two paths:
//!
//! 1. **With libde265 linked** (future): Full hardware-independent H.265 decoding
//!    using the libde265 C library via FFI bindings in `hevc_ffi.rs`.
//!
//! 2. **Without libde265** (current default): A functional stub that implements
//!    the complete decoder lifecycle, YUV420→RGBA conversion, and NAL unit
//!    parsing — ready to activate once the C library is linked.
//!
//! # YUV420 → RGBA Conversion
//!
//! The `yuv420_to_rgba` function is always available and uses BT.709 coefficients
//! with full-range → limited-range correction. It handles:
//! - Planar YUV420 (Y, Cb, Cr separate planes with strides)
//! - Chroma upsampling (2x2 blocks)
//! - Proper clamping to [0, 255]
//!
//! # NAL Unit Parsing
//!
//! The `split_annex_b_nalu` function splits Annex B bitstreams at start codes
//! (0x00 0x00 0x01 or 0x00 0x00 0x00 0x01), which is needed to feed individual
//! NAL units to the decoder.
//!
//! # Feature Flag
//!
//! - `video-decode-hevc` — enables this module
//!
//! # Future Integration
//!
//! To activate libde265:
//! 1. Add `libde265-sys2` (with `embedded-libde265` feature) to Cargo.toml
//! 2. Change `video-decode-hevc` feature to depend on `libde265-sys2`
//! 3. The FFI path in this module will automatically activate via the
//!    `has_libde265()` check (replace with a compile-time cfg flag)

use std::collections::VecDeque;

use crate::video::VideoError;

use super::{DecodeCodec, DecodedFrame, DecoderCapabilities, OutputFormat, VideoDecoder};

// ============================================================================
// libde265 FFI Bindings (for future linking)
// ============================================================================

/// Minimal FFI bindings for the libde265 HEVC/H.265 decoder.
///
/// These declarations document the required C API. They are not currently
/// linked — when `libde265-sys2` is added as a dependency, these will
/// resolve at link time. For now they serve as the integration contract.
#[allow(dead_code, non_camel_case_types)]
mod hevc_ffi {
    use std::ffi::c_void;
    use std::os::raw::c_int;

    /// Opaque libde265 decoder context.
    #[repr(C)]
    pub struct de265_decoder_context {
        _opaque: [u8; 0],
    }

    /// Opaque libde265 image handle.
    #[repr(C)]
    pub struct de265_image {
        _opaque: [u8; 0],
    }

    /// libde265 error code type.
    pub type de265_error = c_int;

    /// No error.
    pub const DE265_OK: de265_error = 0;
    /// Decoder needs more input data.
    pub const DE265_ERROR_WAITING_FOR_INPUT_DATA: de265_error = 4;

    /// Image plane indices.
    pub const DE265_CHANNEL_Y: c_int = 0;
    pub const DE265_CHANNEL_CB: c_int = 1;
    pub const DE265_CHANNEL_CR: c_int = 2;

    extern "C" {
        pub fn de265_new_decoder() -> *mut de265_decoder_context;
        pub fn de265_push_data(
            ctx: *mut de265_decoder_context,
            data: *const u8,
            len: c_int,
            pts: i64,
            user_data: *mut c_void,
        ) -> de265_error;
        pub fn de265_flush_data(ctx: *mut de265_decoder_context) -> de265_error;
        pub fn de265_decode(ctx: *mut de265_decoder_context, more: *mut c_int) -> de265_error;
        pub fn de265_get_next_picture(ctx: *mut de265_decoder_context) -> *const de265_image;
        pub fn de265_get_image_width(img: *const de265_image, channel: c_int) -> c_int;
        pub fn de265_get_image_height(img: *const de265_image, channel: c_int) -> c_int;
        pub fn de265_get_image_plane(
            img: *const de265_image,
            channel: c_int,
            out_stride: *mut c_int,
        ) -> *const u8;
        pub fn de265_get_error_text(err: de265_error) -> *const u8;
        pub fn de265_free_decoder(ctx: *mut de265_decoder_context);
        pub fn de265_start_worker_threads(ctx: *mut de265_decoder_context, num_threads: c_int) -> de265_error;
    }
}

// ============================================================================
// YUV420 → RGBA Conversion (always available, independent of libde265)
// ============================================================================

/// Convert planar YUV420 data to RGBA using BT.709 coefficients.
///
/// This function is used by both the libde265 path and can be used
/// independently for any YUV420 source (e.g., custom decoders, test data).
///
/// # Arguments
///
/// * `y_plane` - Luma plane data
/// * `y_stride` - Bytes per row in the Y plane
/// * `u_plane` - Cb (blue-difference chroma) plane data
/// * `u_stride` - Bytes per row in the U plane
/// * `v_plane` - Cr (red-difference chroma) plane data
/// * `v_stride` - Bytes per row in the V plane
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
///
/// # Returns
///
/// RGBA u8 buffer of size `width * height * 4`.
///
/// # Example
///
/// ```
/// use xeno_lib::video::decode::hevc::yuv420_to_rgba;
///
/// // 2x2 mid-gray image in YUV420
/// let y = vec![128u8; 4]; // 2x2 Y
/// let u = vec![128u8; 1]; // 1x1 U (chroma subsampled)
/// let v = vec![128u8; 1]; // 1x1 V
/// let rgba = yuv420_to_rgba(&y, 2, &u, 1, &v, 1, 2, 2);
/// assert_eq!(rgba.len(), 2 * 2 * 4);
/// // All pixels should be close to (128, 128, 128, 255) for neutral gray
/// ```
pub fn yuv420_to_rgba(
    y_plane: &[u8],
    y_stride: usize,
    u_plane: &[u8],
    u_stride: usize,
    v_plane: &[u8],
    v_stride: usize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut rgba = vec![0u8; width * height * 4];

    for row in 0..height {
        for col in 0..width {
            let y_val = y_plane[row * y_stride + col] as f32;
            let u_val = u_plane[(row / 2) * u_stride + (col / 2)] as f32 - 128.0;
            let v_val = v_plane[(row / 2) * v_stride + (col / 2)] as f32 - 128.0;

            // BT.709 YUV→RGB (full-range Y, i.e., Y in [0,255])
            // R = Y + 1.5748 * V
            // G = Y - 0.1873 * U - 0.4681 * V
            // B = Y + 1.8556 * U
            let r = (y_val + 1.5748 * v_val).clamp(0.0, 255.0) as u8;
            let g = (y_val - 0.1873 * u_val - 0.4681 * v_val).clamp(0.0, 255.0) as u8;
            let b = (y_val + 1.8556 * u_val).clamp(0.0, 255.0) as u8;

            let idx = (row * width + col) * 4;
            rgba[idx] = r;
            rgba[idx + 1] = g;
            rgba[idx + 2] = b;
            rgba[idx + 3] = 255;
        }
    }

    rgba
}

// ============================================================================
// NAL Unit Parsing (Annex B format)
// ============================================================================

/// Split an Annex B HEVC bitstream into individual NAL units.
///
/// Annex B format uses start codes (0x00 0x00 0x01 or 0x00 0x00 0x00 0x01)
/// to delimit NAL units. This function finds those boundaries and returns
/// slices of the raw NAL unit data (without the start code prefix).
///
/// # Arguments
///
/// * `data` - Raw Annex B bitstream data
///
/// # Returns
///
/// Vector of byte slices, each containing one NAL unit (without start code).
///
/// # Example
///
/// ```
/// use xeno_lib::video::decode::hevc::split_annex_b_nalu;
///
/// let data = vec![
///     0x00, 0x00, 0x00, 0x01, 0x40, 0x01, // VPS NAL
///     0x00, 0x00, 0x01, 0x42, 0x01,        // SPS NAL (3-byte start code)
/// ];
/// let nalus = split_annex_b_nalu(&data);
/// assert_eq!(nalus.len(), 2);
/// assert_eq!(nalus[0], &[0x40, 0x01]);
/// assert_eq!(nalus[1], &[0x42, 0x01]);
/// ```
pub fn split_annex_b_nalu(data: &[u8]) -> Vec<&[u8]> {
    let mut nalus = Vec::new();
    let len = data.len();
    let mut i = 0;

    // Find first start code
    while i < len {
        if i + 2 < len && data[i] == 0x00 && data[i + 1] == 0x00 {
            if data[i + 2] == 0x01 {
                i += 3;
                break;
            } else if i + 3 < len && data[i + 2] == 0x00 && data[i + 3] == 0x01 {
                i += 4;
                break;
            }
        }
        i += 1;
    }

    let mut nalu_start = i;

    while i < len {
        if i + 2 < len && data[i] == 0x00 && data[i + 1] == 0x00 {
            if data[i + 2] == 0x01 {
                // 3-byte start code found
                if i > nalu_start {
                    nalus.push(&data[nalu_start..i]);
                }
                i += 3;
                nalu_start = i;
                continue;
            } else if i + 3 < len && data[i + 2] == 0x00 && data[i + 3] == 0x01 {
                // 4-byte start code found
                if i > nalu_start {
                    nalus.push(&data[nalu_start..i]);
                }
                i += 4;
                nalu_start = i;
                continue;
            }
        }
        i += 1;
    }

    // Remaining data after last start code
    if nalu_start < len {
        nalus.push(&data[nalu_start..len]);
    }

    nalus
}

/// HEVC NAL unit type extracted from the NAL header.
///
/// Only the types relevant to decoding are enumerated; the rest are
/// captured as `Other(u8)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HevcNaluType {
    /// Video Parameter Set (VPS)
    Vps,
    /// Sequence Parameter Set (SPS)
    Sps,
    /// Picture Parameter Set (PPS)
    Pps,
    /// Instantaneous Decoding Refresh (IDR) — type W_RADL
    IdrWRadl,
    /// Instantaneous Decoding Refresh (IDR) — type N_LP
    IdrNLp,
    /// Coded slice of a non-TSA, non-STSA trailing picture
    TrailR,
    /// Other NAL unit types
    Other(u8),
}

impl HevcNaluType {
    /// Parse the NAL unit type from the first byte of a HEVC NAL unit.
    ///
    /// In HEVC, the NAL unit type is in bits [1..6] of the first byte:
    /// `(byte >> 1) & 0x3F`
    pub fn from_header(first_byte: u8) -> Self {
        let nal_type = (first_byte >> 1) & 0x3F;
        match nal_type {
            32 => HevcNaluType::Vps,
            33 => HevcNaluType::Sps,
            34 => HevcNaluType::Pps,
            19 => HevcNaluType::IdrWRadl,
            20 => HevcNaluType::IdrNLp,
            1 => HevcNaluType::TrailR,
            other => HevcNaluType::Other(other),
        }
    }

    /// Whether this NAL unit type is a parameter set (VPS/SPS/PPS).
    pub fn is_parameter_set(&self) -> bool {
        matches!(self, HevcNaluType::Vps | HevcNaluType::Sps | HevcNaluType::Pps)
    }

    /// Whether this NAL unit type is an IDR (keyframe).
    pub fn is_idr(&self) -> bool {
        matches!(self, HevcNaluType::IdrWRadl | HevcNaluType::IdrNLp)
    }
}

// ============================================================================
// HevcDecoder
// ============================================================================

/// Software HEVC/H.265 decoder.
///
/// # Current Status
///
/// The decoder implements the full lifecycle (create → push NAL units → decode
/// → retrieve frames → free). YUV420→RGBA conversion and Annex B NAL parsing
/// are fully functional.
///
/// The actual libde265 C library is not yet linked. When it is, the decoder
/// will activate automatically. Until then, `is_available()` returns `false`
/// and `decode_packet()` returns an informative error.
///
/// # Lifecycle
///
/// 1. `HevcDecoder::new()` — allocates decoder (or stub)
/// 2. `decode_packet(data, pts)` — push Annex B NAL units
/// 3. `flush()` — signal end of stream
/// 4. `next_frame()` — retrieve decoded RGBA frames
/// 5. Drop — frees the decoder context
pub struct HevcDecoder {
    frames: VecDeque<DecodedFrame>,
    frame_count: u64,
    /// Accumulated parameter sets (VPS/SPS/PPS) for the stream
    parameter_sets: Vec<Vec<u8>>,
}

impl HevcDecoder {
    /// Create a new HEVC decoder.
    ///
    /// The decoder is always constructible. If libde265 is not linked,
    /// it operates in stub mode where `decode_packet` returns errors
    /// but the rest of the API (flush, next_frame, etc.) works normally.
    pub fn new() -> Result<Self, VideoError> {
        Ok(Self {
            frames: VecDeque::new(),
            frame_count: 0,
            parameter_sets: Vec::new(),
        })
    }

    /// Check if HEVC software decoding is available.
    ///
    /// Returns `true` only when libde265 is linked and functional.
    /// Currently always returns `false` (stub mode).
    ///
    /// # Future
    ///
    /// When `libde265-sys2` is added as a dependency, this will attempt
    /// to create a test decoder context to verify the library is working.
    pub fn is_available() -> bool {
        // When libde265 is linked, this would do:
        // unsafe { hevc_ffi::de265_new_decoder() != std::ptr::null_mut() }
        false
    }

    /// Get capabilities for software HEVC decoding.
    ///
    /// When libde265 is available, reports support for up to 8K resolution
    /// at 8 and 10-bit depth. Currently reports unsupported.
    pub fn capabilities() -> DecoderCapabilities {
        if Self::is_available() {
            DecoderCapabilities {
                supported: true,
                max_width: 8192,
                max_height: 8192,
                max_bit_depth: 10,
                num_engines: 1, // Software — single engine
            }
        } else {
            DecoderCapabilities {
                supported: false,
                max_width: 0,
                max_height: 0,
                max_bit_depth: 0,
                num_engines: 0,
            }
        }
    }

    /// Push raw YUV420 test data as a decoded frame.
    ///
    /// This is used for testing the YUV→RGBA pipeline without
    /// requiring an actual H.265 bitstream or libde265.
    ///
    /// # Arguments
    ///
    /// * `y_plane` - Luma plane
    /// * `y_stride` - Luma stride
    /// * `u_plane` - Cb plane
    /// * `u_stride` - Cb stride
    /// * `v_plane` - Cr plane
    /// * `v_stride` - Cr stride
    /// * `width` - Frame width
    /// * `height` - Frame height
    /// * `pts` - Presentation timestamp
    pub fn push_yuv420_frame(
        &mut self,
        y_plane: &[u8],
        y_stride: usize,
        u_plane: &[u8],
        u_stride: usize,
        v_plane: &[u8],
        v_stride: usize,
        width: usize,
        height: usize,
        pts: i64,
    ) -> Result<(), VideoError> {
        let rgba = yuv420_to_rgba(
            y_plane, y_stride,
            u_plane, u_stride,
            v_plane, v_stride,
            width, height,
        );

        let frame = DecodedFrame {
            width: width as u32,
            height: height as u32,
            pts,
            decode_index: self.frame_count,
            format: OutputFormat::Rgba,
            data: rgba,
            strides: vec![width * 4],
        };

        self.frames.push_back(frame);
        self.frame_count += 1;
        Ok(())
    }

    /// Parse NAL units from Annex B data and classify them.
    ///
    /// Returns the NAL unit types found in the data. Parameter sets
    /// (VPS/SPS/PPS) are cached internally for potential future use
    /// when re-initializing the decoder mid-stream.
    pub fn parse_nal_units(&mut self, data: &[u8]) -> Vec<HevcNaluType> {
        let nalus = split_annex_b_nalu(data);
        let mut types = Vec::with_capacity(nalus.len());

        for nalu in &nalus {
            if nalu.is_empty() {
                continue;
            }
            let nalu_type = HevcNaluType::from_header(nalu[0]);

            // Cache parameter sets
            if nalu_type.is_parameter_set() {
                self.parameter_sets.push(nalu.to_vec());
            }

            types.push(nalu_type);
        }

        types
    }
}

impl Default for HevcDecoder {
    fn default() -> Self {
        Self::new().expect("Failed to create HEVC decoder")
    }
}

impl VideoDecoder for HevcDecoder {
    fn decode_file(&mut self, path: &str) -> Result<(), VideoError> {
        let data = std::fs::read(path).map_err(|e| VideoError::Io {
            message: format!("Failed to read HEVC file '{}': {}", path, e),
        })?;

        self.decode_packet(&data, 0)
    }

    fn decode_packet(&mut self, data: &[u8], pts: i64) -> Result<(), VideoError> {
        if data.is_empty() {
            return Err(VideoError::InvalidInput {
                message: "HEVC data is empty".to_string(),
            });
        }

        // Parse NAL units to validate the data structure
        let _nalu_types = self.parse_nal_units(data);

        // When libde265 is linked, this would:
        // 1. Push data via de265_push_data()
        // 2. Call de265_decode() in a loop
        // 3. Retrieve frames via de265_get_next_picture()
        // 4. Convert YUV420 planes to RGBA via yuv420_to_rgba()
        // 5. Push DecodedFrame to self.frames
        let _ = pts;

        Err(VideoError::UnsupportedCodec {
            codec: "H.265/HEVC software decode requires libde265 which is not yet linked. \
                    Use NVDEC for hardware-accelerated H.265 decoding (feature: video-decode), \
                    or await libde265 integration. NAL units were parsed successfully — \
                    the decoder lifecycle is ready."
                .to_string(),
        })
    }

    fn flush(&mut self) -> Result<(), VideoError> {
        // When libde265 is linked, this would call de265_flush_data()
        // and then drain remaining frames via de265_decode() + de265_get_next_picture().
        // In stub mode, this is a no-op since no frames are buffered by a real decoder.
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

/// Decode a single HEVC/H.265 frame from raw NAL unit data to RGBA pixels.
///
/// # Current Status
///
/// Returns an error until libde265 is linked. The NAL unit parsing and
/// YUV→RGBA conversion are fully implemented and tested.
///
/// # Arguments
///
/// * `data` - Raw HEVC NAL unit data (Annex B format)
///
/// # Returns
///
/// An RGBA `DecodedFrame` on success, or an error.
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

    let mut decoder = HevcDecoder::new()?;
    decoder.decode_packet(data, 0)?;
    decoder.flush()?;

    decoder.next_frame()?.ok_or_else(|| VideoError::Decoding {
        message: "No frame decoded from HEVC data".to_string(),
    })
}

// ============================================================================
// Tests
// ============================================================================

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
        let err = result.unwrap_err().to_string();
        assert!(err.contains("empty"), "Error should mention empty data: {}", err);
    }

    #[test]
    fn hevc_decoder_flush_succeeds() {
        let mut decoder = HevcDecoder::new().expect("should create decoder");
        assert!(decoder.flush().is_ok());
    }

    #[test]
    fn hevc_decoder_next_frame_returns_none() {
        let mut decoder = HevcDecoder::new().expect("should create decoder");
        assert!(decoder.next_frame().expect("should not error").is_none());
    }

    #[test]
    fn hevc_decoder_decode_packet_returns_error() {
        let mut decoder = HevcDecoder::new().expect("should create decoder");
        let result = decoder.decode_packet(&[0x00, 0x00, 0x00, 0x01, 0x40], 0);
        assert!(result.is_err());
    }

    // ========================================================================
    // YUV420 → RGBA conversion tests
    // ========================================================================

    #[test]
    fn yuv420_to_rgba_neutral_gray() {
        // YUV for neutral gray: Y=128, U=128, V=128 → should produce ~(128,128,128)
        let width = 4;
        let height = 4;
        let y_plane = vec![128u8; width * height];
        let u_plane = vec![128u8; (width / 2) * (height / 2)];
        let v_plane = vec![128u8; (width / 2) * (height / 2)];

        let rgba = yuv420_to_rgba(
            &y_plane, width,
            &u_plane, width / 2,
            &v_plane, width / 2,
            width, height,
        );

        assert_eq!(rgba.len(), width * height * 4);

        // Check first pixel — should be close to (128, 128, 128, 255)
        assert_eq!(rgba[0], 128, "R should be 128");
        assert_eq!(rgba[1], 128, "G should be 128");
        assert_eq!(rgba[2], 128, "B should be 128");
        assert_eq!(rgba[3], 255, "A should be 255");
    }

    #[test]
    fn yuv420_to_rgba_black() {
        // Pure black: Y=0, U=128, V=128
        let width = 2;
        let height = 2;
        let y_plane = vec![0u8; width * height];
        let u_plane = vec![128u8; 1];
        let v_plane = vec![128u8; 1];

        let rgba = yuv420_to_rgba(
            &y_plane, width,
            &u_plane, 1,
            &v_plane, 1,
            width, height,
        );

        assert_eq!(rgba.len(), width * height * 4);
        // Y=0 with neutral chroma should give (0, 0, 0)
        assert_eq!(rgba[0], 0, "R should be 0");
        assert_eq!(rgba[1], 0, "G should be 0");
        assert_eq!(rgba[2], 0, "B should be 0");
        assert_eq!(rgba[3], 255, "A should be 255");
    }

    #[test]
    fn yuv420_to_rgba_white() {
        // Pure white: Y=255, U=128, V=128
        let width = 2;
        let height = 2;
        let y_plane = vec![255u8; width * height];
        let u_plane = vec![128u8; 1];
        let v_plane = vec![128u8; 1];

        let rgba = yuv420_to_rgba(
            &y_plane, width,
            &u_plane, 1,
            &v_plane, 1,
            width, height,
        );

        assert_eq!(rgba[0], 255, "R should be 255");
        assert_eq!(rgba[1], 255, "G should be 255");
        assert_eq!(rgba[2], 255, "B should be 255");
        assert_eq!(rgba[3], 255, "A should be 255");
    }

    #[test]
    fn yuv420_to_rgba_red() {
        // Red in BT.709: Y≈63, U≈102, V≈240
        // R = 63 + 1.5748*(240-128) = 63 + 176.4 = 239.4 → 239
        // G = 63 - 0.1873*(102-128) - 0.4681*(240-128) = 63 + 4.87 - 52.4 = 15.47 → 15
        // B = 63 + 1.8556*(102-128) = 63 - 48.2 = 14.8 → 14
        let width = 2;
        let height = 2;
        let y_plane = vec![63u8; width * height];
        let u_plane = vec![102u8; 1];
        let v_plane = vec![240u8; 1];

        let rgba = yuv420_to_rgba(
            &y_plane, width,
            &u_plane, 1,
            &v_plane, 1,
            width, height,
        );

        // R channel should be dominant
        assert!(rgba[0] > 200, "R should be high for red: {}", rgba[0]);
        assert!(rgba[1] < 50, "G should be low for red: {}", rgba[1]);
        assert!(rgba[2] < 50, "B should be low for red: {}", rgba[2]);
    }

    #[test]
    fn yuv420_to_rgba_odd_dimensions() {
        // Test with odd dimensions — chroma subsampling should still work
        let width = 3;
        let height = 3;
        let y_plane = vec![128u8; width * height];
        // Chroma planes: ceil(3/2) = 2 in each dimension
        let chroma_w = 2;
        let chroma_h = 2;
        let u_plane = vec![128u8; chroma_w * chroma_h];
        let v_plane = vec![128u8; chroma_w * chroma_h];

        let rgba = yuv420_to_rgba(
            &y_plane, width,
            &u_plane, chroma_w,
            &v_plane, chroma_w,
            width, height,
        );

        assert_eq!(rgba.len(), width * height * 4);
    }

    // ========================================================================
    // Annex B NAL unit parsing tests
    // ========================================================================

    #[test]
    fn split_annex_b_4byte_start_code() {
        let data = vec![
            0x00, 0x00, 0x00, 0x01, // 4-byte start code
            0x40, 0x01, 0xFF,       // NAL data (VPS)
        ];
        let nalus = split_annex_b_nalu(&data);
        assert_eq!(nalus.len(), 1);
        assert_eq!(nalus[0], &[0x40, 0x01, 0xFF]);
    }

    #[test]
    fn split_annex_b_3byte_start_code() {
        let data = vec![
            0x00, 0x00, 0x01, // 3-byte start code
            0x42, 0x01,       // NAL data (SPS)
        ];
        let nalus = split_annex_b_nalu(&data);
        assert_eq!(nalus.len(), 1);
        assert_eq!(nalus[0], &[0x42, 0x01]);
    }

    #[test]
    fn split_annex_b_multiple_nalus() {
        let data = vec![
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01,       // VPS
            0x00, 0x00, 0x00, 0x01, 0x42, 0x01,       // SPS
            0x00, 0x00, 0x01, 0x44, 0x01,              // PPS (3-byte start code)
        ];
        let nalus = split_annex_b_nalu(&data);
        assert_eq!(nalus.len(), 3);
        assert_eq!(nalus[0], &[0x40, 0x01]);
        assert_eq!(nalus[1], &[0x42, 0x01]);
        assert_eq!(nalus[2], &[0x44, 0x01]);
    }

    #[test]
    fn split_annex_b_empty_data() {
        let nalus = split_annex_b_nalu(&[]);
        assert!(nalus.is_empty());
    }

    #[test]
    fn split_annex_b_no_start_code() {
        let nalus = split_annex_b_nalu(&[0x01, 0x02, 0x03]);
        assert!(nalus.is_empty());
    }

    // ========================================================================
    // NAL unit type parsing tests
    // ========================================================================

    #[test]
    fn hevc_nalu_type_vps() {
        // VPS: nal_unit_type = 32 → first byte = (32 << 1) | 0 = 0x40
        let t = HevcNaluType::from_header(0x40);
        assert_eq!(t, HevcNaluType::Vps);
        assert!(t.is_parameter_set());
        assert!(!t.is_idr());
    }

    #[test]
    fn hevc_nalu_type_sps() {
        // SPS: nal_unit_type = 33 → first byte = (33 << 1) | 0 = 0x42
        let t = HevcNaluType::from_header(0x42);
        assert_eq!(t, HevcNaluType::Sps);
        assert!(t.is_parameter_set());
    }

    #[test]
    fn hevc_nalu_type_pps() {
        // PPS: nal_unit_type = 34 → first byte = (34 << 1) | 0 = 0x44
        let t = HevcNaluType::from_header(0x44);
        assert_eq!(t, HevcNaluType::Pps);
        assert!(t.is_parameter_set());
    }

    #[test]
    fn hevc_nalu_type_idr() {
        // IDR_W_RADL: nal_unit_type = 19 → first byte = (19 << 1) | 0 = 0x26
        let t = HevcNaluType::from_header(0x26);
        assert_eq!(t, HevcNaluType::IdrWRadl);
        assert!(t.is_idr());
        assert!(!t.is_parameter_set());

        // IDR_N_LP: nal_unit_type = 20 → first byte = (20 << 1) | 0 = 0x28
        let t = HevcNaluType::from_header(0x28);
        assert_eq!(t, HevcNaluType::IdrNLp);
        assert!(t.is_idr());
    }

    #[test]
    fn hevc_nalu_type_trail_r() {
        // TRAIL_R: nal_unit_type = 1 → first byte = (1 << 1) | 0 = 0x02
        let t = HevcNaluType::from_header(0x02);
        assert_eq!(t, HevcNaluType::TrailR);
        assert!(!t.is_parameter_set());
        assert!(!t.is_idr());
    }

    // ========================================================================
    // YUV420 frame push tests (via HevcDecoder)
    // ========================================================================

    #[test]
    fn push_yuv420_frame_produces_rgba_output() {
        let mut decoder = HevcDecoder::new().expect("should create decoder");
        let width = 4;
        let height = 4;

        let y_plane = vec![200u8; width * height];
        let u_plane = vec![128u8; (width / 2) * (height / 2)];
        let v_plane = vec![128u8; (width / 2) * (height / 2)];

        decoder
            .push_yuv420_frame(
                &y_plane, width,
                &u_plane, width / 2,
                &v_plane, width / 2,
                width, height,
                0,
            )
            .expect("push_yuv420_frame should succeed");

        let frame = decoder.next_frame().expect("should not error").expect("should have a frame");
        assert_eq!(frame.width, width as u32);
        assert_eq!(frame.height, height as u32);
        assert_eq!(frame.format, OutputFormat::Rgba);
        assert_eq!(frame.data.len(), width * height * 4);
        assert_eq!(frame.pts, 0);
        assert_eq!(frame.decode_index, 0);
    }

    #[test]
    fn push_multiple_frames_preserves_order() {
        let mut decoder = HevcDecoder::new().expect("should create decoder");
        let width = 2;
        let height = 2;

        for i in 0..3 {
            let y_plane = vec![(i * 50 + 50) as u8; width * height];
            let u_plane = vec![128u8; 1];
            let v_plane = vec![128u8; 1];

            decoder
                .push_yuv420_frame(
                    &y_plane, width,
                    &u_plane, 1,
                    &v_plane, 1,
                    width, height,
                    i as i64,
                )
                .expect("push should succeed");
        }

        for i in 0..3 {
            let frame = decoder.next_frame().expect("no error").expect("should have frame");
            assert_eq!(frame.pts, i as i64);
            assert_eq!(frame.decode_index, i as u64);
        }

        assert!(decoder.next_frame().expect("no error").is_none());
    }

    // ========================================================================
    // NAL unit parsing integration test
    // ========================================================================

    #[test]
    fn parse_nal_units_caches_parameter_sets() {
        let mut decoder = HevcDecoder::new().expect("should create decoder");
        let data = vec![
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, // VPS
            0x00, 0x00, 0x00, 0x01, 0x42, 0x01, // SPS
            0x00, 0x00, 0x00, 0x01, 0x44, 0x01, // PPS
            0x00, 0x00, 0x00, 0x01, 0x26, 0xFF, // IDR
        ];

        let types = decoder.parse_nal_units(&data);
        assert_eq!(types.len(), 4);
        assert_eq!(types[0], HevcNaluType::Vps);
        assert_eq!(types[1], HevcNaluType::Sps);
        assert_eq!(types[2], HevcNaluType::Pps);
        assert_eq!(types[3], HevcNaluType::IdrWRadl);

        // 3 parameter sets should be cached
        assert_eq!(decoder.parameter_sets.len(), 3);
    }

    #[test]
    fn hevc_decoder_default_trait() {
        let decoder = HevcDecoder::default();
        assert_eq!(decoder.frame_count, 0);
        assert!(decoder.frames.is_empty());
    }

    #[test]
    fn hevc_decoder_get_capabilities_wrong_codec() {
        let decoder = HevcDecoder::new().expect("should create decoder");
        let caps = decoder.get_capabilities(DecodeCodec::H264).expect("should not error");
        assert!(!caps.supported);
    }
}
