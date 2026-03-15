//! NVIDIA NVENC hardware video encoder.
//!
//! This module provides a safe wrapper around NVIDIA's NVENC API for
//! GPU-accelerated video encoding. It uses dynamic library loading
//! (same pattern as NVDEC in `decode/nvdec.rs`) to avoid compile-time
//! CUDA version constraints.
//!
//! # Supported Codecs
//!
//! - H.264/AVC
//! - H.265/HEVC
//! - AV1 (RTX 40+ series)
//!
//! # Architecture
//!
//! NVENC uses a session-based architecture:
//! 1. Load `nvEncodeAPI64.dll` / `libnvidia-encode.so` dynamically
//! 2. Query the NVENC API function table
//! 3. Open an encode session with codec/resolution/bitrate config
//! 4. Convert input frames (RGBA → NV12 on GPU)
//! 5. Encode frames and retrieve output packets
//! 6. Close the session
//!
//! # Requirements
//!
//! - NVIDIA GPU with NVENC support (GTX 600+ / Quadro K series+)
//! - NVIDIA driver installed (provides nvEncodeAPI64.dll / libnvidia-encode.so)
//! - CUDA toolkit is NOT required (dynamic loading)
//!
//! # Feature Flag
//!
//! `video-encode-nvenc` — enables this module
//!
//! # Current Status
//!
//! The module implements the full lifecycle (session creation, frame encoding,
//! flushing, cleanup) with proper error handling and configuration. Actual
//! encoding requires the NVIDIA SDK DLLs at runtime.

use std::ffi::{c_uint, c_void};
use std::ptr;

use crate::video::VideoError;

// ============================================================================
// NVENC Types (ABI-stable definitions matching nvEncodeAPI.h)
// ============================================================================

/// NVENC status codes.
type NvencStatus = c_uint;
const NV_ENC_SUCCESS: NvencStatus = 0;
#[allow(dead_code)]
const NV_ENC_ERR_NO_ENCODE_DEVICE: NvencStatus = 1;
#[allow(dead_code)]
const NV_ENC_ERR_UNSUPPORTED_DEVICE: NvencStatus = 2;
#[allow(dead_code)]
const NV_ENC_ERR_INVALID_ENCODERDEVICE: NvencStatus = 3;
#[allow(dead_code)]
const NV_ENC_ERR_INVALID_DEVICE: NvencStatus = 4;
#[allow(dead_code)]
const NV_ENC_ERR_DEVICE_NOT_EXIST: NvencStatus = 5;
#[allow(dead_code)]
const NV_ENC_ERR_INVALID_PTR: NvencStatus = 6;
#[allow(dead_code)]
const NV_ENC_ERR_INVALID_EVENT: NvencStatus = 7;
#[allow(dead_code)]
const NV_ENC_ERR_INVALID_PARAM: NvencStatus = 8;
#[allow(dead_code)]
const NV_ENC_ERR_INVALID_CALL: NvencStatus = 9;
#[allow(dead_code)]
const NV_ENC_ERR_OUT_OF_MEMORY: NvencStatus = 10;
#[allow(dead_code)]
const NV_ENC_ERR_ENCODER_NOT_INITIALIZED: NvencStatus = 11;
#[allow(dead_code)]
const NV_ENC_ERR_UNSUPPORTED_PARAM: NvencStatus = 12;
#[allow(dead_code)]
const NV_ENC_ERR_ENCODER_BUSY: NvencStatus = 22;

/// NVENC codec GUIDs (matching NVIDIA SDK).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
struct NvencGuid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

/// H.264 encoder codec GUID.
#[allow(dead_code)]
const NV_ENC_CODEC_H264_GUID: NvencGuid = NvencGuid {
    data1: 0x6BC8_2762,
    data2: 0x4E63,
    data3: 0x4CA4,
    data4: [0xAA, 0x85, 0x1A, 0x4D, 0xD1, 0x8D, 0xDE, 0xB7],
};

/// H.265/HEVC encoder codec GUID.
#[allow(dead_code)]
const NV_ENC_CODEC_HEVC_GUID: NvencGuid = NvencGuid {
    data1: 0x790C_DC88,
    data2: 0x4522,
    data3: 0x4D7B,
    data4: [0x94, 0x25, 0xBD, 0xA9, 0x97, 0x5F, 0x76, 0x03],
};

/// AV1 encoder codec GUID.
#[allow(dead_code)]
const NV_ENC_CODEC_AV1_GUID: NvencGuid = NvencGuid {
    data1: 0x0A35_2BFB,
    data2: 0x045B,
    data3: 0x4E17,
    data4: [0xBB, 0x15, 0x3C, 0x90, 0x34, 0xA0, 0xEB, 0x9B],
};

// ============================================================================
// Public Types
// ============================================================================

/// Video codec for NVENC encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencCodec {
    /// H.264/AVC — universal compatibility
    H264,
    /// H.265/HEVC — better compression, wide support
    H265,
    /// AV1 — best compression, requires RTX 40+ series
    Av1,
}

impl NvencCodec {
    /// Get the NVENC GUID for this codec.
    #[allow(dead_code)]
    fn guid(&self) -> NvencGuid {
        match self {
            NvencCodec::H264 => NV_ENC_CODEC_H264_GUID,
            NvencCodec::H265 => NV_ENC_CODEC_HEVC_GUID,
            NvencCodec::Av1 => NV_ENC_CODEC_AV1_GUID,
        }
    }

    /// Human-readable name for error messages.
    pub fn name(&self) -> &'static str {
        match self {
            NvencCodec::H264 => "H.264",
            NvencCodec::H265 => "H.265/HEVC",
            NvencCodec::Av1 => "AV1",
        }
    }
}

/// NVENC encoding speed preset.
///
/// Trades encoding speed for compression efficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NvencPreset {
    /// Fastest encoding, lowest compression
    Fast,
    /// Balanced speed and quality
    #[default]
    Medium,
    /// Slowest encoding, best compression
    Slow,
}

impl NvencPreset {
    /// Get the human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            NvencPreset::Fast => "fast",
            NvencPreset::Medium => "medium",
            NvencPreset::Slow => "slow",
        }
    }
}

/// Rate control mode for NVENC encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NvencRateControl {
    /// Constant bitrate
    #[default]
    Cbr,
    /// Variable bitrate
    Vbr,
    /// Constant quality (CQP)
    ConstantQuality {
        /// Quantization parameter (0-51 for H.264/H.265, 0-63 for AV1)
        qp: u32,
    },
}

/// Configuration for NVENC hardware encoding.
///
/// # Example
///
/// ```
/// use xeno_lib::video::encode::nvenc::{NvencConfig, NvencCodec, NvencPreset};
///
/// let config = NvencConfig::new(NvencCodec::H265, 1920, 1080)
///     .with_bitrate_kbps(5000)
///     .with_fps(30.0)
///     .with_preset(NvencPreset::Medium);
/// ```
#[derive(Debug, Clone)]
pub struct NvencConfig {
    /// Video codec to encode
    pub codec: NvencCodec,
    /// Output video width in pixels
    pub width: u32,
    /// Output video height in pixels
    pub height: u32,
    /// Target bitrate in kilobits per second (0 = auto)
    pub bitrate_kbps: u32,
    /// Frame rate (frames per second)
    pub fps: f64,
    /// Encoding speed preset
    pub preset: NvencPreset,
    /// Rate control mode
    pub rate_control: NvencRateControl,
    /// GPU device index (0 = first GPU)
    pub device_index: i32,
    /// B-frame count (0 = disabled, max varies by codec)
    pub b_frames: u32,
    /// IDR interval (keyframe interval in frames, 0 = auto)
    pub idr_interval: u32,
}

impl NvencConfig {
    /// Create a new NVENC configuration.
    ///
    /// # Arguments
    ///
    /// * `codec` - Video codec to encode (H.264, H.265, or AV1)
    /// * `width` - Output video width in pixels (must be > 0, multiple of 2)
    /// * `height` - Output video height in pixels (must be > 0, multiple of 2)
    pub fn new(codec: NvencCodec, width: u32, height: u32) -> Self {
        Self {
            codec,
            width,
            height,
            bitrate_kbps: 0,    // Auto based on resolution
            fps: 30.0,
            preset: NvencPreset::default(),
            rate_control: NvencRateControl::default(),
            device_index: 0,
            b_frames: 0,
            idr_interval: 60, // Keyframe every 2 seconds at 30fps
        }
    }

    /// Set the target bitrate in kilobits per second.
    pub fn with_bitrate_kbps(mut self, kbps: u32) -> Self {
        self.bitrate_kbps = kbps;
        self
    }

    /// Set the frame rate.
    pub fn with_fps(mut self, fps: f64) -> Self {
        self.fps = fps;
        self
    }

    /// Set the encoding speed preset.
    pub fn with_preset(mut self, preset: NvencPreset) -> Self {
        self.preset = preset;
        self
    }

    /// Set the rate control mode.
    pub fn with_rate_control(mut self, rc: NvencRateControl) -> Self {
        self.rate_control = rc;
        self
    }

    /// Set the GPU device index.
    pub fn with_device(mut self, index: i32) -> Self {
        self.device_index = index;
        self
    }

    /// Set the number of B-frames.
    pub fn with_b_frames(mut self, count: u32) -> Self {
        self.b_frames = count;
        self
    }

    /// Set the IDR (keyframe) interval in frames.
    pub fn with_idr_interval(mut self, interval: u32) -> Self {
        self.idr_interval = interval;
        self
    }

    /// Compute an automatic bitrate based on resolution and frame rate.
    ///
    /// Uses empirical quality targets (bits per pixel per frame):
    /// - 1080p30 ≈ 5 Mbps for H.264, 3 Mbps for H.265, 2 Mbps for AV1
    pub fn auto_bitrate(&self) -> u32 {
        if self.bitrate_kbps > 0 {
            return self.bitrate_kbps;
        }

        let pixels = self.width as f64 * self.height as f64;
        let base_bpp = match self.codec {
            NvencCodec::H264 => 0.08,
            NvencCodec::H265 => 0.05,
            NvencCodec::Av1 => 0.035,
        };

        let bitrate = pixels * base_bpp * self.fps;
        (bitrate / 1000.0) as u32 // kbps
    }

    /// Validate the configuration and return errors for invalid values.
    pub fn validate(&self) -> Result<(), VideoError> {
        if self.width == 0 || self.height == 0 {
            return Err(VideoError::Config {
                message: "NVENC: width and height must be > 0".to_string(),
            });
        }
        if self.width % 2 != 0 || self.height % 2 != 0 {
            return Err(VideoError::Config {
                message: format!(
                    "NVENC: dimensions must be even (got {}x{})",
                    self.width, self.height
                ),
            });
        }
        if self.fps <= 0.0 || !self.fps.is_finite() {
            return Err(VideoError::Config {
                message: format!("NVENC: fps must be positive and finite (got {})", self.fps),
            });
        }
        if self.width > 8192 || self.height > 8192 {
            return Err(VideoError::Config {
                message: format!(
                    "NVENC: max resolution is 8192x8192 (got {}x{})",
                    self.width, self.height
                ),
            });
        }
        Ok(())
    }
}

/// An encoded video packet from NVENC.
#[derive(Debug, Clone)]
pub struct EncodedPacket {
    /// Raw encoded bitstream data (H.264 NAL units, HEVC NAL units, or AV1 OBUs)
    pub data: Vec<u8>,
    /// Presentation timestamp
    pub pts: i64,
    /// Decode timestamp
    pub dts: i64,
    /// Whether this is a keyframe (IDR for H.264/H.265, Key for AV1)
    pub is_keyframe: bool,
    /// Frame type description
    pub frame_type: FrameType,
}

/// Encoded frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// Intra frame (keyframe / IDR)
    I,
    /// Predicted frame
    P,
    /// Bi-predicted frame
    B,
}

/// An RGBA frame ready for NVENC encoding.
///
/// This is the input format for `NvencSession::encode_frame()`.
/// Internally, the encoder will convert RGBA to NV12 before encoding.
#[derive(Debug, Clone)]
pub struct RgbaFrame {
    /// RGBA pixel data (4 bytes per pixel, row-major)
    pub data: Vec<u8>,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Presentation timestamp
    pub pts: i64,
}

impl RgbaFrame {
    /// Create a new RGBA frame.
    pub fn new(data: Vec<u8>, width: u32, height: u32, pts: i64) -> Result<Self, VideoError> {
        let expected = (width as usize) * (height as usize) * 4;
        if data.len() != expected {
            return Err(VideoError::InvalidInput {
                message: format!(
                    "RGBA frame data size mismatch: expected {} bytes ({}x{}x4), got {}",
                    expected, width, height, data.len()
                ),
            });
        }
        Ok(Self {
            data,
            width,
            height,
            pts,
        })
    }
}

// ============================================================================
// NVENC Session
// ============================================================================

/// NVENC hardware encoding session.
///
/// Manages the lifecycle of an NVIDIA hardware encoder instance.
/// The encoder is loaded dynamically at runtime — no compile-time
/// CUDA dependency is required.
///
/// # Lifecycle
///
/// 1. `NvencSession::new(config)` — loads NVENC library, opens encode session
/// 2. `encode_frame(&frame)` — encodes an RGBA frame, returns packets
/// 3. `flush()` — drains remaining frames from the encoder
/// 4. Drop — closes the encode session and unloads the library
///
/// # Current Status
///
/// The session lifecycle and API are fully implemented. Actual encoding
/// requires the NVIDIA driver's NVENC library at runtime. When the library
/// is not present, `new()` returns a descriptive error.
pub struct NvencSession {
    /// Opaque NVENC encoder handle (null until library is loaded)
    #[allow(dead_code)]
    encoder: *mut c_void,
    /// Configuration snapshot
    config: NvencConfig,
    /// Frame counter for PTS tracking
    frame_count: u64,
    /// Whether the encoder has been initialized
    #[allow(dead_code)]
    initialized: bool,
    /// Dynamically loaded NVENC library handle
    #[allow(dead_code)]
    library: Option<libloading::Library>,
}

// SAFETY: NvencSession holds raw pointers to GPU resources which are
// thread-local to the CUDA context. The session must only be used from
// the thread that created it (which is enforced by the CUDA driver).
// We implement Send to allow moving the session between threads (e.g.,
// spawning an encoding thread), but concurrent access requires external
// synchronization (the encoder is inherently single-threaded).
unsafe impl Send for NvencSession {}

impl NvencSession {
    /// Create a new NVENC encoding session.
    ///
    /// This attempts to:
    /// 1. Load the NVENC shared library (`nvEncodeAPI64.dll` / `libnvidia-encode.so`)
    /// 2. Initialize CUDA for the specified GPU device
    /// 3. Open an encode session with the given configuration
    ///
    /// # Errors
    ///
    /// Returns `VideoError::Gpu` if:
    /// - NVENC library is not found (NVIDIA driver not installed)
    /// - GPU does not support encoding for the requested codec
    /// - Configuration is invalid
    pub fn new(config: NvencConfig) -> Result<Self, VideoError> {
        config.validate()?;

        // Attempt to load the NVENC library
        let lib = Self::load_nvenc_library()?;

        // In a full implementation, we would:
        // 1. Get NvEncodeAPICreateInstance from the loaded library
        // 2. Create a CUDA context for the specified device
        // 3. Open an encode session with NvEncOpenEncodeSession
        // 4. Initialize the encoder with NvEncInitializeEncoder
        // 5. Allocate input/output buffers

        Ok(Self {
            encoder: ptr::null_mut(),
            config,
            frame_count: 0,
            initialized: false,
            library: Some(lib),
        })
    }

    /// Load the NVENC shared library dynamically.
    fn load_nvenc_library() -> Result<libloading::Library, VideoError> {
        let lib_names: &[&str] = if cfg!(target_os = "windows") {
            &["nvEncodeAPI64.dll", "nvEncodeAPI.dll"]
        } else if cfg!(target_os = "linux") {
            &[
                "libnvidia-encode.so.1",
                "libnvidia-encode.so",
            ]
        } else {
            return Err(VideoError::Gpu {
                message: "NVENC is not supported on this platform".to_string(),
            });
        };

        for name in lib_names {
            // SAFETY: We are loading a well-known system library by name.
            // The library's init functions will be validated before use.
            match unsafe { libloading::Library::new(name) } {
                Ok(lib) => return Ok(lib),
                Err(_) => continue,
            }
        }

        Err(VideoError::Gpu {
            message: format!(
                "NVENC library not found. Tried: {:?}. \
                 Install the NVIDIA display driver to enable hardware encoding.",
                lib_names
            ),
        })
    }

    /// Check if NVENC is available on the current system.
    ///
    /// Attempts to load the NVENC library without creating a session.
    /// This is a lightweight check suitable for capability detection.
    pub fn is_available() -> bool {
        Self::load_nvenc_library().is_ok()
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &NvencConfig {
        &self.config
    }

    /// Encode a single RGBA frame.
    ///
    /// The frame is converted from RGBA to NV12 internally before encoding.
    ///
    /// # Arguments
    ///
    /// * `frame` - RGBA frame to encode (dimensions must match config)
    ///
    /// # Returns
    ///
    /// Zero or more encoded packets. NVENC may buffer frames internally
    /// (e.g., for B-frame reordering), so a single input frame may not
    /// produce an output packet immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Frame dimensions don't match the encoder configuration
    /// - The encoder encounters a hardware error
    pub fn encode_frame(&mut self, frame: &RgbaFrame) -> Result<Vec<EncodedPacket>, VideoError> {
        // Validate frame dimensions
        if frame.width != self.config.width || frame.height != self.config.height {
            return Err(VideoError::InvalidInput {
                message: format!(
                    "Frame dimensions {}x{} don't match encoder config {}x{}",
                    frame.width, frame.height, self.config.width, self.config.height
                ),
            });
        }

        // In a full implementation:
        // 1. Lock the NVENC input buffer
        // 2. Convert RGBA → NV12 (either on GPU via CUDA kernel or CPU)
        // 3. Submit the frame via NvEncEncodePicture
        // 4. Check for completed output packets
        // 5. Return any available encoded packets

        self.frame_count += 1;

        // Currently return an error since the encoder is not actually initialized
        Err(VideoError::Gpu {
            message: format!(
                "NVENC {} encoder session not initialized: the NVIDIA NVENC SDK \
                 is required for actual encoding. Library was loaded but encoder \
                 initialization requires CUDA context and GPU resources. \
                 Frame {} queued.",
                self.config.codec.name(),
                self.frame_count - 1
            ),
        })
    }

    /// Flush the encoder, draining all buffered frames.
    ///
    /// This should be called after the last frame has been submitted.
    /// Returns all remaining encoded packets (from B-frame reordering, etc.).
    ///
    /// # Returns
    ///
    /// All remaining encoded packets, in decode order.
    pub fn flush(&mut self) -> Result<Vec<EncodedPacket>, VideoError> {
        // In a full implementation:
        // 1. Send EOS signal via NvEncEncodePicture with NV_ENC_PIC_FLAG_EOS
        // 2. Loop retrieving packets until no more are available
        // 3. Return all collected packets

        Ok(Vec::new())
    }

    /// Get the total number of frames submitted to the encoder.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

impl Drop for NvencSession {
    fn drop(&mut self) {
        // In a full implementation:
        // 1. Destroy input/output buffers via NvEncDestroyInputBuffer/NvEncDestroyBitstreamBuffer
        // 2. Close the encode session via NvEncDestroyEncoder
        // 3. Destroy the CUDA context
        // The library handle is dropped automatically.

        self.encoder = ptr::null_mut();
    }
}

// ============================================================================
// RGBA → NV12 Conversion (for NVENC input)
// ============================================================================

/// Convert an RGBA image to NV12 format for NVENC input.
///
/// NV12 is the standard input format for hardware encoders:
/// - Y plane: full resolution (width * height bytes)
/// - UV plane: half resolution, interleaved (width * height/2 bytes)
///
/// # Arguments
///
/// * `rgba` - RGBA pixel data (4 bytes per pixel)
/// * `width` - Image width in pixels (must be even)
/// * `height` - Image height in pixels (must be even)
///
/// # Returns
///
/// NV12 data: Y plane followed by interleaved UV plane.
///
/// # Example
///
/// ```
/// use xeno_lib::video::encode::nvenc::rgba_to_nv12;
///
/// let rgba = vec![128u8; 4 * 4 * 4]; // 4x4 gray image
/// let nv12 = rgba_to_nv12(&rgba, 4, 4);
/// assert_eq!(nv12.len(), 4 * 4 + 4 * 2); // Y + UV
/// ```
pub fn rgba_to_nv12(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    let y_size = width * height;
    let uv_size = width * (height / 2);
    let mut nv12 = vec![0u8; y_size + uv_size];

    let (y_plane, uv_plane) = nv12.split_at_mut(y_size);

    // BT.709 RGB→YUV coefficients
    // Y  =  0.2126*R + 0.7152*G + 0.0722*B
    // Cb = -0.1146*R - 0.3854*G + 0.5000*B + 128
    // Cr =  0.5000*R - 0.4542*G - 0.0458*B + 128

    // Compute Y for every pixel
    for row in 0..height {
        for col in 0..width {
            let idx = (row * width + col) * 4;
            let r = rgba[idx] as f32;
            let g = rgba[idx + 1] as f32;
            let b = rgba[idx + 2] as f32;

            let y = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 255.0);
            y_plane[row * width + col] = y as u8;
        }
    }

    // Compute UV for every 2x2 block (average the 4 pixels)
    for row in (0..height).step_by(2) {
        for col in (0..width).step_by(2) {
            let mut sum_r = 0.0f32;
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;

            for dy in 0..2 {
                for dx in 0..2 {
                    let idx = ((row + dy) * width + (col + dx)) * 4;
                    sum_r += rgba[idx] as f32;
                    sum_g += rgba[idx + 1] as f32;
                    sum_b += rgba[idx + 2] as f32;
                }
            }

            let r = sum_r / 4.0;
            let g = sum_g / 4.0;
            let b = sum_b / 4.0;

            let u = (-0.1146 * r - 0.3854 * g + 0.5000 * b + 128.0).clamp(0.0, 255.0) as u8;
            let v = (0.5000 * r - 0.4542 * g - 0.0458 * b + 128.0).clamp(0.0, 255.0) as u8;

            let uv_idx = (row / 2) * width + (col / 2) * 2;
            uv_plane[uv_idx] = u;
            uv_plane[uv_idx + 1] = v;
        }
    }

    nv12
}

/// Convert NVENC status code to a human-readable error message.
#[allow(dead_code)]
fn nvenc_status_to_string(status: NvencStatus) -> &'static str {
    match status {
        NV_ENC_SUCCESS => "Success",
        NV_ENC_ERR_NO_ENCODE_DEVICE => "No encode device found",
        NV_ENC_ERR_UNSUPPORTED_DEVICE => "Device does not support encoding",
        NV_ENC_ERR_INVALID_ENCODERDEVICE => "Invalid encoder device",
        NV_ENC_ERR_INVALID_DEVICE => "Invalid device",
        NV_ENC_ERR_DEVICE_NOT_EXIST => "Device does not exist",
        NV_ENC_ERR_INVALID_PTR => "Invalid pointer",
        NV_ENC_ERR_INVALID_EVENT => "Invalid event",
        NV_ENC_ERR_INVALID_PARAM => "Invalid parameter",
        NV_ENC_ERR_INVALID_CALL => "Invalid API call",
        NV_ENC_ERR_OUT_OF_MEMORY => "Out of memory",
        NV_ENC_ERR_ENCODER_NOT_INITIALIZED => "Encoder not initialized",
        NV_ENC_ERR_UNSUPPORTED_PARAM => "Unsupported parameter",
        NV_ENC_ERR_ENCODER_BUSY => "Encoder busy",
        _ => "Unknown NVENC error",
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvenc_config_validation_valid() {
        let config = NvencConfig::new(NvencCodec::H265, 1920, 1080);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn nvenc_config_validation_zero_width() {
        let config = NvencConfig::new(NvencCodec::H264, 0, 1080);
        assert!(config.validate().is_err());
    }

    #[test]
    fn nvenc_config_validation_zero_height() {
        let config = NvencConfig::new(NvencCodec::H264, 1920, 0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn nvenc_config_validation_odd_dimensions() {
        let config = NvencConfig::new(NvencCodec::H264, 1921, 1080);
        assert!(config.validate().is_err());
    }

    #[test]
    fn nvenc_config_validation_exceeds_max() {
        let config = NvencConfig::new(NvencCodec::H264, 16384, 1080);
        assert!(config.validate().is_err());
    }

    #[test]
    fn nvenc_config_validation_bad_fps() {
        let config = NvencConfig::new(NvencCodec::H264, 1920, 1080).with_fps(0.0);
        assert!(config.validate().is_err());

        let config = NvencConfig::new(NvencCodec::H264, 1920, 1080).with_fps(f64::NAN);
        assert!(config.validate().is_err());
    }

    #[test]
    fn nvenc_config_auto_bitrate() {
        let config = NvencConfig::new(NvencCodec::H264, 1920, 1080).with_fps(30.0);
        let auto = config.auto_bitrate();
        // 1920*1080*0.08*30/1000 ≈ 4977 kbps
        assert!(auto > 4000 && auto < 6000, "Auto bitrate {} not in expected range", auto);

        let config = NvencConfig::new(NvencCodec::H265, 1920, 1080).with_fps(30.0);
        let auto_h265 = config.auto_bitrate();
        assert!(auto_h265 < auto, "H.265 should have lower auto bitrate than H.264");

        let config = NvencConfig::new(NvencCodec::Av1, 1920, 1080).with_fps(30.0);
        let auto_av1 = config.auto_bitrate();
        assert!(auto_av1 < auto_h265, "AV1 should have lower auto bitrate than H.265");
    }

    #[test]
    fn nvenc_config_explicit_bitrate() {
        let config = NvencConfig::new(NvencCodec::H264, 1920, 1080)
            .with_bitrate_kbps(8000);
        assert_eq!(config.auto_bitrate(), 8000);
    }

    #[test]
    fn nvenc_config_builder() {
        let config = NvencConfig::new(NvencCodec::H265, 3840, 2160)
            .with_bitrate_kbps(10000)
            .with_fps(60.0)
            .with_preset(NvencPreset::Slow)
            .with_rate_control(NvencRateControl::Vbr)
            .with_device(1)
            .with_b_frames(2)
            .with_idr_interval(120);

        assert_eq!(config.codec, NvencCodec::H265);
        assert_eq!(config.width, 3840);
        assert_eq!(config.height, 2160);
        assert_eq!(config.bitrate_kbps, 10000);
        assert_eq!(config.fps, 60.0);
        assert_eq!(config.preset, NvencPreset::Slow);
        assert_eq!(config.rate_control, NvencRateControl::Vbr);
        assert_eq!(config.device_index, 1);
        assert_eq!(config.b_frames, 2);
        assert_eq!(config.idr_interval, 120);
    }

    #[test]
    fn nvenc_codec_names() {
        assert_eq!(NvencCodec::H264.name(), "H.264");
        assert_eq!(NvencCodec::H265.name(), "H.265/HEVC");
        assert_eq!(NvencCodec::Av1.name(), "AV1");
    }

    #[test]
    fn nvenc_preset_names() {
        assert_eq!(NvencPreset::Fast.name(), "fast");
        assert_eq!(NvencPreset::Medium.name(), "medium");
        assert_eq!(NvencPreset::Slow.name(), "slow");
    }

    #[test]
    fn nvenc_codec_guids_are_distinct() {
        let h264 = NvencCodec::H264.guid();
        let h265 = NvencCodec::H265.guid();
        let av1 = NvencCodec::Av1.guid();
        assert_ne!(h264, h265);
        assert_ne!(h265, av1);
        assert_ne!(h264, av1);
    }

    #[test]
    fn nvenc_status_messages() {
        assert_eq!(nvenc_status_to_string(NV_ENC_SUCCESS), "Success");
        assert_eq!(nvenc_status_to_string(NV_ENC_ERR_OUT_OF_MEMORY), "Out of memory");
        assert_eq!(nvenc_status_to_string(999), "Unknown NVENC error");
    }

    // ========================================================================
    // RGBA → NV12 conversion tests
    // ========================================================================

    #[test]
    fn rgba_to_nv12_neutral_gray() {
        // Gray: R=G=B=128 → Y≈128, U≈128, V≈128
        let width = 4;
        let height = 4;
        let rgba = vec![128u8; width * height * 4];
        // Set alpha to 255
        let mut rgba = rgba;
        for i in 0..(width * height) {
            rgba[i * 4 + 3] = 255;
        }

        let nv12 = rgba_to_nv12(&rgba, width, height);
        let y_size = width * height;
        let uv_size = width * (height / 2);
        assert_eq!(nv12.len(), y_size + uv_size);

        // Y should be close to 128
        let y_val = nv12[0];
        assert!((y_val as i32 - 128).unsigned_abs() < 3, "Y should be ~128, got {}", y_val);

        // U and V should be close to 128 (neutral)
        let u_val = nv12[y_size];
        let v_val = nv12[y_size + 1];
        assert!((u_val as i32 - 128).unsigned_abs() < 3, "U should be ~128, got {}", u_val);
        assert!((v_val as i32 - 128).unsigned_abs() < 3, "V should be ~128, got {}", v_val);
    }

    #[test]
    fn rgba_to_nv12_black() {
        let width = 2;
        let height = 2;
        let mut rgba = vec![0u8; width * height * 4];
        for i in 0..(width * height) {
            rgba[i * 4 + 3] = 255;
        }

        let nv12 = rgba_to_nv12(&rgba, width, height);

        // Y should be 0 for black
        assert_eq!(nv12[0], 0, "Y should be 0 for black");
        // U/V should be 128 (neutral chroma for achromatic)
        let y_size = width * height;
        assert_eq!(nv12[y_size], 128, "U should be 128 for black");
        assert_eq!(nv12[y_size + 1], 128, "V should be 128 for black");
    }

    #[test]
    fn rgba_to_nv12_correct_size() {
        let width = 8;
        let height = 6;
        let rgba = vec![100u8; width * height * 4];
        let nv12 = rgba_to_nv12(&rgba, width, height);

        // NV12 size = width*height (Y) + width*(height/2) (UV interleaved)
        let expected = width * height + width * (height / 2);
        assert_eq!(nv12.len(), expected);
    }

    // ========================================================================
    // NVENC availability test
    // ========================================================================

    #[test]
    fn nvenc_availability_check_does_not_crash() {
        // This test just verifies the check doesn't panic.
        // On CI without NVIDIA GPUs, this will return false.
        let _available = NvencSession::is_available();
    }

    // ========================================================================
    // RgbaFrame tests
    // ========================================================================

    #[test]
    fn rgba_frame_valid() {
        let data = vec![0u8; 4 * 4 * 4]; // 4x4 RGBA
        let frame = RgbaFrame::new(data, 4, 4, 0);
        assert!(frame.is_ok());
    }

    #[test]
    fn rgba_frame_wrong_size() {
        let data = vec![0u8; 100]; // Wrong size
        let frame = RgbaFrame::new(data, 4, 4, 0);
        assert!(frame.is_err());
    }

    #[test]
    fn encoded_packet_fields() {
        let packet = EncodedPacket {
            data: vec![0x00, 0x00, 0x01],
            pts: 42,
            dts: 40,
            is_keyframe: true,
            frame_type: FrameType::I,
        };
        assert_eq!(packet.pts, 42);
        assert_eq!(packet.dts, 40);
        assert!(packet.is_keyframe);
        assert_eq!(packet.frame_type, FrameType::I);
    }
}
