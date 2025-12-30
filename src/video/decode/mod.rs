//! GPU-accelerated video decoding via NVIDIA NVDEC.
//!
//! This module provides a safe Rust wrapper around NVIDIA's NVDEC hardware decoder,
//! enabling 10-100x faster video decoding compared to CPU-based solutions.
//!
//! # Supported Codecs
//!
//! - AV1 (RTX 30/40 series)
//! - H.264/AVC
//! - H.265/HEVC
//! - VP8/VP9
//!
//! # Architecture
//!
//! The decoder uses NVIDIA's callback-based architecture:
//! 1. VideoParser parses the bitstream
//! 2. NVDEC hardware decodes frames
//! 3. Frames are available in GPU memory (CUDA)
//! 4. Optional: copy to CPU for image processing
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::video::decode::{NvDecoder, DecoderConfig, VideoCodec};
//!
//! let config = DecoderConfig::new(VideoCodec::Av1)
//!     .with_output_format(OutputFormat::Nv12);
//!
//! let mut decoder = NvDecoder::new(config)?;
//! decoder.decode_file("input.ivf")?;
//!
//! while let Some(frame) = decoder.next_frame()? {
//!     let rgba = frame.to_rgba()?; // Convert to RGBA on GPU
//!     let image = rgba.to_cpu()?;  // Copy to CPU if needed
//! }
//! ```

#[cfg(feature = "video-decode")]
mod nvdec;

#[cfg(feature = "video-decode-sw")]
mod dav1d_decoder;

#[cfg(feature = "video-decode")]
pub use nvdec::*;

#[cfg(feature = "video-decode-sw")]
pub use dav1d_decoder::Dav1dDecoder;

use crate::video::VideoError;

/// Video codec for decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeCodec {
    /// AV1 codec (RTX 30/40 series required)
    Av1,
    /// H.264/AVC codec
    H264,
    /// H.265/HEVC codec
    H265,
    /// VP8 codec
    Vp8,
    /// VP9 codec
    Vp9,
}

impl DecodeCodec {
    /// Get the FourCC code for this codec.
    pub fn fourcc(&self) -> &'static str {
        match self {
            DecodeCodec::Av1 => "AV01",
            DecodeCodec::H264 => "H264",
            DecodeCodec::H265 => "H265",
            DecodeCodec::Vp8 => "VP80",
            DecodeCodec::Vp9 => "VP90",
        }
    }

    /// Detect codec from FourCC code.
    pub fn from_fourcc(fourcc: &str) -> Option<Self> {
        match fourcc {
            "AV01" | "av01" => Some(DecodeCodec::Av1),
            "H264" | "h264" | "avc1" | "AVC1" => Some(DecodeCodec::H264),
            "H265" | "h265" | "hvc1" | "HVC1" | "hev1" | "HEV1" => Some(DecodeCodec::H265),
            "VP80" | "vp80" => Some(DecodeCodec::Vp8),
            "VP90" | "vp90" => Some(DecodeCodec::Vp9),
            _ => None,
        }
    }
}

/// Output pixel format for decoded frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// NV12: Y plane + interleaved UV (most efficient for NVDEC)
    #[default]
    Nv12,
    /// Planar YUV420
    Yuv420,
    /// BGRA 8-bit (for direct display/processing)
    Bgra,
    /// RGBA 8-bit
    Rgba,
}

/// Configuration for video decoder.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Video codec to decode
    pub codec: DecodeCodec,
    /// Output pixel format
    pub output_format: OutputFormat,
    /// Maximum width (for decoder allocation)
    pub max_width: u32,
    /// Maximum height (for decoder allocation)
    pub max_height: u32,
    /// Number of decode surfaces (affects memory usage)
    pub num_surfaces: u32,
    /// GPU device index (0 = first GPU)
    pub device_index: i32,
    /// Enable low-latency mode (for streaming)
    pub low_latency: bool,
}

impl DecoderConfig {
    /// Create a new decoder configuration.
    pub fn new(codec: DecodeCodec) -> Self {
        Self {
            codec,
            output_format: OutputFormat::default(),
            max_width: 4096,
            max_height: 4096,
            num_surfaces: 20,
            device_index: 0,
            low_latency: false,
        }
    }

    /// Set the output pixel format.
    pub fn with_output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Set maximum resolution for decoder allocation.
    pub fn with_max_resolution(mut self, width: u32, height: u32) -> Self {
        self.max_width = width;
        self.max_height = height;
        self
    }

    /// Set number of decode surfaces.
    pub fn with_surfaces(mut self, num: u32) -> Self {
        self.num_surfaces = num;
        self
    }

    /// Set GPU device index.
    pub fn with_device(mut self, index: i32) -> Self {
        self.device_index = index;
        self
    }

    /// Enable low-latency mode for streaming.
    pub fn with_low_latency(mut self, enabled: bool) -> Self {
        self.low_latency = enabled;
        self
    }
}

/// A decoded video frame in GPU memory.
#[derive(Debug)]
pub struct DecodedFrame {
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Presentation timestamp (in decoder timebase)
    pub pts: i64,
    /// Frame index in decode order
    pub decode_index: u64,
    /// Pixel format of the frame data
    pub format: OutputFormat,
    /// Frame data (in GPU memory if NVDEC, CPU memory otherwise)
    pub data: Vec<u8>,
    /// Stride (bytes per row) for each plane
    pub strides: Vec<usize>,
}

impl DecodedFrame {
    /// Convert to RGBA8 image (for CPU processing).
    pub fn to_rgba_image(&self) -> Result<image::DynamicImage, VideoError> {
        match self.format {
            OutputFormat::Rgba => {
                // Already RGBA, just create image
                let rgba = image::RgbaImage::from_raw(self.width, self.height, self.data.clone())
                    .ok_or_else(|| VideoError::Decoding {
                        message: "Invalid RGBA data dimensions".to_string(),
                    })?;
                Ok(image::DynamicImage::ImageRgba8(rgba))
            }
            OutputFormat::Bgra => {
                // Convert BGRA to RGBA
                let mut rgba_data = self.data.clone();
                for chunk in rgba_data.chunks_exact_mut(4) {
                    chunk.swap(0, 2); // Swap B and R
                }
                let rgba = image::RgbaImage::from_raw(self.width, self.height, rgba_data)
                    .ok_or_else(|| VideoError::Decoding {
                        message: "Invalid BGRA data dimensions".to_string(),
                    })?;
                Ok(image::DynamicImage::ImageRgba8(rgba))
            }
            OutputFormat::Nv12 | OutputFormat::Yuv420 => {
                // YUV to RGB conversion
                self.yuv_to_rgba()
            }
        }
    }

    /// Convert YUV frame to RGBA.
    fn yuv_to_rgba(&self) -> Result<image::DynamicImage, VideoError> {
        let width = self.width as usize;
        let height = self.height as usize;

        // For NV12: Y plane is first, then interleaved UV
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2) * 2; // Interleaved UV

        if self.data.len() < y_size + uv_size {
            return Err(VideoError::Decoding {
                message: format!(
                    "Invalid YUV data size: {} < {}",
                    self.data.len(),
                    y_size + uv_size
                ),
            });
        }

        let y_plane = &self.data[0..y_size];
        let uv_plane = &self.data[y_size..];

        let mut rgba = vec![0u8; width * height * 4];

        // BT.709 YUV to RGB conversion
        for y in 0..height {
            for x in 0..width {
                let y_val = y_plane[y * width + x] as f32;
                let uv_idx = (y / 2) * width + (x / 2) * 2;
                let u_val = uv_plane[uv_idx] as f32 - 128.0;
                let v_val = uv_plane[uv_idx + 1] as f32 - 128.0;

                // BT.709 coefficients
                let r = (y_val + 1.5748 * v_val).clamp(0.0, 255.0) as u8;
                let g = (y_val - 0.1873 * u_val - 0.4681 * v_val).clamp(0.0, 255.0) as u8;
                let b = (y_val + 1.8556 * u_val).clamp(0.0, 255.0) as u8;

                let rgba_idx = (y * width + x) * 4;
                rgba[rgba_idx] = r;
                rgba[rgba_idx + 1] = g;
                rgba[rgba_idx + 2] = b;
                rgba[rgba_idx + 3] = 255;
            }
        }

        let img = image::RgbaImage::from_raw(self.width, self.height, rgba).ok_or_else(|| {
            VideoError::Decoding {
                message: "Failed to create RGBA image from YUV".to_string(),
            }
        })?;

        Ok(image::DynamicImage::ImageRgba8(img))
    }
}

/// Trait for video decoders.
pub trait VideoDecoder {
    /// Decode a video file and return an iterator over frames.
    fn decode_file(&mut self, path: &str) -> Result<(), VideoError>;

    /// Feed raw bitstream data to the decoder.
    fn decode_packet(&mut self, data: &[u8], pts: i64) -> Result<(), VideoError>;

    /// Flush the decoder (call after all packets are sent).
    fn flush(&mut self) -> Result<(), VideoError>;

    /// Get the next decoded frame.
    fn next_frame(&mut self) -> Result<Option<DecodedFrame>, VideoError>;

    /// Get decoder capabilities for a codec.
    fn get_capabilities(&self, codec: DecodeCodec) -> Result<DecoderCapabilities, VideoError>;
}

/// Decoder capabilities for a specific codec.
#[derive(Debug, Clone)]
pub struct DecoderCapabilities {
    /// Whether this codec is supported
    pub supported: bool,
    /// Maximum supported width
    pub max_width: u32,
    /// Maximum supported height
    pub max_height: u32,
    /// Maximum bit depth (8, 10, 12)
    pub max_bit_depth: u32,
    /// Number of NVDEC engines on this GPU
    pub num_engines: u32,
}

/// Decoder backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderBackend {
    /// NVIDIA NVDEC hardware decoder
    Nvdec,
    /// dav1d software decoder (AV1 only)
    Dav1d,
    /// No decoder available
    None,
}

/// Get the best available decoder backend for a codec.
#[cfg(feature = "video-decode")]
pub fn best_decoder_for(codec: DecodeCodec) -> DecoderBackend {
    // Try NVDEC (fastest)
    if NvDecoder::is_available() {
        if let Ok(caps) = NvDecoder::query_capabilities(codec, 0) {
            if caps.supported {
                return DecoderBackend::Nvdec;
            }
        }
    }

    // Fall back to software decoder for AV1
    #[cfg(feature = "video-decode-sw")]
    if codec == DecodeCodec::Av1 && Dav1dDecoder::is_available() {
        return DecoderBackend::Dav1d;
    }

    DecoderBackend::None
}

/// Get the best available decoder backend for a codec (software only version).
#[cfg(all(feature = "video-decode-sw", not(feature = "video-decode")))]
pub fn best_decoder_for(codec: DecodeCodec) -> DecoderBackend {
    // Software decoder for AV1
    if codec == DecodeCodec::Av1 && Dav1dDecoder::is_available() {
        return DecoderBackend::Dav1d;
    }

    DecoderBackend::None
}

/// Decode an IVF file and return frames.
/// Uses NVDEC for GPU-accelerated decoding.
#[cfg(feature = "video-decode")]
pub fn decode_ivf<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<DecodedFrame>, VideoError> {
    use std::fs::File;
    use std::io::{BufReader, Read};

    let path = path.as_ref();

    // Read IVF header to detect codec
    let file = File::open(path).map_err(|e| VideoError::Io {
        message: format!("Failed to open file: {}", e),
    })?;
    let mut reader = BufReader::new(file);

    let mut header = [0u8; 32];
    reader.read_exact(&mut header).map_err(|e| VideoError::Io {
        message: format!("Failed to read IVF header: {}", e),
    })?;

    if &header[0..4] != b"DKIF" {
        return Err(VideoError::Decoding {
            message: "Invalid IVF file: bad signature".to_string(),
        });
    }

    let fourcc = std::str::from_utf8(&header[8..12]).unwrap_or("????");
    let codec = DecodeCodec::from_fourcc(fourcc).ok_or_else(|| VideoError::Decoding {
        message: format!("Unknown codec: {}", fourcc),
    })?;

    let backend = best_decoder_for(codec);

    match backend {
        DecoderBackend::Nvdec => {
            let config = DecoderConfig::new(codec);
            let mut decoder = NvDecoder::new(config)?;
            decoder.decode_ivf_file(path)
        }
        #[cfg(feature = "video-decode-sw")]
        DecoderBackend::Dav1d => {
            let mut decoder = Dav1dDecoder::new()?;
            decoder.decode_ivf_file(path)
        }
        #[cfg(not(feature = "video-decode-sw"))]
        DecoderBackend::Dav1d => Err(VideoError::Decoding {
            message: "dav1d software decoder not available. Enable 'video-decode-sw' feature.".to_string(),
        }),
        DecoderBackend::None => Err(VideoError::Decoding {
            message: format!("No decoder available for {:?}. NVDEC or dav1d required.", codec),
        }),
    }
}

/// Extract frames from a video file to images.
#[cfg(feature = "video-decode")]
pub fn extract_frames<P: AsRef<std::path::Path>>(
    input: P,
    output_dir: P,
    format: &str,
) -> Result<Vec<std::path::PathBuf>, VideoError> {
    let input = input.as_ref();
    let output_dir = output_dir.as_ref();

    // Create output directory if needed
    std::fs::create_dir_all(output_dir).map_err(|e| VideoError::Io {
        message: format!("Failed to create output directory: {}", e),
    })?;

    // Decode video
    let frames = decode_ivf(input)?;

    let mut output_paths = Vec::with_capacity(frames.len());

    for (idx, frame) in frames.iter().enumerate() {
        let output_path = output_dir.join(format!("frame_{:06}.{}", idx, format));

        // Convert to image and save
        let img = frame.to_rgba_image()?;
        img.save(&output_path).map_err(|e| VideoError::Io {
            message: format!("Failed to save frame: {}", e),
        })?;

        output_paths.push(output_path);
    }

    Ok(output_paths)
}
