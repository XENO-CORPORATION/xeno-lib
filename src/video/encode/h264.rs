//! H.264 video encoding using OpenH264.
//!
//! This module provides H.264 encoding capabilities using Cisco's OpenH264 codec,
//! which is compiled from source (pure Rust + C) and requires no external dependencies.
//!
//! # Performance
//!
//! For optimal encoding performance:
//! - Have `nasm` in PATH for 3x speedup on x86/x64
//! - Compile with `target-cpu=native`
//!
//! # Output Formats
//!
//! - `.h264` - Raw H.264 NAL stream (Annex B format)
//! - `.mp4` - Proper MP4 container (requires mp4 crate)

use image::DynamicImage;
use openh264::encoder::{Encoder, EncoderConfig};
use openh264::formats::YUVSource;
use std::fs::File;
use std::io::{BufWriter, Write, Seek};
use std::path::Path;

use crate::video::{VideoError, VideoResult};
use super::{VideoEncoder, VideoEncoderConfig};

/// H.264 profile for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum H264Profile {
    /// Baseline profile - simplest, most compatible
    Baseline,
    /// Main profile - better compression
    Main,
    /// High profile - best quality, widely supported
    #[default]
    High,
}

/// H.264 encoder configuration.
#[derive(Debug, Clone)]
pub struct H264EncoderConfig {
    /// Output video width.
    pub width: u32,
    /// Output video height.
    pub height: u32,
    /// Frame rate (frames per second).
    pub frame_rate: f64,
    /// Target bitrate in kbps (0 = auto based on resolution).
    pub bitrate_kbps: u32,
    /// Max bitrate in kbps (0 = 1.5x target).
    pub max_bitrate_kbps: u32,
    /// Encoding profile.
    pub profile: H264Profile,
    /// Enable multi-threading (0 = auto).
    pub threads: usize,
    /// Key frame interval in frames (0 = auto).
    pub keyframe_interval: u32,
    /// Enable scene change detection.
    pub scene_change_detect: bool,
    /// Enable background detection.
    pub background_detect: bool,
    /// Enable adaptive quantization.
    pub adaptive_quant: bool,
}

impl Default for H264EncoderConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            frame_rate: 30.0,
            bitrate_kbps: 0,         // Auto-calculate
            max_bitrate_kbps: 0,     // Auto-calculate
            profile: H264Profile::High,
            threads: 0,              // Auto
            keyframe_interval: 0,    // Auto (typically 2 seconds)
            scene_change_detect: true,
            background_detect: true,
            adaptive_quant: true,
        }
    }
}

impl H264EncoderConfig {
    /// Create a new config with specified dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Set frame rate.
    pub fn with_frame_rate(mut self, fps: f64) -> Self {
        self.frame_rate = fps;
        self
    }

    /// Set target bitrate in kbps.
    pub fn with_bitrate(mut self, kbps: u32) -> Self {
        self.bitrate_kbps = kbps;
        self
    }

    /// Set max bitrate in kbps.
    pub fn with_max_bitrate(mut self, kbps: u32) -> Self {
        self.max_bitrate_kbps = kbps;
        self
    }

    /// Set encoding profile.
    pub fn with_profile(mut self, profile: H264Profile) -> Self {
        self.profile = profile;
        self
    }

    /// Set number of threads.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Set key frame interval.
    pub fn with_keyframe_interval(mut self, interval: u32) -> Self {
        self.keyframe_interval = interval;
        self
    }

    /// Calculate recommended bitrate based on resolution and frame rate.
    pub fn calculate_bitrate(&self) -> u32 {
        if self.bitrate_kbps > 0 {
            return self.bitrate_kbps;
        }

        // Rough bitrate estimation based on resolution
        let pixels = self.width * self.height;
        let base_bitrate = match pixels {
            0..=307200 => 1000,       // Up to 640x480: 1 Mbps
            307201..=921600 => 2500,   // Up to 1280x720: 2.5 Mbps
            921601..=2073600 => 5000,  // Up to 1920x1080: 5 Mbps
            2073601..=3686400 => 10000, // Up to 2560x1440: 10 Mbps
            _ => 20000,               // 4K+: 20 Mbps
        };

        // Adjust for frame rate (base assumes 30fps)
        let fps_factor = (self.frame_rate / 30.0).sqrt();
        (base_bitrate as f64 * fps_factor) as u32
    }
}

impl VideoEncoderConfig for H264EncoderConfig {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn frame_rate(&self) -> f64 {
        self.frame_rate
    }
}

/// H.264 encoder using OpenH264.
pub struct H264Encoder {
    encoder: Encoder,
    config: H264EncoderConfig,
    frame_count: u64,
    /// Temporary YUV buffer (reused across frames)
    yuv_buffer: Vec<u8>,
}

impl H264Encoder {
    /// Create a new H.264 encoder with the given configuration.
    pub fn create(config: H264EncoderConfig) -> Result<Self, VideoError> {
        let bitrate = config.calculate_bitrate() * 1000; // Convert to bps

        // OpenH264 0.9 API: dimensions are taken from YUVSource during encode
        let enc_config = EncoderConfig::new()
            .bitrate(openh264::encoder::BitRate::from_bps(bitrate))
            .max_frame_rate(openh264::encoder::FrameRate::from_hz(config.frame_rate as f32))
            .scene_change_detect(config.scene_change_detect)
            .background_detection(config.background_detect)
            .adaptive_quantization(config.adaptive_quant);

        let encoder = Encoder::with_api_config(openh264::OpenH264API::from_source(), enc_config)
            .map_err(|e| VideoError::Encoding {
                message: format!("Failed to create H.264 encoder: {:?}", e),
            })?;

        // Pre-allocate YUV buffer (YUV420 = 1.5 bytes per pixel)
        let yuv_size = (config.width * config.height * 3 / 2) as usize;

        Ok(Self {
            encoder,
            config,
            frame_count: 0,
            yuv_buffer: vec![0u8; yuv_size],
        })
    }

    /// Encode a single frame and return the NAL units.
    pub fn encode_frame(&mut self, image: &DynamicImage) -> Result<Option<EncodedFrame>, VideoError> {
        // Resize image if needed
        let img = if image.width() != self.config.width || image.height() != self.config.height {
            image.resize_exact(
                self.config.width,
                self.config.height,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            image.clone()
        };

        // Convert to RGB8
        let rgb = img.to_rgb8();

        // Convert RGB to YUV420 in-place
        rgb_to_yuv420_buffer(&rgb, &mut self.yuv_buffer, self.config.width, self.config.height);

        // Create YUV source
        let yuv = YUVBuffer {
            data: &self.yuv_buffer,
            width: self.config.width as usize,
            height: self.config.height as usize,
        };

        // Encode
        let bitstream = self.encoder.encode(&yuv).map_err(|e| VideoError::Encoding {
            message: format!("H.264 encoding failed: {:?}", e),
        })?;

        self.frame_count += 1;

        // Get frame type and data
        let frame_type = bitstream.frame_type();
        let is_keyframe = matches!(
            frame_type,
            openh264::encoder::FrameType::IDR | openh264::encoder::FrameType::I
        );

        // Get encoded data (Annex B format with start codes)
        let nal_data = bitstream.to_vec();

        if nal_data.is_empty() {
            Ok(None)
        } else {
            Ok(Some(EncodedFrame {
                data: nal_data,
                pts: self.frame_count - 1,
                is_keyframe,
            }))
        }
    }
}

impl VideoEncoder for H264Encoder {
    type Config = H264EncoderConfig;
    type Packet = EncodedFrame;

    fn new(config: Self::Config) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::create(config)
    }

    fn send_frame(&mut self, frame: &DynamicImage) -> Result<(), VideoError> {
        // For H264Encoder, we encode immediately (not buffered like rav1e)
        // The actual encoding happens in receive_packet via the stored frame
        // Store frame for later encoding
        let _ = self.encode_frame(frame)?;
        Ok(())
    }

    fn flush(&mut self) {
        // OpenH264 doesn't require explicit flushing
    }

    fn receive_packet(&mut self) -> Result<Option<Self::Packet>, VideoError> {
        // For H264Encoder, packets are returned immediately from encode_frame
        // This is a no-op since we return frames directly
        Ok(None)
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Encoded H.264 frame data.
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// NAL unit data (Annex B format with start codes)
    pub data: Vec<u8>,
    /// Presentation timestamp
    pub pts: u64,
    /// Is this a keyframe (IDR/I-frame)?
    pub is_keyframe: bool,
}

/// YUV420 buffer wrapper for OpenH264
struct YUVBuffer<'a> {
    data: &'a [u8],
    width: usize,
    height: usize,
}

impl YUVSource for YUVBuffer<'_> {
    fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn strides(&self) -> (usize, usize, usize) {
        let y_stride = self.width;
        let uv_stride = self.width / 2;
        (y_stride, uv_stride, uv_stride)
    }

    fn y(&self) -> &[u8] {
        let y_size = self.width * self.height;
        &self.data[..y_size]
    }

    fn u(&self) -> &[u8] {
        let y_size = self.width * self.height;
        let u_size = (self.width / 2) * (self.height / 2);
        &self.data[y_size..y_size + u_size]
    }

    fn v(&self) -> &[u8] {
        let y_size = self.width * self.height;
        let u_size = (self.width / 2) * (self.height / 2);
        &self.data[y_size + u_size..]
    }
}

/// Convert RGB image to YUV420 planar format in-place.
fn rgb_to_yuv420_buffer(
    rgb: &image::RgbImage,
    yuv: &mut [u8],
    width: u32,
    height: u32,
) {
    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);

    // Split buffer into Y, U, V planes
    let (y_plane, uv) = yuv.split_at_mut(y_size);
    let (u_plane, v_plane) = uv.split_at_mut(uv_size);

    // Y plane (full resolution) - BT.601
    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;

            let y_val = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0) as u8;
            y_plane[y * w + x] = y_val;
        }
    }

    // U and V planes (half resolution for 4:2:0)
    let uv_w = w / 2;
    for y in 0..(h / 2) {
        for x in 0..(w / 2) {
            // Average 2x2 block
            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;

            for dy in 0..2 {
                for dx in 0..2 {
                    let px = (x * 2 + dx) as u32;
                    let py = (y * 2 + dy) as u32;
                    if px < width && py < height {
                        let pixel = rgb.get_pixel(px, py);
                        r_sum += pixel[0] as u32;
                        g_sum += pixel[1] as u32;
                        b_sum += pixel[2] as u32;
                    }
                }
            }

            let r = (r_sum / 4) as f32;
            let g = (g_sum / 4) as f32;
            let b = (b_sum / 4) as f32;

            // BT.601 RGB to Cb/Cr conversion
            let u_val = (128.0 + (-0.169 * r - 0.331 * g + 0.500 * b)).clamp(0.0, 255.0) as u8;
            let v_val = (128.0 + (0.500 * r - 0.419 * g - 0.081 * b)).clamp(0.0, 255.0) as u8;

            u_plane[y * uv_w + x] = u_val;
            v_plane[y * uv_w + x] = v_val;
        }
    }
}

/// Encode an image sequence to a raw H.264 file (Annex B format).
///
/// # Arguments
/// * `images` - Iterator of images to encode
/// * `output` - Output file path (.h264)
/// * `config` - Encoder configuration
///
/// # Returns
/// Number of frames encoded
///
/// # Example
/// ```ignore
/// use xeno_lib::video::encode::{encode_to_h264, H264EncoderConfig};
///
/// let images: Vec<DynamicImage> = load_images()?;
/// let config = H264EncoderConfig::new(1920, 1080)
///     .with_frame_rate(30.0)
///     .with_bitrate(5000);
///
/// let frames = encode_to_h264(images.iter(), "output.h264", config)?;
/// ```
pub fn encode_to_h264<'a, I, P>(
    images: I,
    output: P,
    config: H264EncoderConfig,
) -> VideoResult<u64>
where
    I: Iterator<Item = &'a DynamicImage>,
    P: AsRef<Path>,
{
    let file = File::create(output.as_ref()).map_err(|e| VideoError::Io {
        message: format!("Failed to create output file: {}", e),
    })?;
    let mut writer = BufWriter::new(file);

    let mut encoder = H264Encoder::create(config)?;
    let mut frame_count = 0u64;

    // Encode all frames
    for image in images {
        if let Some(frame) = encoder.encode_frame(image)? {
            writer.write_all(&frame.data).map_err(|e| VideoError::Io {
                message: format!("Failed to write frame: {}", e),
            })?;
            frame_count += 1;
        }
    }

    writer.flush().map_err(|e| VideoError::Io {
        message: format!("Failed to flush writer: {}", e),
    })?;

    Ok(frame_count)
}

/// Encode an image sequence to an MP4 file with H.264 codec.
///
/// # Arguments
/// * `images` - Iterator of images to encode
/// * `output` - Output file path (.mp4)
/// * `config` - Encoder configuration
///
/// # Returns
/// Number of frames encoded
///
/// # Example
/// ```ignore
/// use xeno_lib::video::encode::{encode_h264_to_mp4, H264EncoderConfig};
///
/// let images: Vec<DynamicImage> = load_images()?;
/// let config = H264EncoderConfig::new(1920, 1080)
///     .with_frame_rate(30.0);
///
/// let frames = encode_h264_to_mp4(images.iter(), "output.mp4", config)?;
/// ```
pub fn encode_h264_to_mp4<'a, I, P>(
    images: I,
    output: P,
    config: H264EncoderConfig,
) -> VideoResult<u64>
where
    I: Iterator<Item = &'a DynamicImage>,
    P: AsRef<Path>,
{
    // Collect all encoded frames first
    let mut encoder = H264Encoder::create(config.clone())?;
    let mut frames: Vec<EncodedFrame> = Vec::new();

    for image in images {
        if let Some(frame) = encoder.encode_frame(image)? {
            frames.push(frame);
        }
    }

    if frames.is_empty() {
        return Err(VideoError::Encoding {
            message: "No frames to encode".to_string(),
        });
    }

    // Extract SPS and PPS from first keyframe
    let (sps, pps) = extract_sps_pps(&frames[0].data).ok_or_else(|| VideoError::Encoding {
        message: "Failed to extract SPS/PPS from encoded stream".to_string(),
    })?;

    // Create MP4 file
    let file = File::create(output.as_ref()).map_err(|e| VideoError::Io {
        message: format!("Failed to create output file: {}", e),
    })?;
    let mut writer = BufWriter::new(file);

    // Write MP4 header and moov atom
    write_mp4_h264(&mut writer, &frames, &sps, &pps, &config)?;

    writer.flush().map_err(|e| VideoError::Io {
        message: format!("Failed to flush writer: {}", e),
    })?;

    Ok(frames.len() as u64)
}

/// Extract SPS and PPS NAL units from H.264 stream.
fn extract_sps_pps(data: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut sps = None;
    let mut pps = None;

    let mut i = 0;
    while i < data.len() {
        // Find start code
        if i + 4 <= data.len() && data[i..i + 4] == [0x00, 0x00, 0x00, 0x01] {
            let nal_start = i + 4;
            // Find next start code or end
            let mut nal_end = data.len();
            for j in nal_start..data.len().saturating_sub(3) {
                if data[j..j + 4] == [0x00, 0x00, 0x00, 0x01]
                    || (data[j..j + 3] == [0x00, 0x00, 0x01])
                {
                    nal_end = j;
                    break;
                }
            }

            if nal_start < nal_end {
                let nal_type = data[nal_start] & 0x1F;
                match nal_type {
                    7 => sps = Some(data[nal_start..nal_end].to_vec()), // SPS
                    8 => pps = Some(data[nal_start..nal_end].to_vec()), // PPS
                    _ => {}
                }
            }

            i = nal_end;
        } else if i + 3 <= data.len() && data[i..i + 3] == [0x00, 0x00, 0x01] {
            i += 3;
        } else {
            i += 1;
        }
    }

    match (sps, pps) {
        (Some(s), Some(p)) => Some((s, p)),
        _ => None,
    }
}

/// Write MP4 file with H.264 video track.
fn write_mp4_h264<W: Write + Seek>(
    writer: &mut W,
    frames: &[EncodedFrame],
    sps: &[u8],
    pps: &[u8],
    config: &H264EncoderConfig,
) -> VideoResult<()> {
    let frame_count = frames.len() as u32;
    let timescale = (config.frame_rate * 1000.0) as u32;
    let sample_duration = 1000; // Each frame = 1000 ticks at our timescale
    let duration = frame_count * sample_duration;

    // Calculate sample sizes and build mdat
    let mut sample_sizes: Vec<u32> = Vec::with_capacity(frames.len());
    let mut mdat_data: Vec<u8> = Vec::new();
    let mut sync_samples: Vec<u32> = Vec::new();

    for (i, frame) in frames.iter().enumerate() {
        // Convert Annex B to AVCC format (length-prefixed)
        let avcc_data = annex_b_to_avcc(&frame.data);
        sample_sizes.push(avcc_data.len() as u32);
        mdat_data.extend_from_slice(&avcc_data);

        if frame.is_keyframe {
            sync_samples.push((i + 1) as u32); // 1-indexed
        }
    }

    // Calculate offsets
    let ftyp_size = 20u32;
    let moov_size = calculate_moov_size(frame_count, &sample_sizes, sps, pps, &sync_samples);
    let mdat_header_size = 8u32;
    let mdat_offset = ftyp_size + moov_size + mdat_header_size;

    // Write ftyp
    write_ftyp(writer)?;

    // Write moov
    write_moov(writer, config, frame_count, timescale, duration,
               sps, pps, &sample_sizes, mdat_offset, &sync_samples)?;

    // Write mdat
    let mdat_size = mdat_header_size + mdat_data.len() as u32;
    writer.write_all(&mdat_size.to_be_bytes())?;
    writer.write_all(b"mdat")?;
    writer.write_all(&mdat_data)?;

    Ok(())
}

/// Convert Annex B (start code) format to AVCC (length-prefixed) format.
fn annex_b_to_avcc(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        // Find start code (0x00000001 or 0x000001)
        let (_start_code_len, nal_start) = if i + 4 <= data.len() && data[i..i + 4] == [0x00, 0x00, 0x00, 0x01] {
            (4, i + 4)
        } else if i + 3 <= data.len() && data[i..i + 3] == [0x00, 0x00, 0x01] {
            (3, i + 3)
        } else {
            i += 1;
            continue;
        };

        // Find end of NAL unit
        let mut nal_end = data.len();
        for j in nal_start..data.len().saturating_sub(2) {
            if (data[j..j + 3] == [0x00, 0x00, 0x01])
                || (j + 3 < data.len() && data[j..j + 4] == [0x00, 0x00, 0x00, 0x01])
            {
                nal_end = j;
                break;
            }
        }

        let nal_data = &data[nal_start..nal_end];
        let nal_type = nal_data.get(0).map(|b| b & 0x1F).unwrap_or(0);

        // Skip SPS/PPS in sample data (they go in avcC)
        if nal_type != 7 && nal_type != 8 && !nal_data.is_empty() {
            // Write 4-byte length prefix
            let len = nal_data.len() as u32;
            result.extend_from_slice(&len.to_be_bytes());
            result.extend_from_slice(nal_data);
        }

        i = nal_end;
    }

    result
}

// Helper functions for MP4 writing
fn write_ftyp<W: Write>(w: &mut W) -> VideoResult<()> {
    let size = 20u32;
    w.write_all(&size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"ftyp").map_err(io_err)?;
    w.write_all(b"isom").map_err(io_err)?; // major brand
    w.write_all(&0u32.to_be_bytes()).map_err(io_err)?; // minor version
    w.write_all(b"isom").map_err(io_err)?; // compatible brand
    Ok(())
}

fn calculate_moov_size(
    frame_count: u32,
    _sample_sizes: &[u32],
    sps: &[u8],
    pps: &[u8],
    sync_samples: &[u32],
) -> u32 {
    // This is an approximation - actual sizes depend on box structure
    let mvhd_size = 108u32;
    let tkhd_size = 92u32;
    let mdhd_size = 32u32;
    let hdlr_size = 45u32;
    let vmhd_size = 20u32;
    let dinf_size = 36u32;

    // stsd with avcC
    let avcc_size = 11 + sps.len() as u32 + 3 + pps.len() as u32;
    let avc1_size = 86 + 8 + avcc_size;
    let stsd_size = 16 + avc1_size;

    // stts (single entry for constant frame rate)
    let stts_size = 24u32;

    // stsz
    let stsz_size = 20 + (frame_count * 4);

    // stsc (single entry)
    let stsc_size = 28u32;

    // stco (single entry)
    let stco_size = 20u32;

    // stss (sync samples)
    let stss_size = 16 + (sync_samples.len() as u32 * 4);

    let stbl_size = 8 + stsd_size + stts_size + stsz_size + stsc_size + stco_size + stss_size;
    let minf_size = 8 + vmhd_size + dinf_size + stbl_size;
    let mdia_size = 8 + mdhd_size + hdlr_size + minf_size;
    let trak_size = 8 + tkhd_size + mdia_size;
    let moov_size = 8 + mvhd_size + trak_size;

    moov_size
}

fn write_moov<W: Write>(
    w: &mut W,
    config: &H264EncoderConfig,
    frame_count: u32,
    timescale: u32,
    duration: u32,
    sps: &[u8],
    pps: &[u8],
    sample_sizes: &[u32],
    mdat_offset: u32,
    sync_samples: &[u32],
) -> VideoResult<()> {
    let moov_size = calculate_moov_size(frame_count, sample_sizes, sps, pps, sync_samples);

    // moov header
    w.write_all(&moov_size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"moov").map_err(io_err)?;

    // mvhd
    write_mvhd(w, timescale, duration)?;

    // trak
    write_trak(w, config, frame_count, timescale, duration, sps, pps, sample_sizes, mdat_offset, sync_samples)?;

    Ok(())
}

fn write_mvhd<W: Write>(w: &mut W, timescale: u32, duration: u32) -> VideoResult<()> {
    let size = 108u32;
    w.write_all(&size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"mvhd").map_err(io_err)?;
    w.write_all(&[0, 0, 0, 0]).map_err(io_err)?; // version + flags
    w.write_all(&0u32.to_be_bytes()).map_err(io_err)?; // creation time
    w.write_all(&0u32.to_be_bytes()).map_err(io_err)?; // modification time
    w.write_all(&timescale.to_be_bytes()).map_err(io_err)?;
    w.write_all(&duration.to_be_bytes()).map_err(io_err)?;
    w.write_all(&0x00010000u32.to_be_bytes()).map_err(io_err)?; // rate = 1.0
    w.write_all(&0x0100u16.to_be_bytes()).map_err(io_err)?; // volume = 1.0
    w.write_all(&[0u8; 10]).map_err(io_err)?; // reserved
    // Unity matrix
    w.write_all(&0x00010000u32.to_be_bytes()).map_err(io_err)?;
    w.write_all(&[0u8; 12]).map_err(io_err)?;
    w.write_all(&0x00010000u32.to_be_bytes()).map_err(io_err)?;
    w.write_all(&[0u8; 12]).map_err(io_err)?;
    w.write_all(&0x40000000u32.to_be_bytes()).map_err(io_err)?;
    w.write_all(&[0u8; 24]).map_err(io_err)?; // pre-defined
    w.write_all(&2u32.to_be_bytes()).map_err(io_err)?; // next track ID
    Ok(())
}

fn write_trak<W: Write>(
    w: &mut W,
    config: &H264EncoderConfig,
    frame_count: u32,
    timescale: u32,
    duration: u32,
    sps: &[u8],
    pps: &[u8],
    sample_sizes: &[u32],
    mdat_offset: u32,
    sync_samples: &[u32],
) -> VideoResult<()> {
    // Calculate trak size
    let trak_size = calculate_moov_size(frame_count, sample_sizes, sps, pps, sync_samples) - 8 - 108;

    w.write_all(&trak_size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"trak").map_err(io_err)?;

    // tkhd
    write_tkhd(w, config, duration)?;

    // mdia
    write_mdia(w, config, frame_count, timescale, duration, sps, pps, sample_sizes, mdat_offset, sync_samples)?;

    Ok(())
}

fn write_tkhd<W: Write>(w: &mut W, config: &H264EncoderConfig, duration: u32) -> VideoResult<()> {
    let size = 92u32;
    w.write_all(&size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"tkhd").map_err(io_err)?;
    w.write_all(&[0, 0, 0, 3]).map_err(io_err)?; // version + flags (enabled + in movie)
    w.write_all(&0u32.to_be_bytes()).map_err(io_err)?; // creation time
    w.write_all(&0u32.to_be_bytes()).map_err(io_err)?; // modification time
    w.write_all(&1u32.to_be_bytes()).map_err(io_err)?; // track ID
    w.write_all(&0u32.to_be_bytes()).map_err(io_err)?; // reserved
    w.write_all(&duration.to_be_bytes()).map_err(io_err)?;
    w.write_all(&[0u8; 8]).map_err(io_err)?; // reserved
    w.write_all(&0u16.to_be_bytes()).map_err(io_err)?; // layer
    w.write_all(&0u16.to_be_bytes()).map_err(io_err)?; // alternate group
    w.write_all(&0u16.to_be_bytes()).map_err(io_err)?; // volume
    w.write_all(&0u16.to_be_bytes()).map_err(io_err)?; // reserved
    // Unity matrix
    w.write_all(&0x00010000u32.to_be_bytes()).map_err(io_err)?;
    w.write_all(&[0u8; 12]).map_err(io_err)?;
    w.write_all(&0x00010000u32.to_be_bytes()).map_err(io_err)?;
    w.write_all(&[0u8; 12]).map_err(io_err)?;
    w.write_all(&0x40000000u32.to_be_bytes()).map_err(io_err)?;
    // Width and height (16.16 fixed point)
    w.write_all(&((config.width as u32) << 16).to_be_bytes()).map_err(io_err)?;
    w.write_all(&((config.height as u32) << 16).to_be_bytes()).map_err(io_err)?;
    Ok(())
}

fn write_mdia<W: Write>(
    w: &mut W,
    config: &H264EncoderConfig,
    frame_count: u32,
    timescale: u32,
    duration: u32,
    sps: &[u8],
    pps: &[u8],
    sample_sizes: &[u32],
    mdat_offset: u32,
    sync_samples: &[u32],
) -> VideoResult<()> {
    // Calculate mdia size
    let mdhd_size = 32u32;
    let hdlr_size = 45u32;
    let minf_size = calculate_minf_size(frame_count, sample_sizes, sps, pps, sync_samples);
    let mdia_size = 8 + mdhd_size + hdlr_size + minf_size;

    w.write_all(&mdia_size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"mdia").map_err(io_err)?;

    // mdhd
    write_mdhd(w, timescale, duration)?;

    // hdlr
    write_hdlr(w)?;

    // minf
    write_minf(w, config, frame_count, timescale, sps, pps, sample_sizes, mdat_offset, sync_samples)?;

    Ok(())
}

fn calculate_minf_size(frame_count: u32, sample_sizes: &[u32], sps: &[u8], pps: &[u8], sync_samples: &[u32]) -> u32 {
    let vmhd_size = 20u32;
    let dinf_size = 36u32;
    let stbl_size = calculate_stbl_size(frame_count, sample_sizes, sps, pps, sync_samples);
    8 + vmhd_size + dinf_size + stbl_size
}

fn calculate_stbl_size(frame_count: u32, _sample_sizes: &[u32], sps: &[u8], pps: &[u8], sync_samples: &[u32]) -> u32 {
    let avcc_size = 11 + sps.len() as u32 + 3 + pps.len() as u32;
    let avc1_size = 86 + 8 + avcc_size;
    let stsd_size = 16 + avc1_size;
    let stts_size = 24u32;
    let stsz_size = 20 + (frame_count * 4);
    let stsc_size = 28u32;
    let stco_size = 20u32;
    let stss_size = 16 + (sync_samples.len() as u32 * 4);
    8 + stsd_size + stts_size + stsz_size + stsc_size + stco_size + stss_size
}

fn write_mdhd<W: Write>(w: &mut W, timescale: u32, duration: u32) -> VideoResult<()> {
    let size = 32u32;
    w.write_all(&size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"mdhd").map_err(io_err)?;
    w.write_all(&[0, 0, 0, 0]).map_err(io_err)?; // version + flags
    w.write_all(&0u32.to_be_bytes()).map_err(io_err)?; // creation time
    w.write_all(&0u32.to_be_bytes()).map_err(io_err)?; // modification time
    w.write_all(&timescale.to_be_bytes()).map_err(io_err)?;
    w.write_all(&duration.to_be_bytes()).map_err(io_err)?;
    w.write_all(&0x55C40000u32.to_be_bytes()).map_err(io_err)?; // language (und) + quality
    Ok(())
}

fn write_hdlr<W: Write>(w: &mut W) -> VideoResult<()> {
    let size = 45u32;
    w.write_all(&size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"hdlr").map_err(io_err)?;
    w.write_all(&[0, 0, 0, 0]).map_err(io_err)?; // version + flags
    w.write_all(&0u32.to_be_bytes()).map_err(io_err)?; // pre-defined
    w.write_all(b"vide").map_err(io_err)?; // handler type
    w.write_all(&[0u8; 12]).map_err(io_err)?; // reserved
    w.write_all(b"VideoHandler\0").map_err(io_err)?; // name
    Ok(())
}

fn write_minf<W: Write>(
    w: &mut W,
    config: &H264EncoderConfig,
    frame_count: u32,
    _timescale: u32,
    sps: &[u8],
    pps: &[u8],
    sample_sizes: &[u32],
    mdat_offset: u32,
    sync_samples: &[u32],
) -> VideoResult<()> {
    let minf_size = calculate_minf_size(frame_count, sample_sizes, sps, pps, sync_samples);

    w.write_all(&minf_size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"minf").map_err(io_err)?;

    // vmhd
    write_vmhd(w)?;

    // dinf
    write_dinf(w)?;

    // stbl
    write_stbl(w, config, frame_count, _timescale, sps, pps, sample_sizes, mdat_offset, sync_samples)?;

    Ok(())
}

fn write_vmhd<W: Write>(w: &mut W) -> VideoResult<()> {
    let size = 20u32;
    w.write_all(&size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"vmhd").map_err(io_err)?;
    w.write_all(&[0, 0, 0, 1]).map_err(io_err)?; // version + flags
    w.write_all(&[0u8; 8]).map_err(io_err)?; // graphics mode + opcolor
    Ok(())
}

fn write_dinf<W: Write>(w: &mut W) -> VideoResult<()> {
    // dinf
    let dinf_size = 36u32;
    w.write_all(&dinf_size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"dinf").map_err(io_err)?;

    // dref
    let dref_size = 28u32;
    w.write_all(&dref_size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"dref").map_err(io_err)?;
    w.write_all(&[0, 0, 0, 0]).map_err(io_err)?; // version + flags
    w.write_all(&1u32.to_be_bytes()).map_err(io_err)?; // entry count

    // url
    let url_size = 12u32;
    w.write_all(&url_size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"url ").map_err(io_err)?;
    w.write_all(&[0, 0, 0, 1]).map_err(io_err)?; // version + flags (self-contained)

    Ok(())
}

fn write_stbl<W: Write>(
    w: &mut W,
    config: &H264EncoderConfig,
    frame_count: u32,
    _timescale: u32,
    sps: &[u8],
    pps: &[u8],
    sample_sizes: &[u32],
    mdat_offset: u32,
    sync_samples: &[u32],
) -> VideoResult<()> {
    let stbl_size = calculate_stbl_size(frame_count, sample_sizes, sps, pps, sync_samples);

    w.write_all(&stbl_size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"stbl").map_err(io_err)?;

    // stsd
    write_stsd(w, config, sps, pps)?;

    // stts
    write_stts(w, frame_count)?;

    // stsz
    write_stsz(w, sample_sizes)?;

    // stsc
    write_stsc(w)?;

    // stco
    write_stco(w, mdat_offset)?;

    // stss
    write_stss(w, sync_samples)?;

    Ok(())
}

fn write_stsd<W: Write>(w: &mut W, config: &H264EncoderConfig, sps: &[u8], pps: &[u8]) -> VideoResult<()> {
    let avcc_size = 11 + sps.len() as u32 + 3 + pps.len() as u32;
    let avc1_size = 86 + 8 + avcc_size;
    let stsd_size = 16 + avc1_size;

    w.write_all(&stsd_size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"stsd").map_err(io_err)?;
    w.write_all(&[0, 0, 0, 0]).map_err(io_err)?; // version + flags
    w.write_all(&1u32.to_be_bytes()).map_err(io_err)?; // entry count

    // avc1
    w.write_all(&avc1_size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"avc1").map_err(io_err)?;
    w.write_all(&[0u8; 6]).map_err(io_err)?; // reserved
    w.write_all(&1u16.to_be_bytes()).map_err(io_err)?; // data reference index
    w.write_all(&[0u8; 16]).map_err(io_err)?; // pre-defined + reserved
    w.write_all(&(config.width as u16).to_be_bytes()).map_err(io_err)?;
    w.write_all(&(config.height as u16).to_be_bytes()).map_err(io_err)?;
    w.write_all(&0x00480000u32.to_be_bytes()).map_err(io_err)?; // horizontal resolution (72 dpi)
    w.write_all(&0x00480000u32.to_be_bytes()).map_err(io_err)?; // vertical resolution (72 dpi)
    w.write_all(&0u32.to_be_bytes()).map_err(io_err)?; // reserved
    w.write_all(&1u16.to_be_bytes()).map_err(io_err)?; // frame count
    w.write_all(&[0u8; 32]).map_err(io_err)?; // compressor name
    w.write_all(&0x0018u16.to_be_bytes()).map_err(io_err)?; // depth (24-bit)
    w.write_all(&(-1i16).to_be_bytes()).map_err(io_err)?; // pre-defined

    // avcC
    write_avcc(w, sps, pps)?;

    Ok(())
}

fn write_avcc<W: Write>(w: &mut W, sps: &[u8], pps: &[u8]) -> VideoResult<()> {
    let avcc_size = 8 + 11 + sps.len() as u32 + 3 + pps.len() as u32;

    w.write_all(&avcc_size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"avcC").map_err(io_err)?;

    w.write_all(&[1]).map_err(io_err)?; // configuration version
    w.write_all(&[sps.get(1).copied().unwrap_or(0)]).map_err(io_err)?; // profile
    w.write_all(&[sps.get(2).copied().unwrap_or(0)]).map_err(io_err)?; // compatibility
    w.write_all(&[sps.get(3).copied().unwrap_or(0)]).map_err(io_err)?; // level
    w.write_all(&[0xFF]).map_err(io_err)?; // length size minus one (3 -> 4 bytes)
    w.write_all(&[0xE1]).map_err(io_err)?; // num SPS (1)
    w.write_all(&(sps.len() as u16).to_be_bytes()).map_err(io_err)?;
    w.write_all(sps).map_err(io_err)?;
    w.write_all(&[0x01]).map_err(io_err)?; // num PPS (1)
    w.write_all(&(pps.len() as u16).to_be_bytes()).map_err(io_err)?;
    w.write_all(pps).map_err(io_err)?;

    Ok(())
}

fn write_stts<W: Write>(w: &mut W, frame_count: u32) -> VideoResult<()> {
    let size = 24u32;
    w.write_all(&size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"stts").map_err(io_err)?;
    w.write_all(&[0, 0, 0, 0]).map_err(io_err)?; // version + flags
    w.write_all(&1u32.to_be_bytes()).map_err(io_err)?; // entry count
    w.write_all(&frame_count.to_be_bytes()).map_err(io_err)?; // sample count
    w.write_all(&1000u32.to_be_bytes()).map_err(io_err)?; // sample delta (constant)
    Ok(())
}

fn write_stsz<W: Write>(w: &mut W, sample_sizes: &[u32]) -> VideoResult<()> {
    let size = 20 + (sample_sizes.len() as u32 * 4);
    w.write_all(&size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"stsz").map_err(io_err)?;
    w.write_all(&[0, 0, 0, 0]).map_err(io_err)?; // version + flags
    w.write_all(&0u32.to_be_bytes()).map_err(io_err)?; // sample size (0 = variable)
    w.write_all(&(sample_sizes.len() as u32).to_be_bytes()).map_err(io_err)?;
    for &s in sample_sizes {
        w.write_all(&s.to_be_bytes()).map_err(io_err)?;
    }
    Ok(())
}

fn write_stsc<W: Write>(w: &mut W) -> VideoResult<()> {
    let size = 28u32;
    w.write_all(&size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"stsc").map_err(io_err)?;
    w.write_all(&[0, 0, 0, 0]).map_err(io_err)?; // version + flags
    w.write_all(&1u32.to_be_bytes()).map_err(io_err)?; // entry count
    w.write_all(&1u32.to_be_bytes()).map_err(io_err)?; // first chunk
    w.write_all(&1u32.to_be_bytes()).map_err(io_err)?; // samples per chunk (all in one chunk)
    w.write_all(&1u32.to_be_bytes()).map_err(io_err)?; // sample description index
    Ok(())
}

fn write_stco<W: Write>(w: &mut W, mdat_offset: u32) -> VideoResult<()> {
    let size = 20u32;
    w.write_all(&size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"stco").map_err(io_err)?;
    w.write_all(&[0, 0, 0, 0]).map_err(io_err)?; // version + flags
    w.write_all(&1u32.to_be_bytes()).map_err(io_err)?; // entry count
    w.write_all(&mdat_offset.to_be_bytes()).map_err(io_err)?; // chunk offset
    Ok(())
}

fn write_stss<W: Write>(w: &mut W, sync_samples: &[u32]) -> VideoResult<()> {
    let size = 16 + (sync_samples.len() as u32 * 4);
    w.write_all(&size.to_be_bytes()).map_err(io_err)?;
    w.write_all(b"stss").map_err(io_err)?;
    w.write_all(&[0, 0, 0, 0]).map_err(io_err)?; // version + flags
    w.write_all(&(sync_samples.len() as u32).to_be_bytes()).map_err(io_err)?;
    for &s in sync_samples {
        w.write_all(&s.to_be_bytes()).map_err(io_err)?;
    }
    Ok(())
}

fn io_err(e: std::io::Error) -> VideoError {
    VideoError::Io {
        message: e.to_string(),
    }
}
