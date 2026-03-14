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
use mp4::{AvcConfig, Bytes, MediaConfig, Mp4Config, Mp4Sample, Mp4Writer, TrackConfig, TrackType};
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
    let mp4_config = Mp4Config {
        major_brand: "isom".parse().unwrap(),
        minor_version: 512,
        compatible_brands: vec![
            "isom".parse().unwrap(),
            "iso2".parse().unwrap(),
            "avc1".parse().unwrap(),
            "mp41".parse().unwrap(),
        ],
        timescale: 1000,
    };

    let mut mp4_writer =
        Mp4Writer::write_start(writer, &mp4_config).map_err(|e| VideoError::Container {
            message: format!("Failed to start MP4 writer: {}", e),
        })?;

    let video_timescale = ((config.frame_rate * 1000.0).round() as u32).max(1);
    let frame_duration = 1000u32;
    let track_config = TrackConfig {
        track_type: TrackType::Video,
        timescale: video_timescale,
        language: String::from("und"),
        media_conf: MediaConfig::AvcConfig(AvcConfig {
            width: config.width as u16,
            height: config.height as u16,
            seq_param_set: sps.to_vec(),
            pic_param_set: pps.to_vec(),
        }),
    };

    mp4_writer
        .add_track(&track_config)
        .map_err(|e| VideoError::Container {
            message: format!("Failed to add H.264 MP4 track: {}", e),
        })?;

    for (index, frame) in frames.iter().enumerate() {
        let avcc_data = annex_b_to_avcc(&frame.data);
        let sample = Mp4Sample {
            start_time: index as u64 * frame_duration as u64,
            duration: frame_duration,
            rendering_offset: 0,
            is_sync: frame.is_keyframe,
            bytes: Bytes::copy_from_slice(&avcc_data),
        };

        mp4_writer
            .write_sample(1, &sample)
            .map_err(|e| VideoError::Container {
                message: format!("Failed to write H.264 MP4 sample: {}", e),
            })?;
    }

    mp4_writer.write_end().map_err(|e| VideoError::Container {
        message: format!("Failed to finalize H.264 MP4: {}", e),
    })?;

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

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgba, RgbaImage};
    use tempfile::tempdir;

    use crate::video::container::open_container;
    use crate::video::VideoCodec;

    use super::{H264EncoderConfig, encode_h264_to_mp4};

    #[test]
    fn h264_mp4_output_round_trips_through_mp4_demuxer() {
        let mut frame_a = RgbaImage::new(32, 24);
        let mut frame_b = RgbaImage::new(32, 24);

        for (x, y, pixel) in frame_a.enumerate_pixels_mut() {
            *pixel = Rgba([(x * 7) as u8, (y * 9) as u8, 48, 255]);
        }
        for (x, y, pixel) in frame_b.enumerate_pixels_mut() {
            *pixel = Rgba([48, (x * 7) as u8, (y * 9) as u8, 255]);
        }

        let images = vec![
            DynamicImage::ImageRgba8(frame_a),
            DynamicImage::ImageRgba8(frame_b),
        ];

        let tempdir = tempdir().unwrap();
        let output = tempdir.path().join("roundtrip.mp4");

        encode_h264_to_mp4(
            images.iter(),
            &output,
            H264EncoderConfig::new(32, 24).with_frame_rate(2.0),
        )
        .unwrap();

        let mut demuxer = open_container(&output).unwrap();
        let video_info = demuxer.video_info().unwrap();
        assert_eq!(video_info.codec, VideoCodec::H264);
        assert_eq!(video_info.width, 32);
        assert_eq!(video_info.height, 24);

        let mut packet_count = 0usize;
        while let Some(packet) = demuxer.next_video_packet().unwrap() {
            assert!(!packet.data.is_empty());
            packet_count += 1;
        }

        assert_eq!(packet_count, 2);
    }
}

