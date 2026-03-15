//! AV1 video encoding using rav1e.
//!
//! This module provides AV1 encoding capabilities using the rav1e encoder,
//! which is a pure Rust + ASM implementation of the AV1 video codec.

use image::DynamicImage;
use rav1e::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::video::{VideoError, VideoResult};
use super::{VideoEncoder, VideoEncoderConfig};

/// Encoding speed preset (0-10).
/// Lower values = better quality but slower encoding.
/// Higher values = faster encoding but lower quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingSpeed {
    /// Slowest, highest quality (speed 0)
    Placebo,
    /// Very slow, very high quality (speed 2)
    VerySlow,
    /// Slow, high quality (speed 4)
    Slow,
    /// Medium speed/quality tradeoff (speed 6)
    Medium,
    /// Fast encoding (speed 8)
    Fast,
    /// Very fast encoding (speed 9)
    VeryFast,
    /// Fastest, lowest quality (speed 10)
    Ultrafast,
    /// Custom speed value (0-10)
    Custom(u8),
}

impl EncodingSpeed {
    /// Get the numeric speed value (0-10).
    pub fn value(&self) -> u8 {
        match self {
            EncodingSpeed::Placebo => 0,
            EncodingSpeed::VerySlow => 2,
            EncodingSpeed::Slow => 4,
            EncodingSpeed::Medium => 6,
            EncodingSpeed::Fast => 8,
            EncodingSpeed::VeryFast => 9,
            EncodingSpeed::Ultrafast => 10,
            EncodingSpeed::Custom(v) => (*v).min(10),
        }
    }
}

impl Default for EncodingSpeed {
    fn default() -> Self {
        EncodingSpeed::Medium
    }
}

/// AV1 encoder configuration.
#[derive(Debug, Clone)]
pub struct Av1EncoderConfig {
    /// Output video width.
    pub width: u32,
    /// Output video height.
    pub height: u32,
    /// Frame rate (frames per second).
    pub frame_rate: f64,
    /// Encoding speed preset (0-10).
    pub speed: EncodingSpeed,
    /// Constant quality value (0-255, lower = better quality).
    /// Default: 100 (good balance of quality/size).
    pub quantizer: u8,
    /// Minimum quantizer for rate control.
    pub min_quantizer: u8,
    /// Target bitrate in kbps (0 = use quantizer mode).
    pub bitrate_kbps: u32,
    /// Number of encoding threads (0 = auto).
    pub threads: usize,
    /// Key frame interval in frames.
    pub keyframe_interval: u64,
    /// Bit depth (8, 10, or 12).
    pub bit_depth: u8,
}

impl Default for Av1EncoderConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            frame_rate: 30.0,
            speed: EncodingSpeed::Medium,
            quantizer: 100,
            min_quantizer: 0,
            bitrate_kbps: 0,
            threads: 0,
            keyframe_interval: 240,
            bit_depth: 8,
        }
    }
}

impl Av1EncoderConfig {
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

    /// Set encoding speed.
    pub fn with_speed(mut self, speed: EncodingSpeed) -> Self {
        self.speed = speed;
        self
    }

    /// Set quality (quantizer, 0-255).
    pub fn with_quality(mut self, quantizer: u8) -> Self {
        self.quantizer = quantizer;
        self
    }

    /// Set target bitrate in kbps.
    pub fn with_bitrate(mut self, kbps: u32) -> Self {
        self.bitrate_kbps = kbps;
        self
    }

    /// Set number of threads.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }
}

impl VideoEncoderConfig for Av1EncoderConfig {
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

/// AV1 encoder using rav1e.
pub struct Av1Encoder {
    context: Context<u8>,
    config: Av1EncoderConfig,
    frame_count: u64,
}

impl Av1Encoder {
    /// Create encoder config from our config.
    fn create_rav1e_config(config: &Av1EncoderConfig) -> Result<Config, VideoError> {
        let mut enc_config = EncoderConfig::with_speed_preset(config.speed.value());

        enc_config.width = config.width as usize;
        enc_config.height = config.height as usize;
        enc_config.bit_depth = config.bit_depth as usize;
        enc_config.chroma_sampling = ChromaSampling::Cs420;
        enc_config.quantizer = config.quantizer as usize;
        enc_config.min_quantizer = config.min_quantizer;

        // Set time base for frame rate
        let (num, den) = rational_from_f64(config.frame_rate);
        enc_config.time_base = Rational { num: den, den: num };

        // Key frame settings
        enc_config.min_key_frame_interval = 12;
        enc_config.max_key_frame_interval = config.keyframe_interval;

        // Bitrate mode
        if config.bitrate_kbps > 0 {
            enc_config.bitrate = (config.bitrate_kbps as i32) * 1000;
        }

        let rav1e_config = Config::new()
            .with_encoder_config(enc_config)
            .with_threads(config.threads);

        Ok(rav1e_config)
    }
}

impl VideoEncoder for Av1Encoder {
    type Config = Av1EncoderConfig;
    type Packet = Packet<u8>;

    fn new(config: Av1EncoderConfig) -> Result<Self, VideoError> {
        // YUV420 chroma subsampling requires even dimensions
        if config.width == 0 || config.height == 0 {
            return Err(VideoError::Config {
                message: format!("dimensions must be non-zero, got {}x{}", config.width, config.height),
            });
        }
        if config.width % 2 != 0 || config.height % 2 != 0 {
            return Err(VideoError::Config {
                message: format!(
                    "AV1 YUV420 requires even dimensions, got {}x{} — pad or crop to even values",
                    config.width, config.height
                ),
            });
        }
        if config.frame_rate <= 0.0 || !config.frame_rate.is_finite() {
            return Err(VideoError::Config {
                message: format!("frame_rate must be positive and finite, got {}", config.frame_rate),
            });
        }

        let rav1e_config = Self::create_rav1e_config(&config)?;
        let context = rav1e_config.new_context().map_err(|e| VideoError::Encoding {
            message: format!("Failed to create encoder context: {}", e),
        })?;

        Ok(Self {
            context,
            config,
            frame_count: 0,
        })
    }

    fn send_frame(&mut self, image: &DynamicImage) -> Result<(), VideoError> {
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

        // Create rav1e frame
        let mut frame = self.context.new_frame();

        // Convert RGB to YUV420
        rgb_to_yuv420(&rgb, &mut frame, self.config.width, self.config.height);

        // Send frame to encoder
        self.context
            .send_frame(frame)
            .map_err(|e| VideoError::Encoding {
                message: format!("Failed to send frame {}: {:?}", self.frame_count, e),
            })?;

        self.frame_count += 1;
        Ok(())
    }

    fn flush(&mut self) {
        let _ = self.context.flush();
    }

    fn receive_packet(&mut self) -> Result<Option<Packet<u8>>, VideoError> {
        match self.context.receive_packet() {
            Ok(packet) => Ok(Some(packet)),
            Err(EncoderStatus::Encoded) => Ok(None),
            Err(EncoderStatus::LimitReached) => Ok(None),
            Err(EncoderStatus::NeedMoreData) => Ok(None),
            Err(e) => Err(VideoError::Encoding {
                message: format!("Encoder error: {:?}", e),
            }),
        }
    }

    fn config(&self) -> &Av1EncoderConfig {
        &self.config
    }
}

/// Convert RGB image to YUV420 planar format for rav1e.
fn rgb_to_yuv420(
    rgb: &image::RgbImage,
    frame: &mut Frame<u8>,
    width: u32,
    height: u32,
) {
    let w = width as usize;
    let h = height as usize;

    // Get strides before mutable borrows
    let y_stride = frame.planes[0].cfg.stride;
    let u_stride = frame.planes[1].cfg.stride;
    let v_stride = frame.planes[2].cfg.stride;

    // Y plane (full resolution)
    {
        let y_plane = frame.planes[0].data_origin_mut();
        for y in 0..h {
            for x in 0..w {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let r = pixel[0] as f32;
                let g = pixel[1] as f32;
                let b = pixel[2] as f32;

                // BT.601 RGB to Y conversion
                let y_val = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0) as u8;
                y_plane[y * y_stride + x] = y_val;
            }
        }
    }

    // U and V planes (half resolution for 4:2:0)
    // We need to split the planes to avoid borrowing issues
    let (y_plane, uv_planes) = frame.planes.split_at_mut(1);
    let (u_plane_slice, v_plane_slice) = uv_planes.split_at_mut(1);
    let u_plane = u_plane_slice[0].data_origin_mut();
    let v_plane = v_plane_slice[0].data_origin_mut();
    let _ = y_plane; // silence unused warning

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

            u_plane[y * u_stride + x] = u_val;
            v_plane[y * v_stride + x] = v_val;
        }
    }
}

/// Convert f64 frame rate to rational.
fn rational_from_f64(fps: f64) -> (u64, u64) {
    // Common frame rates
    let common = [
        (24000, 1001, 23.976),
        (24, 1, 24.0),
        (25, 1, 25.0),
        (30000, 1001, 29.97),
        (30, 1, 30.0),
        (50, 1, 50.0),
        (60000, 1001, 59.94),
        (60, 1, 60.0),
    ];

    for (num, den, rate) in common {
        if (fps - rate).abs() < 0.01 {
            return (num, den);
        }
    }

    // Fall back to simple integer approximation
    let rounded = fps.round() as u64;
    (rounded, 1)
}

/// IVF file header structure.
struct IvfHeader {
    width: u16,
    height: u16,
    timebase_num: u32,
    timebase_den: u32,
    frame_count: u32,
}

impl IvfHeader {
    fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(b"DKIF")?; // Signature
        writer.write_all(&0u16.to_le_bytes())?; // Version
        writer.write_all(&32u16.to_le_bytes())?; // Header size
        writer.write_all(b"AV01")?; // FourCC (AV1)
        writer.write_all(&self.width.to_le_bytes())?;
        writer.write_all(&self.height.to_le_bytes())?;
        writer.write_all(&self.timebase_den.to_le_bytes())?;
        writer.write_all(&self.timebase_num.to_le_bytes())?;
        writer.write_all(&self.frame_count.to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?; // Unused
        Ok(())
    }
}

/// Write IVF frame header.
fn write_ivf_frame<W: Write>(
    writer: &mut W,
    data: &[u8],
    pts: u64,
) -> std::io::Result<()> {
    writer.write_all(&(data.len() as u32).to_le_bytes())?;
    writer.write_all(&pts.to_le_bytes())?;
    writer.write_all(data)?;
    Ok(())
}

/// Encode an image sequence to an IVF file (raw AV1 bitstream container).
///
/// # Arguments
/// * `images` - Iterator of images to encode
/// * `output` - Output file path (.ivf)
/// * `config` - Encoder configuration
///
/// # Returns
/// Number of frames encoded
///
/// # Example
/// ```ignore
/// use xeno_lib::video::encode::{encode_to_ivf, Av1EncoderConfig, EncodingSpeed};
///
/// let images: Vec<DynamicImage> = load_images()?;
/// let config = Av1EncoderConfig::new(1920, 1080)
///     .with_frame_rate(30.0)
///     .with_speed(EncodingSpeed::Fast);
///
/// let frames = encode_to_ivf(images.iter(), "output.ivf", config)?;
/// ```
pub fn encode_to_ivf<'a, I, P>(
    images: I,
    output: P,
    config: Av1EncoderConfig,
) -> VideoResult<u64>
where
    I: Iterator<Item = &'a DynamicImage>,
    P: AsRef<Path>,
{
    let file = File::create(output.as_ref()).map_err(|e| VideoError::Io {
        message: format!("Failed to create output file: {}", e),
    })?;
    let mut writer = BufWriter::new(file);

    // Write placeholder header (will update frame count later)
    let (num, den) = rational_from_f64(config.frame_rate);
    let header = IvfHeader {
        width: config.width as u16,
        height: config.height as u16,
        timebase_num: num as u32,
        timebase_den: den as u32,
        frame_count: 0,
    };
    header.write(&mut writer).map_err(|e| VideoError::Io {
        message: format!("Failed to write IVF header: {}", e),
    })?;

    // Create encoder
    let mut encoder = Av1Encoder::new(config)?;
    let mut frame_count = 0u64;
    let mut pts = 0u64;

    // Encode all frames
    for image in images {
        encoder.send_frame(image)?;
        frame_count += 1;

        // Drain any available packets
        while let Some(packet) = encoder.receive_packet()? {
            write_ivf_frame(&mut writer, &packet.data, pts).map_err(|e| VideoError::Io {
                message: format!("Failed to write frame: {}", e),
            })?;
            pts += 1;
        }
    }

    // Flush encoder
    encoder.flush();

    // Drain remaining packets
    loop {
        match encoder.receive_packet()? {
            Some(packet) => {
                write_ivf_frame(&mut writer, &packet.data, pts).map_err(|e| VideoError::Io {
                    message: format!("Failed to write frame: {}", e),
                })?;
                pts += 1;
            }
            None => {
                // Check if we're really done
                if encoder.context.receive_packet().is_err() {
                    break;
                }
            }
        }
    }

    // Update frame count in header
    writer.flush().map_err(|e| VideoError::Io {
        message: format!("Failed to flush writer: {}", e),
    })?;
    drop(writer);

    // Reopen and update header
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(output.as_ref())
        .map_err(|e| VideoError::Io {
            message: format!("Failed to reopen file: {}", e),
        })?;

    use std::io::Seek;
    file.seek(std::io::SeekFrom::Start(24)).map_err(|e| VideoError::Io {
        message: format!("Failed to seek: {}", e),
    })?;
    file.write_all(&(frame_count as u32).to_le_bytes())
        .map_err(|e| VideoError::Io {
            message: format!("Failed to update frame count: {}", e),
        })?;

    Ok(frame_count)
}

/// Encode an image sequence to an MP4 file with AV1 codec.
///
/// # Arguments
/// * `images` - Iterator of images to encode
/// * `output` - Output file path (.mp4)
/// * `config` - Encoder configuration
///
/// # Returns
/// Number of frames encoded
///
/// # Note
/// Currently outputs IVF format. Full MP4/AV1 muxing is planned.
/// The output can be remuxed with: `ffmpeg -i output.ivf -c copy output.mp4`
pub fn encode_to_mp4<'a, I, P>(
    images: I,
    output: P,
    config: Av1EncoderConfig,
) -> VideoResult<u64>
where
    I: Iterator<Item = &'a DynamicImage>,
    P: AsRef<Path>,
{
    // Compatibility shim:
    // AV1-in-MP4 writing is not supported by our current muxing stack.
    // We intentionally encode IVF next to the requested output path.
    // Callers should remux the IVF stream externally when MP4 is required.

    let ivf_path = output.as_ref().with_extension("ivf");
    let frames = encode_to_ivf(images, &ivf_path, config)?;

    // Note: Full MP4 muxing for AV1 would require:
    // 1. Writing AV1 samples with proper timing
    // 2. Creating av1C configuration box
    // 3. Proper sample grouping

    // For now, return the IVF file
    // Users can remux with ffmpeg: ffmpeg -i output.ivf -c copy output.mp4
    eprintln!(
        "Note: AV1 MP4 muxing not yet implemented. Output saved as IVF: {}",
        ivf_path.display()
    );

    Ok(frames)
}
