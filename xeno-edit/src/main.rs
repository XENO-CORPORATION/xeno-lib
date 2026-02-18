//! xeno-edit: CLI tool for image editing powered by xeno-lib
//!
//! # Usage
//!
//! ```bash
//! xeno-edit remove-bg input.jpg output.png
//! xeno-edit convert png image.jpg
//! xeno-edit gif output.gif frame1.png frame2.png frame3.png
//! xeno-edit awebp output.webp frame1.png frame2.png frame3.png
//! ```

use std::ffi::OsStr;
use std::io::{self, Read as IoRead};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

use xeno_lib::agent::{Capabilities, ToAgentJson};
use xeno_lib::background::{load_model, remove_background, BackgroundRemovalConfig};
use xeno_lib::video::{encode_to_ivf, Av1EncoderConfig, EncodingSpeed};
use xeno_lib::video::{encode_h264_to_mp4, encode_to_h264, H264EncoderConfig};
use xeno_lib::video::open_container;
use xeno_lib::video::decode::{DecodeCodec, NvDecoder, decode_ivf, best_decoder_for, DecoderBackend};
use xeno_lib::audio::{AudioInfo, decode_file, encode_wav, encode_flac, WavConfig, FlacConfig};
use xeno_lib::transforms::{
    flip_horizontal, flip_vertical, flip_both, rotate_90_cw, rotate_180, rotate_270_cw, crop,
};
use xeno_lib::adjustments::{
    adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue, adjust_gamma,
    grayscale as to_grayscale, invert as invert_colors,
};
use xeno_lib::filters::{gaussian_blur, unsharp_mask, sepia as apply_sepia};
use xeno_lib::composite::watermark as apply_watermark;
use xeno_lib::text::{TextOverlay, TextConfig, Anchor};

#[derive(Parser)]
#[command(name = "xeno-edit")]
#[command(author, version, about = "CLI tool for image editing powered by xeno-lib")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Supported output image formats
#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImageFormat {
    /// PNG - lossless compression, supports transparency
    Png,
    /// JPEG - lossy compression, no transparency
    #[value(alias = "jpg")]
    Jpeg,
    /// WebP - modern format, lossy/lossless, supports transparency
    Webp,
    /// GIF - limited colors, supports animation
    Gif,
    /// BMP - uncompressed bitmap
    Bmp,
    /// TIFF - high quality, large files
    #[value(alias = "tif")]
    Tiff,
    /// ICO - Windows icon format
    Ico,
}

impl ImageFormat {
    fn extension(&self) -> &'static str {
        match self {
            ImageFormat::Png => "png",
            ImageFormat::Jpeg => "jpg",
            ImageFormat::Webp => "webp",
            ImageFormat::Gif => "gif",
            ImageFormat::Bmp => "bmp",
            ImageFormat::Tiff => "tiff",
            ImageFormat::Ico => "ico",
        }
    }

    fn name(&self) -> &'static str {
        match self {
            ImageFormat::Png => "PNG",
            ImageFormat::Jpeg => "JPEG",
            ImageFormat::Webp => "WebP",
            ImageFormat::Gif => "GIF",
            ImageFormat::Bmp => "BMP",
            ImageFormat::Tiff => "TIFF",
            ImageFormat::Ico => "ICO",
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Remove background from an image using AI
    #[command(name = "remove-bg", alias = "rmbg")]
    RemoveBg {
        /// Input image path(s) - supports multiple files
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Output directory for batch processing (default: same as input)
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Custom model path (default: ~/.xeno-lib/models/birefnet-general.onnx)
        #[arg(short, long)]
        model: Option<PathBuf>,

        /// Confidence threshold (0.0-1.0, default: 0.5)
        #[arg(short, long, default_value = "0.5")]
        threshold: f32,

        /// Disable GPU acceleration (use CPU only)
        #[arg(long)]
        cpu: bool,

        /// Suppress output messages
        #[arg(short, long)]
        quiet: bool,
    },

    /// Convert image(s) to a different format
    #[command(name = "convert", alias = "cvt")]
    Convert {
        /// Target format (png, jpeg/jpg, webp, gif, bmp, tiff/tif, ico)
        format: ImageFormat,

        /// Input image path(s) - supports multiple files
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Output directory (default: same as input)
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// JPEG quality (1-100, default: 90)
        #[arg(long, default_value = "90")]
        quality: u8,

        /// Suppress output messages
        #[arg(short, long)]
        quiet: bool,
    },

    /// Apply filters to image(s) - FFmpeg-like image processing
    #[command(name = "image-filter", alias = "imgf")]
    ImageFilter {
        /// Input image path(s) - supports multiple files
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Output directory (default: same as input with _filtered suffix)
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Output format (default: same as input)
        #[arg(short, long)]
        format: Option<ImageFormat>,

        // === Resize & Crop ===
        /// Resize width (0 = no resize)
        #[arg(short, long, default_value = "0")]
        width: u32,

        /// Resize height (0 = proportional if width set, otherwise no resize)
        #[arg(long, default_value = "0")]
        height: u32,

        /// Crop region: "x,y,width,height" (e.g., "100,50,800,600")
        #[arg(long)]
        crop: Option<String>,

        // === Color Adjustments ===
        /// Brightness adjustment (-100 to 100, 0 = no change)
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        brightness: i32,

        /// Contrast adjustment (-100 to 100, 0 = no change)
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        contrast: i32,

        /// Saturation adjustment (-100 to 100, 0 = no change)
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        saturation: i32,

        /// Hue rotation in degrees (-180 to 180, 0 = no change)
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        hue: i32,

        /// Gamma correction (0.1 to 10.0, 1.0 = no change)
        #[arg(long, default_value = "1.0")]
        gamma: f32,

        // === Filters ===
        /// Gaussian blur radius (0 = no blur)
        #[arg(long, default_value = "0")]
        blur: u32,

        /// Sharpen amount (0 = no sharpen)
        #[arg(long, default_value = "0")]
        sharpen: u32,

        /// Convert to grayscale
        #[arg(long)]
        grayscale: bool,

        /// Apply sepia tone
        #[arg(long)]
        sepia: bool,

        /// Invert colors (negative)
        #[arg(long)]
        invert: bool,

        // === Advanced Filters (FFmpeg-equivalent) ===
        /// Vignette strength (0.0-2.0, 0 = off)
        #[arg(long, default_value = "0.0")]
        vignette: f32,

        /// Denoise strength (0-10, 0 = off)
        #[arg(long, default_value = "0")]
        denoise: u32,

        /// Remove green screen (tolerance 0.0-1.0, 0 = off)
        #[arg(long, default_value = "0.0")]
        chromakey: f32,

        /// Posterize levels (2-256, 0 = off)
        #[arg(long, default_value = "0")]
        posterize: u8,

        /// Solarize threshold (0-255, 256 = off)
        #[arg(long, default_value = "256")]
        solarize: u16,

        /// Color temperature (-100 to 100, 0 = no change)
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        temperature: i32,

        /// Tint/green-magenta (-100 to 100, 0 = no change)
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        tint: i32,

        /// Vibrance (-100 to 100, 0 = no change)
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        vibrance: i32,

        /// Edge detect strength (0.0 = off)
        #[arg(long, default_value = "0.0")]
        edges: f32,

        /// Emboss strength (0.0 = off)
        #[arg(long, default_value = "0.0")]
        emboss: f32,

        // === Transform ===
        /// Rotation: 0, 90, 180, 270 degrees clockwise
        #[arg(short, long, default_value = "0")]
        rotate: u16,

        /// Flip: "none", "h" (horizontal), "v" (vertical), "hv" (both)
        #[arg(long, default_value = "none")]
        flip: String,

        // === Overlay ===
        /// Watermark image path
        #[arg(long)]
        watermark: Option<PathBuf>,

        /// Watermark position: tl, tr, bl, br, center
        #[arg(long, default_value = "br")]
        watermark_pos: String,

        /// Watermark opacity (0.0 to 1.0)
        #[arg(long, default_value = "0.5")]
        watermark_opacity: f32,

        /// Watermark scale (0.1 to 1.0, relative to image size)
        #[arg(long, default_value = "0.2")]
        watermark_scale: f32,

        /// JPEG/WebP output quality (1-100)
        #[arg(long, default_value = "90")]
        quality: u8,

        /// Suppress output messages
        #[arg(short = 'Q', long)]
        quiet: bool,
    },

    /// Create animated GIF from image sequence (native, no FFmpeg)
    #[command(name = "gif")]
    Gif {
        /// Output GIF path
        output: PathBuf,

        /// Input image paths (in order)
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Frame delay in milliseconds (default: 100)
        #[arg(short, long, default_value = "100")]
        delay: u16,

        /// Resize width (auto-scales height, 0 = original size)
        #[arg(short, long, default_value = "0")]
        width: u32,

        /// Loop count (0 = infinite)
        #[arg(short, long, default_value = "0")]
        loops: u16,

        /// Suppress output messages
        #[arg(short, long)]
        quiet: bool,
    },

    /// Create animated WebP from image sequence (native, no FFmpeg)
    #[command(name = "awebp", alias = "webp-anim")]
    AnimatedWebp {
        /// Output WebP path
        output: PathBuf,

        /// Input image paths (in order)
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Frame delay in milliseconds (default: 100)
        #[arg(short, long, default_value = "100")]
        delay: u16,

        /// Resize width (auto-scales height, 0 = original size)
        #[arg(short, long, default_value = "0")]
        width: u32,

        /// Encoding quality (0-100, default: 80)
        #[arg(long, default_value = "80")]
        quality: u8,

        /// Use lossless encoding (larger files, perfect quality)
        #[arg(long)]
        lossless: bool,

        /// Loop count (0 = infinite)
        #[arg(short, long, default_value = "0")]
        loops: u16,

        /// Suppress output messages
        #[arg(short, long)]
        quiet: bool,
    },

    /// Show video file metadata (MP4, IVF)
    #[command(name = "video-info", alias = "vinfo")]
    VideoInfo {
        /// Input video file(s)
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Suppress headers (just show data)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Encode image sequence to AV1 video (native, no FFmpeg)
    #[command(name = "video-encode", alias = "av1")]
    VideoEncode {
        /// Output video path (.ivf for AV1 bitstream)
        output: PathBuf,

        /// Input image paths (in order)
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Frame rate (default: 30)
        #[arg(short, long, default_value = "30")]
        fps: f64,

        /// Encoding speed preset (0-10, higher = faster but lower quality)
        #[arg(short, long, default_value = "6")]
        speed: u8,

        /// Quality (quantizer 0-255, lower = better quality, default: 100)
        #[arg(long, default_value = "100")]
        quality: u8,

        /// Target bitrate in kbps (0 = use quality mode)
        #[arg(long, default_value = "0")]
        bitrate: u32,

        /// Output width (0 = use first image's width)
        #[arg(short, long, default_value = "0")]
        width: u32,

        /// Output height (0 = use first image's height)
        #[arg(long, default_value = "0")]
        height: u32,

        /// Number of encoding threads (0 = auto)
        #[arg(short, long, default_value = "0")]
        threads: usize,

        /// Suppress output messages
        #[arg(short = 'Q', long)]
        quiet: bool,
    },

    /// Show GPU video decode capabilities (NVDEC)
    #[command(name = "gpu-info", alias = "nvdec")]
    GpuInfo {
        /// GPU device index (default: 0)
        #[arg(short, long, default_value = "0")]
        device: i32,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show audio file metadata (pure Rust via Symphonia)
    #[command(name = "audio-info", alias = "ainfo")]
    AudioInfo {
        /// Input audio file(s)
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Suppress headers (just show data)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Extract audio from video/audio file to WAV
    #[command(name = "extract-audio", alias = "xaudio")]
    ExtractAudio {
        /// Input video/audio file
        input: PathBuf,

        /// Output WAV file (default: input_name.wav)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Convert to mono
        #[arg(long)]
        mono: bool,

        /// Suppress output messages
        #[arg(short, long)]
        quiet: bool,
    },

    /// Extract frames from IVF video to images (NVDEC GPU-accelerated)
    #[command(name = "video-frames", alias = "vframes")]
    VideoFrames {
        /// Input IVF video file
        input: PathBuf,

        /// Output directory (default: ./frames)
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Output image format (png, jpg, webp)
        #[arg(short, long, default_value = "png")]
        format: String,

        /// Extract only every Nth frame
        #[arg(long, default_value = "1")]
        every: u32,

        /// Maximum number of frames to extract (0 = all)
        #[arg(long, default_value = "0")]
        max_frames: u32,

        /// Suppress output messages
        #[arg(short, long)]
        quiet: bool,
    },

    /// Convert IVF video to animated GIF (NVDEC + native GIF encoder)
    #[command(name = "video-to-gif", alias = "v2gif")]
    VideoToGif {
        /// Input IVF video file
        input: PathBuf,

        /// Output GIF file
        output: PathBuf,

        /// Resize width (0 = original)
        #[arg(short, long, default_value = "0")]
        width: u32,

        /// Frame rate for GIF (fps, 0 = use video fps)
        #[arg(long, default_value = "0")]
        fps: u32,

        /// Skip every N frames (1 = keep all, 2 = keep every other, etc.)
        #[arg(long, default_value = "1")]
        skip: u32,

        /// Maximum frames to include (0 = all)
        #[arg(long, default_value = "0")]
        max_frames: u32,

        /// Loop count (0 = infinite)
        #[arg(short, long, default_value = "0")]
        loops: u16,

        /// Suppress output messages
        #[arg(short, long)]
        quiet: bool,
    },

    /// Generate thumbnail from IVF video (NVDEC GPU-accelerated)
    #[command(name = "video-thumbnail", alias = "vthumb")]
    VideoThumbnail {
        /// Input IVF video file
        input: PathBuf,

        /// Output image file
        output: PathBuf,

        /// Frame position: "first", "middle", "last", or frame number
        #[arg(short, long, default_value = "middle")]
        position: String,

        /// Resize width (0 = original)
        #[arg(short, long, default_value = "0")]
        width: u32,

        /// Suppress output messages
        #[arg(short, long)]
        quiet: bool,
    },

    /// Encode image sequence with pattern (frame_%04d.png) to AV1 video
    #[command(name = "encode-sequence", alias = "eseq")]
    EncodeSequence {
        /// Input pattern (e.g., frame_%04d.png, output_%05d.jpg)
        pattern: String,

        /// Output video path (.ivf)
        output: PathBuf,

        /// Start frame number (default: 0)
        #[arg(long, default_value = "0")]
        start: usize,

        /// End frame number (inclusive, 0 = auto-detect)
        #[arg(long, default_value = "0")]
        end: usize,

        /// Frame rate (default: 30)
        #[arg(short, long, default_value = "30")]
        fps: f64,

        /// Encoding speed preset (0-10, higher = faster but lower quality)
        #[arg(short, long, default_value = "6")]
        speed: u8,

        /// Quality (quantizer 0-255, lower = better quality, default: 100)
        #[arg(long, default_value = "100")]
        quality: u8,

        /// Target bitrate in kbps (0 = use quality mode)
        #[arg(long, default_value = "0")]
        bitrate: u32,

        /// Output width (0 = use first frame's width)
        #[arg(short, long, default_value = "0")]
        width: u32,

        /// Output height (0 = use first frame's height)
        #[arg(long, default_value = "0")]
        height: u32,

        /// Number of encoding threads (0 = auto)
        #[arg(short, long, default_value = "0")]
        threads: usize,

        /// Suppress output messages
        #[arg(short = 'Q', long)]
        quiet: bool,
    },

    /// Transcode video with optional transforms (decode → transform → encode)
    #[command(name = "video-transcode", alias = "vtrans")]
    VideoTranscode {
        /// Input video file (IVF, MP4)
        input: PathBuf,

        /// Output video path (.ivf or .mp4)
        output: PathBuf,

        /// Output width (0 = original, or use source aspect ratio if only height specified)
        #[arg(short, long, default_value = "0")]
        width: u32,

        /// Output height (0 = original, or use source aspect ratio if only width specified)
        #[arg(long, default_value = "0")]
        height: u32,

        /// Frame rate (0 = use source fps, or specify new fps)
        #[arg(short, long, default_value = "0")]
        fps: f64,

        /// Rotation: 0, 90, 180, 270 degrees clockwise
        #[arg(short, long, default_value = "0")]
        rotate: u16,

        /// Flip: "none", "h" (horizontal), "v" (vertical), "hv" (both)
        #[arg(long, default_value = "none")]
        flip: String,

        /// Encoding speed preset (0-10, higher = faster but lower quality)
        #[arg(short, long, default_value = "6")]
        speed: u8,

        /// Quality (quantizer 0-255, lower = better quality, default: 100)
        #[arg(long, default_value = "100")]
        quality: u8,

        /// Target bitrate in kbps (0 = use quality mode)
        #[arg(long, default_value = "0")]
        bitrate: u32,

        /// Number of encoding threads (0 = auto)
        #[arg(short, long, default_value = "0")]
        threads: usize,

        /// Maximum frames to process (0 = all)
        #[arg(long, default_value = "0")]
        max_frames: u32,

        /// Start time in seconds (skip frames before this)
        #[arg(long, default_value = "0")]
        start: f64,

        /// End time in seconds (stop after this, 0 = to end)
        #[arg(long, default_value = "0")]
        end: f64,

        // === Color Adjustments (FFmpeg-equivalent) ===

        /// Brightness adjustment (-100 to 100, 0 = no change)
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        brightness: i32,

        /// Contrast adjustment (-100 to 100, 0 = no change)
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        contrast: i32,

        /// Saturation adjustment (-100 to 100, 0 = no change)
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        saturation: i32,

        /// Hue rotation in degrees (-180 to 180, 0 = no change)
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        hue: i32,

        /// Gamma adjustment (0.1 to 10.0, 1.0 = no change)
        #[arg(long, default_value = "1.0")]
        gamma: f32,

        // === Filters ===

        /// Gaussian blur radius (0 = no blur, 1-50)
        #[arg(long, default_value = "0")]
        blur: u32,

        /// Sharpen amount (0 = no sharpen, 1-100)
        #[arg(long, default_value = "0")]
        sharpen: u32,

        /// Convert to grayscale
        #[arg(long)]
        grayscale: bool,

        /// Apply sepia tone
        #[arg(long)]
        sepia: bool,

        /// Invert colors (negative)
        #[arg(long)]
        invert: bool,

        // === Crop ===

        /// Crop region: "x,y,width,height" (e.g., "100,50,1280,720")
        #[arg(long)]
        crop: Option<String>,

        // === Effects ===

        /// Fade in duration in seconds (0 = no fade)
        #[arg(long, default_value = "0")]
        fade_in: f64,

        /// Fade out duration in seconds (0 = no fade)
        #[arg(long, default_value = "0")]
        fade_out: f64,

        /// Speed factor (0.25 = 4x slower, 2.0 = 2x faster, 1.0 = normal)
        #[arg(long, default_value = "1.0")]
        speed_factor: f64,

        // === Overlay ===

        /// Watermark image path
        #[arg(long)]
        watermark: Option<PathBuf>,

        /// Watermark position: "x,y" or preset (tl, tr, bl, br, center)
        #[arg(long, default_value = "br")]
        watermark_pos: String,

        /// Watermark opacity (0.0 to 1.0)
        #[arg(long, default_value = "0.5")]
        watermark_opacity: f32,

        /// Watermark scale (0.1 to 1.0, relative to video size)
        #[arg(long, default_value = "0.2")]
        watermark_scale: f32,

        // === Output Codec ===

        /// Output codec: av1, h264 (default: av1)
        #[arg(long, default_value = "av1")]
        codec: String,

        /// Suppress output messages
        #[arg(short = 'Q', long)]
        quiet: bool,
    },

    /// Show library capabilities and available features (JSON for AI agents)
    #[command(name = "capabilities", alias = "caps")]
    Capabilities,

    /// Trim/cut video to a specific time range
    #[command(name = "video-trim", alias = "vtrim")]
    VideoTrim {
        /// Input video file (IVF)
        input: PathBuf,

        /// Output video path (.ivf)
        output: PathBuf,

        /// Start time in seconds
        #[arg(short, long, default_value = "0")]
        start: f64,

        /// End time in seconds (0 = to end)
        #[arg(short, long, default_value = "0")]
        end: f64,

        /// Encoding speed preset (0-10)
        #[arg(long, default_value = "6")]
        speed: u8,

        /// Quality (0-255, lower = better)
        #[arg(long, default_value = "100")]
        quality: u8,

        /// Suppress output messages
        #[arg(short, long)]
        quiet: bool,
    },

    /// Concatenate multiple videos into one
    #[command(name = "video-concat", alias = "vcat")]
    VideoConcat {
        /// Output video path (.ivf)
        output: PathBuf,

        /// Input video files (IVF)
        #[arg(required = true, num_args = 2..)]
        inputs: Vec<PathBuf>,

        /// Encoding speed preset (0-10)
        #[arg(short, long, default_value = "6")]
        speed: u8,

        /// Quality (0-255, lower = better)
        #[arg(long, default_value = "100")]
        quality: u8,

        /// Target frame rate (0 = use first video's fps)
        #[arg(short, long, default_value = "0")]
        fps: f64,

        /// Suppress output messages
        #[arg(short = 'Q', long)]
        quiet: bool,
    },

    /// Draw text overlay on image(s) (pure Rust, no FFmpeg)
    #[command(name = "text-overlay", alias = "drawtext")]
    TextOverlay {
        /// Input image path(s)
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Text to draw
        #[arg(short, long, required = true)]
        text: String,

        /// Font file path (.ttf, .otf)
        #[arg(short, long, required = true)]
        font: PathBuf,

        /// Font size in pixels
        #[arg(short = 's', long, default_value = "32")]
        font_size: f32,

        /// X position
        #[arg(short = 'x', long, default_value = "10")]
        x: i32,

        /// Y position
        #[arg(short = 'y', long, default_value = "10")]
        y: i32,

        /// Text color as hex (e.g., FFFFFF for white, FF0000 for red)
        #[arg(short, long, default_value = "FFFFFF")]
        color: String,

        /// Anchor position: tl, tc, tr, ml, c, mr, bl, bc, br
        #[arg(short, long, default_value = "tl")]
        anchor: String,

        /// Shadow offset (dx,dy) e.g., "2,2"
        #[arg(long)]
        shadow: Option<String>,

        /// Outline thickness (pixels)
        #[arg(long, default_value = "0")]
        outline: u32,

        /// Output directory (default: same as input with _text suffix)
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Suppress output messages
        #[arg(short = 'Q', long)]
        quiet: bool,
    },

    /// Encode image sequence to H.264 video (native, no FFmpeg)
    ///
    /// H.264 is universally playable on all devices and platforms.
    /// Outputs either raw .h264 bitstream or proper .mp4 container.
    #[command(name = "h264-encode", alias = "h264")]
    H264Encode {
        /// Output video path (.mp4 for container, .h264 for raw)
        output: PathBuf,

        /// Input image paths (in order)
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Frame rate (default: 30)
        #[arg(short, long, default_value = "30")]
        fps: f64,

        /// Target bitrate in kbps (0 = auto based on resolution)
        #[arg(short, long, default_value = "0")]
        bitrate: u32,

        /// Output width (0 = use first image's width)
        #[arg(short, long, default_value = "0")]
        width: u32,

        /// Output height (0 = use first image's height)
        #[arg(long, default_value = "0")]
        height: u32,

        /// Output raw H.264 instead of MP4 container
        #[arg(long)]
        raw: bool,

        /// Suppress output messages
        #[arg(short = 'Q', long)]
        quiet: bool,
    },

    /// Encode audio to WAV or FLAC format (native, no FFmpeg)
    ///
    /// Converts audio files to lossless WAV or FLAC format.
    /// WAV is uncompressed, FLAC provides ~60% compression.
    #[command(name = "audio-encode", alias = "aenc")]
    AudioEncode {
        /// Output audio path (.wav or .flac)
        output: PathBuf,

        /// Input audio file(s) to concatenate and encode
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Output format: wav, flac (auto-detected from extension if not specified)
        #[arg(short, long)]
        format: Option<String>,

        /// Sample rate (0 = preserve source, e.g., 44100, 48000)
        #[arg(short = 'r', long, default_value = "0")]
        sample_rate: u32,

        /// Bits per sample: 8, 16, 24, 32 for WAV; 16, 24 for FLAC
        #[arg(short, long, default_value = "16")]
        bits: u16,

        /// Suppress output messages
        #[arg(short = 'Q', long)]
        quiet: bool,
    },

    /// Execute operation from JSON config (agent-friendly, programmatic control)
    ///
    /// AI agents can construct JSON configurations to execute any operation
    /// without needing to know CLI flags. Accepts JSON via file, stdin, or argument.
    ///
    /// Example JSON:
    /// {
    ///   "input": "video.ivf",
    ///   "output": "output.ivf",
    ///   "operation": "transcode",
    ///   "video": { "codec": "av1", "quality": 80, "speed": 6 },
    ///   "transforms": ["rotate:90", "flip:h"],
    ///   "trim": { "start": 0, "end": 10 }
    /// }
    #[command(name = "exec", alias = "run")]
    Exec {
        /// JSON config: file path, "-" for stdin, or inline JSON string starting with "{"
        #[arg(required = true)]
        config: String,

        /// Override output path (ignores JSON output field)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Force JSON output format for all responses
        #[arg(long)]
        json: bool,
    },

    /// Generate a JSON config template for the specified operation
    #[command(name = "template", alias = "tpl")]
    Template {
        /// Operation type: transcode, trim, concat, encode, decode, remove-bg, text
        operation: String,

        /// Pretty print JSON (default: true)
        #[arg(long, default_value = "true")]
        pretty: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::RemoveBg {
            inputs,
            output_dir,
            model,
            threshold,
            cpu,
            quiet,
        } => cmd_remove_bg(inputs, output_dir, model, threshold, cpu, quiet),

        Commands::Convert {
            format,
            inputs,
            output_dir,
            quality,
            quiet,
        } => cmd_convert(format, inputs, output_dir, quality, quiet),

        Commands::ImageFilter {
            inputs, output_dir, format, width, height, crop,
            brightness, contrast, saturation, hue, gamma,
            blur, sharpen, grayscale, sepia, invert,
            vignette, denoise, chromakey, posterize, solarize,
            temperature, tint, vibrance, edges, emboss,
            rotate, flip, watermark, watermark_pos, watermark_opacity, watermark_scale,
            quality, quiet,
        } => cmd_image_filter(
            inputs, output_dir, format, width, height, crop,
            brightness, contrast, saturation, hue, gamma,
            blur, sharpen, grayscale, sepia, invert,
            vignette, denoise, chromakey, posterize, solarize,
            temperature, tint, vibrance, edges, emboss,
            rotate, flip, watermark, watermark_pos, watermark_opacity, watermark_scale,
            quality, quiet,
        ),

        Commands::Gif {
            output,
            inputs,
            delay,
            width,
            loops,
            quiet,
        } => cmd_gif(output, inputs, delay, width, loops, quiet),

        Commands::AnimatedWebp {
            output,
            inputs,
            delay,
            width,
            quality,
            lossless,
            loops,
            quiet,
        } => cmd_awebp(output, inputs, delay, width, quality, lossless, loops, quiet),

        Commands::VideoInfo {
            inputs,
            json,
            quiet,
        } => cmd_video_info(inputs, json, quiet),

        Commands::VideoEncode {
            output,
            inputs,
            fps,
            speed,
            quality,
            bitrate,
            width,
            height,
            threads,
            quiet,
        } => cmd_video_encode(output, inputs, fps, speed, quality, bitrate, width, height, threads, quiet),

        Commands::GpuInfo { device, json } => cmd_gpu_info(device, json),

        Commands::AudioInfo { inputs, json, quiet } => cmd_audio_info(inputs, json, quiet),

        Commands::ExtractAudio { input, output, mono, quiet } => cmd_extract_audio(input, output, mono, quiet),

        Commands::VideoFrames { input, output_dir, format, every, max_frames, quiet } =>
            cmd_video_frames(input, output_dir, format, every, max_frames, quiet),

        Commands::VideoToGif { input, output, width, fps, skip, max_frames, loops, quiet } =>
            cmd_video_to_gif(input, output, width, fps, skip, max_frames, loops, quiet),

        Commands::VideoThumbnail { input, output, position, width, quiet } =>
            cmd_video_thumbnail(input, output, position, width, quiet),

        Commands::EncodeSequence {
            pattern, output, start, end, fps, speed, quality, bitrate, width, height, threads, quiet
        } => cmd_encode_sequence(
            pattern, output, start, end, fps, speed, quality, bitrate, width, height, threads, quiet
        ),

        Commands::VideoTranscode {
            input, output, width, height, fps, rotate, flip,
            speed, quality, bitrate, threads, max_frames, start, end,
            brightness, contrast, saturation, hue, gamma,
            blur, sharpen, grayscale, sepia, invert,
            crop, fade_in, fade_out, speed_factor,
            watermark, watermark_pos, watermark_opacity, watermark_scale,
            codec, quiet
        } => cmd_video_transcode(
            input, output, width, height, fps, rotate, flip,
            speed, quality, bitrate, threads, max_frames, start, end,
            brightness, contrast, saturation, hue, gamma,
            blur, sharpen, grayscale, sepia, invert,
            crop, fade_in, fade_out, speed_factor,
            watermark, watermark_pos, watermark_opacity, watermark_scale,
            codec, quiet
        ),

        Commands::Capabilities => cmd_capabilities(),

        Commands::VideoTrim { input, output, start, end, speed, quality, quiet } =>
            cmd_video_trim(input, output, start, end, speed, quality, quiet),

        Commands::VideoConcat { output, inputs, speed, quality, fps, quiet } =>
            cmd_video_concat(output, inputs, speed, quality, fps, quiet),

        Commands::TextOverlay {
            inputs, text, font, font_size, x, y, color, anchor, shadow, outline, output_dir, quiet
        } => cmd_text_overlay(
            inputs, text, font, font_size, x, y, color, anchor, shadow, outline, output_dir, quiet
        ),

        Commands::H264Encode {
            output, inputs, fps, bitrate, width, height, raw, quiet
        } => cmd_h264_encode(output, inputs, fps, bitrate, width, height, raw, quiet),

        Commands::AudioEncode {
            output, inputs, format, sample_rate, bits, quiet
        } => cmd_audio_encode(output, inputs, format, sample_rate, bits, quiet),

        Commands::Exec { config, output, json } => cmd_exec(config, output, json),

        Commands::Template { operation, pretty } => cmd_template(operation, pretty),
    }
}

// ============================================================================
// Capabilities Command (Agent-friendly)
// ============================================================================

fn cmd_capabilities() -> Result<()> {
    let caps = Capabilities::query();
    println!("{}", caps.to_agent_json());
    Ok(())
}

// ============================================================================
// Remove Background Command
// ============================================================================

fn cmd_remove_bg(
    inputs: Vec<PathBuf>,
    output_dir: Option<PathBuf>,
    model: Option<PathBuf>,
    threshold: f32,
    cpu: bool,
    quiet: bool,
) -> Result<()> {
    let total_images = inputs.len();
    let is_batch = total_images > 1;

    if !quiet {
        println!("xeno-edit remove-bg");
        println!("==================");
        if is_batch {
            println!("Batch mode: {} images", total_images);
        }
        println!();
    }

    let mut config = BackgroundRemovalConfig::default();
    if let Some(model_path) = model {
        config.model_path = model_path;
    }
    config.use_gpu = !cpu;
    config.confidence_threshold = threshold;

    if !quiet {
        println!("Model: {}", config.model_path.display());
        println!("GPU: {}", if config.use_gpu { "enabled" } else { "disabled" });
        println!("Threshold: {}", config.confidence_threshold);
        println!();
    }

    if !quiet {
        print!("Loading model... ");
    }
    let load_start = Instant::now();
    let mut session = load_model(&config).context("Failed to load model")?;
    if !quiet {
        println!("done ({:.0?})", load_start.elapsed());
        println!();
    }

    let batch_start = Instant::now();
    let mut success_count = 0;
    let mut fail_count = 0;

    for (idx, input) in inputs.iter().enumerate() {
        let output = if let Some(ref dir) = output_dir {
            let stem = input.file_stem().unwrap_or_default().to_string_lossy();
            dir.join(format!("{}_nobg.png", stem))
        } else {
            let stem = input.file_stem().unwrap_or_default().to_string_lossy();
            let parent = input.parent().unwrap_or(std::path::Path::new("."));
            parent.join(format!("{}_nobg.png", stem))
        };

        if !quiet && is_batch {
            println!("[{}/{}] {}", idx + 1, total_images, input.display());
        }

        match process_single_image(input, &output, &mut session, quiet && !is_batch) {
            Ok(_) => {
                success_count += 1;
                if quiet {
                    println!("{}", output.display());
                } else if !is_batch {
                    println!("Output: {}", output.display());
                } else {
                    println!("  -> {}", output.display());
                }
            }
            Err(e) => {
                fail_count += 1;
                if !quiet {
                    eprintln!("  x Error: {}", e);
                } else {
                    eprintln!("Error processing {}: {}", input.display(), e);
                }
            }
        }
    }

    if !quiet && is_batch {
        println!();
        println!("Batch complete: {} succeeded, {} failed ({:.1?} total)",
            success_count, fail_count, batch_start.elapsed());
    }

    if fail_count > 0 && success_count == 0 {
        anyhow::bail!("All images failed to process");
    }

    Ok(())
}

fn process_single_image(
    input: &PathBuf,
    output: &PathBuf,
    session: &mut xeno_lib::background::ModelSession,
    quiet: bool,
) -> Result<()> {
    if !quiet {
        print!("Loading image... ");
    }
    let img_start = Instant::now();
    // Use guessed format from file contents, not extension (handles misnamed files)
    let input_image = load_image_auto(input)
        .with_context(|| format!("Failed to open image: {}", input.display()))?;
    if !quiet {
        println!("done ({:.0?}, {}x{})",
            img_start.elapsed(), input_image.width(), input_image.height());
    }

    if !quiet {
        print!("Removing background... ");
    }
    let inference_start = Instant::now();
    let output_image = remove_background(&input_image, session)
        .context("Failed to remove background")?;
    if !quiet {
        println!("done ({:.0?})", inference_start.elapsed());
    }

    if !quiet {
        print!("Saving result... ");
    }
    let save_start = Instant::now();
    output_image.save(output)
        .with_context(|| format!("Failed to save image: {}", output.display()))?;
    if !quiet {
        println!("done ({:.0?})", save_start.elapsed());
        println!();
    }

    Ok(())
}

// ============================================================================
// Convert Command
// ============================================================================

fn cmd_convert(
    format: ImageFormat,
    inputs: Vec<PathBuf>,
    output_dir: Option<PathBuf>,
    quality: u8,
    quiet: bool,
) -> Result<()> {
    let total_images = inputs.len();
    let is_batch = total_images > 1;

    if !quiet {
        println!("xeno-edit convert");
        println!("=================");
        println!("Target format: {}", format.name());
        if is_batch {
            println!("Batch mode: {} images", total_images);
        }
        if matches!(format, ImageFormat::Jpeg) {
            println!("JPEG quality: {}", quality);
        }
        println!();
    }

    let batch_start = Instant::now();
    let mut success_count = 0;
    let mut fail_count = 0;

    for (idx, input) in inputs.iter().enumerate() {
        let output = if let Some(ref dir) = output_dir {
            let stem = input.file_stem().unwrap_or(OsStr::new("output")).to_string_lossy();
            dir.join(format!("{}.{}", stem, format.extension()))
        } else {
            let stem = input.file_stem().unwrap_or(OsStr::new("output")).to_string_lossy();
            let parent = input.parent().unwrap_or(std::path::Path::new("."));
            parent.join(format!("{}.{}", stem, format.extension()))
        };

        if input == &output {
            if !quiet {
                println!("Skipping {} (already in {} format)", input.display(), format.name());
            }
            continue;
        }

        if !quiet && is_batch {
            println!("[{}/{}] {}", idx + 1, total_images, input.display());
        }

        match convert_single_image(input, &output, format, quality, quiet && !is_batch) {
            Ok(_) => {
                success_count += 1;
                if quiet {
                    println!("{}", output.display());
                } else if !is_batch {
                    println!("Output: {}", output.display());
                } else {
                    println!("  -> {}", output.display());
                }
            }
            Err(e) => {
                fail_count += 1;
                if !quiet {
                    eprintln!("  x Error: {}", e);
                } else {
                    eprintln!("Error converting {}: {}", input.display(), e);
                }
            }
        }
    }

    if !quiet && is_batch {
        println!();
        println!("Batch complete: {} succeeded, {} failed ({:.1?} total)",
            success_count, fail_count, batch_start.elapsed());
    }

    if fail_count > 0 && success_count == 0 {
        anyhow::bail!("All images failed to convert");
    }

    Ok(())
}

fn convert_single_image(
    input: &PathBuf,
    output: &PathBuf,
    format: ImageFormat,
    quality: u8,
    quiet: bool,
) -> Result<()> {
    use image::codecs::jpeg::JpegEncoder;
    use image::ImageEncoder;
    use std::fs::File;
    use std::io::BufWriter;

    if !quiet {
        print!("Loading... ");
    }
    let load_start = Instant::now();
    let img = image::open(input)
        .with_context(|| format!("Failed to open image: {}", input.display()))?;
    if !quiet {
        println!("done ({:.0?}, {}x{})", load_start.elapsed(), img.width(), img.height());
    }

    if !quiet {
        print!("Converting to {}... ", format.name());
    }
    let save_start = Instant::now();

    if matches!(format, ImageFormat::Jpeg) {
        let file = File::create(output)
            .with_context(|| format!("Failed to create file: {}", output.display()))?;
        let writer = BufWriter::new(file);
        let rgb_img = img.to_rgb8();
        let encoder = JpegEncoder::new_with_quality(writer, quality);
        encoder.write_image(
            rgb_img.as_raw(),
            rgb_img.width(),
            rgb_img.height(),
            image::ExtendedColorType::Rgb8,
        ).with_context(|| format!("Failed to encode JPEG: {}", output.display()))?;
    } else {
        img.save(output)
            .with_context(|| format!("Failed to save image: {}", output.display()))?;
    }

    if !quiet {
        println!("done ({:.0?})", save_start.elapsed());
        println!();
    }

    Ok(())
}

// ============================================================================
// GIF Command (Native - No FFmpeg!)
// ============================================================================
// ============================================================================
// Image Filter Command (FFmpeg-like image processing)
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn cmd_image_filter(
    inputs: Vec<PathBuf>,
    output_dir: Option<PathBuf>,
    format: Option<ImageFormat>,
    width: u32,
    height: u32,
    crop_region: Option<String>,
    brightness: i32,
    contrast: i32,
    saturation: i32,
    hue: i32,
    gamma: f32,
    blur: u32,
    sharpen: u32,
    grayscale: bool,
    sepia: bool,
    invert: bool,
    vignette: f32,
    denoise: u32,
    chromakey: f32,
    posterize: u8,
    solarize: u16,
    temperature: i32,
    tint: i32,
    vibrance: i32,
    edges: f32,
    emboss: f32,
    rotate: u16,
    flip: String,
    watermark: Option<PathBuf>,
    watermark_pos: String,
    watermark_opacity: f32,
    watermark_scale: f32,
    quality: u8,
    quiet: bool,
) -> Result<()> {
    let total_images = inputs.len();
    let is_batch = total_images > 1;

    if !quiet {
        println!("xeno-edit image-filter");
        println!("======================");
        if is_batch {
            println!("Batch mode: {} images", total_images);
        }

        // Print active filters
        let mut filters = Vec::new();
        if width > 0 || height > 0 { filters.push(format!("resize({}x{})", width, height)); }
        if crop_region.is_some() { filters.push(format!("crop({})", crop_region.as_ref().unwrap())); }
        if brightness != 0 { filters.push(format!("brightness({})", brightness)); }
        if contrast != 0 { filters.push(format!("contrast({})", contrast)); }
        if saturation != 0 { filters.push(format!("saturation({})", saturation)); }
        if hue != 0 { filters.push(format!("hue({})", hue)); }
        if (gamma - 1.0).abs() > 0.001 { filters.push(format!("gamma({:.2})", gamma)); }
        if temperature != 0 { filters.push(format!("temperature({})", temperature)); }
        if tint != 0 { filters.push(format!("tint({})", tint)); }
        if vibrance != 0 { filters.push(format!("vibrance({})", vibrance)); }
        if blur > 0 { filters.push(format!("blur({})", blur)); }
        if sharpen > 0 { filters.push(format!("sharpen({})", sharpen)); }
        if denoise > 0 { filters.push(format!("denoise({})", denoise)); }
        if vignette > 0.0 { filters.push(format!("vignette({:.1})", vignette)); }
        if edges > 0.0 { filters.push(format!("edges({:.1})", edges)); }
        if emboss > 0.0 { filters.push(format!("emboss({:.1})", emboss)); }
        if chromakey > 0.0 { filters.push(format!("chromakey({:.2})", chromakey)); }
        if posterize > 0 { filters.push(format!("posterize({})", posterize)); }
        if solarize < 256 { filters.push(format!("solarize({})", solarize)); }
        if grayscale { filters.push("grayscale".to_string()); }
        if sepia { filters.push("sepia".to_string()); }
        if invert { filters.push("invert".to_string()); }
        if rotate != 0 { filters.push(format!("rotate({}°)", rotate)); }
        if flip != "none" { filters.push(format!("flip({})", flip)); }
        if watermark.is_some() { filters.push("watermark".to_string()); }

        if filters.is_empty() {
            println!("Filters: (none - copying images)");
        } else {
            println!("Filters: {}", filters.join(", "));
        }
        println!();
    }

    // Load watermark if specified
    let watermark_img = if let Some(ref wm_path) = watermark {
        if !quiet {
            println!("Loading watermark: {}", wm_path.display());
        }
        Some(image::open(wm_path)
            .with_context(|| format!("Failed to load watermark: {}", wm_path.display()))?)
    } else {
        None
    };

    let batch_start = Instant::now();
    let mut success_count = 0;
    let mut fail_count = 0;

    for (idx, input) in inputs.iter().enumerate() {
        // Determine output path and format
        let out_format = format.unwrap_or_else(|| {
            // Detect format from input extension
            input.extension()
                .and_then(|e| e.to_str())
                .map(|ext| match ext.to_lowercase().as_str() {
                    "jpg" | "jpeg" => ImageFormat::Jpeg,
                    "png" => ImageFormat::Png,
                    "webp" => ImageFormat::Webp,
                    "gif" => ImageFormat::Gif,
                    "bmp" => ImageFormat::Bmp,
                    "tiff" | "tif" => ImageFormat::Tiff,
                    "ico" => ImageFormat::Ico,
                    _ => ImageFormat::Png,
                })
                .unwrap_or(ImageFormat::Png)
        });

        let output = if let Some(ref dir) = output_dir {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("Failed to create output directory: {}", dir.display()))?;
            let stem = input.file_stem().unwrap_or(OsStr::new("output")).to_string_lossy();
            dir.join(format!("{}_filtered.{}", stem, out_format.extension()))
        } else {
            let stem = input.file_stem().unwrap_or(OsStr::new("output")).to_string_lossy();
            let parent = input.parent().unwrap_or(std::path::Path::new("."));
            parent.join(format!("{}_filtered.{}", stem, out_format.extension()))
        };

        if !quiet && is_batch {
            print!("[{}/{}] {}... ", idx + 1, total_images, input.display());
        } else if !quiet {
            print!("Processing {}... ", input.display());
        }

        // Process single image
        match process_image_filter(
            input, &output, out_format,
            width, height, crop_region.as_ref(),
            brightness, contrast, saturation, hue, gamma,
            blur, sharpen, grayscale, sepia, invert,
            vignette, denoise, chromakey, posterize, solarize,
            temperature, tint, vibrance, edges, emboss,
            rotate, &flip, watermark_img.as_ref(), &watermark_pos, watermark_opacity, watermark_scale,
            quality,
        ) {
            Ok(_) => {
                success_count += 1;
                if quiet {
                    println!("{}", output.display());
                } else {
                    println!("-> {}", output.display());
                }
            }
            Err(e) => {
                fail_count += 1;
                if !quiet {
                    eprintln!("ERROR: {}", e);
                } else {
                    eprintln!("Error processing {}: {}", input.display(), e);
                }
            }
        }
    }

    if !quiet && is_batch {
        println!();
        println!("Batch complete: {} succeeded, {} failed ({:.2?} total)",
            success_count, fail_count, batch_start.elapsed());
    }

    if fail_count > 0 && success_count == 0 {
        anyhow::bail!("All images failed to process");
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_image_filter(
    input: &PathBuf,
    output: &PathBuf,
    format: ImageFormat,
    width: u32,
    height: u32,
    crop_region: Option<&String>,
    brightness: i32,
    contrast: i32,
    saturation: i32,
    hue: i32,
    gamma: f32,
    blur: u32,
    sharpen: u32,
    grayscale: bool,
    sepia: bool,
    invert: bool,
    vignette_strength: f32,
    denoise_strength: u32,
    chromakey_tolerance: f32,
    posterize_levels: u8,
    solarize_threshold: u16,
    temperature: i32,
    tint_value: i32,
    vibrance_amount: i32,
    edge_strength: f32,
    emboss_strength: f32,
    rotate: u16,
    flip: &str,
    watermark_img: Option<&image::DynamicImage>,
    watermark_pos: &str,
    watermark_opacity: f32,
    watermark_scale: f32,
    quality: u8,
) -> Result<()> {
    use image::imageops::FilterType;

    // Load image
    let mut img = image::open(input)
        .with_context(|| format!("Failed to open: {}", input.display()))?;

    // Apply crop first (before resize)
    if let Some(crop_str) = crop_region {
        let parts: Vec<u32> = crop_str
            .split(|c| c == ',' || c == ':')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        if parts.len() == 4 {
            let (x, y, w, h) = if crop_str.contains(':') {
                // FFmpeg style: w:h:x:y
                (parts[2], parts[3], parts[0], parts[1])
            } else {
                // Our style: x,y,w,h
                (parts[0], parts[1], parts[2], parts[3])
            };
            img = crop(&img, x, y, w, h).unwrap_or(img);
        }
    }

    // Apply resize
    if width > 0 || height > 0 {
        let (new_width, new_height) = match (width, height) {
            (w, 0) => {
                let aspect = img.height() as f32 / img.width() as f32;
                (w, (w as f32 * aspect) as u32)
            }
            (0, h) => {
                let aspect = img.width() as f32 / img.height() as f32;
                ((h as f32 * aspect) as u32, h)
            }
            (w, h) => (w, h),
        };
        img = img.resize_exact(new_width, new_height, FilterType::Lanczos3);
    }

    // Apply rotation
    img = match rotate {
        90 => rotate_90_cw(&img).unwrap_or(img),
        180 => rotate_180(&img).unwrap_or(img),
        270 => rotate_270_cw(&img).unwrap_or(img),
        _ => img,
    };

    // Apply flip
    img = match flip {
        "h" | "horizontal" => flip_horizontal(&img).unwrap_or(img),
        "v" | "vertical" => flip_vertical(&img).unwrap_or(img),
        "hv" | "both" => flip_both(&img).unwrap_or(img),
        _ => img,
    };

    // Apply color adjustments
    if brightness != 0 {
        img = adjust_brightness(&img, brightness as f32).unwrap_or(img);
    }
    if contrast != 0 {
        img = adjust_contrast(&img, contrast as f32).unwrap_or(img);
    }
    if saturation != 0 {
        img = adjust_saturation(&img, saturation as f32).unwrap_or(img);
    }
    if hue != 0 {
        img = adjust_hue(&img, hue as f32).unwrap_or(img);
    }
    if (gamma - 1.0).abs() > 0.001 {
        img = adjust_gamma(&img, gamma).unwrap_or(img);
    }

    // Apply filters
    if blur > 0 {
        img = gaussian_blur(&img, blur as f32).unwrap_or(img);
    }
    if sharpen > 0 {
        img = unsharp_mask(&img, 1.0, sharpen as i32).unwrap_or(img);
    }

    // Apply color effects
    if grayscale {
        img = to_grayscale(&img).unwrap_or(img);
    }
    if sepia {
        img = apply_sepia(&img).unwrap_or(img);
    }
    if invert {
        img = invert_colors(&img).unwrap_or(img);
    }

    // Apply advanced FFmpeg-equivalent filters
    if temperature != 0 {
        img = xeno_lib::color_temperature(&img, temperature as f32).unwrap_or(img);
    }
    if tint_value != 0 {
        img = xeno_lib::tint(&img, tint_value as f32).unwrap_or(img);
    }
    if vibrance_amount != 0 {
        img = xeno_lib::vibrance(&img, vibrance_amount as f32).unwrap_or(img);
    }
    if denoise_strength > 0 {
        img = xeno_lib::denoise(&img, denoise_strength).unwrap_or(img);
    }
    if vignette_strength > 0.0 {
        img = xeno_lib::vignette(&img, vignette_strength, 0.5).unwrap_or(img);
    }
    if edge_strength > 0.0 {
        img = xeno_lib::edge_detect(&img, edge_strength).unwrap_or(img);
    }
    if emboss_strength > 0.0 {
        img = xeno_lib::emboss(&img, emboss_strength).unwrap_or(img);
    }
    if chromakey_tolerance > 0.0 {
        img = xeno_lib::remove_green_screen(&img, chromakey_tolerance, 0.1).unwrap_or(img);
    }
    if posterize_levels > 0 {
        img = xeno_lib::posterize(&img, posterize_levels).unwrap_or(img);
    }
    if solarize_threshold < 256 {
        img = xeno_lib::solarize(&img, solarize_threshold as u8).unwrap_or(img);
    }

    // Apply watermark
    if let Some(wm) = watermark_img {
        let wm_target_width = (img.width() as f32 * watermark_scale) as u32;
        let wm_aspect = wm.height() as f32 / wm.width() as f32;
        let wm_target_height = (wm_target_width as f32 * wm_aspect) as u32;

        let scaled_wm = wm.resize_exact(wm_target_width, wm_target_height, FilterType::Lanczos3);

        let (x, y) = match watermark_pos {
            "tl" | "top-left" => (10u32, 10u32),
            "tr" | "top-right" => (img.width().saturating_sub(wm_target_width + 10), 10),
            "bl" | "bottom-left" => (10, img.height().saturating_sub(wm_target_height + 10)),
            "c" | "center" => (
                (img.width().saturating_sub(wm_target_width)) / 2,
                (img.height().saturating_sub(wm_target_height)) / 2,
            ),
            "br" | "bottom-right" | _ => (
                img.width().saturating_sub(wm_target_width + 10),
                img.height().saturating_sub(wm_target_height + 10),
            ),
        };

        img = apply_watermark(&img, &scaled_wm, x, y, watermark_opacity).unwrap_or(img);
    }

    // Save with appropriate format
    match format {
        ImageFormat::Jpeg => {
            use image::codecs::jpeg::JpegEncoder;
            use image::ImageEncoder;
            use std::fs::File;
            use std::io::BufWriter;

            let file = File::create(output)?;
            let writer = BufWriter::new(file);
            let encoder = JpegEncoder::new_with_quality(writer, quality);
            let rgb = img.to_rgb8();
            encoder.write_image(&rgb, rgb.width(), rgb.height(), image::ExtendedColorType::Rgb8)?;
        }
        ImageFormat::Webp => {
            // Use WebP quality
            img.save(output)?;
        }
        _ => {
            img.save(output)?;
        }
    }

    Ok(())
}

fn cmd_gif(
    output: PathBuf,
    inputs: Vec<PathBuf>,
    delay: u16,
    width: u32,
    loops: u16,
    quiet: bool,
) -> Result<()> {
    use gif::{Encoder, Frame, Repeat};
    use image::imageops::FilterType;
    use std::fs::File;

    if inputs.is_empty() {
        anyhow::bail!("No input images provided");
    }

    if !quiet {
        println!("xeno-edit gif (native)");
        println!("======================");
        println!("Output: {}", output.display());
        println!("Frames: {}", inputs.len());
        println!("Delay: {}ms", delay);
        if width > 0 {
            println!("Width: {}px", width);
        }
        println!("Loop: {}", if loops == 0 { "infinite".to_string() } else { loops.to_string() });
        println!();
    }

    let start = Instant::now();

    // Load first image to get dimensions
    if !quiet {
        print!("Loading frames... ");
    }

    let first_img = image::open(&inputs[0])
        .with_context(|| format!("Failed to open: {}", inputs[0].display()))?;

    let (target_width, target_height) = if width > 0 {
        let aspect = first_img.height() as f32 / first_img.width() as f32;
        (width, (width as f32 * aspect) as u32)
    } else {
        (first_img.width(), first_img.height())
    };

    // Create GIF encoder
    let file = File::create(&output)
        .with_context(|| format!("Failed to create: {}", output.display()))?;

    let mut encoder = Encoder::new(file, target_width as u16, target_height as u16, &[])
        .context("Failed to create GIF encoder")?;

    encoder.set_repeat(if loops == 0 { Repeat::Infinite } else { Repeat::Finite(loops) })
        .context("Failed to set repeat")?;

    if !quiet {
        println!("done");
    }

    // Process each frame
    for (idx, input_path) in inputs.iter().enumerate() {
        if !quiet {
            print!("  Frame {}/{}... ", idx + 1, inputs.len());
        }

        let img = image::open(input_path)
            .with_context(|| format!("Failed to open: {}", input_path.display()))?;

        // Resize if needed
        let img = if width > 0 {
            img.resize_exact(target_width, target_height, FilterType::Lanczos3)
        } else {
            img
        };

        // Convert to RGBA
        let rgba = img.to_rgba8();

        // Create frame with color quantization
        let mut frame = Frame::from_rgba_speed(
            target_width as u16,
            target_height as u16,
            &mut rgba.into_raw(),
            10, // Speed (1-30, higher = faster but lower quality)
        );
        frame.delay = delay / 10; // GIF delay is in 10ms units

        encoder.write_frame(&frame)
            .with_context(|| format!("Failed to write frame {}", idx + 1))?;

        if !quiet {
            println!("done");
        }
    }

    let elapsed = start.elapsed();

    if !quiet {
        println!();
        println!("GIF created: {} ({:.1?})", output.display(), elapsed);

        // Show file size
        if let Ok(meta) = std::fs::metadata(&output) {
            let size_kb = meta.len() as f64 / 1024.0;
            if size_kb > 1024.0 {
                println!("Size: {:.2} MB", size_kb / 1024.0);
            } else {
                println!("Size: {:.2} KB", size_kb);
            }
        }
    } else {
        println!("{}", output.display());
    }

    Ok(())
}

// ============================================================================
// Animated WebP Command (Native - No FFmpeg!)
// ============================================================================

fn cmd_awebp(
    output: PathBuf,
    inputs: Vec<PathBuf>,
    delay: u16,
    width: u32,
    quality: u8,
    lossless: bool,
    _loops: u16,  // WebP loop support handled by encoder options
    quiet: bool,
) -> Result<()> {
    use image::imageops::FilterType;
    use webp_animation::prelude::*;
    use std::fs;

    if inputs.is_empty() {
        anyhow::bail!("No input images provided");
    }

    if !quiet {
        println!("xeno-edit awebp (native)");
        println!("========================");
        println!("Output: {}", output.display());
        println!("Frames: {}", inputs.len());
        println!("Delay: {}ms", delay);
        if width > 0 {
            println!("Width: {}px", width);
        }
        println!("Quality: {}", quality);
        println!("Mode: {}", if lossless { "lossless" } else { "lossy" });
        println!();
    }

    let start = Instant::now();

    // Load first image to get dimensions
    if !quiet {
        print!("Analyzing frames... ");
    }

    let first_img = image::open(&inputs[0])
        .with_context(|| format!("Failed to open: {}", inputs[0].display()))?;

    let (target_width, target_height) = if width > 0 {
        let aspect = first_img.height() as f32 / first_img.width() as f32;
        (width, (width as f32 * aspect) as u32)
    } else {
        (first_img.width(), first_img.height())
    };

    if !quiet {
        println!("done ({}x{})", target_width, target_height);
    }

    // Create WebP encoder
    let mut encoder = Encoder::new((target_width, target_height))
        .context("Failed to create WebP encoder")?;

    // Set encoding config (lossy with quality)
    // Note: webp-animation uses lossy encoding; lossless flag adjusts quality to 100
    let effective_quality = if lossless { 100.0 } else { quality as f32 };
    encoder.set_default_encoding_config(EncodingConfig::new_lossy(effective_quality))
        .context("Failed to set encoding config")?;

    // Process each frame
    let mut timestamp_ms: i32 = 0;

    for (idx, input_path) in inputs.iter().enumerate() {
        if !quiet {
            print!("  Frame {}/{}... ", idx + 1, inputs.len());
        }

        let img = image::open(input_path)
            .with_context(|| format!("Failed to open: {}", input_path.display()))?;

        // Resize if needed
        let img = if width > 0 {
            img.resize_exact(target_width, target_height, FilterType::Lanczos3)
        } else {
            img
        };

        // Convert to RGBA
        let rgba = img.to_rgba8();
        let pixels = rgba.as_raw();

        // Add frame at current timestamp
        encoder.add_frame(pixels, timestamp_ms)
            .with_context(|| format!("Failed to encode frame {}", idx + 1))?;

        timestamp_ms += delay as i32;

        if !quiet {
            println!("done");
        }
    }

    // Finalize - the final timestamp marks the end of the last frame
    if !quiet {
        print!("Finalizing... ");
    }

    let webp_data = encoder.finalize(timestamp_ms)
        .context("Failed to finalize WebP animation")?;

    // Write to file
    fs::write(&output, webp_data.to_vec())
        .with_context(|| format!("Failed to write: {}", output.display()))?;

    if !quiet {
        println!("done");
    }

    let elapsed = start.elapsed();

    if !quiet {
        println!();
        println!("WebP created: {} ({:.1?})", output.display(), elapsed);

        // Show file size
        if let Ok(meta) = fs::metadata(&output) {
            let size_kb = meta.len() as f64 / 1024.0;
            if size_kb > 1024.0 {
                println!("Size: {:.2} MB", size_kb / 1024.0);
            } else {
                println!("Size: {:.2} KB", size_kb);
            }
        }
    } else {
        println!("{}", output.display());
    }

    Ok(())
}

// ============================================================================
// Helper: Load image with auto-detected format (ignores extension)
// ============================================================================

/// Load an image by detecting format from file contents, not extension.
/// This handles misnamed files (e.g., JPEG saved as .png).
fn load_image_auto(path: &PathBuf) -> Result<image::DynamicImage> {
    use image::ImageReader;
    use std::io::BufReader;
    use std::fs::File;

    let file = File::open(path)
        .with_context(|| format!("Cannot open file: {}", path.display()))?;

    let reader = ImageReader::new(BufReader::new(file))
        .with_guessed_format()
        .with_context(|| format!("Cannot detect image format: {}", path.display()))?;

    reader.decode()
        .with_context(|| format!("Cannot decode image: {}", path.display()))
}

// ============================================================================
// Video Encode Command (AV1)
// ============================================================================

fn cmd_video_encode(
    output: PathBuf,
    inputs: Vec<PathBuf>,
    fps: f64,
    speed: u8,
    quality: u8,
    bitrate: u32,
    width: u32,
    height: u32,
    threads: usize,
    quiet: bool,
) -> Result<()> {
    let start = Instant::now();

    if inputs.is_empty() {
        anyhow::bail!("No input images specified");
    }

    if !quiet {
        println!("xeno-edit video-encode (AV1)");
        println!("============================");
        println!("Frames: {}", inputs.len());
        println!("Output: {}", output.display());
        println!();
    }

    // Load first image to get dimensions
    if !quiet {
        print!("Scanning input images... ");
    }

    let first_img = load_image_auto(&inputs[0])
        .with_context(|| format!("Failed to load first image: {}", inputs[0].display()))?;

    let target_width = if width > 0 { width } else { first_img.width() };
    let target_height = if height > 0 { height } else { first_img.height() };

    if !quiet {
        println!("done");
        println!("Resolution: {}x{}", target_width, target_height);
        println!("Frame rate: {} fps", fps);
        println!("Speed preset: {} (0=slowest/best, 10=fastest)", speed);
        println!("Quality: {} (lower=better)", quality);
        if bitrate > 0 {
            println!("Bitrate: {} kbps", bitrate);
        }
        if threads > 0 {
            println!("Threads: {}", threads);
        }
        println!();
    }

    // Load all images
    if !quiet {
        print!("Loading {} images... ", inputs.len());
    }

    let images: Vec<image::DynamicImage> = inputs
        .iter()
        .map(|p| load_image_auto(p))
        .collect::<Result<Vec<_>>>()
        .context("Failed to load images")?;

    if !quiet {
        println!("done");
    }

    // Create encoder config
    let encoding_speed = match speed {
        0 => EncodingSpeed::Placebo,
        1..=2 => EncodingSpeed::VerySlow,
        3..=4 => EncodingSpeed::Slow,
        5..=6 => EncodingSpeed::Medium,
        7..=8 => EncodingSpeed::Fast,
        9 => EncodingSpeed::VeryFast,
        _ => EncodingSpeed::Ultrafast,
    };

    let config = Av1EncoderConfig::new(target_width, target_height)
        .with_frame_rate(fps)
        .with_speed(encoding_speed)
        .with_quality(quality)
        .with_bitrate(bitrate)
        .with_threads(threads);

    // Ensure output has .ivf extension
    let output_path = if output.extension().map_or(true, |e| e != "ivf") {
        output.with_extension("ivf")
    } else {
        output
    };

    // Encode
    if !quiet {
        println!("Encoding to AV1...");
    }

    let encode_start = Instant::now();
    let frame_count = encode_to_ivf(images.iter(), &output_path, config)
        .context("Failed to encode video")?;

    let encode_time = encode_start.elapsed();
    let total_time = start.elapsed();

    if !quiet {
        println!();
        println!("Encoding complete!");
        println!("==================");
        println!("Output: {}", output_path.display());
        println!("Frames: {}", frame_count);
        println!("Encode time: {:.2?}", encode_time);
        println!("Total time: {:.2?}", total_time);

        // Calculate FPS
        let encode_fps = frame_count as f64 / encode_time.as_secs_f64();
        println!("Encode speed: {:.2} fps", encode_fps);

        // Show file size
        if let Ok(meta) = std::fs::metadata(&output_path) {
            let size_kb = meta.len() as f64 / 1024.0;
            if size_kb > 1024.0 {
                println!("Size: {:.2} MB", size_kb / 1024.0);
            } else {
                println!("Size: {:.2} KB", size_kb);
            }

            // Calculate bitrate
            let duration_secs = frame_count as f64 / fps;
            let actual_bitrate = (meta.len() as f64 * 8.0) / (duration_secs * 1000.0);
            println!("Actual bitrate: {:.0} kbps", actual_bitrate);
        }

        println!();
        println!("To play: ffplay {}", output_path.display());
        println!("To remux to MP4: ffmpeg -i {} -c copy output.mp4", output_path.display());
    } else {
        println!("{}", output_path.display());
    }

    Ok(())
}

// ============================================================================
// Encode Sequence Command (pattern-based)
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn cmd_encode_sequence(
    pattern: String,
    output: PathBuf,
    start: usize,
    end: usize,
    fps: f64,
    speed: u8,
    quality: u8,
    bitrate: u32,
    width: u32,
    height: u32,
    threads: usize,
    quiet: bool,
) -> Result<()> {
    let start_time = Instant::now();

    if !quiet {
        println!("xeno-edit encode-sequence (AV1)");
        println!("===============================");
        println!("Pattern: {}", pattern);
        println!("Output: {}", output.display());
        println!();
    }

    // Auto-detect end frame if not specified
    let end_frame = if end == 0 {
        if !quiet {
            print!("Auto-detecting sequence range... ");
        }
        let detected = detect_sequence_range(&pattern, start)?;
        if !quiet {
            println!("found {} frames ({}–{})", detected.1 - detected.0 + 1, detected.0, detected.1);
        }
        detected.1
    } else {
        end
    };

    if start > end_frame {
        anyhow::bail!("Start frame {} is greater than end frame {}", start, end_frame);
    }

    let frame_count = end_frame - start + 1;

    if !quiet {
        println!("Frame range: {} to {} ({} frames)", start, end_frame, frame_count);
        println!("Frame rate: {} fps", fps);
        println!("Speed preset: {} (0=slowest/best, 10=fastest)", speed);
        println!("Quality: {} (lower=better)", quality);
        if bitrate > 0 {
            println!("Bitrate: {} kbps", bitrate);
        }
        if threads > 0 {
            println!("Threads: {}", threads);
        }
        println!();
    }

    // Load images
    if !quiet {
        println!("Loading {} images...", frame_count);
    }

    let load_start = Instant::now();
    let mut images: Vec<image::DynamicImage> = Vec::with_capacity(frame_count);
    let mut loaded = 0;

    for i in start..=end_frame {
        let path = format_pattern(&pattern, i);
        let img = load_image_auto(&PathBuf::from(&path))
            .with_context(|| format!("Failed to load frame {}: {}", i, path))?;
        images.push(img);
        loaded += 1;

        if !quiet && loaded % 100 == 0 {
            print!("\r  Loaded {}/{} frames", loaded, frame_count);
        }
    }

    if !quiet {
        println!("\r  Loaded {}/{} frames ({:.2?})", loaded, frame_count, load_start.elapsed());
    }

    if images.is_empty() {
        anyhow::bail!("No images loaded from sequence");
    }

    // Get dimensions from first image
    let first_img = &images[0];
    let target_width = if width > 0 { width } else { first_img.width() };
    let target_height = if height > 0 { height } else { first_img.height() };

    if !quiet {
        println!("Resolution: {}x{}", target_width, target_height);
        println!();
    }

    // Resize images if needed
    if target_width != first_img.width() || target_height != first_img.height() {
        if !quiet {
            print!("Resizing frames to {}x{}... ", target_width, target_height);
        }
        let resize_start = Instant::now();
        use image::imageops::FilterType;
        images = images.into_iter()
            .map(|img| img.resize_exact(target_width, target_height, FilterType::Lanczos3))
            .collect();
        if !quiet {
            println!("done ({:.2?})", resize_start.elapsed());
        }
    }

    // Create encoder config
    let encoding_speed = match speed {
        0 => EncodingSpeed::Placebo,
        1..=2 => EncodingSpeed::VerySlow,
        3..=4 => EncodingSpeed::Slow,
        5..=6 => EncodingSpeed::Medium,
        7..=8 => EncodingSpeed::Fast,
        9 => EncodingSpeed::VeryFast,
        _ => EncodingSpeed::Ultrafast,
    };

    let config = Av1EncoderConfig::new(target_width, target_height)
        .with_frame_rate(fps)
        .with_speed(encoding_speed)
        .with_quality(quality)
        .with_bitrate(bitrate)
        .with_threads(threads);

    // Ensure output has .ivf extension
    let output_path = if output.extension().map_or(true, |e| e != "ivf") {
        output.with_extension("ivf")
    } else {
        output
    };

    // Encode
    if !quiet {
        println!("Encoding to AV1...");
    }

    let encode_start = Instant::now();
    let encoded_frames = encode_to_ivf(images.iter(), &output_path, config)
        .context("Failed to encode video")?;

    let encode_time = encode_start.elapsed();
    let total_time = start_time.elapsed();

    if !quiet {
        println!();
        println!("Encoding complete!");
        println!("==================");
        println!("Output: {}", output_path.display());
        println!("Frames: {}", encoded_frames);
        println!("Encode time: {:.2?}", encode_time);
        println!("Total time: {:.2?}", total_time);

        // Calculate FPS
        let encode_fps = encoded_frames as f64 / encode_time.as_secs_f64();
        println!("Encode speed: {:.2} fps", encode_fps);

        // Show file size
        if let Ok(meta) = std::fs::metadata(&output_path) {
            let size_kb = meta.len() as f64 / 1024.0;
            if size_kb > 1024.0 {
                println!("Size: {:.2} MB", size_kb / 1024.0);
            } else {
                println!("Size: {:.2} KB", size_kb);
            }

            // Calculate bitrate
            let duration_secs = encoded_frames as f64 / fps;
            let actual_bitrate = (meta.len() as f64 * 8.0) / (duration_secs * 1000.0);
            println!("Actual bitrate: {:.0} kbps", actual_bitrate);
        }

        println!();
        println!("To play: ffplay {}", output_path.display());
        println!("To remux to MP4: ffmpeg -i {} -c copy output.mp4", output_path.display());
    } else {
        println!("{}", output_path.display());
    }

    Ok(())
}

/// Formats a path pattern with a frame index.
/// Supports printf-style %04d, %05d, etc. and {} placeholder.
fn format_pattern(pattern: &str, index: usize) -> String {
    // Support %04d, %05d, etc. format specifiers
    if let Some(pos) = pattern.find('%') {
        if let Some(end) = pattern[pos..].find('d') {
            let format_spec = &pattern[pos..pos + end + 1];
            let width = format_spec
                .chars()
                .filter(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<usize>()
                .unwrap_or(4);

            let formatted = format!("{:0width$}", index, width = width);
            return pattern.replace(format_spec, &formatted);
        }
    }

    // Fallback: simple {} replacement
    pattern.replace("{}", &index.to_string())
}

/// Auto-detect the range of a frame sequence.
fn detect_sequence_range(pattern: &str, start: usize) -> Result<(usize, usize)> {
    let mut first = None;
    let mut last = start;

    // Search forward from start
    for i in start..100000 {
        let path = format_pattern(pattern, i);
        if std::path::Path::new(&path).exists() {
            if first.is_none() {
                first = Some(i);
            }
            last = i;
        } else if first.is_some() {
            // Gap found, stop searching
            break;
        }
    }

    match first {
        Some(f) => Ok((f, last)),
        None => anyhow::bail!("No frames found matching pattern: {}", pattern),
    }
}

// ============================================================================
// Video Info Command
// ============================================================================

fn cmd_video_info(inputs: Vec<PathBuf>, json: bool, quiet: bool) -> Result<()> {
    for input in &inputs {
        let ext = input.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        match ext.as_str() {
            "mp4" | "m4v" | "mov" => {
                show_mp4_info(input, json, quiet)?;
            }
            "ivf" => {
                show_ivf_info(input, json, quiet)?;
            }
            "mkv" | "webm" => {
                show_container_info(input, json, quiet)?;
            }
            _ => {
                if !quiet && !json {
                    println!("File: {}", input.display());
                    println!("Error: Unsupported format '{}'", ext);
                    println!("Supported: mp4, m4v, mov, ivf, mkv, webm");
                    println!();
                }
            }
        }
    }

    Ok(())
}

fn show_container_info(path: &PathBuf, json: bool, quiet: bool) -> Result<()> {
    let file_size = std::fs::metadata(path)
        .with_context(|| format!("Cannot stat file: {}", path.display()))?
        .len();

    let demuxer = match open_container(path) {
        Ok(d) => d,
        Err(e) => {
            if json {
                println!("{{");
                println!("  \"file\": \"{}\",", path.display());
                println!("  \"error\": \"{}\"", e.to_string().replace('"', "\\\""));
                println!("}}");
            } else if !quiet {
                println!("File: {}", path.display());
                println!("Error: {}", e);
                println!();
            }
            return Ok(());
        }
    };

    let v = demuxer.video_info().cloned();
    let a = demuxer.audio_info().cloned();
    let container = format!("{:?}", demuxer.container_type());

    if json {
        println!("{{");
        println!("  \"file\": \"{}\",", path.display());
        println!("  \"format\": \"{}\",", container);
        println!("  \"size_bytes\": {},", file_size);

        if let Some(info) = &v {
            println!("  \"video\": {{");
            println!("    \"codec\": \"{}\",", info.codec);
            println!("    \"width\": {},", info.width);
            println!("    \"height\": {},", info.height);
            println!(
                "    \"frame_rate\": {},",
                info.frame_rate.map(|f| format!("{:.3}", f)).unwrap_or_else(|| "null".to_string())
            );
            println!(
                "    \"duration_secs\": {},",
                info.duration.map(|d| format!("{:.3}", d)).unwrap_or_else(|| "null".to_string())
            );
            println!(
                "    \"frame_count\": {}",
                info.frame_count.map(|c| c.to_string()).unwrap_or_else(|| "null".to_string())
            );
            println!("  }},");
        } else {
            println!("  \"video\": null,");
        }

        if let Some(info) = &a {
            println!("  \"audio\": {{");
            println!("    \"codec\": \"{}\",", info.codec);
            println!("    \"sample_rate\": {},", info.sample_rate);
            println!("    \"channels\": {},", info.channels);
            println!(
                "    \"duration_secs\": {}",
                info.duration.map(|d| format!("{:.3}", d)).unwrap_or_else(|| "null".to_string())
            );
            println!("  }}");
        } else {
            println!("  \"audio\": null");
        }

        println!("}}");
        return Ok(());
    }

    if !quiet {
        println!("xeno-edit video-info");
        println!("====================");
        println!();
    }

    println!("File: {}", path.display());
    println!("Format: {}", container);
    println!("Size: {}", format_size(file_size));

    if let Some(info) = v {
        println!("Video Codec: {}", info.codec);
        println!("Resolution: {}x{}", info.width, info.height);
        if let Some(fps) = info.frame_rate {
            println!("Frame rate: {:.2} fps", fps);
        }
        if let Some(frames) = info.frame_count {
            println!("Frames: {}", frames);
        }
        if let Some(duration) = info.duration {
            println!("Duration: {}", format_duration((duration * 1000.0) as u64));
        }
    } else {
        println!("Video: none");
    }

    if let Some(info) = a {
        println!("Audio Codec: {}", info.codec);
        println!("Sample rate: {} Hz", info.sample_rate);
        println!("Channels: {}", info.channels);
    }

    println!();
    Ok(())
}

fn show_mp4_info(path: &PathBuf, json: bool, quiet: bool) -> Result<()> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path)
        .with_context(|| format!("Cannot open file: {}", path.display()))?;
    let size = file.metadata()?.len();
    let reader = BufReader::new(file);

    let mp4 = mp4::Mp4Reader::read_header(reader, size)
        .with_context(|| format!("Failed to parse MP4: {}", path.display()))?;

    if json {
        // JSON output
        println!("{{");
        println!("  \"file\": \"{}\",", path.display());
        println!("  \"format\": \"MP4\",");
        println!("  \"size_bytes\": {},", size);
        println!("  \"duration_ms\": {},", mp4.duration().as_millis());
        println!("  \"tracks\": [");

        let tracks: Vec<_> = mp4.tracks().values().collect();
        for (i, track) in tracks.iter().enumerate() {
            println!("    {{");
            println!("      \"id\": {},", track.track_id());
            println!("      \"type\": \"{:?}\",", track.track_type());
            let codec = match track.media_type() {
                Ok(mt) => format!("{:?}", mt),
                Err(_) => "unknown".to_string(),
            };
            println!("      \"codec\": \"{}\",", codec);
            println!("      \"duration_ms\": {},", track.duration().as_millis());

            if let Ok(mp4::TrackType::Video) = track.track_type() {
                println!("      \"width\": {},", track.width());
                println!("      \"height\": {},", track.height());
                println!("      \"frame_rate\": {:.2},", track.frame_rate());
                println!("      \"bitrate\": {}", track.bitrate());
            }

            if i < tracks.len() - 1 {
                println!("    }},");
            } else {
                println!("    }}");
            }
        }
        println!("  ]");
        println!("}}");
    } else {
        // Human-readable output
        if !quiet {
            println!("xeno-edit video-info");
            println!("====================");
            println!();
        }

        println!("File: {}", path.display());
        println!("Format: MP4");
        println!("Size: {}", format_size(size));
        println!("Duration: {}", format_duration(mp4.duration().as_millis() as u64));
        println!();

        for track in mp4.tracks().values() {
            let track_type = track.track_type().unwrap_or(mp4::TrackType::Video);
            println!("Track #{} ({:?})", track.track_id(), track_type);
            let codec = match track.media_type() {
                Ok(mt) => format!("{:?}", mt),
                Err(_) => "unknown".to_string(),
            };
            println!("  Codec: {}", codec);
            println!("  Duration: {}", format_duration(track.duration().as_millis() as u64));

            if let mp4::TrackType::Video = track_type {
                println!("  Resolution: {}x{}", track.width(), track.height());
                println!("  Frame rate: {:.2} fps", track.frame_rate());
                println!("  Bitrate: {} kbps", track.bitrate() / 1000);
            }

            if let mp4::TrackType::Audio = track_type {
                // Sample rate from sample_freq_index or default to 44100
                let sample_rate = match track.sample_freq_index() {
                    Ok(sfi) => match sfi {
                        mp4::SampleFreqIndex::Freq96000 => 96000,
                        mp4::SampleFreqIndex::Freq88200 => 88200,
                        mp4::SampleFreqIndex::Freq64000 => 64000,
                        mp4::SampleFreqIndex::Freq48000 => 48000,
                        mp4::SampleFreqIndex::Freq44100 => 44100,
                        mp4::SampleFreqIndex::Freq32000 => 32000,
                        mp4::SampleFreqIndex::Freq24000 => 24000,
                        mp4::SampleFreqIndex::Freq22050 => 22050,
                        mp4::SampleFreqIndex::Freq16000 => 16000,
                        mp4::SampleFreqIndex::Freq12000 => 12000,
                        mp4::SampleFreqIndex::Freq11025 => 11025,
                        mp4::SampleFreqIndex::Freq8000 => 8000,
                        mp4::SampleFreqIndex::Freq7350 => 7350,
                    },
                    Err(_) => 44100,
                };
                let channels = match track.channel_config() {
                    Ok(cc) => cc as u8,
                    Err(_) => 2,
                };
                println!("  Sample rate: {} Hz", sample_rate);
                println!("  Channels: {}", channels);
            }

            println!();
        }
    }

    Ok(())
}

fn show_ivf_info(path: &PathBuf, json: bool, quiet: bool) -> Result<()> {
    use std::fs::File;
    use std::io::{BufReader, Read};

    let file = File::open(path)
        .with_context(|| format!("Cannot open file: {}", path.display()))?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    // Read IVF header (32 bytes)
    let mut header = [0u8; 32];
    reader.read_exact(&mut header)
        .with_context(|| "Failed to read IVF header")?;

    // Parse IVF header
    // Bytes 0-3: "DKIF" signature
    // Bytes 4-5: Version (should be 0)
    // Bytes 6-7: Header size (should be 32)
    // Bytes 8-11: FourCC codec
    // Bytes 12-13: Width
    // Bytes 14-15: Height
    // Bytes 16-19: Frame rate denominator
    // Bytes 20-23: Frame rate numerator
    // Bytes 24-27: Frame count
    // Bytes 28-31: Unused

    let signature = &header[0..4];
    if signature != b"DKIF" {
        anyhow::bail!("Invalid IVF file: bad signature");
    }

    let codec = std::str::from_utf8(&header[8..12]).unwrap_or("????");
    let width = u16::from_le_bytes([header[12], header[13]]);
    let height = u16::from_le_bytes([header[14], header[15]]);
    let fps_den = u32::from_le_bytes([header[16], header[17], header[18], header[19]]);
    let fps_num = u32::from_le_bytes([header[20], header[21], header[22], header[23]]);
    let frame_count = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);

    let frame_rate = if fps_den > 0 { fps_num as f64 / fps_den as f64 } else { 0.0 };
    let duration_ms = if frame_rate > 0.0 { (frame_count as f64 / frame_rate * 1000.0) as u64 } else { 0 };
    let bitrate = if duration_ms > 0 { (file_size * 8 * 1000) / duration_ms } else { 0 };

    let codec_name = match codec {
        "AV01" => "AV1",
        "VP90" => "VP9",
        "VP80" => "VP8",
        _ => codec,
    };

    if json {
        println!("{{");
        println!("  \"file\": \"{}\",", path.display());
        println!("  \"format\": \"IVF\",");
        println!("  \"codec\": \"{}\",", codec_name);
        println!("  \"size_bytes\": {},", file_size);
        println!("  \"width\": {},", width);
        println!("  \"height\": {},", height);
        println!("  \"frame_rate\": {:.2},", frame_rate);
        println!("  \"frame_count\": {},", frame_count);
        println!("  \"duration_ms\": {},", duration_ms);
        println!("  \"bitrate_kbps\": {}", bitrate / 1000);
        println!("}}");
    } else {
        if !quiet {
            println!("xeno-edit video-info");
            println!("====================");
            println!();
        }

        println!("File: {}", path.display());
        println!("Format: IVF ({})", codec_name);
        println!("Size: {}", format_size(file_size));
        println!("Resolution: {}x{}", width, height);
        println!("Frame rate: {:.2} fps", frame_rate);
        println!("Frames: {}", frame_count);
        println!("Duration: {}", format_duration(duration_ms));
        println!("Bitrate: {} kbps", bitrate / 1000);
        println!();
    }

    Ok(())
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} bytes", bytes)
    }
}

fn format_duration(ms: u64) -> String {
    let secs = ms / 1000;
    let mins = secs / 60;
    let hours = mins / 60;

    if hours > 0 {
        format!("{}:{:02}:{:02}", hours, mins % 60, secs % 60)
    } else if mins > 0 {
        format!("{}:{:02}", mins, secs % 60)
    } else {
        format!("{}.{:03}s", secs, ms % 1000)
    }
}

// ============================================================================
// GPU Info Command
// ============================================================================

fn cmd_gpu_info(device: i32, json: bool) -> Result<()> {
    if !json {
        println!("xeno-edit gpu-info (NVDEC)");
        println!("==========================");
        println!();
    }

    // Check if NVDEC is available
    if !NvDecoder::is_available() {
        if json {
            println!("{{");
            println!("  \"available\": false,");
            println!("  \"error\": \"NVDEC not available. Is NVIDIA driver installed?\"");
            println!("}}");
        } else {
            println!("NVDEC Status: NOT AVAILABLE");
            println!("Error: NVIDIA driver not found or GPU not supported");
            println!();
            println!("Requirements:");
            println!("  - NVIDIA GPU with NVDEC support");
            println!("  - NVIDIA driver installed");
        }
        return Ok(());
    }

    // Query capabilities for each codec
    let codecs = [
        (DecodeCodec::Av1, "AV1"),
        (DecodeCodec::H264, "H.264"),
        (DecodeCodec::H265, "H.265/HEVC"),
        (DecodeCodec::Vp9, "VP9"),
        (DecodeCodec::Vp8, "VP8"),
    ];

    if json {
        println!("{{");
        println!("  \"available\": true,");
        println!("  \"device_index\": {},", device);
        println!("  \"codecs\": [");
    } else {
        println!("Device: GPU #{}", device);
        println!("NVDEC Status: AVAILABLE");
        println!();
        println!("Supported Codecs:");
        println!("-----------------");
    }

    let mut first = true;
    for (codec, name) in codecs.iter() {
        match NvDecoder::query_capabilities(*codec, device) {
            Ok(caps) => {
                if json {
                    if !first {
                        println!(",");
                    }
                    first = false;
                    print!("    {{");
                    print!("\"name\": \"{}\", ", name);
                    print!("\"supported\": {}, ", caps.supported);
                    if caps.supported {
                        print!("\"max_width\": {}, ", caps.max_width);
                        print!("\"max_height\": {}, ", caps.max_height);
                        print!("\"max_bit_depth\": {}, ", caps.max_bit_depth);
                        print!("\"nvdec_engines\": {}", caps.num_engines);
                    }
                    print!("}}");
                } else {
                    if caps.supported {
                        println!(
                            "  {} ✓ (max {}x{}, {}-bit, {} NVDEC engine(s))",
                            name, caps.max_width, caps.max_height, caps.max_bit_depth, caps.num_engines
                        );
                    } else {
                        println!("  {} ✗ (not supported)", name);
                    }
                }
            }
            Err(e) => {
                if json {
                    if !first {
                        println!(",");
                    }
                    first = false;
                    print!("    {{\"name\": \"{}\", \"error\": \"{}\"}}", name, e);
                } else {
                    println!("  {} ✗ (error: {})", name, e);
                }
            }
        }
    }

    if json {
        println!();
        println!("  ]");
        println!("}}");
    } else {
        println!();
        println!("Use 'xeno-edit video-frames' to extract frames from video.");
    }

    Ok(())
}

// ============================================================================
// Audio Info Command (Pure Rust via Symphonia)
// ============================================================================

fn cmd_audio_info(inputs: Vec<PathBuf>, json: bool, quiet: bool) -> Result<()> {
    for (idx, input) in inputs.iter().enumerate() {
        if idx > 0 && !json {
            println!();
        }

        match AudioInfo::from_file(input) {
            Ok(info) => {
                if json {
                    println!("{{");
                    println!("  \"file\": \"{}\",", input.display());
                    println!("  \"format\": \"{}\",", info.format.name());
                    println!("  \"codec\": \"{}\",", info.codec.name());
                    println!("  \"lossless\": {},", info.codec.is_lossless());
                    println!("  \"sample_rate\": {},", info.sample_rate);
                    println!("  \"channels\": {},", info.channels);
                    if let Some(bps) = info.bits_per_sample {
                        println!("  \"bits_per_sample\": {},", bps);
                    }
                    println!("  \"duration_secs\": {:.3},", info.duration_secs);
                    if let Some(samples) = info.total_samples {
                        println!("  \"total_samples\": {},", samples);
                    }
                    if let Some(bitrate) = info.bitrate_kbps {
                        println!("  \"bitrate_kbps\": {},", bitrate);
                    }
                    println!("  \"file_size\": {},", info.file_size);
                    if let Some(ref title) = info.title {
                        println!("  \"title\": \"{}\",", title);
                    }
                    if let Some(ref artist) = info.artist {
                        println!("  \"artist\": \"{}\",", artist);
                    }
                    if let Some(ref album) = info.album {
                        println!("  \"album\": \"{}\"", album);
                    }
                    println!("}}");
                } else {
                    if !quiet {
                        println!("xeno-edit audio-info");
                        println!("====================");
                        println!();
                    }

                    println!("File: {}", input.display());
                    println!("Format: {} ({})", info.format.name(), info.codec.name());
                    println!("Quality: {}", if info.codec.is_lossless() { "Lossless" } else { "Lossy" });
                    println!("Sample Rate: {} Hz", info.sample_rate);
                    println!("Channels: {}", info.channels);
                    if let Some(bps) = info.bits_per_sample {
                        println!("Bit Depth: {}-bit", bps);
                    }
                    println!("Duration: {}", info.duration_string());
                    if let Some(bitrate) = info.bitrate_kbps {
                        println!("Bitrate: {} kbps", bitrate);
                    }
                    println!("Size: {}", format_size(info.file_size));

                    // Show metadata if present
                    if info.title.is_some() || info.artist.is_some() || info.album.is_some() {
                        println!();
                        println!("Metadata:");
                        if let Some(ref title) = info.title {
                            println!("  Title: {}", title);
                        }
                        if let Some(ref artist) = info.artist {
                            println!("  Artist: {}", artist);
                        }
                        if let Some(ref album) = info.album {
                            println!("  Album: {}", album);
                        }
                    }
                }
            }
            Err(e) => {
                if json {
                    println!("{{");
                    println!("  \"file\": \"{}\",", input.display());
                    println!("  \"error\": \"{}\"", e);
                    println!("}}");
                } else {
                    eprintln!("Error: {} - {}", input.display(), e);
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Extract Audio Command (Pure Rust via Symphonia)
// ============================================================================

fn cmd_extract_audio(
    input: PathBuf,
    output: Option<PathBuf>,
    mono: bool,
    quiet: bool,
) -> Result<()> {
    let start = Instant::now();

    // Determine output path
    let output_path = output.unwrap_or_else(|| {
        let stem = input.file_stem().unwrap_or_default().to_string_lossy();
        let parent = input.parent().unwrap_or(std::path::Path::new("."));
        parent.join(format!("{}.wav", stem))
    });

    if !quiet {
        println!("xeno-edit extract-audio");
        println!("=======================");
        println!("Input: {}", input.display());
        println!("Output: {}", output_path.display());
        if mono {
            println!("Mode: Convert to mono");
        }
        println!();
    }

    // Decode audio
    if !quiet {
        print!("Decoding audio... ");
    }

    let decode_start = Instant::now();
    let mut audio = decode_file(&input)
        .with_context(|| format!("Failed to decode audio: {}", input.display()))?;

    if !quiet {
        println!("done ({:.2?})", decode_start.elapsed());
        println!("  Sample rate: {} Hz", audio.sample_rate);
        println!("  Channels: {}", audio.channels);
        println!("  Duration: {:.2}s", audio.duration_secs());
        println!("  Samples: {}", audio.samples.len());
    }

    // Convert to mono if requested
    if mono && audio.channels > 1 {
        if !quiet {
            print!("Converting to mono... ");
        }
        audio = audio.to_mono();
        if !quiet {
            println!("done");
        }
    }

    // Save as WAV
    if !quiet {
        print!("Saving WAV... ");
    }

    let save_start = Instant::now();
    audio.save_wav(&output_path)
        .with_context(|| format!("Failed to save WAV: {}", output_path.display()))?;

    if !quiet {
        println!("done ({:.2?})", save_start.elapsed());
    }

    let elapsed = start.elapsed();

    if !quiet {
        println!();
        println!("Audio extracted: {} ({:.2?})", output_path.display(), elapsed);

        // Show file size
        if let Ok(meta) = std::fs::metadata(&output_path) {
            println!("Size: {}", format_size(meta.len()));
        }
    } else {
        println!("{}", output_path.display());
    }

    Ok(())
}

// ============================================================================
// Video Frames Command (NVDEC GPU-accelerated)
// ============================================================================

fn cmd_video_frames(
    input: PathBuf,
    output_dir: Option<PathBuf>,
    format: String,
    every: u32,
    max_frames: u32,
    quiet: bool,
) -> Result<()> {
    let start = Instant::now();

    // Determine output directory
    let output_dir = output_dir.unwrap_or_else(|| PathBuf::from("./frames"));

    if !quiet {
        println!("xeno-edit video-frames (NVDEC)");
        println!("==============================");
        println!("Input: {}", input.display());
        println!("Output: {}", output_dir.display());
        println!("Format: {}", format);
        if every > 1 {
            println!("Extract every {}th frame", every);
        }
        if max_frames > 0 {
            println!("Max frames: {}", max_frames);
        }
        println!();
    }

    // Check if input exists
    if !input.exists() {
        anyhow::bail!("Input file does not exist: {}", input.display());
    }

    // Check decoder availability
    if !quiet {
        print!("Checking decoder... ");
    }

    // Read IVF header to get codec
    let header = std::fs::read(&input)
        .with_context(|| format!("Failed to read file: {}", input.display()))?;

    if header.len() < 32 || &header[0..4] != b"DKIF" {
        anyhow::bail!("Not a valid IVF file");
    }

    let fourcc = std::str::from_utf8(&header[8..12]).unwrap_or("????");
    let codec = DecodeCodec::from_fourcc(fourcc)
        .ok_or_else(|| anyhow::anyhow!("Unknown codec: {}", fourcc))?;

    let backend = best_decoder_for(codec);
    match backend {
        DecoderBackend::Nvdec => {
            if !quiet {
                println!("NVDEC (GPU)");
            }
        }
        DecoderBackend::Dav1d => {
            if !quiet {
                println!("dav1d (software)");
            }
        }
        DecoderBackend::None => {
            anyhow::bail!("No decoder available for {:?}. NVDEC GPU required.", codec);
        }
    }

    // Create output directory
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create output directory: {}", output_dir.display()))?;

    // Decode video
    if !quiet {
        print!("Decoding video... ");
    }

    let decode_start = Instant::now();
    let frames = decode_ivf(&input)
        .with_context(|| format!("Failed to decode video: {}", input.display()))?;

    if !quiet {
        println!("done ({:.2?}, {} frames)", decode_start.elapsed(), frames.len());
    }

    // Extract and save frames
    if !quiet {
        println!("Saving frames...");
    }

    let save_start = Instant::now();
    let mut saved_count = 0;
    let mut frame_idx = 0;

    for (idx, frame) in frames.iter().enumerate() {
        // Skip frames based on 'every' parameter
        if every > 1 && idx % every as usize != 0 {
            continue;
        }

        // Check max frames limit
        if max_frames > 0 && saved_count >= max_frames {
            break;
        }

        let output_path = output_dir.join(format!("frame_{:06}.{}", frame_idx, format));

        // Convert to image and save
        let img = frame.to_rgba_image()
            .with_context(|| format!("Failed to convert frame {}", idx))?;

        img.save(&output_path)
            .with_context(|| format!("Failed to save frame: {}", output_path.display()))?;

        if !quiet {
            print!("\r  Frame {}/{} saved", saved_count + 1,
                if max_frames > 0 { max_frames.min(frames.len() as u32 / every.max(1)) } else { frames.len() as u32 / every.max(1) });
        }

        saved_count += 1;
        frame_idx += 1;
    }

    if !quiet {
        println!();
    }

    let elapsed = start.elapsed();

    if !quiet {
        println!();
        println!("Extraction complete!");
        println!("====================");
        println!("Frames saved: {}", saved_count);
        println!("Output: {}", output_dir.display());
        println!("Decode time: {:.2?}", decode_start.elapsed());
        println!("Save time: {:.2?}", save_start.elapsed());
        println!("Total time: {:.2?}", elapsed);

        // Calculate FPS
        let extract_fps = saved_count as f64 / elapsed.as_secs_f64();
        println!("Speed: {:.2} frames/sec", extract_fps);
    } else {
        println!("{}", output_dir.display());
    }

    Ok(())
}

// ============================================================================
// Video to GIF Command (NVDEC + Native GIF Encoder)
// ============================================================================

fn cmd_video_to_gif(
    input: PathBuf,
    output: PathBuf,
    width: u32,
    fps: u32,
    skip: u32,
    max_frames: u32,
    loops: u16,
    quiet: bool,
) -> Result<()> {
    use gif::{Encoder, Frame, Repeat};
    use image::imageops::FilterType;
    use std::fs::File;

    let start = Instant::now();

    if !quiet {
        println!("xeno-edit video-to-gif (NVDEC + native)");
        println!("=======================================");
        println!("Input: {}", input.display());
        println!("Output: {}", output.display());
        if width > 0 {
            println!("Resize width: {}px", width);
        }
        if fps > 0 {
            println!("Target FPS: {}", fps);
        }
        if skip > 1 {
            println!("Skip: every {}th frame", skip);
        }
        if max_frames > 0 {
            println!("Max frames: {}", max_frames);
        }
        println!("Loop: {}", if loops == 0 { "infinite".to_string() } else { loops.to_string() });
        println!();
    }

    // Check if input exists
    if !input.exists() {
        anyhow::bail!("Input file does not exist: {}", input.display());
    }

    // Decode video
    if !quiet {
        print!("Decoding video... ");
    }

    let decode_start = Instant::now();
    let frames = decode_ivf(&input)
        .with_context(|| format!("Failed to decode video: {}", input.display()))?;

    if !quiet {
        println!("done ({:.2?}, {} frames)", decode_start.elapsed(), frames.len());
    }

    if frames.is_empty() {
        anyhow::bail!("No frames decoded from video");
    }

    // Determine dimensions
    let first_frame = &frames[0];
    let (target_width, target_height) = if width > 0 {
        let aspect = first_frame.height as f32 / first_frame.width as f32;
        (width, (width as f32 * aspect) as u32)
    } else {
        (first_frame.width, first_frame.height)
    };

    // Calculate delay
    let delay_ms = if fps > 0 {
        1000 / fps
    } else {
        // Default to 100ms (10 fps) if no fps specified
        100
    };

    // Create GIF encoder
    if !quiet {
        print!("Creating GIF encoder... ");
    }

    let file = File::create(&output)
        .with_context(|| format!("Failed to create: {}", output.display()))?;

    let mut encoder = Encoder::new(file, target_width as u16, target_height as u16, &[])
        .context("Failed to create GIF encoder")?;

    encoder.set_repeat(if loops == 0 { Repeat::Infinite } else { Repeat::Finite(loops) })
        .context("Failed to set repeat")?;

    if !quiet {
        println!("done ({}x{})", target_width, target_height);
    }

    // Process frames
    if !quiet {
        println!("Encoding frames...");
    }

    let encode_start = Instant::now();
    let mut frame_count = 0;
    let skip = skip.max(1) as usize;

    for (idx, frame) in frames.iter().enumerate() {
        // Skip frames
        if idx % skip != 0 {
            continue;
        }

        // Check max frames
        if max_frames > 0 && frame_count >= max_frames {
            break;
        }

        // Convert to RGBA image
        let img = frame.to_rgba_image()
            .with_context(|| format!("Failed to convert frame {}", idx))?;

        // Resize if needed
        let img = if width > 0 {
            img.resize_exact(target_width, target_height, FilterType::Lanczos3)
        } else {
            img
        };

        // Convert to RGBA8
        let rgba = img.to_rgba8();

        // Create GIF frame
        let mut gif_frame = Frame::from_rgba_speed(
            target_width as u16,
            target_height as u16,
            &mut rgba.into_raw(),
            10, // Speed (1-30, higher = faster)
        );
        gif_frame.delay = (delay_ms / 10) as u16; // GIF delay is in 10ms units

        encoder.write_frame(&gif_frame)
            .with_context(|| format!("Failed to write frame {}", idx))?;

        frame_count += 1;

        if !quiet {
            print!("\r  Frame {}/{}", frame_count,
                if max_frames > 0 { max_frames.min(frames.len() as u32 / skip as u32) } else { frames.len() as u32 / skip as u32 });
        }
    }

    if !quiet {
        println!();
    }

    let elapsed = start.elapsed();

    if !quiet {
        println!();
        println!("GIF created: {}", output.display());
        println!("================");
        println!("Frames: {}", frame_count);
        println!("Resolution: {}x{}", target_width, target_height);
        println!("Decode time: {:.2?}", decode_start.elapsed());
        println!("Encode time: {:.2?}", encode_start.elapsed());
        println!("Total time: {:.2?}", elapsed);

        // Show file size
        if let Ok(meta) = std::fs::metadata(&output) {
            println!("Size: {}", format_size(meta.len()));
        }
    } else {
        println!("{}", output.display());
    }

    Ok(())
}

// ============================================================================
// Video Thumbnail Command (NVDEC GPU-accelerated)
// ============================================================================

fn cmd_video_thumbnail(
    input: PathBuf,
    output: PathBuf,
    position: String,
    width: u32,
    quiet: bool,
) -> Result<()> {
    use image::imageops::FilterType;

    let start = Instant::now();

    if !quiet {
        println!("xeno-edit video-thumbnail (NVDEC)");
        println!("=================================");
        println!("Input: {}", input.display());
        println!("Output: {}", output.display());
        println!("Position: {}", position);
        if width > 0 {
            println!("Resize width: {}px", width);
        }
        println!();
    }

    // Check if input exists
    if !input.exists() {
        anyhow::bail!("Input file does not exist: {}", input.display());
    }

    // Decode video
    if !quiet {
        print!("Decoding video... ");
    }

    let decode_start = Instant::now();
    let frames = decode_ivf(&input)
        .with_context(|| format!("Failed to decode video: {}", input.display()))?;

    if !quiet {
        println!("done ({:.2?}, {} frames)", decode_start.elapsed(), frames.len());
    }

    if frames.is_empty() {
        anyhow::bail!("No frames decoded from video");
    }

    // Determine which frame to use
    let frame_idx = match position.as_str() {
        "first" | "0" => 0,
        "middle" => frames.len() / 2,
        "last" => frames.len() - 1,
        s => {
            // Try to parse as number
            s.parse::<usize>()
                .with_context(|| format!("Invalid position: {}. Use 'first', 'middle', 'last', or a number.", s))?
                .min(frames.len() - 1)
        }
    };

    if !quiet {
        println!("Using frame {} of {}", frame_idx + 1, frames.len());
    }

    // Convert frame to image
    if !quiet {
        print!("Converting frame... ");
    }

    let frame = &frames[frame_idx];
    let img = frame.to_rgba_image()
        .context("Failed to convert frame to image")?;

    // Resize if needed
    let img = if width > 0 {
        let aspect = img.height() as f32 / img.width() as f32;
        let new_height = (width as f32 * aspect) as u32;
        if !quiet {
            print!("resizing to {}x{}... ", width, new_height);
        }
        img.resize_exact(width, new_height, FilterType::Lanczos3)
    } else {
        img
    };

    if !quiet {
        println!("done");
    }

    // Save image
    if !quiet {
        print!("Saving thumbnail... ");
    }

    img.save(&output)
        .with_context(|| format!("Failed to save thumbnail: {}", output.display()))?;

    if !quiet {
        println!("done");
    }

    let elapsed = start.elapsed();

    if !quiet {
        println!();
        println!("Thumbnail created: {}", output.display());
        println!("==================");
        println!("Resolution: {}x{}", img.width(), img.height());
        println!("Time: {:.2?}", elapsed);

        // Show file size
        if let Ok(meta) = std::fs::metadata(&output) {
            println!("Size: {}", format_size(meta.len()));
        }
    } else {
        println!("{}", output.display());
    }

    Ok(())
}

// ============================================================================
// Video Transcode Command (decode → transform → encode)
// FFmpeg-equivalent filters: brightness, contrast, saturation, hue, gamma,
// blur, sharpen, grayscale, sepia, invert, crop, fade, speed, watermark
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn cmd_video_transcode(
    input: PathBuf,
    output: PathBuf,
    width: u32,
    height: u32,
    fps: f64,
    rotate: u16,
    flip: String,
    speed: u8,
    quality: u8,
    bitrate: u32,
    threads: usize,
    max_frames: u32,
    start_time: f64,
    end_time: f64,
    // Color adjustments
    brightness: i32,
    contrast: i32,
    saturation: i32,
    hue: i32,
    gamma: f32,
    // Filters
    blur: u32,
    sharpen: u32,
    grayscale: bool,
    sepia: bool,
    invert: bool,
    // Crop
    crop_region: Option<String>,
    // Effects
    fade_in: f64,
    fade_out: f64,
    speed_factor: f64,
    // Watermark
    watermark: Option<PathBuf>,
    watermark_pos: String,
    watermark_opacity: f32,
    watermark_scale: f32,
    // Output
    codec: String,
    quiet: bool,
) -> Result<()> {
    use image::imageops::FilterType;

    let start = Instant::now();

    if !quiet {
        println!("xeno-edit video-transcode");
        println!("=========================");
        println!("Input: {}", input.display());
        println!("Output: {}", output.display());
        println!();
    }

    // Check if input exists
    if !input.exists() {
        anyhow::bail!("Input file does not exist: {}", input.display());
    }

    // Determine input format from extension
    let ext = input.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    // Decode video based on format
    if !quiet {
        print!("Decoding video... ");
    }

    let decode_start = Instant::now();
    let (frames, source_fps) = match ext.as_str() {
        "ivf" => {
            // Read IVF header to get source fps
            let header_data = std::fs::read(&input)
                .with_context(|| format!("Failed to read file: {}", input.display()))?;

            if header_data.len() < 32 || &header_data[0..4] != b"DKIF" {
                anyhow::bail!("Not a valid IVF file");
            }

            let fps_den = u32::from_le_bytes([header_data[16], header_data[17], header_data[18], header_data[19]]);
            let fps_num = u32::from_le_bytes([header_data[20], header_data[21], header_data[22], header_data[23]]);
            let source_fps = if fps_den > 0 { fps_num as f64 / fps_den as f64 } else { 30.0 };

            let frames = decode_ivf(&input)
                .with_context(|| format!("Failed to decode IVF: {}", input.display()))?;

            (frames, source_fps)
        }
        "mp4" | "m4v" | "mov" => {
            // Use container demuxer for MP4
            // For now, return an error since we need to implement MP4 frame decoding
            // The demuxer provides packets, but we need NVDEC to decode them
            anyhow::bail!("MP4 transcoding requires FFmpeg-based decoding (not yet implemented). Use IVF input for now.");
        }
        _ => {
            anyhow::bail!("Unsupported input format: {}. Supported: ivf, mp4, m4v, mov", ext);
        }
    };

    if !quiet {
        println!("done ({:.2?}, {} frames at {:.2} fps)", decode_start.elapsed(), frames.len(), source_fps);
    }

    if frames.is_empty() {
        anyhow::bail!("No frames decoded from video");
    }

    // Determine output fps
    let output_fps = if fps > 0.0 { fps } else { source_fps };

    // Get source dimensions from first frame
    let first_frame = &frames[0];
    let source_width = first_frame.width;
    let source_height = first_frame.height;

    // Calculate output dimensions
    let (target_width, target_height) = match (width, height) {
        (0, 0) => {
            // Apply rotation to dimensions if needed
            if rotate == 90 || rotate == 270 {
                (source_height, source_width)
            } else {
                (source_width, source_height)
            }
        }
        (w, 0) => {
            // Width specified, calculate height maintaining aspect
            let aspect = source_height as f64 / source_width as f64;
            let h = (w as f64 * aspect) as u32;
            if rotate == 90 || rotate == 270 {
                (h, w)
            } else {
                (w, h)
            }
        }
        (0, h) => {
            // Height specified, calculate width maintaining aspect
            let aspect = source_width as f64 / source_height as f64;
            let w = (h as f64 * aspect) as u32;
            if rotate == 90 || rotate == 270 {
                (h, w)
            } else {
                (w, h)
            }
        }
        (w, h) => {
            if rotate == 90 || rotate == 270 {
                (h, w)
            } else {
                (w, h)
            }
        }
    };

    // Calculate frame range based on time
    let start_frame = if start_time > 0.0 {
        (start_time * source_fps) as usize
    } else {
        0
    };

    let end_frame = if end_time > 0.0 {
        ((end_time * source_fps) as usize).min(frames.len())
    } else {
        frames.len()
    };

    // Apply max_frames limit
    let frame_limit = if max_frames > 0 {
        (start_frame + max_frames as usize).min(end_frame)
    } else {
        end_frame
    };

    let frames_to_process = &frames[start_frame..frame_limit];
    let total_frames = frames_to_process.len();

    if !quiet {
        println!("Source: {}x{} @ {:.2} fps", source_width, source_height, source_fps);
        println!("Output: {}x{} @ {:.2} fps", target_width, target_height, output_fps);
        println!("Codec: {}", if codec.to_lowercase().starts_with("h264") || codec.to_lowercase().starts_with("x264") || codec.to_lowercase() == "avc" { "H.264" } else { "AV1" });
        if rotate != 0 {
            println!("Rotation: {}°", rotate);
        }
        if flip != "none" {
            println!("Flip: {}", flip);
        }
        println!("Frames: {} ({}–{})", total_frames, start_frame, frame_limit);

        // Print active filters
        let mut filters = Vec::new();
        if brightness != 0 { filters.push(format!("brightness({})", brightness)); }
        if contrast != 0 { filters.push(format!("contrast({})", contrast)); }
        if saturation != 0 { filters.push(format!("saturation({})", saturation)); }
        if hue != 0 { filters.push(format!("hue({})", hue)); }
        if (gamma - 1.0).abs() > 0.001 { filters.push(format!("gamma({:.2})", gamma)); }
        if blur > 0 { filters.push(format!("blur({})", blur)); }
        if sharpen > 0 { filters.push(format!("sharpen({})", sharpen)); }
        if grayscale { filters.push("grayscale".to_string()); }
        if sepia { filters.push("sepia".to_string()); }
        if invert { filters.push("invert".to_string()); }
        if crop_region.is_some() { filters.push(format!("crop({})", crop_region.as_ref().unwrap())); }
        if fade_in > 0.0 { filters.push(format!("fade_in({:.1}s)", fade_in)); }
        if fade_out > 0.0 { filters.push(format!("fade_out({:.1}s)", fade_out)); }
        if (speed_factor - 1.0).abs() > 0.001 { filters.push(format!("speed({:.2}x)", speed_factor)); }
        if watermark.is_some() { filters.push("watermark".to_string()); }

        if !filters.is_empty() {
            println!("Filters: {}", filters.join(", "));
        }
        println!();
    }

    // Transform frames
    if !quiet {
        println!("Transforming frames...");
    }

    // Load watermark if specified
    let watermark_img = if let Some(ref wm_path) = watermark {
        if !quiet {
            println!("Loading watermark: {}", wm_path.display());
        }
        Some(image::open(wm_path)
            .with_context(|| format!("Failed to load watermark: {}", wm_path.display()))?)
    } else {
        None
    };

    let transform_start = Instant::now();
    let mut transformed_images: Vec<image::DynamicImage> = Vec::with_capacity(total_frames);

    for (idx, frame) in frames_to_process.iter().enumerate() {
        // Convert decoded frame to image
        let mut img = frame.to_rgba_image()
            .with_context(|| format!("Failed to convert frame {}", idx))?;

        // Apply resize if needed
        let needs_resize = target_width != source_width || target_height != source_height
            || rotate == 90 || rotate == 270;

        if needs_resize && (rotate == 0 || rotate == 180) {
            img = img.resize_exact(target_width, target_height, FilterType::Lanczos3);
        }

        // Apply rotation
        let img: image::DynamicImage = match rotate {
            90 => {
                let rotated = rotate_90_cw(&img.into())
                    .context("Failed to rotate 90°")?;
                if width > 0 || height > 0 {
                    rotated.resize_exact(target_width, target_height, FilterType::Lanczos3)
                } else {
                    rotated
                }
            }
            180 => rotate_180(&img.into())
                .context("Failed to rotate 180°")?,
            270 => {
                let rotated = rotate_270_cw(&img.into())
                    .context("Failed to rotate 270°")?;
                if width > 0 || height > 0 {
                    rotated.resize_exact(target_width, target_height, FilterType::Lanczos3)
                } else {
                    rotated
                }
            }
            _ => img.into(),
        };

        // Apply flip
        let img = match flip.as_str() {
            "h" | "horizontal" => flip_horizontal(&img)
                .context("Failed to flip horizontal")?,
            "v" | "vertical" => flip_vertical(&img)
                .context("Failed to flip vertical")?,
            "hv" | "both" => flip_both(&img)
                .context("Failed to flip both")?,
            _ => img,
        };

        // ====================================================================
        // FFmpeg-equivalent Filters
        // ====================================================================

        // Apply crop if specified (format: "x,y,w,h" or "w:h:x:y" like FFmpeg)
        let img = if let Some(ref crop_str) = crop_region {
            // Parse crop string - support both "x,y,w,h" and "w:h:x:y" (FFmpeg style)
            let parts: Vec<u32> = crop_str
                .split(|c| c == ',' || c == ':')
                .filter_map(|s| s.trim().parse().ok())
                .collect();

            if parts.len() == 4 {
                // Determine format: if first two are larger, it's likely w:h:x:y
                let (x, y, w, h) = if crop_str.contains(':') {
                    // FFmpeg style: w:h:x:y
                    (parts[2], parts[3], parts[0], parts[1])
                } else {
                    // Our style: x,y,w,h
                    (parts[0], parts[1], parts[2], parts[3])
                };
                crop(&img, x, y, w, h).unwrap_or(img)
            } else {
                img
            }
        } else {
            img
        };

        // Apply color adjustments
        let img = if brightness != 0 {
            adjust_brightness(&img, brightness as f32).unwrap_or(img)
        } else {
            img
        };

        let img = if contrast != 0 {
            adjust_contrast(&img, contrast as f32).unwrap_or(img)
        } else {
            img
        };

        let img = if saturation != 0 {
            adjust_saturation(&img, saturation as f32).unwrap_or(img)
        } else {
            img
        };

        let img = if hue != 0 {
            adjust_hue(&img, hue as f32).unwrap_or(img)
        } else {
            img
        };

        let img = if (gamma - 1.0).abs() > 0.001 {
            adjust_gamma(&img, gamma).unwrap_or(img)
        } else {
            img
        };

        // Apply filters
        let img = if blur > 0 {
            gaussian_blur(&img, blur as f32).unwrap_or(img)
        } else {
            img
        };

        let img = if sharpen > 0 {
            // unsharp_mask(image, sigma, amount) - higher amount = more sharpening
            let amount = sharpen as i32; // Scale sharpen level to reasonable amount
            unsharp_mask(&img, 1.0, amount).unwrap_or(img)
        } else {
            img
        };

        // Apply color effects
        let img = if grayscale {
            to_grayscale(&img).unwrap_or(img)
        } else {
            img
        };

        let img = if sepia {
            apply_sepia(&img).unwrap_or(img)
        } else {
            img
        };

        let img = if invert {
            invert_colors(&img).unwrap_or(img)
        } else {
            img
        };

        // Apply fade effects
        let img = if fade_in > 0.0 || fade_out > 0.0 {
            let frame_time = idx as f64 / output_fps;
            let total_duration = total_frames as f64 / output_fps;

            let mut alpha = 1.0f32;

            // Fade in: increase alpha from 0 to 1 over fade_in seconds
            if fade_in > 0.0 && frame_time < fade_in {
                alpha = (frame_time / fade_in) as f32;
            }

            // Fade out: decrease alpha from 1 to 0 over fade_out seconds
            if fade_out > 0.0 && frame_time > (total_duration - fade_out) {
                let fade_progress = (total_duration - frame_time) / fade_out;
                alpha = alpha.min(fade_progress as f32);
            }

            // Apply alpha by darkening the image (fade to black)
            if alpha < 1.0 {
                let mut rgba = img.to_rgba8();
                for pixel in rgba.pixels_mut() {
                    pixel[0] = (pixel[0] as f32 * alpha) as u8;
                    pixel[1] = (pixel[1] as f32 * alpha) as u8;
                    pixel[2] = (pixel[2] as f32 * alpha) as u8;
                }
                image::DynamicImage::ImageRgba8(rgba)
            } else {
                img
            }
        } else {
            img
        };

        // Apply watermark overlay
        let img = if let Some(ref wm) = watermark_img {
            // Scale watermark based on watermark_scale (relative to frame size)
            let wm_target_width = (img.width() as f32 * watermark_scale) as u32;
            let wm_aspect = wm.height() as f32 / wm.width() as f32;
            let wm_target_height = (wm_target_width as f32 * wm_aspect) as u32;

            let scaled_wm = wm.resize_exact(wm_target_width, wm_target_height, FilterType::Lanczos3);

            // Calculate position based on watermark_pos
            let (x, y) = match watermark_pos.as_str() {
                "tl" | "top-left" => (10u32, 10u32),
                "tr" | "top-right" => (img.width().saturating_sub(wm_target_width + 10), 10),
                "bl" | "bottom-left" => (10, img.height().saturating_sub(wm_target_height + 10)),
                "c" | "center" => (
                    (img.width().saturating_sub(wm_target_width)) / 2,
                    (img.height().saturating_sub(wm_target_height)) / 2,
                ),
                "br" | "bottom-right" | _ => (
                    img.width().saturating_sub(wm_target_width + 10),
                    img.height().saturating_sub(wm_target_height + 10),
                ),
            };

            apply_watermark(&img, &scaled_wm, x, y, watermark_opacity).unwrap_or(img)
        } else {
            img
        };

        transformed_images.push(img);

        if !quiet && (idx + 1) % 100 == 0 {
            print!("\r  Transformed {}/{} frames", idx + 1, total_frames);
        }
    }

    if !quiet {
        println!("\r  Transformed {}/{} frames ({:.2?})", total_frames, total_frames, transform_start.elapsed());
    }

    // Apply speed factor (frame selection for slow-mo / fast-forward)
    let final_frames: Vec<image::DynamicImage> = if (speed_factor - 1.0).abs() > 0.001 {
        if !quiet {
            println!("Applying speed factor: {:.2}x", speed_factor);
        }

        if speed_factor > 1.0 {
            // Fast forward: skip frames
            // speed_factor of 2.0 means keep every 2nd frame
            let step = speed_factor;
            let mut selected = Vec::new();
            let mut idx = 0.0;
            while (idx as usize) < transformed_images.len() {
                selected.push(transformed_images[idx as usize].clone());
                idx += step;
            }
            selected
        } else {
            // Slow motion: duplicate frames
            // speed_factor of 0.5 means duplicate each frame once
            let repeat_count = (1.0 / speed_factor).round() as usize;
            let mut expanded = Vec::with_capacity(transformed_images.len() * repeat_count);
            for img in transformed_images {
                for _ in 0..repeat_count {
                    expanded.push(img.clone());
                }
            }
            expanded
        }
    } else {
        transformed_images
    };

    let _final_frame_count = final_frames.len();

    // Determine output codec and path
    let codec_lower = codec.to_lowercase();
    let (output_path, codec_name) = match codec_lower.as_str() {
        "h264" | "x264" | "avc" => {
            let path = if output.extension().map_or(true, |e| e != "mp4") {
                output.with_extension("mp4")
            } else {
                output
            };
            (path, "H.264")
        }
        "av1" | "aom" | _ => {
            let path = if output.extension().map_or(true, |e| e != "ivf") {
                output.with_extension("ivf")
            } else {
                output
            };
            (path, "AV1")
        }
    };

    // Create encoder config and encode
    let encoding_speed = match speed {
        0 => EncodingSpeed::Placebo,
        1..=2 => EncodingSpeed::VerySlow,
        3..=4 => EncodingSpeed::Slow,
        5..=6 => EncodingSpeed::Medium,
        7..=8 => EncodingSpeed::Fast,
        9 => EncodingSpeed::VeryFast,
        _ => EncodingSpeed::Ultrafast,
    };

    // Encode
    if !quiet {
        println!("Encoding to {}...", codec_name);
    }

    let encode_start = Instant::now();

    let frame_count = match codec_lower.as_str() {
        "h264" | "x264" | "avc" => {
            // Use H.264 encoder
            let h264_config = H264EncoderConfig::new(target_width, target_height)
                .with_frame_rate(output_fps);

            encode_h264_to_mp4(final_frames.iter(), &output_path, h264_config)
                .context("Failed to encode H.264 video")?
        }
        "av1" | "aom" | _ => {
            // Use AV1 encoder
            let config = Av1EncoderConfig::new(target_width, target_height)
                .with_frame_rate(output_fps)
                .with_speed(encoding_speed)
                .with_quality(quality)
                .with_bitrate(bitrate)
                .with_threads(threads);

            encode_to_ivf(final_frames.iter(), &output_path, config)
                .context("Failed to encode AV1 video")?
        }
    };

    let encode_time = encode_start.elapsed();
    let total_time = start.elapsed();

    if !quiet {
        println!();
        println!("Transcoding complete!");
        println!("=====================");
        println!("Output: {}", output_path.display());
        println!("Frames: {}", frame_count);
        println!();
        println!("Timings:");
        println!("  Decode: {:.2?}", decode_start.elapsed() - transform_start.elapsed());
        println!("  Transform: {:.2?}", transform_start.elapsed() - encode_start.elapsed() + encode_time);
        println!("  Encode: {:.2?}", encode_time);
        println!("  Total: {:.2?}", total_time);
        println!();

        // Calculate speeds
        let decode_fps = total_frames as f64 / decode_start.elapsed().as_secs_f64();
        let encode_fps = frame_count as f64 / encode_time.as_secs_f64();
        println!("Speed: {:.2} fps decode, {:.2} fps encode", decode_fps, encode_fps);

        // Show file sizes
        if let (Ok(input_meta), Ok(output_meta)) = (std::fs::metadata(&input), std::fs::metadata(&output_path)) {
            let input_size = input_meta.len();
            let output_size = output_meta.len();
            let ratio = output_size as f64 / input_size as f64;

            println!();
            println!("Size: {} → {} ({:.1}x)", format_size(input_size), format_size(output_size), ratio);

            // Calculate bitrate
            let duration_secs = frame_count as f64 / output_fps;
            let actual_bitrate = (output_size as f64 * 8.0) / (duration_secs * 1000.0);
            println!("Bitrate: {:.0} kbps", actual_bitrate);
        }
    } else {
        println!("{}", output_path.display());
    }

    Ok(())
}

// ============================================================================
// Video Trim Command (cut to time range)
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn cmd_video_trim(
    input: PathBuf,
    output: PathBuf,
    start_time: f64,
    end_time: f64,
    speed: u8,
    quality: u8,
    quiet: bool,
) -> Result<()> {
    let start = Instant::now();

    if !quiet {
        println!("xeno-edit video-trim");
        println!("====================");
        println!("Input: {}", input.display());
        println!("Output: {}", output.display());
        println!("Time range: {}s → {}", start_time, if end_time > 0.0 { format!("{}s", end_time) } else { "end".to_string() });
        println!();
    }

    // Check if input exists
    if !input.exists() {
        anyhow::bail!("Input file does not exist: {}", input.display());
    }

    // Read IVF header to get source fps
    let header_data = std::fs::read(&input)
        .with_context(|| format!("Failed to read file: {}", input.display()))?;

    if header_data.len() < 32 || &header_data[0..4] != b"DKIF" {
        anyhow::bail!("Not a valid IVF file. Video trim currently supports IVF format only.");
    }

    let fps_den = u32::from_le_bytes([header_data[16], header_data[17], header_data[18], header_data[19]]);
    let fps_num = u32::from_le_bytes([header_data[20], header_data[21], header_data[22], header_data[23]]);
    let source_fps = if fps_den > 0 { fps_num as f64 / fps_den as f64 } else { 30.0 };

    // Decode video
    if !quiet {
        print!("Decoding video... ");
    }

    let decode_start = Instant::now();
    let frames = decode_ivf(&input)
        .with_context(|| format!("Failed to decode video: {}", input.display()))?;

    if !quiet {
        println!("done ({:.2?}, {} frames at {:.2} fps)", decode_start.elapsed(), frames.len(), source_fps);
    }

    if frames.is_empty() {
        anyhow::bail!("No frames decoded from video");
    }

    // Calculate frame range
    let start_frame = (start_time * source_fps) as usize;
    let end_frame = if end_time > 0.0 {
        ((end_time * source_fps) as usize).min(frames.len())
    } else {
        frames.len()
    };

    if start_frame >= frames.len() {
        anyhow::bail!("Start time {:.2}s is beyond video duration ({:.2}s)",
            start_time, frames.len() as f64 / source_fps);
    }

    if start_frame >= end_frame {
        anyhow::bail!("Start frame {} >= end frame {}. Nothing to trim.", start_frame, end_frame);
    }

    let frames_to_keep = &frames[start_frame..end_frame];
    let total_frames = frames_to_keep.len();

    if !quiet {
        println!("Trimming: frame {} to {} ({} frames)", start_frame, end_frame, total_frames);
        println!("Duration: {:.2}s → {:.2}s", start_time, (end_frame as f64) / source_fps);
        println!();
    }

    // Convert frames to images
    if !quiet {
        println!("Converting frames...");
    }

    let convert_start = Instant::now();
    let mut images: Vec<image::DynamicImage> = Vec::with_capacity(total_frames);

    for (idx, frame) in frames_to_keep.iter().enumerate() {
        let img = frame.to_rgba_image()
            .with_context(|| format!("Failed to convert frame {}", idx))?;
        images.push(img.into());

        if !quiet && (idx + 1) % 100 == 0 {
            print!("\r  Converted {}/{} frames", idx + 1, total_frames);
        }
    }

    if !quiet {
        println!("\r  Converted {}/{} frames ({:.2?})", total_frames, total_frames, convert_start.elapsed());
    }

    // Get dimensions from first frame
    let first_img = &images[0];
    let width = first_img.width();
    let height = first_img.height();

    // Create encoder config
    let encoding_speed = match speed {
        0 => EncodingSpeed::Placebo,
        1..=2 => EncodingSpeed::VerySlow,
        3..=4 => EncodingSpeed::Slow,
        5..=6 => EncodingSpeed::Medium,
        7..=8 => EncodingSpeed::Fast,
        9 => EncodingSpeed::VeryFast,
        _ => EncodingSpeed::Ultrafast,
    };

    let config = Av1EncoderConfig::new(width, height)
        .with_frame_rate(source_fps)
        .with_speed(encoding_speed)
        .with_quality(quality);

    // Ensure output has .ivf extension
    let output_path = if output.extension().map_or(true, |e| e != "ivf") {
        output.with_extension("ivf")
    } else {
        output
    };

    // Encode
    if !quiet {
        println!("Encoding to AV1...");
    }

    let encode_start = Instant::now();
    let frame_count = encode_to_ivf(images.iter(), &output_path, config)
        .context("Failed to encode video")?;

    let encode_time = encode_start.elapsed();
    let total_time = start.elapsed();

    if !quiet {
        println!();
        println!("Trim complete!");
        println!("==============");
        println!("Output: {}", output_path.display());
        println!("Frames: {} → {}", frames.len(), frame_count);
        println!("Duration: {:.2}s → {:.2}s", frames.len() as f64 / source_fps, frame_count as f64 / source_fps);
        println!();
        println!("Timings:");
        println!("  Decode: {:.2?}", decode_start.elapsed());
        println!("  Convert: {:.2?}", convert_start.elapsed());
        println!("  Encode: {:.2?}", encode_time);
        println!("  Total: {:.2?}", total_time);

        // Show file sizes
        if let (Ok(input_meta), Ok(output_meta)) = (std::fs::metadata(&input), std::fs::metadata(&output_path)) {
            let input_size = input_meta.len();
            let output_size = output_meta.len();
            println!();
            println!("Size: {} → {}", format_size(input_size), format_size(output_size));
        }
    } else {
        println!("{}", output_path.display());
    }

    Ok(())
}

// ============================================================================
// Video Concat Command (join multiple videos)
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn cmd_video_concat(
    output: PathBuf,
    inputs: Vec<PathBuf>,
    speed: u8,
    quality: u8,
    target_fps: f64,
    quiet: bool,
) -> Result<()> {
    let start = Instant::now();

    if !quiet {
        println!("xeno-edit video-concat");
        println!("======================");
        println!("Output: {}", output.display());
        println!("Inputs: {} files", inputs.len());
        for (i, input) in inputs.iter().enumerate() {
            println!("  [{}] {}", i + 1, input.display());
        }
        println!();
    }

    // Validate all inputs exist
    for input in &inputs {
        if !input.exists() {
            anyhow::bail!("Input file does not exist: {}", input.display());
        }
    }

    // Decode all videos and collect frames
    let mut all_images: Vec<image::DynamicImage> = Vec::new();
    let mut first_width: Option<u32> = None;
    let mut first_height: Option<u32> = None;
    let mut detected_fps: f64 = 30.0;

    for (file_idx, input) in inputs.iter().enumerate() {
        // Read IVF header
        let header_data = std::fs::read(input)
            .with_context(|| format!("Failed to read file: {}", input.display()))?;

        if header_data.len() < 32 || &header_data[0..4] != b"DKIF" {
            anyhow::bail!("Not a valid IVF file: {}. Video concat currently supports IVF format only.", input.display());
        }

        // Get FPS from first file if not specified
        if file_idx == 0 && target_fps == 0.0 {
            let fps_den = u32::from_le_bytes([header_data[16], header_data[17], header_data[18], header_data[19]]);
            let fps_num = u32::from_le_bytes([header_data[20], header_data[21], header_data[22], header_data[23]]);
            detected_fps = if fps_den > 0 { fps_num as f64 / fps_den as f64 } else { 30.0 };
        }

        if !quiet {
            print!("Decoding video {} of {}... ", file_idx + 1, inputs.len());
        }

        let decode_start = Instant::now();
        let frames = decode_ivf(input)
            .with_context(|| format!("Failed to decode video: {}", input.display()))?;

        if !quiet {
            println!("done ({:.2?}, {} frames)", decode_start.elapsed(), frames.len());
        }

        if frames.is_empty() {
            if !quiet {
                println!("  Warning: No frames in {}, skipping", input.display());
            }
            continue;
        }

        // Check dimensions consistency
        let w = frames[0].width;
        let h = frames[0].height;

        match (first_width, first_height) {
            (None, None) => {
                first_width = Some(w);
                first_height = Some(h);
            }
            (Some(fw), Some(fh)) => {
                if w != fw || h != fh {
                    anyhow::bail!(
                        "Dimension mismatch: {} is {}x{} but first video is {}x{}. All videos must have same resolution.",
                        input.display(), w, h, fw, fh
                    );
                }
            }
            _ => unreachable!(),
        }

        // Convert frames to images
        for frame in frames.iter() {
            let img = frame.to_rgba_image()
                .with_context(|| format!("Failed to convert frame from {}", input.display()))?;
            all_images.push(img.into());
        }
    }

    if all_images.is_empty() {
        anyhow::bail!("No frames collected from any input video");
    }

    let total_frames = all_images.len();
    let width = first_width.unwrap();
    let height = first_height.unwrap();
    let output_fps = if target_fps > 0.0 { target_fps } else { detected_fps };

    if !quiet {
        println!();
        println!("Combined: {} frames total ({}x{} @ {:.2} fps)", total_frames, width, height, output_fps);
        println!("Duration: {:.2}s", total_frames as f64 / output_fps);
        println!();
    }

    // Create encoder config
    let encoding_speed = match speed {
        0 => EncodingSpeed::Placebo,
        1..=2 => EncodingSpeed::VerySlow,
        3..=4 => EncodingSpeed::Slow,
        5..=6 => EncodingSpeed::Medium,
        7..=8 => EncodingSpeed::Fast,
        9 => EncodingSpeed::VeryFast,
        _ => EncodingSpeed::Ultrafast,
    };

    let config = Av1EncoderConfig::new(width, height)
        .with_frame_rate(output_fps)
        .with_speed(encoding_speed)
        .with_quality(quality);

    // Ensure output has .ivf extension
    let output_path = if output.extension().map_or(true, |e| e != "ivf") {
        output.with_extension("ivf")
    } else {
        output
    };

    // Encode
    if !quiet {
        println!("Encoding to AV1...");
    }

    let encode_start = Instant::now();
    let frame_count = encode_to_ivf(all_images.iter(), &output_path, config)
        .context("Failed to encode video")?;

    let encode_time = encode_start.elapsed();
    let total_time = start.elapsed();

    if !quiet {
        println!();
        println!("Concatenation complete!");
        println!("=======================");
        println!("Output: {}", output_path.display());
        println!("Frames: {}", frame_count);
        println!("Duration: {:.2}s", frame_count as f64 / output_fps);
        println!();
        println!("Encode time: {:.2?}", encode_time);
        println!("Total time: {:.2?}", total_time);

        // Show file size
        if let Ok(meta) = std::fs::metadata(&output_path) {
            println!("Size: {}", format_size(meta.len()));

            // Calculate bitrate
            let duration_secs = frame_count as f64 / output_fps;
            let actual_bitrate = (meta.len() as f64 * 8.0) / (duration_secs * 1000.0);
            println!("Bitrate: {:.0} kbps", actual_bitrate);
        }
    } else {
        println!("{}", output_path.display());
    }

    Ok(())
}

// ============================================================================
// Text Overlay Command (pure Rust font rendering)
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn cmd_text_overlay(
    inputs: Vec<PathBuf>,
    text: String,
    font_path: PathBuf,
    font_size: f32,
    x: i32,
    y: i32,
    color: String,
    anchor_str: String,
    shadow: Option<String>,
    outline: u32,
    output_dir: Option<PathBuf>,
    quiet: bool,
) -> Result<()> {
    let start = Instant::now();
    let total_images = inputs.len();
    let is_batch = total_images > 1;

    if !quiet {
        println!("xeno-edit text-overlay");
        println!("======================");
        if is_batch {
            println!("Batch mode: {} images", total_images);
        }
        println!("Text: \"{}\"", text);
        println!("Font: {}", font_path.display());
        println!("Size: {}px", font_size);
        println!("Position: ({}, {})", x, y);
        println!();
    }

    // Load font file
    if !font_path.exists() {
        anyhow::bail!("Font file not found: {}", font_path.display());
    }

    let font_data = std::fs::read(&font_path)
        .with_context(|| format!("Failed to read font file: {}", font_path.display()))?;

    let overlay = TextOverlay::new(&font_data)
        .with_context(|| "Failed to load font")?;

    // Parse color
    let rgba = parse_hex_color(&color)?;

    // Parse anchor
    let anchor = parse_anchor(&anchor_str)?;

    // Parse shadow
    let shadow_offset = if let Some(ref s) = shadow {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() == 2 {
            let dx = parts[0].trim().parse::<i32>()
                .with_context(|| format!("Invalid shadow dx: {}", parts[0]))?;
            let dy = parts[1].trim().parse::<i32>()
                .with_context(|| format!("Invalid shadow dy: {}", parts[1]))?;
            Some((dx, dy))
        } else {
            anyhow::bail!("Shadow format must be 'dx,dy' (e.g., '2,2')");
        }
    } else {
        None
    };

    // Build text config
    let mut config = TextConfig::new(&text)
        .with_font_size(font_size)
        .with_color(rgba)
        .with_position(x, y)
        .with_anchor(anchor);

    if let Some((dx, dy)) = shadow_offset {
        config = config.with_shadow(dx, dy);
    }

    if outline > 0 {
        config = config.with_outline(outline);
    }

    let mut success_count = 0;
    let mut fail_count = 0;

    for (idx, input) in inputs.iter().enumerate() {
        // Determine output path
        let output = if let Some(ref dir) = output_dir {
            let stem = input.file_stem().unwrap_or_default().to_string_lossy();
            let ext = input.extension().unwrap_or_default().to_string_lossy();
            dir.join(format!("{}_text.{}", stem, ext))
        } else {
            let stem = input.file_stem().unwrap_or_default().to_string_lossy();
            let ext = input.extension().unwrap_or_default().to_string_lossy();
            let parent = input.parent().unwrap_or(std::path::Path::new("."));
            parent.join(format!("{}_text.{}", stem, ext))
        };

        if !quiet && is_batch {
            println!("[{}/{}] {}", idx + 1, total_images, input.display());
        }

        match process_text_overlay(input, &output, &overlay, &config, quiet && !is_batch) {
            Ok(_) => {
                success_count += 1;
                if quiet {
                    println!("{}", output.display());
                } else if !is_batch {
                    println!("Output: {}", output.display());
                } else {
                    println!("  -> {}", output.display());
                }
            }
            Err(e) => {
                fail_count += 1;
                if !quiet {
                    eprintln!("  x Error: {}", e);
                } else {
                    eprintln!("Error processing {}: {}", input.display(), e);
                }
            }
        }
    }

    if !quiet && is_batch {
        println!();
        println!("Batch complete: {} succeeded, {} failed ({:.1?} total)",
            success_count, fail_count, start.elapsed());
    }

    if fail_count > 0 && success_count == 0 {
        anyhow::bail!("All images failed to process");
    }

    Ok(())
}

fn process_text_overlay(
    input: &PathBuf,
    output: &PathBuf,
    overlay: &TextOverlay,
    config: &TextConfig,
    quiet: bool,
) -> Result<()> {
    if !quiet {
        print!("Loading image... ");
    }

    let img_start = Instant::now();
    let input_image = load_image_auto(input)
        .with_context(|| format!("Failed to open image: {}", input.display()))?;

    if !quiet {
        println!("done ({:.0?}, {}x{})",
            img_start.elapsed(), input_image.width(), input_image.height());
    }

    if !quiet {
        print!("Drawing text... ");
    }

    let draw_start = Instant::now();
    let output_image = overlay.draw(&input_image, config)
        .context("Failed to draw text")?;

    if !quiet {
        println!("done ({:.0?})", draw_start.elapsed());
    }

    if !quiet {
        print!("Saving image... ");
    }

    let save_start = Instant::now();
    output_image.save(output)
        .with_context(|| format!("Failed to save image: {}", output.display()))?;

    if !quiet {
        println!("done ({:.0?})", save_start.elapsed());
    }

    Ok(())
}

/// Parse hex color string to RGBA array
fn parse_hex_color(hex: &str) -> Result<[u8; 4]> {
    let hex = hex.trim_start_matches('#');

    match hex.len() {
        6 => {
            // RGB format
            let r = u8::from_str_radix(&hex[0..2], 16)
                .with_context(|| format!("Invalid red component in color: {}", hex))?;
            let g = u8::from_str_radix(&hex[2..4], 16)
                .with_context(|| format!("Invalid green component in color: {}", hex))?;
            let b = u8::from_str_radix(&hex[4..6], 16)
                .with_context(|| format!("Invalid blue component in color: {}", hex))?;
            Ok([r, g, b, 255])
        }
        8 => {
            // RGBA format
            let r = u8::from_str_radix(&hex[0..2], 16)
                .with_context(|| format!("Invalid red component in color: {}", hex))?;
            let g = u8::from_str_radix(&hex[2..4], 16)
                .with_context(|| format!("Invalid green component in color: {}", hex))?;
            let b = u8::from_str_radix(&hex[4..6], 16)
                .with_context(|| format!("Invalid blue component in color: {}", hex))?;
            let a = u8::from_str_radix(&hex[6..8], 16)
                .with_context(|| format!("Invalid alpha component in color: {}", hex))?;
            Ok([r, g, b, a])
        }
        _ => anyhow::bail!("Color must be 6 (RGB) or 8 (RGBA) hex digits, got: {}", hex),
    }
}

/// Parse anchor string to Anchor enum
fn parse_anchor(s: &str) -> Result<Anchor> {
    match s.to_lowercase().as_str() {
        "tl" | "topleft" | "top-left" => Ok(Anchor::TopLeft),
        "tc" | "topcenter" | "top-center" => Ok(Anchor::TopCenter),
        "tr" | "topright" | "top-right" => Ok(Anchor::TopRight),
        "ml" | "middleleft" | "middle-left" => Ok(Anchor::MiddleLeft),
        "c" | "center" => Ok(Anchor::Center),
        "mr" | "middleright" | "middle-right" => Ok(Anchor::MiddleRight),
        "bl" | "bottomleft" | "bottom-left" => Ok(Anchor::BottomLeft),
        "bc" | "bottomcenter" | "bottom-center" => Ok(Anchor::BottomCenter),
        "br" | "bottomright" | "bottom-right" => Ok(Anchor::BottomRight),
        _ => anyhow::bail!(
            "Invalid anchor '{}'. Valid values: tl, tc, tr, ml, c, mr, bl, bc, br",
            s
        ),
    }
}

// ============================================================================
// H.264 Encode Command (native, no FFmpeg)
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn cmd_h264_encode(
    output: PathBuf,
    inputs: Vec<PathBuf>,
    fps: f64,
    bitrate: u32,
    width: u32,
    height: u32,
    raw: bool,
    quiet: bool,
) -> Result<()> {
    let start = Instant::now();

    if inputs.is_empty() {
        anyhow::bail!("No input images specified");
    }

    if !quiet {
        println!("xeno-edit h264-encode (H.264)");
        println!("=============================");
        println!("Frames: {}", inputs.len());
        println!("Output: {}", output.display());
        println!();
    }

    // Load first image to get dimensions
    if !quiet {
        print!("Scanning input images... ");
    }

    let first_img = load_image_auto(&inputs[0])
        .with_context(|| format!("Failed to load first image: {}", inputs[0].display()))?;

    let target_width = if width > 0 { width } else { first_img.width() };
    let target_height = if height > 0 { height } else { first_img.height() };

    if !quiet {
        println!("done");
        println!("Resolution: {}x{}", target_width, target_height);
        println!("Frame rate: {} fps", fps);
        if bitrate > 0 {
            println!("Bitrate: {} kbps", bitrate);
        } else {
            println!("Bitrate: auto (quality-based)");
        }
        println!("Output format: {}", if raw { "raw H.264 bitstream" } else { "MP4 container" });
        println!();
    }

    // Load all images
    if !quiet {
        print!("Loading {} images... ", inputs.len());
    }

    let images: Vec<image::DynamicImage> = inputs
        .iter()
        .map(|p| load_image_auto(p))
        .collect::<Result<Vec<_>>>()
        .context("Failed to load images")?;

    if !quiet {
        println!("done");
    }

    // Create encoder config
    let config = H264EncoderConfig::new(target_width, target_height)
        .with_frame_rate(fps)
        .with_bitrate(bitrate);

    // Ensure proper extension
    let output_path = if raw {
        if output.extension().map_or(true, |e| e != "h264" && e != "264") {
            output.with_extension("h264")
        } else {
            output
        }
    } else {
        if output.extension().map_or(true, |e| e != "mp4") {
            output.with_extension("mp4")
        } else {
            output
        }
    };

    // Encode
    if !quiet {
        println!("Encoding to H.264...");
    }

    let encode_start = Instant::now();
    let frame_count = if raw {
        encode_to_h264(images.iter(), &output_path, config)
            .context("Failed to encode H.264 video")?
    } else {
        encode_h264_to_mp4(images.iter(), &output_path, config)
            .context("Failed to encode H.264 video")?
    };

    let encode_time = encode_start.elapsed();
    let total_time = start.elapsed();

    if !quiet {
        println!();
        println!("Encoding complete!");
        println!("==================");
        println!("Output: {}", output_path.display());
        println!("Frames: {}", frame_count);
        println!("Encode time: {:.2?}", encode_time);
        println!("Total time: {:.2?}", total_time);

        // Calculate FPS
        let encode_fps = frame_count as f64 / encode_time.as_secs_f64();
        println!("Encode speed: {:.2} fps", encode_fps);

        // Show file size
        if let Ok(meta) = std::fs::metadata(&output_path) {
            let size_kb = meta.len() as f64 / 1024.0;
            if size_kb > 1024.0 {
                println!("Size: {:.2} MB", size_kb / 1024.0);
            } else {
                println!("Size: {:.2} KB", size_kb);
            }

            // Calculate bitrate
            let duration_secs = frame_count as f64 / fps;
            let actual_bitrate = (meta.len() as f64 * 8.0) / (duration_secs * 1000.0);
            println!("Actual bitrate: {:.0} kbps", actual_bitrate);
        }

        println!();
        if raw {
            println!("To play: ffplay {}", output_path.display());
            println!("To remux to MP4: ffmpeg -i {} -c copy output.mp4", output_path.display());
        } else {
            println!("To play: ffplay {} or any media player", output_path.display());
        }
    } else {
        println!("{}", output_path.display());
    }

    Ok(())
}

// ============================================================================
// Audio Encode Command (WAV/FLAC)
// ============================================================================

fn cmd_audio_encode(
    output: PathBuf,
    inputs: Vec<PathBuf>,
    format: Option<String>,
    sample_rate: u32,
    bits: u16,
    quiet: bool,
) -> Result<()> {
    let start = Instant::now();

    if inputs.is_empty() {
        anyhow::bail!("No input audio files specified");
    }

    // Determine output format
    let out_format = if let Some(ref fmt) = format {
        fmt.to_lowercase()
    } else {
        output
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("wav")
            .to_lowercase()
    };

    let is_flac = out_format == "flac";
    let is_wav = out_format == "wav" || out_format == "wave";

    if !is_flac && !is_wav {
        anyhow::bail!("Unsupported output format: {}. Use 'wav' or 'flac'.", out_format);
    }

    if !quiet {
        println!("xeno-edit audio-encode ({})", if is_flac { "FLAC" } else { "WAV" });
        println!("================================");
        println!("Input files: {}", inputs.len());
        println!("Output: {}", output.display());
        println!();
    }

    // Decode all input files
    if !quiet {
        print!("Decoding {} input file(s)... ", inputs.len());
    }

    let decode_start = Instant::now();
    let mut all_samples: Vec<f32> = Vec::new();
    let mut source_sample_rate: u32 = 0;
    let mut source_channels: u32 = 0;

    for (idx, input) in inputs.iter().enumerate() {
        let audio = decode_file(input)
            .with_context(|| format!("Failed to decode audio: {}", input.display()))?;

        // Check for sample rate/channel consistency
        if idx == 0 {
            source_sample_rate = audio.sample_rate;
            source_channels = audio.channels;
        } else if audio.sample_rate != source_sample_rate || audio.channels != source_channels {
            anyhow::bail!(
                "Audio format mismatch in {}: expected {}Hz {}ch, got {}Hz {}ch",
                input.display(),
                source_sample_rate,
                source_channels,
                audio.sample_rate,
                audio.channels
            );
        }

        all_samples.extend(audio.samples);
    }

    if !quiet {
        println!("done ({:.1?})", decode_start.elapsed());
        println!("Sample rate: {} Hz", source_sample_rate);
        println!("Channels: {}", source_channels);
        println!("Total samples: {}", all_samples.len());
        println!("Duration: {:.2}s", all_samples.len() as f64 / (source_sample_rate as f64 * source_channels as f64));
        println!();
    }

    // Use specified sample rate or source rate
    let target_sample_rate = if sample_rate > 0 { sample_rate } else { source_sample_rate };

    // Ensure proper extension
    let output_path = if is_flac {
        if output.extension().map_or(true, |e| e != "flac") {
            output.with_extension("flac")
        } else {
            output
        }
    } else {
        if output.extension().map_or(true, |e| e != "wav" && e != "wave") {
            output.with_extension("wav")
        } else {
            output
        }
    };

    // Encode
    if !quiet {
        println!("Encoding to {}...", if is_flac { "FLAC" } else { "WAV" });
    }

    let encode_start = Instant::now();
    let sample_count = if is_flac {
        let config = FlacConfig::new(target_sample_rate, source_channels as u16)
            .with_bits(bits);
        encode_flac(&all_samples, &output_path, config)
            .with_context(|| "Failed to encode FLAC")?
    } else {
        let config = WavConfig::new(target_sample_rate, source_channels as u16)
            .with_bits(bits);
        encode_wav(&all_samples, &output_path, config)
            .with_context(|| "Failed to encode WAV")?
    };

    let encode_time = encode_start.elapsed();
    let total_time = start.elapsed();

    if !quiet {
        println!();
        println!("Encoding complete!");
        println!("==================");
        println!("Output: {}", output_path.display());
        println!("Format: {}", if is_flac { "FLAC" } else { "WAV" });
        println!("Samples written: {}", sample_count);
        println!("Bits per sample: {}", bits);
        println!("Encode time: {:.2?}", encode_time);
        println!("Total time: {:.2?}", total_time);

        // Show file size
        if let Ok(meta) = std::fs::metadata(&output_path) {
            let size_kb = meta.len() as f64 / 1024.0;
            if size_kb > 1024.0 {
                println!("Size: {:.2} MB", size_kb / 1024.0);
            } else {
                println!("Size: {:.2} KB", size_kb);
            }

            // Calculate compression ratio if we have source data
            let uncompressed_size = all_samples.len() * 2; // 16-bit reference
            let ratio = uncompressed_size as f64 / meta.len() as f64;
            if is_flac {
                println!("Compression ratio: {:.1}x ({:.1}% of uncompressed)", ratio, 100.0 / ratio);
            }
        }
    } else {
        println!("{}", output_path.display());
    }

    Ok(())
}

// ============================================================================
// Agent API - JSON Execution (exec command)
// ============================================================================

/// JSON configuration for agent-driven operations.
/// This is the schema that AI agents should use to construct commands.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ExecConfig {
    /// Input file path (required for most operations)
    pub input: Option<String>,
    /// Output file path (required for most operations)
    pub output: Option<String>,
    /// Operation type: transcode, trim, concat, encode, decode, frames, gif, webp, remove-bg, text, info
    pub operation: String,
    /// Additional inputs for concat/batch operations
    #[serde(default)]
    pub inputs: Vec<String>,
    /// Video encoding settings
    pub video: Option<ExecVideoSettings>,
    /// Audio settings
    pub audio: Option<ExecAudioSettings>,
    /// List of transforms to apply
    #[serde(default)]
    pub transforms: Vec<String>,
    /// Trim settings
    pub trim: Option<ExecTrimSettings>,
    /// Text overlay settings
    pub text: Option<ExecTextSettings>,
    /// Frame extraction settings
    pub frames: Option<ExecFrameSettings>,
    /// Background removal settings
    pub remove_bg: Option<ExecRemoveBgSettings>,
    /// Video filter settings (brightness, contrast, blur, etc.)
    pub filters: Option<ExecFilterSettings>,
    /// Suppress normal output (quiet mode)
    #[serde(default)]
    pub quiet: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ExecVideoSettings {
    /// Codec: av1, h264 (future)
    #[serde(default = "default_codec")]
    pub codec: String,
    /// Output width (0 = preserve source)
    #[serde(default)]
    pub width: u32,
    /// Output height (0 = preserve source)
    #[serde(default)]
    pub height: u32,
    /// Frame rate (0 = preserve source)
    #[serde(default)]
    pub fps: f64,
    /// Quality (0-255 for AV1, lower = better)
    #[serde(default = "default_quality")]
    pub quality: u8,
    /// Speed preset (0-10, higher = faster)
    #[serde(default = "default_speed")]
    pub speed: u8,
    /// Target bitrate in kbps (0 = quality mode)
    #[serde(default)]
    pub bitrate: u32,
    /// Number of threads (0 = auto)
    #[serde(default)]
    pub threads: usize,
}

fn default_codec() -> String { "av1".to_string() }
fn default_quality() -> u8 { 100 }
fn default_speed() -> u8 { 6 }

impl Default for ExecVideoSettings {
    fn default() -> Self {
        Self {
            codec: "av1".to_string(),
            width: 0,
            height: 0,
            fps: 0.0,
            quality: 100,
            speed: 6,
            bitrate: 0,
            threads: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ExecAudioSettings {
    /// Audio codec
    #[serde(default)]
    pub codec: String,
    /// Sample rate
    #[serde(default)]
    pub sample_rate: u32,
    /// Channels
    #[serde(default)]
    pub channels: u8,
    /// Bitrate in kbps
    #[serde(default)]
    pub bitrate: u32,
}

/// Video filter settings for FFmpeg-like filters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct ExecFilterSettings {
    /// Brightness adjustment (-100 to 100)
    #[serde(default)]
    pub brightness: i32,
    /// Contrast adjustment (-100 to 100)
    #[serde(default)]
    pub contrast: i32,
    /// Saturation adjustment (-100 to 100)
    #[serde(default)]
    pub saturation: i32,
    /// Hue rotation in degrees (-180 to 180)
    #[serde(default)]
    pub hue: i32,
    /// Gamma correction (0.1 to 10.0, 1.0 = no change)
    #[serde(default = "default_gamma")]
    pub gamma: f32,
    /// Gaussian blur radius (0 = disabled)
    #[serde(default)]
    pub blur: u32,
    /// Sharpen amount (0 = disabled)
    #[serde(default)]
    pub sharpen: u32,
    /// Convert to grayscale
    #[serde(default)]
    pub grayscale: bool,
    /// Apply sepia tone
    #[serde(default)]
    pub sepia: bool,
    /// Invert colors
    #[serde(default)]
    pub invert: bool,
    /// Crop region: "x,y,w,h" or "w:h:x:y" (FFmpeg style)
    #[serde(default)]
    pub crop: Option<String>,
    /// Fade in duration in seconds
    #[serde(default)]
    pub fade_in: f64,
    /// Fade out duration in seconds
    #[serde(default)]
    pub fade_out: f64,
    /// Speed factor (0.5 = slow-mo, 2.0 = fast forward)
    #[serde(default = "default_speed_factor")]
    pub speed_factor: f64,
    /// Watermark image path
    #[serde(default)]
    pub watermark: Option<String>,
    /// Watermark position: tl, tr, bl, br, center
    #[serde(default = "default_watermark_pos")]
    pub watermark_pos: String,
    /// Watermark opacity (0.0 to 1.0)
    #[serde(default = "default_watermark_opacity")]
    pub watermark_opacity: f32,
    /// Watermark scale relative to frame (0.0 to 1.0)
    #[serde(default = "default_watermark_scale")]
    pub watermark_scale: f32,
}

fn default_gamma() -> f32 { 1.0 }
fn default_speed_factor() -> f64 { 1.0 }
fn default_watermark_pos() -> String { "br".to_string() }
fn default_watermark_opacity() -> f32 { 0.5 }
fn default_watermark_scale() -> f32 { 0.2 }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ExecTrimSettings {
    /// Start time in seconds
    #[serde(default)]
    pub start: f64,
    /// End time in seconds (0 = to end)
    #[serde(default)]
    pub end: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ExecTextSettings {
    /// Text to draw
    pub content: String,
    /// Font file path
    pub font: String,
    /// Font size in pixels
    #[serde(default = "default_font_size")]
    pub font_size: f32,
    /// X position
    #[serde(default)]
    pub x: i32,
    /// Y position
    #[serde(default)]
    pub y: i32,
    /// Color as hex (RRGGBB or RRGGBBAA)
    #[serde(default = "default_color")]
    pub color: String,
    /// Anchor: tl, tc, tr, ml, c, mr, bl, bc, br
    #[serde(default = "default_anchor")]
    pub anchor: String,
    /// Shadow offset "dx,dy" or null
    pub shadow: Option<String>,
    /// Outline thickness
    #[serde(default)]
    pub outline: u32,
}

fn default_font_size() -> f32 { 32.0 }
fn default_color() -> String { "FFFFFF".to_string() }
fn default_anchor() -> String { "tl".to_string() }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ExecFrameSettings {
    /// Output directory
    pub output_dir: String,
    /// Format: png, jpg, webp
    #[serde(default = "default_frame_format")]
    pub format: String,
    /// Extract every Nth frame
    #[serde(default = "default_every")]
    pub every: u32,
    /// Max frames to extract (0 = all)
    #[serde(default)]
    pub max_frames: u32,
}

fn default_frame_format() -> String { "png".to_string() }
fn default_every() -> u32 { 1 }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ExecRemoveBgSettings {
    /// Custom model path
    pub model: Option<String>,
    /// Confidence threshold (0.0-1.0)
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    /// Use CPU only
    #[serde(default)]
    pub cpu: bool,
}

fn default_threshold() -> f32 { 0.5 }

/// Result structure for JSON output
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ExecResult {
    pub success: bool,
    pub operation: String,
    pub output: Option<String>,
    pub outputs: Vec<String>,
    pub error: Option<String>,
    pub timing_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_size: Option<u64>,
}

/// Execute operation from JSON config
fn cmd_exec(config_input: String, output_override: Option<PathBuf>, json_output: bool) -> Result<()> {
    let start = Instant::now();

    // Parse JSON config from file, stdin, or inline
    let json_str = if config_input == "-" {
        // Read from stdin
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)
            .context("Failed to read JSON from stdin")?;
        buffer
    } else if config_input.starts_with('{') {
        // Inline JSON
        config_input.clone()
    } else {
        // File path
        std::fs::read_to_string(&config_input)
            .with_context(|| format!("Failed to read config file: {}", config_input))?
    };

    let mut config: ExecConfig = serde_json::from_str(&json_str)
        .context("Failed to parse JSON config")?;

    // Apply output override if specified
    if let Some(out) = output_override {
        config.output = Some(out.to_string_lossy().to_string());
    }

    // Execute based on operation type
    let result = match config.operation.as_str() {
        "transcode" | "video-transcode" | "vtrans" => exec_transcode(&config),
        "trim" | "video-trim" | "vtrim" => exec_trim(&config),
        "concat" | "video-concat" | "vcat" => exec_concat(&config),
        "encode" | "video-encode" | "av1" => exec_encode(&config),
        "frames" | "video-frames" | "vframes" => exec_frames(&config),
        "gif" | "video-to-gif" | "v2gif" => exec_gif(&config),
        "info" | "video-info" | "vinfo" => exec_info(&config),
        "remove-bg" | "rmbg" => exec_remove_bg(&config),
        "text" | "text-overlay" | "drawtext" => exec_text(&config),
        "capabilities" | "caps" => {
            let caps = Capabilities::query();
            if json_output {
                println!("{}", caps.to_agent_json());
            } else {
                println!("{}", caps.to_agent_json());
            }
            return Ok(());
        }
        _ => {
            let err_result = ExecResult {
                success: false,
                operation: config.operation.clone(),
                output: None,
                outputs: vec![],
                error: Some(format!(
                    "Unknown operation '{}'. Valid operations: transcode, trim, concat, encode, frames, gif, info, remove-bg, text, capabilities",
                    config.operation
                )),
                timing_ms: start.elapsed().as_millis() as u64,
                frame_count: None,
                file_size: None,
            };
            if json_output {
                println!("{}", serde_json::to_string_pretty(&err_result)?);
            } else {
                eprintln!("Error: {}", err_result.error.as_ref().unwrap());
            }
            anyhow::bail!("Unknown operation: {}", config.operation);
        }
    };

    // Build result
    let timing_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok((outputs, frame_count, file_size)) => {
            let result = ExecResult {
                success: true,
                operation: config.operation.clone(),
                output: outputs.first().cloned(),
                outputs: outputs.clone(),
                error: None,
                timing_ms,
                frame_count,
                file_size,
            };

            if json_output {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                for out in &outputs {
                    println!("{}", out);
                }
            }
            Ok(())
        }
        Err(e) => {
            let result = ExecResult {
                success: false,
                operation: config.operation.clone(),
                output: None,
                outputs: vec![],
                error: Some(e.to_string()),
                timing_ms,
                frame_count: None,
                file_size: None,
            };

            if json_output {
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            Err(e)
        }
    }
}

// Execution helpers - returns (output_paths, frame_count, file_size)
type ExecReturn = Result<(Vec<String>, Option<u32>, Option<u64>)>;

fn exec_transcode(config: &ExecConfig) -> ExecReturn {
    let input = config.input.as_ref()
        .ok_or_else(|| anyhow::anyhow!("'input' is required for transcode"))?;
    let output = config.output.as_ref()
        .ok_or_else(|| anyhow::anyhow!("'output' is required for transcode"))?;

    let video = config.video.as_ref().cloned().unwrap_or_default();
    let filters = config.filters.as_ref().cloned().unwrap_or_default();
    let trim = config.trim.as_ref();

    // Parse flip/rotate from transforms
    let mut rotate: u16 = 0;
    let mut flip = "none".to_string();
    for t in &config.transforms {
        if t.starts_with("rotate:") {
            rotate = t.trim_start_matches("rotate:").parse().unwrap_or(0);
        } else if t.starts_with("flip:") {
            flip = t.trim_start_matches("flip:").to_string();
        }
    }

    cmd_video_transcode(
        PathBuf::from(input),
        PathBuf::from(output),
        video.width,
        video.height,
        video.fps,
        rotate,
        flip,
        video.speed,
        video.quality,
        video.bitrate,
        video.threads,
        0, // max_frames
        trim.map(|t| t.start).unwrap_or(0.0),
        trim.map(|t| t.end).unwrap_or(0.0),
        // Color adjustments
        filters.brightness,
        filters.contrast,
        filters.saturation,
        filters.hue,
        filters.gamma,
        // Filters
        filters.blur,
        filters.sharpen,
        filters.grayscale,
        filters.sepia,
        filters.invert,
        // Crop
        filters.crop,
        // Effects
        filters.fade_in,
        filters.fade_out,
        filters.speed_factor,
        // Watermark
        filters.watermark.map(PathBuf::from),
        filters.watermark_pos,
        filters.watermark_opacity,
        filters.watermark_scale,
        // Output
        video.codec,
        config.quiet,
    )?;

    let file_size = std::fs::metadata(output).ok().map(|m| m.len());
    Ok((vec![output.clone()], None, file_size))
}

fn exec_trim(config: &ExecConfig) -> ExecReturn {
    let input = config.input.as_ref()
        .ok_or_else(|| anyhow::anyhow!("'input' is required for trim"))?;
    let output = config.output.as_ref()
        .ok_or_else(|| anyhow::anyhow!("'output' is required for trim"))?;
    let trim = config.trim.as_ref()
        .ok_or_else(|| anyhow::anyhow!("'trim' settings required for trim operation"))?;
    let video = config.video.as_ref().cloned().unwrap_or_default();

    cmd_video_trim(
        PathBuf::from(input),
        PathBuf::from(output),
        trim.start,
        trim.end,
        video.speed,
        video.quality,
        config.quiet,
    )?;

    let file_size = std::fs::metadata(output).ok().map(|m| m.len());
    Ok((vec![output.clone()], None, file_size))
}

fn exec_concat(config: &ExecConfig) -> ExecReturn {
    let output = config.output.as_ref()
        .ok_or_else(|| anyhow::anyhow!("'output' is required for concat"))?;

    let mut inputs: Vec<PathBuf> = config.inputs.iter().map(PathBuf::from).collect();
    if let Some(input) = &config.input {
        inputs.insert(0, PathBuf::from(input));
    }

    if inputs.len() < 2 {
        anyhow::bail!("concat requires at least 2 input files");
    }

    let video = config.video.as_ref().cloned().unwrap_or_default();

    cmd_video_concat(
        PathBuf::from(output),
        inputs,
        video.speed,
        video.quality,
        video.fps,
        config.quiet,
    )?;

    let file_size = std::fs::metadata(output).ok().map(|m| m.len());
    Ok((vec![output.clone()], None, file_size))
}

fn exec_encode(config: &ExecConfig) -> ExecReturn {
    let output = config.output.as_ref()
        .ok_or_else(|| anyhow::anyhow!("'output' is required for encode"))?;

    let mut inputs: Vec<PathBuf> = config.inputs.iter().map(PathBuf::from).collect();
    if let Some(input) = &config.input {
        inputs.insert(0, PathBuf::from(input));
    }

    if inputs.is_empty() {
        anyhow::bail!("encode requires at least 1 input image");
    }

    let video = config.video.as_ref().cloned().unwrap_or_default();

    cmd_video_encode(
        PathBuf::from(output),
        inputs,
        video.fps.max(30.0),
        video.speed,
        video.quality,
        video.bitrate,
        video.width,
        video.height,
        video.threads,
        config.quiet,
    )?;

    let file_size = std::fs::metadata(output).ok().map(|m| m.len());
    Ok((vec![output.clone()], None, file_size))
}

fn exec_frames(config: &ExecConfig) -> ExecReturn {
    let input = config.input.as_ref()
        .ok_or_else(|| anyhow::anyhow!("'input' is required for frames"))?;
    let frames = config.frames.as_ref()
        .ok_or_else(|| anyhow::anyhow!("'frames' settings required for frames operation"))?;

    cmd_video_frames(
        PathBuf::from(input),
        Some(PathBuf::from(&frames.output_dir)),
        frames.format.clone(),
        frames.every,
        frames.max_frames,
        config.quiet,
    )?;

    Ok((vec![frames.output_dir.clone()], None, None))
}

fn exec_gif(config: &ExecConfig) -> ExecReturn {
    let input = config.input.as_ref()
        .ok_or_else(|| anyhow::anyhow!("'input' is required for gif"))?;
    let output = config.output.as_ref()
        .ok_or_else(|| anyhow::anyhow!("'output' is required for gif"))?;
    let video = config.video.as_ref().cloned().unwrap_or_default();

    cmd_video_to_gif(
        PathBuf::from(input),
        PathBuf::from(output),
        video.width,
        video.fps as u32,
        1, // skip
        0, // max_frames
        0, // loops (infinite)
        config.quiet,
    )?;

    let file_size = std::fs::metadata(output).ok().map(|m| m.len());
    Ok((vec![output.clone()], None, file_size))
}

fn exec_info(config: &ExecConfig) -> ExecReturn {
    let mut inputs: Vec<PathBuf> = config.inputs.iter().map(PathBuf::from).collect();
    if let Some(input) = &config.input {
        inputs.insert(0, PathBuf::from(input));
    }

    if inputs.is_empty() {
        anyhow::bail!("info requires at least 1 input file");
    }

    cmd_video_info(inputs, true, config.quiet)?;
    Ok((vec![], None, None))
}

fn exec_remove_bg(config: &ExecConfig) -> ExecReturn {
    let mut inputs: Vec<PathBuf> = config.inputs.iter().map(PathBuf::from).collect();
    if let Some(input) = &config.input {
        inputs.insert(0, PathBuf::from(input));
    }

    if inputs.is_empty() {
        anyhow::bail!("remove-bg requires at least 1 input image");
    }

    let output_dir = config.output.as_ref().map(PathBuf::from);
    let rmbg = config.remove_bg.as_ref();

    cmd_remove_bg(
        inputs,
        output_dir.clone(),
        rmbg.and_then(|r| r.model.as_ref()).map(PathBuf::from),
        rmbg.map(|r| r.threshold).unwrap_or(0.5),
        rmbg.map(|r| r.cpu).unwrap_or(false),
        config.quiet,
    )?;

    Ok((output_dir.map(|p| p.to_string_lossy().to_string()).into_iter().collect(), None, None))
}

fn exec_text(config: &ExecConfig) -> ExecReturn {
    let mut inputs: Vec<PathBuf> = config.inputs.iter().map(PathBuf::from).collect();
    if let Some(input) = &config.input {
        inputs.insert(0, PathBuf::from(input));
    }

    if inputs.is_empty() {
        anyhow::bail!("text requires at least 1 input image");
    }

    let text = config.text.as_ref()
        .ok_or_else(|| anyhow::anyhow!("'text' settings required for text operation"))?;

    let output_dir = config.output.as_ref().map(PathBuf::from);

    cmd_text_overlay(
        inputs,
        text.content.clone(),
        PathBuf::from(&text.font),
        text.font_size,
        text.x,
        text.y,
        text.color.clone(),
        text.anchor.clone(),
        text.shadow.clone(),
        text.outline,
        output_dir.clone(),
        config.quiet,
    )?;

    Ok((output_dir.map(|p| p.to_string_lossy().to_string()).into_iter().collect(), None, None))
}

// ============================================================================
// Template Command (generate JSON templates for agents)
// ============================================================================

fn cmd_template(operation: String, pretty: bool) -> Result<()> {
    let template = match operation.as_str() {
        "transcode" | "video-transcode" => ExecConfig {
            input: Some("input.ivf".to_string()),
            output: Some("output.ivf".to_string()),
            operation: "transcode".to_string(),
            inputs: vec![],
            video: Some(ExecVideoSettings {
                codec: "av1".to_string(),
                width: 1920,
                height: 1080,
                fps: 30.0,
                quality: 80,
                speed: 6,
                bitrate: 0,
                threads: 0,
            }),
            audio: None,
            transforms: vec!["rotate:0".to_string(), "flip:none".to_string()],
            trim: Some(ExecTrimSettings { start: 0.0, end: 0.0 }),
            text: None,
            frames: None,
            remove_bg: None,
            filters: Some(ExecFilterSettings {
                brightness: 0,
                contrast: 0,
                saturation: 0,
                hue: 0,
                gamma: 1.0,
                blur: 0,
                sharpen: 0,
                grayscale: false,
                sepia: false,
                invert: false,
                crop: None,
                fade_in: 0.0,
                fade_out: 0.0,
                speed_factor: 1.0,
                watermark: None,
                watermark_pos: "br".to_string(),
                watermark_opacity: 0.5,
                watermark_scale: 0.2,
            }),
            quiet: false,
        },
        "trim" | "video-trim" => ExecConfig {
            input: Some("input.ivf".to_string()),
            output: Some("output.ivf".to_string()),
            operation: "trim".to_string(),
            inputs: vec![],
            video: Some(ExecVideoSettings::default()),
            audio: None,
            transforms: vec![],
            trim: Some(ExecTrimSettings { start: 0.0, end: 10.0 }),
            text: None,
            frames: None,
            remove_bg: None,
            filters: None,
            quiet: false,
        },
        "concat" | "video-concat" => ExecConfig {
            input: None,
            output: Some("output.ivf".to_string()),
            operation: "concat".to_string(),
            inputs: vec!["video1.ivf".to_string(), "video2.ivf".to_string()],
            video: Some(ExecVideoSettings::default()),
            audio: None,
            transforms: vec![],
            trim: None,
            text: None,
            frames: None,
            remove_bg: None,
            filters: None,
            quiet: false,
        },
        "encode" | "video-encode" => ExecConfig {
            input: None,
            output: Some("output.ivf".to_string()),
            operation: "encode".to_string(),
            inputs: vec!["frame_0001.png".to_string(), "frame_0002.png".to_string()],
            video: Some(ExecVideoSettings {
                codec: "av1".to_string(),
                width: 0,
                height: 0,
                fps: 30.0,
                quality: 100,
                speed: 6,
                bitrate: 0,
                threads: 0,
            }),
            audio: None,
            transforms: vec![],
            trim: None,
            text: None,
            frames: None,
            remove_bg: None,
            filters: None,
            quiet: false,
        },
        "frames" | "video-frames" => ExecConfig {
            input: Some("input.ivf".to_string()),
            output: None,
            operation: "frames".to_string(),
            inputs: vec![],
            video: None,
            audio: None,
            transforms: vec![],
            trim: None,
            text: None,
            frames: Some(ExecFrameSettings {
                output_dir: "./frames".to_string(),
                format: "png".to_string(),
                every: 1,
                max_frames: 0,
            }),
            remove_bg: None,
            filters: None,
            quiet: false,
        },
        "gif" | "video-to-gif" => ExecConfig {
            input: Some("input.ivf".to_string()),
            output: Some("output.gif".to_string()),
            operation: "gif".to_string(),
            inputs: vec![],
            video: Some(ExecVideoSettings {
                width: 480,
                fps: 15.0,
                ..Default::default()
            }),
            audio: None,
            transforms: vec![],
            trim: None,
            text: None,
            frames: None,
            remove_bg: None,
            filters: None,
            quiet: false,
        },
        "remove-bg" | "rmbg" => ExecConfig {
            input: Some("input.jpg".to_string()),
            output: Some("./output".to_string()),
            operation: "remove-bg".to_string(),
            inputs: vec![],
            video: None,
            audio: None,
            transforms: vec![],
            trim: None,
            text: None,
            frames: None,
            remove_bg: Some(ExecRemoveBgSettings {
                model: None,
                threshold: 0.5,
                cpu: false,
            }),
            filters: None,
            quiet: false,
        },
        "text" | "text-overlay" => ExecConfig {
            input: Some("input.png".to_string()),
            output: Some("./output".to_string()),
            operation: "text".to_string(),
            inputs: vec![],
            video: None,
            audio: None,
            transforms: vec![],
            trim: None,
            text: Some(ExecTextSettings {
                content: "Hello, World!".to_string(),
                font: "/path/to/font.ttf".to_string(),
                font_size: 32.0,
                x: 10,
                y: 10,
                color: "FFFFFF".to_string(),
                anchor: "tl".to_string(),
                shadow: Some("2,2".to_string()),
                outline: 0,
            }),
            frames: None,
            remove_bg: None,
            filters: None,
            quiet: false,
        },
        "info" | "video-info" => ExecConfig {
            input: Some("input.ivf".to_string()),
            output: None,
            operation: "info".to_string(),
            inputs: vec![],
            video: None,
            audio: None,
            transforms: vec![],
            trim: None,
            text: None,
            frames: None,
            remove_bg: None,
            filters: None,
            quiet: false,
        },
        _ => {
            eprintln!("Unknown operation: {}", operation);
            eprintln!("Valid operations: transcode, trim, concat, encode, frames, gif, remove-bg, text, info");
            anyhow::bail!("Unknown operation: {}", operation);
        }
    };

    let json = if pretty {
        serde_json::to_string_pretty(&template)?
    } else {
        serde_json::to_string(&template)?
    };

    println!("{}", json);
    Ok(())
}
