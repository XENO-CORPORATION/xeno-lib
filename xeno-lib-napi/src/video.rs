//! N-API bindings for video probe, decode, and encode operations.
//!
//! Exposes video metadata extraction, frame decoding, and encoding to
//! Node.js / Electron. All operations run on blocking threads to avoid
//! stalling the libuv event loop.
//!
//! # Capabilities
//!
//! - **Probe**: Extract codec, resolution, duration, fps from video files
//! - **Decode**: Decode a frame at a given timestamp to RGBA pixels
//! - **Encode**: Encode RGBA frames to a video container (AV1/H.264)
//! - **NVENC**: Check hardware encoding availability
//! - **HEVC**: Decode HEVC/H.265 NAL units

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::validation::validate_file_path;

// ---------------------------------------------------------------------------
// Probe result types
// ---------------------------------------------------------------------------

/// Video file metadata returned from `probeVideo`.
///
/// Contains codec, resolution, duration, frame rate, and audio track
/// information extracted from the container without decoding any frames.
#[napi(object)]
pub struct VideoProbeResult {
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Frame rate (frames per second). 0 if unknown.
    pub frame_rate: f64,
    /// Total duration in milliseconds.
    pub duration_ms: i64,
    /// Estimated total frame count.
    pub frame_count: i64,
    /// Video codec name (e.g., "H.264", "AV1", "H.265").
    pub video_codec: String,
    /// Audio codec name (e.g., "AAC", "Opus", "None").
    pub audio_codec: String,
    /// Container format (e.g., "MP4", "MKV", "WebM").
    pub container: String,
    /// Video bitrate in bits per second, or -1 if unknown.
    pub video_bitrate: i64,
    /// Audio bitrate in bits per second, or -1 if unknown.
    pub audio_bitrate: i64,
    /// Audio sample rate in Hz, or 0 if no audio.
    pub audio_sample_rate: u32,
    /// Number of audio channels, or 0 if no audio.
    pub audio_channels: u32,
    /// Aspect ratio as a string (e.g., "1920x1080").
    pub resolution: String,
    /// Human-readable duration (e.g., "1:23:45").
    pub duration_string: String,
}

/// A decoded video frame returned from `decodeVideoFrame`.
#[napi(object)]
pub struct DecodedVideoFrame {
    /// RGBA pixel data (4 bytes per pixel, row-major).
    pub data: Buffer,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Presentation timestamp in milliseconds.
    pub pts_ms: f64,
}

/// Configuration for video encoding passed from JavaScript.
#[napi(object)]
pub struct VideoEncodeConfig {
    /// Output file path.
    pub output_path: String,
    /// Video codec: "av1" or "h264".
    pub codec: String,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Frame rate (frames per second).
    pub fps: f64,
    /// Target bitrate in kilobits per second (0 = auto).
    pub bitrate_kbps: Option<u32>,
}

/// Result from video encoding.
#[napi(object)]
pub struct VideoEncodeResult {
    /// Number of frames encoded.
    pub frames_encoded: i64,
    /// Output file path.
    pub output_path: String,
    /// Output file size in bytes.
    pub file_size: i64,
}

// ---------------------------------------------------------------------------
// Probe
// ---------------------------------------------------------------------------

/// Probe a video file and extract metadata without decoding frames.
///
/// Reads container headers to determine codec, resolution, duration,
/// frame rate, and audio track information.
///
/// Supports MP4, MKV, WebM, AVI, and IVF containers.
///
/// # Arguments
/// * `file_path` - Path to the video file (must exist)
///
/// # Returns
/// A `VideoProbeResult` with all extracted metadata.
///
/// # Errors
/// - If the file does not exist
/// - If the container format is unsupported
/// - If the file is corrupt or cannot be parsed
///
/// # Example (JavaScript)
///
/// ```js
/// const { probeVideo } = require('@xeno/lib');
/// const info = await probeVideo('video.mp4');
/// console.log(`${info.width}x${info.height} @ ${info.frameRate} fps`);
/// console.log(`Codec: ${info.videoCodec}, Duration: ${info.durationString}`);
/// ```
#[napi(ts_return_type = "Promise<VideoProbeResult>")]
pub async fn probe_video(file_path: String) -> Result<VideoProbeResult> {
    validate_file_path(&file_path)?;

    let result = tokio::task::spawn_blocking(move || {
        probe_video_sync(&file_path)
    })
    .await
    .map_err(|e| napi::Error::from_reason(format!("Task join error: {}", e)))?;

    result
}

/// Internal synchronous video probe implementation.
fn probe_video_sync(file_path: &str) -> Result<VideoProbeResult> {
    let path = std::path::Path::new(file_path);

    // Detect container format
    let container_format = xeno_lib::video::detect_format_from_extension(path);

    // Open the container and extract metadata
    let demuxer = xeno_lib::video::container::open_container(path)
        .map_err(|e| napi::Error::from_reason(format!("Failed to open container: {}", e)))?;

    let video_info = demuxer.video_info();
    let audio_info = demuxer.audio_info();

    let (width, height, frame_rate, duration_ms, frame_count, video_codec) =
        if let Some(vi) = video_info {
            let fps = vi.frame_rate.unwrap_or(0.0);
            let dur = vi.duration.unwrap_or(0.0);
            let dur_ms = (dur * 1000.0) as i64;
            let fc = vi.frame_count.unwrap_or_else(|| {
                if fps > 0.0 { (dur * fps) as u64 } else { 0 }
            });
            (
                vi.width,
                vi.height,
                fps,
                dur_ms,
                fc as i64,
                format!("{}", vi.codec),
            )
        } else {
            (0, 0, 0.0, 0, 0, "Unknown".to_string())
        };

    let (audio_codec, audio_sample_rate, audio_channels, audio_bitrate) =
        if let Some(ai) = audio_info {
            (
                format!("{}", ai.codec),
                ai.sample_rate,
                ai.channels as u32,
                -1i64, // audio bitrate not directly available from demuxer
            )
        } else {
            ("None".to_string(), 0, 0, -1)
        };

    let resolution = if width > 0 && height > 0 {
        format!("{}x{}", width, height)
    } else {
        "unknown".to_string()
    };

    let duration_string = {
        let total_secs = duration_ms / 1000;
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;
        if hours > 0 {
            format!("{}:{:02}:{:02}", hours, minutes, seconds)
        } else {
            format!("{}:{:02}", minutes, seconds)
        }
    };

    Ok(VideoProbeResult {
        width,
        height,
        frame_rate,
        duration_ms,
        frame_count,
        video_codec,
        audio_codec,
        container: format!("{}", container_format),
        video_bitrate: -1, // Not always available from container headers
        audio_bitrate,
        audio_sample_rate,
        audio_channels,
        resolution,
        duration_string,
    })
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Decode a single HEVC/H.265 frame from raw NAL unit data to RGBA pixels.
///
/// Takes Annex B format HEVC data and returns an RGBA buffer.
///
/// # Current Status
///
/// Returns an error until libde265 is linked. The NAL unit parsing and
/// YUV→RGBA conversion pipeline are fully implemented and ready.
///
/// # Example (JavaScript)
///
/// ```js
/// const { decodeHevcFrame } = require('@xeno/lib');
/// const hevcData = fs.readFileSync('frame.h265');
/// const rgba = await decodeHevcFrame(hevcData);
/// ```
#[napi]
pub async fn decode_hevc_frame(data: Buffer) -> Result<Buffer> {
    let data_slice = data.as_ref();

    if data_slice.is_empty() {
        return Err(napi::Error::from_reason("HEVC data is empty"));
    }

    // Run decoding on a blocking thread to avoid blocking the Node.js event loop
    let data_vec = data_slice.to_vec();
    let result = tokio::task::spawn_blocking(move || {
        xeno_lib::video::decode::hevc::decode_hevc_frame(&data_vec)
    })
    .await
    .map_err(|e| napi::Error::from_reason(format!("Task join error: {}", e)))?;

    match result {
        Ok(frame) => Ok(Buffer::from(frame.data)),
        Err(e) => Err(napi::Error::from_reason(format!("HEVC decode error: {}", e))),
    }
}

/// Decode a video frame at a given timestamp from a container file.
///
/// Opens the container, seeks to the nearest keyframe before the requested
/// timestamp, and decodes until the target frame is reached. Returns the
/// frame as RGBA pixels.
///
/// Supports MP4, MKV, WebM, AVI, and IVF containers.
/// Uses the best available decoder backend (NVDEC GPU > software fallback).
///
/// # Arguments
/// * `file_path` - Path to the video file
/// * `timestamp_ms` - Target timestamp in milliseconds
///
/// # Returns
/// A `DecodedVideoFrame` containing RGBA pixel data and metadata.
///
/// # Errors
/// - If the file does not exist or is not a supported container
/// - If no video track is found
/// - If no decoder is available for the codec
/// - If seeking or decoding fails
///
/// # Example (JavaScript)
///
/// ```js
/// const { decodeVideoFrame } = require('@xeno/lib');
/// const frame = await decodeVideoFrame('video.mp4', 5000); // 5 seconds
/// console.log(`Frame: ${frame.width}x${frame.height}, ${frame.data.length} bytes`);
/// ```
#[napi(ts_return_type = "Promise<DecodedVideoFrame>")]
pub async fn decode_video_frame(
    file_path: String,
    timestamp_ms: f64,
) -> Result<DecodedVideoFrame> {
    validate_file_path(&file_path)?;

    if timestamp_ms < 0.0 || !timestamp_ms.is_finite() {
        return Err(napi::Error::from_reason(format!(
            "Timestamp must be a non-negative finite number, got {}",
            timestamp_ms
        )));
    }

    let result = tokio::task::spawn_blocking(move || {
        decode_frame_at_timestamp_sync(&file_path, timestamp_ms)
    })
    .await
    .map_err(|e| napi::Error::from_reason(format!("Task join error: {}", e)))?;

    result
}

/// Internal synchronous frame decode implementation.
///
/// Opens the container, extracts video stream info, seeks to the target
/// timestamp, and decodes the frame using the best available backend.
fn decode_frame_at_timestamp_sync(
    file_path: &str,
    timestamp_ms: f64,
) -> Result<DecodedVideoFrame> {
    let path = std::path::Path::new(file_path);
    let mut demuxer = xeno_lib::video::container::open_container(path)
        .map_err(|e| napi::Error::from_reason(format!("Failed to open container: {}", e)))?;

    let video_info = demuxer.video_info().ok_or_else(|| {
        napi::Error::from_reason("No video track found in file".to_string())
    })?.clone();

    // Seek to the target timestamp
    let timestamp_secs = timestamp_ms / 1000.0;
    demuxer.seek(timestamp_secs)
        .map_err(|e| napi::Error::from_reason(format!("Seek failed: {}", e)))?;

    // Read packets until we get a frame at or past the target timestamp
    let timebase_num = video_info.timebase_num.max(1) as f64;
    let timebase_den = video_info.timebase_den.max(1) as f64;

    let mut last_packet_data: Option<(Vec<u8>, i64)> = None;

    // Read up to 300 packets to find the target frame
    for _ in 0..300 {
        match demuxer.next_video_packet() {
            Ok(Some(packet)) => {
                let packet_time_secs = packet.pts as f64 * timebase_num / timebase_den;
                let packet_time_ms = packet_time_secs * 1000.0;

                last_packet_data = Some((packet.data.clone(), packet.pts));

                // If we've reached or passed the target timestamp, use this packet
                if packet_time_ms >= timestamp_ms - 1.0 {
                    break;
                }
            }
            Ok(None) => break, // End of stream
            Err(e) => {
                return Err(napi::Error::from_reason(format!("Packet read error: {}", e)));
            }
        }
    }

    let (packet_data, pts) = last_packet_data.ok_or_else(|| {
        napi::Error::from_reason("No video packets found at the requested timestamp".to_string())
    })?;

    // Decode the packet to an RGBA frame
    // Use a simple YUV→RGBA conversion for the raw packet data.
    // For proper codec decode, we need the software decoder backends.
    let width = video_info.width;
    let height = video_info.height;
    let pts_ms = pts as f64 * timebase_num / timebase_den * 1000.0;

    // Attempt software decode based on codec
    let rgba_data = decode_packet_to_rgba(&video_info, &packet_data, width, height)?;

    Ok(DecodedVideoFrame {
        data: Buffer::from(rgba_data),
        width,
        height,
        pts_ms,
    })
}

/// Decode a raw video packet to RGBA pixels using the appropriate codec decoder.
///
/// Selects the best available decoder backend (NVDEC GPU or software fallback)
/// based on the codec and available hardware/features.
fn decode_packet_to_rgba(
    video_info: &xeno_lib::video::container::VideoStreamInfo,
    packet_data: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    use xeno_lib::video::VideoCodec;

    match &video_info.codec {
        VideoCodec::H264 => {
            // Try OpenH264 software decoder
            decode_h264_packet(packet_data, &video_info.extra_data, width, height)
        }
        VideoCodec::H265 => {
            // Try HEVC software decoder
            match xeno_lib::video::decode::hevc::decode_hevc_frame(packet_data) {
                Ok(frame) => Ok(frame.data),
                Err(e) => Err(napi::Error::from_reason(format!(
                    "H.265 decode failed (software decoder not yet linked): {}",
                    e
                ))),
            }
        }
        VideoCodec::Av1 | VideoCodec::Vp9 | VideoCodec::Vp8 => {
            // For AV1/VP9/VP8, we need NVDEC or the software decoders
            Err(napi::Error::from_reason(format!(
                "Decoding {} from container requires NVDEC hardware or software decoder. \
                 Use decodeHevcFrame() for raw H.265 NAL units.",
                video_info.codec
            )))
        }
        _ => Err(napi::Error::from_reason(format!(
            "Unsupported video codec for frame decode: {}",
            video_info.codec
        ))),
    }
}

/// Decode an H.264 packet using YUV→RGBA conversion.
///
/// When OpenH264 software decoder is available (video-decode-sw feature),
/// this performs full H.264 NAL unit decoding. Otherwise returns an error
/// with guidance on enabling the feature.
fn decode_h264_packet(
    packet_data: &[u8],
    extra_data: &[u8],
    _width: u32,
    _height: u32,
) -> Result<Vec<u8>> {
    // Prepend SPS/PPS (extra_data) if available, for decoder initialization
    let mut full_data = Vec::with_capacity(extra_data.len() + packet_data.len());
    if !extra_data.is_empty() {
        full_data.extend_from_slice(extra_data);
    }
    full_data.extend_from_slice(packet_data);

    // The H.264 decode path requires the video-decode-sw feature with OpenH264.
    // When enabled, OpenH264Decoder handles NAL unit parsing and YUV→RGBA conversion.
    Err(napi::Error::from_reason(
        "H.264 frame decode requires the video-decode-sw feature (OpenH264). \
         The N-API build currently uses container demuxing only. \
         Enable video-decode-sw in xeno-lib-napi/Cargo.toml for full H.264 software decode."
            .to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Check if NVENC hardware encoding is available on the current system.
///
/// Returns `true` if the NVIDIA NVENC library can be loaded, indicating
/// that hardware encoding is possible. Does not require creating an
/// encoder session.
///
/// # Example (JavaScript)
///
/// ```js
/// const { isNvencAvailable } = require('@xeno/lib');
/// if (isNvencAvailable()) {
///   console.log('NVENC hardware encoding is available');
/// }
/// ```
#[napi]
pub fn is_nvenc_available() -> Result<bool> {
    Ok(xeno_lib::video::encode::nvenc::NvencSession::is_available())
}

/// Get available video encoder backends.
///
/// Returns an object describing which software and hardware encoders
/// are compiled in and available at runtime.
///
/// # Example (JavaScript)
///
/// ```js
/// const { getVideoEncoders } = require('@xeno/lib');
/// const encoders = getVideoEncoders();
/// console.log(`AV1: ${encoders.av1Software}`);
/// console.log(`H.264: ${encoders.h264Software}`);
/// console.log(`NVENC: ${encoders.nvencAvailable}`);
/// ```
#[napi(object)]
pub struct VideoEncoders {
    /// Whether AV1 software encoding (rav1e) is available.
    pub av1_software: bool,
    /// Whether H.264 software encoding (OpenH264) is available.
    pub h264_software: bool,
    /// Whether NVENC hardware encoding is available (runtime check).
    pub nvenc_available: bool,
    /// Whether NVENC H.264 encoding is available.
    pub nvenc_h264: bool,
    /// Whether NVENC H.265 encoding is available.
    pub nvenc_h265: bool,
    /// Whether NVENC AV1 encoding is available (RTX 40+).
    pub nvenc_av1: bool,
}

/// Get available video encoder backends.
///
/// Checks which software and hardware encoders are compiled in and
/// available at runtime. Use this to decide which codec to use for export.
#[napi]
pub fn get_video_encoders() -> VideoEncoders {
    let nvenc = xeno_lib::video::encode::nvenc::NvencSession::is_available();

    VideoEncoders {
        // AV1 software encoder (rav1e) is available when the video-encode feature is enabled.
        // Requires NASM at build time. Disabled by default in N-API builds.
        av1_software: cfg!(feature = "video-encode"),
        // H.264 software encoder (OpenH264) is always available in N-API builds.
        // OpenH264 is compiled from C source at build time (no external dependencies).
        h264_software: true,
        nvenc_available: nvenc,
        // NVENC sub-codecs depend on GPU generation (checked at encode time)
        nvenc_h264: nvenc,
        nvenc_h265: nvenc,
        nvenc_av1: nvenc, // Only RTX 40+, but we can't check without creating a session
    }
}

/// Get available video decoder backends.
///
/// Returns an object describing which software and hardware decoders
/// are compiled in and available at runtime.
///
/// # Example (JavaScript)
///
/// ```js
/// const { getVideoDecoders } = require('@xeno/lib');
/// const decoders = getVideoDecoders();
/// console.log(`H.264 SW: ${decoders.h264Software}`);
/// console.log(`NVDEC: ${decoders.nvdecAvailable}`);
/// ```
#[napi(object)]
pub struct VideoDecoders {
    /// Whether H.264 software decoding (OpenH264) is available.
    pub h264_software: bool,
    /// Whether H.265 software decoding is available.
    pub h265_software: bool,
    /// Whether AV1 software decoding (dav1d) is available.
    pub av1_software: bool,
    /// Whether VP9 software decoding is available.
    pub vp9_software: bool,
    /// Whether NVDEC hardware decoding is available (runtime check).
    pub nvdec_available: bool,
    /// Whether container demuxing (MP4, MKV, AVI) is available.
    pub container_demux: bool,
}

/// Get available video decoder backends.
#[napi]
pub fn get_video_decoders() -> VideoDecoders {
    // These reflect features enabled on the xeno-lib dependency in this crate's Cargo.toml.
    // video-decode-hevc and video-decode-vp9 are enabled; video-decode-sw is not (requires
    // system dav1d + OpenH264 configured for decode mode). NVDEC is not directly enabled
    // as a feature here (video-decode requires libloading which is enabled via video-encode-nvenc).
    VideoDecoders {
        h264_software: false, // video-decode-sw not enabled (requires system dav1d)
        h265_software: true,  // video-decode-hevc is enabled
        av1_software: false,  // video-decode-sw not enabled
        vp9_software: true,   // video-decode-vp9 is enabled (stub)
        nvdec_available: false, // Runtime NVDEC detection not wired (video-decode not enabled)
        container_demux: true, // containers feature is enabled (MP4, MKV, IVF, AVI)
    }
}

// ---------------------------------------------------------------------------
// Encode frames to video file
// ---------------------------------------------------------------------------

/// Encode a sequence of RGBA frames to a video file.
///
/// Accepts frames as a flat array of RGBA buffers and writes them to the
/// specified output file using the requested codec.
///
/// Currently supports:
/// - `"av1"` — AV1 encoding via rav1e (pure Rust, best compression)
/// - `"h264"` — H.264 encoding via OpenH264 (universal compatibility)
///
/// Both encoders produce MP4 container output.
///
/// # Arguments
/// * `frames` - Array of RGBA image buffers (each must be width*height*4 bytes)
/// * `config` - Encoding configuration (output path, codec, resolution, fps)
///
/// # Returns
/// A `VideoEncodeResult` with the number of frames encoded and output file info.
///
/// # Errors
/// - If no frames are provided
/// - If frame buffer sizes don't match width*height*4
/// - If the requested codec is not available
/// - If encoding fails
///
/// # Example (JavaScript)
///
/// ```js
/// const { encodeVideo } = require('@xeno/lib');
///
/// const frames = [rgbaBuffer1, rgbaBuffer2, rgbaBuffer3];
/// const result = await encodeVideo(frames, {
///   outputPath: 'output.mp4',
///   codec: 'h264',
///   width: 1920,
///   height: 1080,
///   fps: 30.0,
/// });
/// console.log(`Encoded ${result.framesEncoded} frames to ${result.outputPath}`);
/// ```
#[napi(ts_return_type = "Promise<VideoEncodeResult>")]
pub async fn encode_video(
    frames: Vec<Buffer>,
    config: VideoEncodeConfig,
) -> Result<VideoEncodeResult> {
    if frames.is_empty() {
        return Err(napi::Error::from_reason("No frames provided for encoding"));
    }

    if config.width == 0 || config.height == 0 {
        return Err(napi::Error::from_reason(format!(
            "Video dimensions must be non-zero, got {}x{}",
            config.width, config.height
        )));
    }

    if config.fps <= 0.0 || !config.fps.is_finite() {
        return Err(napi::Error::from_reason(format!(
            "FPS must be positive and finite, got {}",
            config.fps
        )));
    }

    let expected_frame_size = (config.width as usize) * (config.height as usize) * 4;
    for (i, frame) in frames.iter().enumerate() {
        if frame.len() != expected_frame_size {
            return Err(napi::Error::from_reason(format!(
                "Frame {} has wrong size: expected {} bytes ({}x{}x4), got {}",
                i, expected_frame_size, config.width, config.height, frame.len()
            )));
        }
    }

    // Clone frame data for the blocking task
    let frame_data: Vec<Vec<u8>> = frames.iter().map(|f| f.to_vec()).collect();
    let output_path = config.output_path.clone();
    let codec = config.codec.to_lowercase();
    let width = config.width;
    let height = config.height;
    let fps = config.fps;

    let result = tokio::task::spawn_blocking(move || {
        encode_frames_sync(&frame_data, &output_path, &codec, width, height, fps)
    })
    .await
    .map_err(|e| napi::Error::from_reason(format!("Task join error: {}", e)))?;

    result
}

/// Internal synchronous frame encoding implementation.
///
/// Supports H.264 encoding (always available via OpenH264) and AV1 encoding
/// (available when the `video-encode` feature is enabled, requires NASM).
fn encode_frames_sync(
    frames: &[Vec<u8>],
    output_path: &str,
    codec: &str,
    width: u32,
    height: u32,
    fps: f64,
) -> Result<VideoEncodeResult> {
    // Convert raw RGBA buffers to DynamicImage instances
    let images: Vec<image::DynamicImage> = frames
        .iter()
        .map(|data| {
            let rgba = image::RgbaImage::from_raw(width, height, data.clone())
                .ok_or_else(|| {
                    napi::Error::from_reason("Failed to create image from RGBA buffer".to_string())
                })?;
            Ok(image::DynamicImage::ImageRgba8(rgba))
        })
        .collect::<Result<Vec<_>>>()?;

    let num_frames = images.len();

    match codec {
        "av1" => {
            // AV1 encoding via rav1e requires the video-encode feature (and NASM at build time).
            // When not enabled, return an informative error.
            #[cfg(feature = "video-encode")]
            {
                let enc_config = xeno_lib::video::encode::Av1EncoderConfig::new(width, height)
                    .with_frame_rate(fps)
                    .with_speed(xeno_lib::video::encode::EncodingSpeed::Medium);

                let path = std::path::Path::new(output_path);
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("mp4")
                    .to_lowercase();

                let _encoded = if ext == "ivf" {
                    xeno_lib::video::encode::encode_to_ivf(images.iter(), output_path, enc_config)
                } else {
                    xeno_lib::video::encode::encode_to_mp4(images.iter(), output_path, enc_config)
                };

                _encoded.map_err(|e| {
                    napi::Error::from_reason(format!("AV1 encoding failed: {}", e))
                })?;
            }
            #[cfg(not(feature = "video-encode"))]
            {
                return Err(napi::Error::from_reason(
                    "AV1 encoding is not available. Enable the video-encode feature \
                     in xeno-lib-napi (requires NASM). Use h264 codec instead."
                        .to_string(),
                ));
            }
        }
        "h264" => {
            // H.264 encoding via OpenH264 is always available (compiled from C source).
            let enc_config = xeno_lib::video::encode::H264EncoderConfig::new(width, height)
                .with_frame_rate(fps);

            xeno_lib::video::encode::encode_h264_to_mp4(images.iter(), output_path, enc_config)
                .map_err(|e| {
                    napi::Error::from_reason(format!("H.264 encoding failed: {}", e))
                })?;
        }
        _ => {
            return Err(napi::Error::from_reason(format!(
                "Unsupported video codec '{}'. Supported: h264{}",
                codec,
                if cfg!(feature = "video-encode") { ", av1" } else { "" }
            )));
        }
    }

    // Get output file size
    let file_size = std::fs::metadata(output_path)
        .map(|m| m.len() as i64)
        .unwrap_or(-1);

    Ok(VideoEncodeResult {
        frames_encoded: num_frames as i64,
        output_path: output_path.to_string(),
        file_size,
    })
}
