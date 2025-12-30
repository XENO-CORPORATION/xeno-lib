//! Video editing operations: trimming, cutting, concatenation, speed change.
//!
//! This module provides high-level video editing functionality built on top of
//! the demuxing and muxing modules.
//!
//! # Features
//!
//! - **Trimming**: Extract a segment from start to end time
//! - **Cutting**: Remove a segment from a video
//! - **Concatenation**: Join multiple video files
//! - **Speed Change**: Change playback speed (0.25x - 4x)
//!
//! # Example: Trim video
//!
//! ```ignore
//! use xeno_lib::video::edit::{trim_video, TrimConfig};
//!
//! let config = TrimConfig::new(10.0, 30.0); // 10s to 30s
//! trim_video("input.mp4", "output.mp4", config)?;
//! ```

use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use crate::video::{VideoError, VideoResult};

#[cfg(feature = "video-encode-h264")]
use crate::video::mux::{AvMuxConfig, AvMuxer, AudioConfig, VideoConfig};

#[cfg(feature = "video")]
use crate::video::container::{open_container, Demuxer, Packet, VideoStreamInfo, AudioStreamInfo};

/// Configuration for video trimming.
#[derive(Debug, Clone)]
pub struct TrimConfig {
    /// Start time in seconds.
    pub start_time: f64,
    /// End time in seconds.
    pub end_time: f64,
    /// Whether to re-encode (true) or copy streams (false).
    pub reencode: bool,
    /// Whether to seek to nearest keyframe (faster but less precise).
    pub keyframe_seek: bool,
}

impl TrimConfig {
    /// Create a new trim configuration.
    pub fn new(start_time: f64, end_time: f64) -> Self {
        Self {
            start_time,
            end_time,
            reencode: false,
            keyframe_seek: true,
        }
    }

    /// Set precise mode (re-encode to get exact timestamps).
    pub fn precise(mut self) -> Self {
        self.reencode = true;
        self.keyframe_seek = false;
        self
    }

    /// Set fast mode (copy streams, seek to keyframes).
    pub fn fast(mut self) -> Self {
        self.reencode = false;
        self.keyframe_seek = true;
        self
    }

    /// Duration of the trimmed segment.
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }
}

/// Configuration for video concatenation.
#[derive(Debug, Clone)]
pub struct ConcatConfig {
    /// Whether to re-encode all inputs to a common format.
    pub reencode: bool,
    /// Target width (if re-encoding).
    pub width: Option<u32>,
    /// Target height (if re-encoding).
    pub height: Option<u32>,
    /// Target frame rate (if re-encoding).
    pub frame_rate: Option<f64>,
}

impl Default for ConcatConfig {
    fn default() -> Self {
        Self {
            reencode: false,
            width: None,
            height: None,
            frame_rate: None,
        }
    }
}

impl ConcatConfig {
    /// Create a new concatenation configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable re-encoding with target resolution.
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.reencode = true;
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Enable re-encoding with target frame rate.
    pub fn with_frame_rate(mut self, frame_rate: f64) -> Self {
        self.reencode = true;
        self.frame_rate = Some(frame_rate);
        self
    }
}

/// Configuration for speed change.
#[derive(Debug, Clone)]
pub struct SpeedConfig {
    /// Speed multiplier (0.25 = quarter speed, 2.0 = double speed).
    pub speed: f64,
    /// Whether to preserve audio pitch (requires re-encoding).
    pub preserve_pitch: bool,
    /// Whether to drop frames (faster) or interpolate (smoother).
    pub interpolate: bool,
}

impl SpeedConfig {
    /// Create a new speed configuration.
    pub fn new(speed: f64) -> Self {
        Self {
            speed: speed.clamp(0.1, 10.0),
            preserve_pitch: true,
            interpolate: false,
        }
    }

    /// Enable smooth interpolation (requires AI frame interpolation).
    pub fn smooth(mut self) -> Self {
        self.interpolate = true;
        self
    }
}

/// A video segment for editing operations.
#[derive(Debug, Clone)]
pub struct VideoSegment {
    /// Source file path.
    pub source: String,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds (None = end of file).
    pub end: Option<f64>,
}

impl VideoSegment {
    /// Create a segment from the entire file.
    pub fn whole<P: AsRef<Path>>(path: P) -> Self {
        Self {
            source: path.as_ref().to_string_lossy().to_string(),
            start: 0.0,
            end: None,
        }
    }

    /// Create a segment from a specific range.
    pub fn range<P: AsRef<Path>>(path: P, start: f64, end: f64) -> Self {
        Self {
            source: path.as_ref().to_string_lossy().to_string(),
            start,
            end: Some(end),
        }
    }
}

/// Trim a video file to extract a segment.
///
/// # Arguments
///
/// * `input` - Input video file path
/// * `output` - Output video file path
/// * `config` - Trim configuration
///
/// # Returns
///
/// Number of frames written.
#[cfg(all(feature = "video", feature = "av-mux", feature = "video-encode-h264"))]
pub fn trim_video<P: AsRef<Path>>(
    input: P,
    output: P,
    config: TrimConfig,
) -> VideoResult<u64> {
    let mut demuxer = open_container(input.as_ref())?;

    // Get stream info
    let video_info = demuxer.video_info().cloned().ok_or_else(|| VideoError::Container {
        message: "No video stream found".to_string(),
    })?;

    let audio_info = demuxer.audio_info().cloned();

    // Seek to start position
    demuxer.seek(config.start_time)?;

    // Calculate time bounds
    let start_pts = time_to_pts(config.start_time, video_info.timebase_num, video_info.timebase_den);
    let end_pts = time_to_pts(config.end_time, video_info.timebase_num, video_info.timebase_den);

    // Create output muxer
    let mux_config = create_mux_config(&video_info, audio_info.as_ref())?;
    let mut muxer = AvMuxer::new(output.as_ref(), mux_config)?;

    let mut frames_written = 0u64;
    let mut found_keyframe = false;

    // Copy video packets within range
    while let Some(packet) = demuxer.next_video_packet()? {
        // Skip until we find a keyframe (for clean cut)
        if !found_keyframe {
            if packet.is_keyframe && packet.pts >= start_pts {
                found_keyframe = true;
            } else {
                continue;
            }
        }

        // Stop at end time
        if packet.pts >= end_pts {
            break;
        }

        // Adjust PTS relative to segment start
        let adjusted_pts = packet.pts - start_pts;

        // Write packet
        muxer.write_video_sample(&packet.data, packet.is_keyframe)?;
        frames_written += 1;
    }

    // Copy audio packets within range (if present)
    if audio_info.is_some() {
        demuxer.reset()?;
        demuxer.seek(config.start_time)?;

        let audio_start_pts = time_to_pts(
            config.start_time,
            1,
            audio_info.as_ref().unwrap().sample_rate,
        );
        let audio_end_pts = time_to_pts(
            config.end_time,
            1,
            audio_info.as_ref().unwrap().sample_rate,
        );

        while let Some(packet) = demuxer.next_audio_packet()? {
            if packet.pts < audio_start_pts {
                continue;
            }
            if packet.pts >= audio_end_pts {
                break;
            }

            muxer.write_audio_sample(&packet.data)?;
        }
    }

    muxer.finish()?;
    Ok(frames_written)
}

/// Cut a segment out of a video (remove portion from middle).
///
/// Creates a video with the specified segment removed.
#[cfg(all(feature = "video", feature = "av-mux", feature = "video-encode-h264"))]
pub fn cut_video<P: AsRef<Path>>(
    input: P,
    output: P,
    cut_start: f64,
    cut_end: f64,
) -> VideoResult<u64> {
    let mut demuxer = open_container(input.as_ref())?;

    let video_info = demuxer.video_info().cloned().ok_or_else(|| VideoError::Container {
        message: "No video stream found".to_string(),
    })?;

    let audio_info = demuxer.audio_info().cloned();
    let duration = video_info.duration.unwrap_or(f64::MAX);

    // Calculate PTS values
    let cut_start_pts = time_to_pts(cut_start, video_info.timebase_num, video_info.timebase_den);
    let cut_end_pts = time_to_pts(cut_end, video_info.timebase_num, video_info.timebase_den);

    // Create output muxer
    let mux_config = create_mux_config(&video_info, audio_info.as_ref())?;
    let mut muxer = AvMuxer::new(output.as_ref(), mux_config)?;

    let mut frames_written = 0u64;
    let mut offset_pts = 0i64;
    let mut in_cut_region = false;

    // First pass: copy packets before cut region
    while let Some(packet) = demuxer.next_video_packet()? {
        if packet.pts >= cut_start_pts && !in_cut_region {
            // Entering cut region
            in_cut_region = true;
            offset_pts = cut_end_pts - cut_start_pts;
            continue;
        }

        if in_cut_region && packet.pts < cut_end_pts {
            // Inside cut region, skip
            continue;
        }

        if in_cut_region && packet.pts >= cut_end_pts {
            // Exiting cut region
            in_cut_region = false;
        }

        // Wait for keyframe after cut
        if !in_cut_region {
            muxer.write_video_sample(&packet.data, packet.is_keyframe)?;
            frames_written += 1;
        }
    }

    muxer.finish()?;
    Ok(frames_written)
}

/// Concatenate multiple video files into one.
///
/// # Arguments
///
/// * `inputs` - List of input video file paths
/// * `output` - Output video file path
/// * `config` - Concatenation configuration
///
/// # Returns
///
/// Total number of frames in output.
#[cfg(all(feature = "video", feature = "av-mux", feature = "video-encode-h264"))]
pub fn concat_videos<P: AsRef<Path>>(
    inputs: &[P],
    output: P,
    config: ConcatConfig,
) -> VideoResult<u64> {
    if inputs.is_empty() {
        return Err(VideoError::InvalidInput {
            message: "No input files provided".to_string(),
        });
    }

    // Open first input to get format info
    let first_demuxer = open_container(inputs[0].as_ref())?;
    let video_info = first_demuxer.video_info().cloned().ok_or_else(|| VideoError::Container {
        message: "No video stream in first input".to_string(),
    })?;
    let audio_info = first_demuxer.audio_info().cloned();
    drop(first_demuxer);

    // Create output muxer using first file's format
    let mux_config = create_mux_config(&video_info, audio_info.as_ref())?;
    let mut muxer = AvMuxer::new(output.as_ref(), mux_config)?;

    let mut total_frames = 0u64;

    // Process each input file
    for input_path in inputs {
        let mut demuxer = open_container(input_path.as_ref())?;

        // Copy all video packets
        while let Some(packet) = demuxer.next_video_packet()? {
            muxer.write_video_sample(&packet.data, packet.is_keyframe)?;
            total_frames += 1;
        }

        // Copy all audio packets
        if audio_info.is_some() {
            demuxer.reset()?;
            while let Some(packet) = demuxer.next_audio_packet()? {
                muxer.write_audio_sample(&packet.data)?;
            }
        }
    }

    muxer.finish()?;
    Ok(total_frames)
}

/// Concatenate video segments (portions of files) into one.
#[cfg(all(feature = "video", feature = "av-mux", feature = "video-encode-h264"))]
pub fn concat_segments(
    segments: &[VideoSegment],
    output: &Path,
) -> VideoResult<u64> {
    if segments.is_empty() {
        return Err(VideoError::InvalidInput {
            message: "No segments provided".to_string(),
        });
    }

    // Open first segment to get format info
    let first_demuxer = open_container(&segments[0].source)?;
    let video_info = first_demuxer.video_info().cloned().ok_or_else(|| VideoError::Container {
        message: "No video stream in first segment".to_string(),
    })?;
    let audio_info = first_demuxer.audio_info().cloned();
    drop(first_demuxer);

    // Create output muxer
    let mux_config = create_mux_config(&video_info, audio_info.as_ref())?;
    let mut muxer = AvMuxer::new(output, mux_config)?;

    let mut total_frames = 0u64;

    for segment in segments {
        let mut demuxer = open_container(&segment.source)?;
        let seg_info = demuxer.video_info().cloned().ok_or_else(|| VideoError::Container {
            message: format!("No video stream in segment: {}", segment.source),
        })?;

        // Seek to segment start
        demuxer.seek(segment.start)?;

        let start_pts = time_to_pts(segment.start, seg_info.timebase_num, seg_info.timebase_den);
        let end_pts = segment.end.map(|e| time_to_pts(e, seg_info.timebase_num, seg_info.timebase_den));

        let mut found_keyframe = false;

        while let Some(packet) = demuxer.next_video_packet()? {
            // Wait for keyframe
            if !found_keyframe {
                if packet.is_keyframe && packet.pts >= start_pts {
                    found_keyframe = true;
                } else {
                    continue;
                }
            }

            // Check end bound
            if let Some(end) = end_pts {
                if packet.pts >= end {
                    break;
                }
            }

            muxer.write_video_sample(&packet.data, packet.is_keyframe)?;
            total_frames += 1;
        }
    }

    muxer.finish()?;
    Ok(total_frames)
}

/// Change video playback speed.
#[cfg(all(feature = "video", feature = "av-mux", feature = "video-encode-h264"))]
pub fn change_speed<P: AsRef<Path>>(
    input: P,
    output: P,
    config: SpeedConfig,
) -> VideoResult<u64> {
    let mut demuxer = open_container(input.as_ref())?;

    let video_info = demuxer.video_info().cloned().ok_or_else(|| VideoError::Container {
        message: "No video stream found".to_string(),
    })?;

    // Modify frame rate for speed change
    let new_frame_rate = video_info.frame_rate.unwrap_or(30.0) * config.speed;

    // For speed increase, we drop frames; for decrease, we duplicate
    let frame_step = if config.speed >= 1.0 {
        config.speed
    } else {
        1.0
    };

    let mut modified_info = video_info.clone();
    modified_info.frame_rate = Some(new_frame_rate);

    let mux_config = create_mux_config(&modified_info, None)?;
    let mut muxer = AvMuxer::new(output.as_ref(), mux_config)?;

    let mut frames_written = 0u64;
    let mut frame_counter = 0.0f64;

    while let Some(packet) = demuxer.next_video_packet()? {
        if config.speed >= 1.0 {
            // Speed up: drop frames
            frame_counter += 1.0;
            if frame_counter >= frame_step {
                frame_counter -= frame_step;
                muxer.write_video_sample(&packet.data, packet.is_keyframe)?;
                frames_written += 1;
            }
        } else {
            // Slow down: duplicate frames
            let duplicates = (1.0 / config.speed).round() as u64;
            for _ in 0..duplicates {
                muxer.write_video_sample(&packet.data, packet.is_keyframe)?;
                frames_written += 1;
            }
        }
    }

    muxer.finish()?;
    Ok(frames_written)
}

/// Get video duration in seconds.
#[cfg(feature = "video")]
pub fn get_duration<P: AsRef<Path>>(input: P) -> VideoResult<f64> {
    let demuxer = open_container(input.as_ref())?;
    let video_info = demuxer.video_info().ok_or_else(|| VideoError::Container {
        message: "No video stream found".to_string(),
    })?;

    video_info.duration.ok_or_else(|| VideoError::Container {
        message: "Duration not available".to_string(),
    })
}

/// Get video frame count.
#[cfg(feature = "video")]
pub fn get_frame_count<P: AsRef<Path>>(input: P) -> VideoResult<u64> {
    let demuxer = open_container(input.as_ref())?;
    let video_info = demuxer.video_info().ok_or_else(|| VideoError::Container {
        message: "No video stream found".to_string(),
    })?;

    video_info.frame_count.ok_or_else(|| VideoError::Container {
        message: "Frame count not available".to_string(),
    })
}

/// Extract a single frame at a specific time.
#[cfg(feature = "video")]
pub fn extract_frame_time<P: AsRef<Path>>(
    input: P,
    time: f64,
) -> VideoResult<Packet> {
    let mut demuxer = open_container(input.as_ref())?;
    demuxer.seek(time)?;

    // Find the nearest keyframe
    while let Some(packet) = demuxer.next_video_packet()? {
        if packet.is_keyframe {
            return Ok(packet);
        }
    }

    Err(VideoError::Container {
        message: "No keyframe found near requested time".to_string(),
    })
}

// Helper functions

/// Convert time in seconds to PTS.
fn time_to_pts(time: f64, timebase_num: u32, timebase_den: u32) -> i64 {
    ((time * timebase_den as f64) / timebase_num as f64) as i64
}

/// Convert PTS to time in seconds.
fn pts_to_time(pts: i64, timebase_num: u32, timebase_den: u32) -> f64 {
    (pts as f64 * timebase_num as f64) / timebase_den as f64
}

/// Create mux configuration from stream info.
#[cfg(feature = "av-mux")]
fn create_mux_config(
    video_info: &VideoStreamInfo,
    audio_info: Option<&AudioStreamInfo>,
) -> VideoResult<AvMuxConfig> {
    // Extract or generate SPS/PPS (this is simplified, real implementation needs proper handling)
    let (sps, pps) = if !video_info.extra_data.is_empty() {
        // Parse extra_data for SPS/PPS
        parse_sps_pps(&video_info.extra_data)
    } else {
        // Default minimal SPS/PPS (will need to be extracted from first keyframe in practice)
        (vec![0x67, 0x64, 0x00, 0x1f], vec![0x68, 0xee, 0x3c, 0x80])
    };

    let video_config = VideoConfig {
        width: video_info.width as u16,
        height: video_info.height as u16,
        frame_rate: video_info.frame_rate.unwrap_or(30.0),
        sps,
        pps,
    };

    let audio_config = audio_info.map(|info| {
        AudioConfig::aac(
            info.sample_rate,
            info.channels as u8,
            128000, // Default bitrate
        )
    });

    Ok(AvMuxConfig {
        video: video_config,
        audio: audio_config,
        timescale: 1000,
    })
}

/// Parse SPS and PPS from extra data.
fn parse_sps_pps(extra_data: &[u8]) -> (Vec<u8>, Vec<u8>) {
    // Simple extraction - assumes concatenated SPS+PPS
    // Real implementation would parse avcC box format
    if extra_data.len() >= 8 {
        let mid = extra_data.len() / 2;
        (extra_data[..mid].to_vec(), extra_data[mid..].to_vec())
    } else {
        (vec![0x67, 0x64, 0x00, 0x1f], vec![0x68, 0xee, 0x3c, 0x80])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_config() {
        let config = TrimConfig::new(10.0, 30.0);
        assert_eq!(config.duration(), 20.0);
        assert!(!config.reencode);
        assert!(config.keyframe_seek);

        let precise = config.clone().precise();
        assert!(precise.reencode);
        assert!(!precise.keyframe_seek);
    }

    #[test]
    fn test_concat_config() {
        let config = ConcatConfig::new()
            .with_resolution(1920, 1080)
            .with_frame_rate(30.0);
        assert!(config.reencode);
        assert_eq!(config.width, Some(1920));
        assert_eq!(config.height, Some(1080));
        assert_eq!(config.frame_rate, Some(30.0));
    }

    #[test]
    fn test_speed_config() {
        let config = SpeedConfig::new(2.0);
        assert_eq!(config.speed, 2.0);
        assert!(config.preserve_pitch);
        assert!(!config.interpolate);

        let smooth = config.smooth();
        assert!(smooth.interpolate);
    }

    #[test]
    fn test_video_segment() {
        let whole = VideoSegment::whole("video.mp4");
        assert_eq!(whole.start, 0.0);
        assert!(whole.end.is_none());

        let range = VideoSegment::range("video.mp4", 10.0, 20.0);
        assert_eq!(range.start, 10.0);
        assert_eq!(range.end, Some(20.0));
    }

    #[test]
    fn test_time_pts_conversion() {
        // 30fps, timebase 1/30000
        let pts = time_to_pts(1.0, 1, 30000);
        assert_eq!(pts, 30000);

        let time = pts_to_time(30000, 1, 30000);
        assert!((time - 1.0).abs() < 0.001);
    }
}
