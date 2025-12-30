//! MKV/WebM container demuxer.
//!
//! Uses the `matroska` crate for parsing Matroska containers.
//! WebM is a subset of Matroska, so this demuxer handles both.
//!
//! Note: The matroska crate is primarily designed for metadata extraction.
//! Frame-by-frame iteration is limited. For full video decoding workflows,
//! consider using IVF or MP4 containers, or FFmpeg-based demuxers.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use matroska::{Matroska, Settings};

use crate::video::{AudioCodec, VideoCodec, VideoError};

use super::{AudioStreamInfo, ContainerType, Demuxer, Packet, VideoStreamInfo};

/// MKV/WebM container demuxer.
///
/// Note: This demuxer can read MKV metadata and track information,
/// but frame iteration is limited by the matroska crate's API.
pub struct MkvDemuxer {
    video_info: Option<VideoStreamInfo>,
    audio_info: Option<AudioStreamInfo>,
    container_type: ContainerType,
}

impl MkvDemuxer {
    /// Open an MKV/WebM file for demuxing.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, VideoError> {
        let path = path.as_ref();

        // Detect container type from extension
        let container_type = match path.extension().and_then(|e| e.to_str()) {
            Some("webm") => ContainerType::WebM,
            _ => ContainerType::Mkv,
        };

        let file = File::open(path).map_err(|e| VideoError::Io {
            message: format!("Failed to open MKV file: {}", e),
        })?;
        let reader = BufReader::new(file);

        let matroska = Matroska::open(reader).map_err(|e| VideoError::Container {
            message: format!("Failed to parse MKV header: {:?}", e),
        })?;

        let mut video_info = None;
        let mut audio_info = None;

        // Parse tracks - tracks is a public field in matroska crate
        for track in &matroska.tracks {
            match &track.settings {
                Settings::Video(video) => {
                    if video_info.is_none() {
                        // Detect codec from codec ID
                        let codec = Self::detect_video_codec(&track.codec_id);

                        // Get duration from segment info
                        let duration = matroska.info.duration.map(|d| d.as_secs_f64());

                        // Frame rate isn't directly available in matroska
                        let frame_rate = track.default_duration
                            .map(|d| 1_000_000_000.0 / d.as_nanos() as f64);

                        video_info = Some(VideoStreamInfo {
                            codec,
                            width: video.pixel_width as u32,
                            height: video.pixel_height as u32,
                            frame_rate,
                            duration,
                            frame_count: None, // MKV doesn't store frame count
                            timebase_num: 1,
                            timebase_den: 1_000_000_000, // Nanoseconds
                            extra_data: track.codec_private.clone().unwrap_or_default(),
                        });
                    }
                }
                Settings::Audio(audio) => {
                    if audio_info.is_none() {
                        let codec = Self::detect_audio_codec(&track.codec_id);

                        let duration = matroska.info.duration.map(|d| d.as_secs_f64());

                        audio_info = Some(AudioStreamInfo {
                            codec,
                            sample_rate: audio.sample_rate as u32,
                            channels: audio.channels as u16,
                            bits_per_sample: audio.bit_depth.map(|b| b as u16),
                            duration,
                            extra_data: track.codec_private.clone().unwrap_or_default(),
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(Self {
            video_info,
            audio_info,
            container_type,
        })
    }

    /// Detect video codec from MKV codec ID string.
    fn detect_video_codec(codec_id: &str) -> VideoCodec {
        match codec_id {
            "V_MPEG4/ISO/AVC" => VideoCodec::H264,
            "V_MPEGH/ISO/HEVC" => VideoCodec::H265,
            "V_VP8" => VideoCodec::Vp8,
            "V_VP9" => VideoCodec::Vp9,
            "V_AV1" => VideoCodec::Av1,
            _ => VideoCodec::Unknown(codec_id.to_string()),
        }
    }

    /// Detect audio codec from MKV codec ID string.
    fn detect_audio_codec(codec_id: &str) -> AudioCodec {
        match codec_id {
            "A_AAC" | "A_AAC/MPEG2/MAIN" | "A_AAC/MPEG2/LC" | "A_AAC/MPEG4/MAIN" | "A_AAC/MPEG4/LC" => AudioCodec::Aac,
            "A_OPUS" => AudioCodec::Opus,
            "A_VORBIS" => AudioCodec::Vorbis,
            "A_FLAC" => AudioCodec::Flac,
            "A_ALAC" => AudioCodec::Alac,
            "A_MPEG/L3" => AudioCodec::Mp3,
            "A_PCM/INT/LIT" | "A_PCM/INT/BIG" | "A_PCM/FLOAT/IEEE" => AudioCodec::Pcm,
            _ => AudioCodec::Unknown(codec_id.to_string()),
        }
    }
}

impl Demuxer for MkvDemuxer {
    fn container_type(&self) -> ContainerType {
        self.container_type
    }

    fn video_info(&self) -> Option<&VideoStreamInfo> {
        self.video_info.as_ref()
    }

    fn audio_info(&self) -> Option<&AudioStreamInfo> {
        self.audio_info.as_ref()
    }

    fn next_video_packet(&mut self) -> Result<Option<Packet>, VideoError> {
        // Note: The matroska crate doesn't provide easy sequential frame access.
        // It's primarily designed for metadata extraction.
        // For full video decoding, consider using IVF, MP4, or FFmpeg-based demuxers.
        Err(VideoError::Container {
            message: "MKV frame iteration not supported. Use IVF or MP4 for video decoding.".to_string(),
        })
    }

    fn next_audio_packet(&mut self) -> Result<Option<Packet>, VideoError> {
        Err(VideoError::Container {
            message: "MKV audio iteration not supported.".to_string(),
        })
    }

    fn seek(&mut self, _timestamp: f64) -> Result<(), VideoError> {
        Err(VideoError::Container {
            message: "MKV seeking not supported.".to_string(),
        })
    }

    fn reset(&mut self) -> Result<(), VideoError> {
        // Nothing to reset since we don't track position
        Ok(())
    }
}
