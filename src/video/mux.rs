//! Audio+Video muxing module for creating MP4 files with both tracks.
//!
//! This module provides functionality to combine encoded video and audio
//! into a single MP4 container file.
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::video::mux::{AvMuxer, AvMuxConfig, AudioConfig, VideoConfig};
//!
//! // Create muxer configuration
//! let config = AvMuxConfig {
//!     video: VideoConfig {
//!         width: 1920,
//!         height: 1080,
//!         frame_rate: 30.0,
//!         sps: sps_data,
//!         pps: pps_data,
//!     },
//!     audio: Some(AudioConfig::aac(44100, 2, 128000)),
//!     timescale: 1000,
//! };
//!
//! // Create muxer and add samples
//! let mut muxer = AvMuxer::new("output.mp4", config)?;
//! muxer.write_video_sample(&video_data, pts, is_keyframe)?;
//! muxer.write_audio_sample(&audio_data, pts)?;
//! muxer.finish()?;
//! ```

use bytes::Bytes;
use mp4::{
    AacConfig, AudioObjectType, AvcConfig, ChannelConfig, MediaConfig, Mp4Config, Mp4Sample,
    Mp4Writer, SampleFreqIndex, TrackConfig, TrackType,
};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use crate::video::{VideoError, VideoResult};

/// Video track configuration.
#[derive(Debug, Clone)]
pub struct VideoConfig {
    /// Video width in pixels.
    pub width: u16,
    /// Video height in pixels.
    pub height: u16,
    /// Frame rate (frames per second).
    pub frame_rate: f64,
    /// H.264 Sequence Parameter Set (SPS) NAL unit.
    pub sps: Vec<u8>,
    /// H.264 Picture Parameter Set (PPS) NAL unit.
    pub pps: Vec<u8>,
}

impl VideoConfig {
    /// Create a new video configuration.
    pub fn new(width: u16, height: u16, frame_rate: f64, sps: Vec<u8>, pps: Vec<u8>) -> Self {
        Self {
            width,
            height,
            frame_rate,
            sps,
            pps,
        }
    }
}

/// Audio track configuration.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Target bitrate in bits per second.
    pub bitrate: u32,
    /// Audio codec type.
    pub codec: AudioCodecType,
}

/// Supported audio codecs for muxing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodecType {
    /// AAC Low Complexity (most compatible).
    AacLc,
    /// AAC High Efficiency (better compression at low bitrates).
    AacHe,
    /// Opus codec (requires special container support).
    Opus,
}

impl AudioConfig {
    /// Create AAC-LC audio configuration.
    pub fn aac(sample_rate: u32, channels: u8, bitrate: u32) -> Self {
        Self {
            sample_rate,
            channels,
            bitrate,
            codec: AudioCodecType::AacLc,
        }
    }

    /// Create AAC-HE audio configuration.
    pub fn aac_he(sample_rate: u32, channels: u8, bitrate: u32) -> Self {
        Self {
            sample_rate,
            channels,
            bitrate,
            codec: AudioCodecType::AacHe,
        }
    }

    /// Create Opus audio configuration.
    pub fn opus(sample_rate: u32, channels: u8, bitrate: u32) -> Self {
        Self {
            sample_rate,
            channels,
            bitrate,
            codec: AudioCodecType::Opus,
        }
    }

    /// Get sample frequency index for AAC.
    fn sample_freq_index(&self) -> SampleFreqIndex {
        match self.sample_rate {
            96000 => SampleFreqIndex::Freq96000,
            88200 => SampleFreqIndex::Freq88200,
            64000 => SampleFreqIndex::Freq64000,
            48000 => SampleFreqIndex::Freq48000,
            44100 => SampleFreqIndex::Freq44100,
            32000 => SampleFreqIndex::Freq32000,
            24000 => SampleFreqIndex::Freq24000,
            22050 => SampleFreqIndex::Freq22050,
            16000 => SampleFreqIndex::Freq16000,
            12000 => SampleFreqIndex::Freq12000,
            11025 => SampleFreqIndex::Freq11025,
            8000 => SampleFreqIndex::Freq8000,
            _ => SampleFreqIndex::Freq44100, // Default
        }
    }

    /// Get channel configuration for AAC.
    fn channel_config(&self) -> ChannelConfig {
        match self.channels {
            1 => ChannelConfig::Mono,
            2 => ChannelConfig::Stereo,
            3 => ChannelConfig::Three,
            4 => ChannelConfig::Four,
            5 => ChannelConfig::Five,
            6 => ChannelConfig::FiveOne,
            8 => ChannelConfig::SevenOne,
            _ => ChannelConfig::Stereo, // Default
        }
    }

    /// Get audio object type for AAC.
    fn audio_object_type(&self) -> AudioObjectType {
        match self.codec {
            AudioCodecType::AacLc => AudioObjectType::AacLowComplexity,
            AudioCodecType::AacHe => AudioObjectType::AacLowComplexity, // HE not in mp4 crate
            AudioCodecType::Opus => AudioObjectType::AacLowComplexity, // Fallback
        }
    }
}

/// A/V muxer configuration.
#[derive(Debug, Clone)]
pub struct AvMuxConfig {
    /// Video track configuration.
    pub video: VideoConfig,
    /// Optional audio track configuration.
    pub audio: Option<AudioConfig>,
    /// Container timescale (ticks per second).
    pub timescale: u32,
}

impl AvMuxConfig {
    /// Create configuration for video-only output.
    pub fn video_only(video: VideoConfig) -> Self {
        Self {
            video,
            audio: None,
            timescale: 1000,
        }
    }

    /// Create configuration for video with audio.
    pub fn with_audio(video: VideoConfig, audio: AudioConfig) -> Self {
        Self {
            video,
            audio: Some(audio),
            timescale: 1000,
        }
    }
}

/// Audio+Video muxer for creating MP4 files.
pub struct AvMuxer<W: Write + Seek> {
    writer: Mp4Writer<W>,
    video_track_id: u32,
    audio_track_id: Option<u32>,
    video_timescale: u32,
    audio_timescale: u32,
    frame_duration: u32,
    sample_duration: u32,
    video_sample_count: u64,
    audio_sample_count: u64,
}

impl AvMuxer<BufWriter<File>> {
    /// Create a new A/V muxer writing to a file.
    pub fn new<P: AsRef<Path>>(path: P, config: AvMuxConfig) -> VideoResult<Self> {
        let file = File::create(path.as_ref()).map_err(|e| VideoError::Io {
            message: format!("Failed to create output file: {}", e),
        })?;
        let writer = BufWriter::new(file);
        Self::new_with_writer(writer, config)
    }
}

impl<W: Write + Seek> AvMuxer<W> {
    /// Create a new A/V muxer with a custom writer.
    pub fn new_with_writer(writer: W, config: AvMuxConfig) -> VideoResult<Self> {
        // Create MP4 configuration
        let mp4_config = Mp4Config {
            major_brand: "isom".parse().unwrap(),
            minor_version: 512,
            compatible_brands: vec![
                "isom".parse().unwrap(),
                "iso2".parse().unwrap(),
                "avc1".parse().unwrap(),
                "mp41".parse().unwrap(),
            ],
            timescale: config.timescale,
        };

        // Start MP4 writer
        let mut mp4_writer =
            Mp4Writer::write_start(writer, &mp4_config).map_err(|e| VideoError::Container {
                message: format!("Failed to start MP4 writer: {}", e),
            })?;

        // Calculate timing values
        let video_timescale = (config.video.frame_rate * 1000.0) as u32;
        let frame_duration = 1000; // Each frame = 1000 ticks

        // Add video track
        let video_config = TrackConfig {
            track_type: TrackType::Video,
            timescale: video_timescale,
            language: String::from("und"),
            media_conf: MediaConfig::AvcConfig(AvcConfig {
                width: config.video.width,
                height: config.video.height,
                seq_param_set: config.video.sps.clone(),
                pic_param_set: config.video.pps.clone(),
            }),
        };
        mp4_writer
            .add_track(&video_config)
            .map_err(|e| VideoError::Container {
                message: format!("Failed to add video track: {}", e),
            })?;
        let video_track_id = 1;

        // Add audio track if configured
        let (audio_track_id, audio_timescale, sample_duration) = if let Some(ref audio) = config.audio {
            let audio_timescale = audio.sample_rate;
            // For AAC, typical frame size is 1024 samples
            let samples_per_frame = 1024u32;
            let sample_duration = samples_per_frame;

            let audio_config = TrackConfig {
                track_type: TrackType::Audio,
                timescale: audio_timescale,
                language: String::from("und"),
                media_conf: MediaConfig::AacConfig(AacConfig {
                    bitrate: audio.bitrate,
                    profile: audio.audio_object_type(),
                    freq_index: audio.sample_freq_index(),
                    chan_conf: audio.channel_config(),
                }),
            };
            mp4_writer
                .add_track(&audio_config)
                .map_err(|e| VideoError::Container {
                    message: format!("Failed to add audio track: {}", e),
                })?;

            (Some(2), audio_timescale, sample_duration)
        } else {
            (None, 44100, 1024)
        };

        Ok(Self {
            writer: mp4_writer,
            video_track_id,
            audio_track_id,
            video_timescale,
            audio_timescale,
            frame_duration,
            sample_duration,
            video_sample_count: 0,
            audio_sample_count: 0,
        })
    }

    /// Write a video sample (frame).
    ///
    /// # Arguments
    /// * `data` - Encoded video data (AVCC format, length-prefixed NAL units)
    /// * `is_keyframe` - Whether this is a keyframe (I-frame/IDR)
    pub fn write_video_sample(&mut self, data: &[u8], is_keyframe: bool) -> VideoResult<()> {
        let start_time = self.video_sample_count * self.frame_duration as u64;

        let sample = Mp4Sample {
            start_time,
            duration: self.frame_duration,
            rendering_offset: 0,
            is_sync: is_keyframe,
            bytes: Bytes::copy_from_slice(data),
        };

        self.writer
            .write_sample(self.video_track_id, &sample)
            .map_err(|e| VideoError::Container {
                message: format!("Failed to write video sample: {}", e),
            })?;

        self.video_sample_count += 1;
        Ok(())
    }

    /// Write an audio sample (AAC frame).
    ///
    /// # Arguments
    /// * `data` - Encoded audio data (raw AAC frame, no ADTS header)
    pub fn write_audio_sample(&mut self, data: &[u8]) -> VideoResult<()> {
        let audio_track_id = self.audio_track_id.ok_or_else(|| VideoError::Container {
            message: "No audio track configured".to_string(),
        })?;

        let start_time = self.audio_sample_count * self.sample_duration as u64;

        let sample = Mp4Sample {
            start_time,
            duration: self.sample_duration,
            rendering_offset: 0,
            is_sync: true, // All audio frames are sync points
            bytes: Bytes::copy_from_slice(data),
        };

        self.writer
            .write_sample(audio_track_id, &sample)
            .map_err(|e| VideoError::Container {
                message: format!("Failed to write audio sample: {}", e),
            })?;

        self.audio_sample_count += 1;
        Ok(())
    }

    /// Write an audio sample with explicit timing.
    pub fn write_audio_sample_timed(
        &mut self,
        data: &[u8],
        start_time: u64,
        duration: u32,
    ) -> VideoResult<()> {
        let audio_track_id = self.audio_track_id.ok_or_else(|| VideoError::Container {
            message: "No audio track configured".to_string(),
        })?;

        let sample = Mp4Sample {
            start_time,
            duration,
            rendering_offset: 0,
            is_sync: true,
            bytes: Bytes::copy_from_slice(data),
        };

        self.writer
            .write_sample(audio_track_id, &sample)
            .map_err(|e| VideoError::Container {
                message: format!("Failed to write audio sample: {}", e),
            })?;

        self.audio_sample_count += 1;
        Ok(())
    }

    /// Get the number of video samples written.
    pub fn video_sample_count(&self) -> u64 {
        self.video_sample_count
    }

    /// Get the number of audio samples written.
    pub fn audio_sample_count(&self) -> u64 {
        self.audio_sample_count
    }

    /// Get video timescale.
    pub fn video_timescale(&self) -> u32 {
        self.video_timescale
    }

    /// Get audio timescale.
    pub fn audio_timescale(&self) -> u32 {
        self.audio_timescale
    }

    /// Finish writing and finalize the MP4 file.
    pub fn finish(mut self) -> VideoResult<()> {
        self.writer.write_end().map_err(|e| VideoError::Container {
            message: format!("Failed to finalize MP4: {}", e),
        })?;
        Ok(())
    }

    /// Get the inner writer (consumes the muxer).
    pub fn into_writer(self) -> W {
        self.writer.into_writer()
    }
}

/// Convert Annex B (start code) format to AVCC (length-prefixed) format.
///
/// H.264 encoders typically output Annex B format with start codes (0x00000001).
/// MP4 containers require AVCC format with 4-byte length prefixes.
pub fn annex_b_to_avcc(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        // Find start code (0x00000001 or 0x000001)
        let (_start_code_len, nal_start) =
            if i + 4 <= data.len() && data[i..i + 4] == [0x00, 0x00, 0x00, 0x01] {
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
        let nal_type = nal_data.first().map(|b| b & 0x1F).unwrap_or(0);

        // Skip SPS/PPS in sample data (they go in avcC configuration)
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

/// Extract SPS and PPS NAL units from H.264 Annex B stream.
pub fn extract_sps_pps(data: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annex_b_to_avcc() {
        // Test simple NAL unit with start code
        let annex_b = vec![
            0x00, 0x00, 0x00, 0x01, // Start code
            0x65, 0x01, 0x02, 0x03, // IDR frame (NAL type 5)
        ];
        let avcc = annex_b_to_avcc(&annex_b);
        assert_eq!(avcc.len(), 8); // 4 bytes length + 4 bytes data
        assert_eq!(&avcc[0..4], &[0x00, 0x00, 0x00, 0x04]); // Length = 4
        assert_eq!(&avcc[4..8], &[0x65, 0x01, 0x02, 0x03]); // Data
    }

    #[test]
    fn test_extract_sps_pps() {
        let stream = vec![
            0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x1f, // SPS
            0x00, 0x00, 0x00, 0x01, 0x68, 0xee, 0x3c, 0x80, // PPS
        ];
        let result = extract_sps_pps(&stream);
        assert!(result.is_some());
        let (sps, pps) = result.unwrap();
        assert_eq!(sps, vec![0x67, 0x64, 0x00, 0x1f]);
        assert_eq!(pps, vec![0x68, 0xee, 0x3c, 0x80]);
    }
}
