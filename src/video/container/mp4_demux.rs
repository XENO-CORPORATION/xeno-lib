//! MP4/M4V/MOV container demuxer.
//!
//! Uses the `mp4` crate for parsing ISOBMFF (ISO Base Media File Format) containers.
//! Supports video tracks with H.264, H.265, VP9 codecs.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use mp4::{Mp4Reader, TrackType, MediaType};

use crate::video::{AudioCodec, VideoCodec, VideoError};

use super::{AudioStreamInfo, ContainerType, Demuxer, Packet, VideoStreamInfo};

/// MP4 container demuxer.
pub struct Mp4Demuxer {
    reader: Mp4Reader<BufReader<File>>,
    video_track_id: Option<u32>,
    audio_track_id: Option<u32>,
    video_info: Option<VideoStreamInfo>,
    audio_info: Option<AudioStreamInfo>,
    video_sample_idx: u32,
    audio_sample_idx: u32,
    video_sample_count: u32,
    audio_sample_count: u32,
}

impl Mp4Demuxer {
    /// Open an MP4 file for demuxing.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, VideoError> {
        let file = File::open(path.as_ref()).map_err(|e| VideoError::Io {
            message: format!("Failed to open MP4 file: {}", e),
        })?;
        let size = file.metadata().map(|m| m.len()).unwrap_or(0);
        let reader = BufReader::new(file);

        let mp4 = Mp4Reader::read_header(reader, size).map_err(|e| VideoError::Container {
            message: format!("Failed to parse MP4 header: {}", e),
        })?;

        let mut video_track_id = None;
        let mut audio_track_id = None;
        let mut video_info = None;
        let mut audio_info = None;
        let mut video_sample_count = 0u32;
        let mut audio_sample_count = 0u32;

        // Find video and audio tracks
        for (track_id, track) in mp4.tracks().iter() {
            match track.track_type() {
                Ok(TrackType::Video) => {
                    if video_track_id.is_none() {
                        video_track_id = Some(*track_id);
                        video_sample_count = track.sample_count();

                        // Detect codec from media type
                        let codec = match track.media_type() {
                            Ok(MediaType::H264) => VideoCodec::H264,
                            Ok(MediaType::H265) => VideoCodec::H265,
                            Ok(MediaType::VP9) => VideoCodec::Vp9,
                            _ => VideoCodec::Unknown("unknown".to_string()),
                        };

                        // Calculate frame rate and duration
                        let timescale = track.timescale();
                        let duration_secs = track.duration().as_secs_f64();
                        let duration = if duration_secs > 0.0 {
                            Some(duration_secs)
                        } else {
                            None
                        };

                        let frame_rate = duration.map(|d| {
                            if d > 0.0 {
                                video_sample_count as f64 / d
                            } else {
                                30.0 // Default
                            }
                        });

                        // Get codec extra data (SPS/PPS for H.264)
                        let extra_data = match track.media_type() {
                            Ok(MediaType::H264) => {
                                // Combine SPS and PPS with length prefixes
                                let mut data = Vec::new();
                                if let Ok(sps) = track.sequence_parameter_set() {
                                    data.extend_from_slice(sps);
                                }
                                if let Ok(pps) = track.picture_parameter_set() {
                                    data.extend_from_slice(pps);
                                }
                                data
                            }
                            _ => Vec::new(),
                        };

                        video_info = Some(VideoStreamInfo {
                            codec,
                            width: track.width() as u32,
                            height: track.height() as u32,
                            frame_rate,
                            duration,
                            frame_count: Some(video_sample_count as u64),
                            timebase_num: 1,
                            timebase_den: timescale,
                            extra_data,
                        });
                    }
                }
                Ok(TrackType::Audio) => {
                    if audio_track_id.is_none() {
                        audio_track_id = Some(*track_id);
                        audio_sample_count = track.sample_count();

                        // Detect audio codec
                        let codec = match track.media_type() {
                            Ok(MediaType::AAC) => AudioCodec::Aac,
                            _ => AudioCodec::Unknown("unknown".to_string()),
                        };

                        let timescale = track.timescale();
                        let duration_secs = track.duration().as_secs_f64();
                        let duration = if duration_secs > 0.0 {
                            Some(duration_secs)
                        } else {
                            None
                        };

                        // Get audio configuration
                        let sample_rate = track.sample_freq_index()
                            .ok()
                            .map(|idx| idx.freq())
                            .unwrap_or(timescale);

                        let channels = track.channel_config()
                            .ok()
                            .map(|c| c as u16)
                            .unwrap_or(2);

                        audio_info = Some(AudioStreamInfo {
                            codec,
                            sample_rate,
                            channels,
                            bits_per_sample: Some(16), // Most common
                            duration,
                            extra_data: Vec::new(),
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(Self {
            reader: mp4,
            video_track_id,
            audio_track_id,
            video_info,
            audio_info,
            video_sample_idx: 1, // MP4 samples are 1-indexed
            audio_sample_idx: 1,
            video_sample_count,
            audio_sample_count,
        })
    }

    /// Read the next video sample.
    fn read_video_sample(&mut self) -> Result<Option<Packet>, VideoError> {
        let track_id = match self.video_track_id {
            Some(id) => id,
            None => return Ok(None),
        };

        if self.video_sample_idx > self.video_sample_count {
            return Ok(None);
        }

        let sample = self.reader.read_sample(track_id, self.video_sample_idx)
            .map_err(|e| VideoError::Container {
                message: format!("Failed to read video sample: {}", e),
            })?;

        let sample = match sample {
            Some(s) => s,
            None => return Ok(None),
        };

        self.video_sample_idx += 1;

        Ok(Some(Packet {
            data: sample.bytes.to_vec(),
            pts: sample.start_time as i64,
            dts: sample.start_time as i64, // MP4 doesn't expose DTS separately in this API
            duration: sample.duration as i64,
            is_keyframe: sample.is_sync,
            stream_index: 0,
        }))
    }

    /// Read the next audio sample.
    fn read_audio_sample(&mut self) -> Result<Option<Packet>, VideoError> {
        let track_id = match self.audio_track_id {
            Some(id) => id,
            None => return Ok(None),
        };

        if self.audio_sample_idx > self.audio_sample_count {
            return Ok(None);
        }

        let sample = self.reader.read_sample(track_id, self.audio_sample_idx)
            .map_err(|e| VideoError::Container {
                message: format!("Failed to read audio sample: {}", e),
            })?;

        let sample = match sample {
            Some(s) => s,
            None => return Ok(None),
        };

        self.audio_sample_idx += 1;

        Ok(Some(Packet {
            data: sample.bytes.to_vec(),
            pts: sample.start_time as i64,
            dts: sample.start_time as i64,
            duration: sample.duration as i64,
            is_keyframe: sample.is_sync,
            stream_index: 1,
        }))
    }
}

impl Demuxer for Mp4Demuxer {
    fn container_type(&self) -> ContainerType {
        ContainerType::Mp4
    }

    fn video_info(&self) -> Option<&VideoStreamInfo> {
        self.video_info.as_ref()
    }

    fn audio_info(&self) -> Option<&AudioStreamInfo> {
        self.audio_info.as_ref()
    }

    fn next_video_packet(&mut self) -> Result<Option<Packet>, VideoError> {
        self.read_video_sample()
    }

    fn next_audio_packet(&mut self) -> Result<Option<Packet>, VideoError> {
        self.read_audio_sample()
    }

    fn seek(&mut self, timestamp: f64) -> Result<(), VideoError> {
        // Find the sample index closest to the target timestamp
        if let Some(ref info) = self.video_info {
            if info.timebase_den > 0 {
                // Reset to beginning first
                self.video_sample_idx = 1;
                self.audio_sample_idx = 1;

                // Skip to approximately the right position
                if let Some(frame_rate) = info.frame_rate {
                    let approx_frame = (timestamp * frame_rate) as u32;
                    self.video_sample_idx = approx_frame.max(1).min(self.video_sample_count);
                }
            }
        }

        Ok(())
    }

    fn reset(&mut self) -> Result<(), VideoError> {
        self.video_sample_idx = 1;
        self.audio_sample_idx = 1;
        Ok(())
    }
}
