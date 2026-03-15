//! MP4/M4V/MOV container demuxer.
//!
//! Uses the `mp4` crate for parsing ISOBMFF (ISO Base Media File Format) containers.
//! Supports video tracks with H.264, H.265, VP9 codecs.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use mp4::{MediaType, Mp4Reader, TrackType};

use crate::video::{AudioCodec, VideoCodec, VideoError};

use super::{AudioStreamInfo, ContainerType, Demuxer, Packet, VideoStreamInfo};

/// MP4 container demuxer.
pub struct Mp4Demuxer {
    reader: Mp4Reader<BufReader<File>>,
    video_track_id: Option<u32>,
    audio_track_id: Option<u32>,
    video_info: Option<VideoStreamInfo>,
    audio_info: Option<AudioStreamInfo>,
    h264_nal_length_size: Option<usize>,
    video_sample_idx: u32,
    audio_sample_idx: u32,
    video_sample_count: u32,
    audio_sample_count: u32,
}

fn append_annex_b_nal(buffer: &mut Vec<u8>, nal: &[u8]) {
    buffer.extend_from_slice(&[0, 0, 0, 1]);
    buffer.extend_from_slice(nal);
}

fn build_h264_parameter_sets(track: &mp4::Mp4Track) -> Vec<u8> {
    let mut data = Vec::new();

    if let Ok(sps) = track.sequence_parameter_set() {
        append_annex_b_nal(&mut data, sps);
    }
    if let Ok(pps) = track.picture_parameter_set() {
        append_annex_b_nal(&mut data, pps);
    }

    data
}

fn avcc_sample_to_annex_b(
    sample: &[u8],
    nal_length_size: usize,
    parameter_sets: Option<&[u8]>,
) -> Result<Vec<u8>, VideoError> {
    if sample.is_empty() {
        return Ok(Vec::new());
    }

    if sample.starts_with(&[0, 0, 1]) || sample.starts_with(&[0, 0, 0, 1]) {
        return Ok(sample.to_vec());
    }

    if !(1..=4).contains(&nal_length_size) {
        return Err(VideoError::Container {
            message: format!("Unsupported H.264 NAL length size: {}", nal_length_size),
        });
    }

    let mut offset = 0usize;
    let mut converted =
        Vec::with_capacity(sample.len() + parameter_sets.map_or(0, |data| data.len()));

    if let Some(parameter_sets) = parameter_sets {
        converted.extend_from_slice(parameter_sets);
    }

    while offset + nal_length_size <= sample.len() {
        let nal_len = match nal_length_size {
            1 => sample[offset] as usize,
            2 => u16::from_be_bytes([sample[offset], sample[offset + 1]]) as usize,
            3 => {
                ((sample[offset] as usize) << 16)
                    | ((sample[offset + 1] as usize) << 8)
                    | sample[offset + 2] as usize
            }
            4 => u32::from_be_bytes([
                sample[offset],
                sample[offset + 1],
                sample[offset + 2],
                sample[offset + 3],
            ]) as usize,
            _ => unreachable!(),
        };
        offset += nal_length_size;

        if nal_len == 0 {
            continue;
        }
        if offset + nal_len > sample.len() {
            return Err(VideoError::Container {
                message: format!(
                    "Invalid H.264 sample: NAL unit length {} exceeds remaining bytes {}",
                    nal_len,
                    sample.len().saturating_sub(offset)
                ),
            });
        }

        append_annex_b_nal(&mut converted, &sample[offset..offset + nal_len]);
        offset += nal_len;
    }

    if offset != sample.len() {
        return Err(VideoError::Container {
            message: format!(
                "Invalid H.264 sample: trailing {} bytes after AVCC conversion",
                sample.len() - offset
            ),
        });
    }

    Ok(converted)
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
        let mut h264_nal_length_size = None;
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
                                if let Some(avc1) = track.trak.mdia.minf.stbl.stsd.avc1.as_ref() {
                                    h264_nal_length_size =
                                        Some((avc1.avcc.length_size_minus_one as usize & 0x3) + 1);
                                }
                                build_h264_parameter_sets(track)
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
                        let sample_rate = track
                            .sample_freq_index()
                            .ok()
                            .map(|idx| idx.freq())
                            .unwrap_or(timescale);

                        let channels = track.channel_config().ok().map(|c| c as u16).unwrap_or(2);

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
            h264_nal_length_size,
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

        let sample = self
            .reader
            .read_sample(track_id, self.video_sample_idx)
            .map_err(|e| VideoError::Container {
                message: format!("Failed to read video sample: {}", e),
            })?;

        let sample = match sample {
            Some(s) => s,
            None => return Ok(None),
        };

        self.video_sample_idx += 1;

        let codec = self.video_info.as_ref().map(|info| info.codec.clone());
        let data = if codec == Some(VideoCodec::H264) {
            let parameter_sets = if sample.is_sync {
                self.video_info
                    .as_ref()
                    .map(|info| info.extra_data.as_slice())
                    .filter(|data| !data.is_empty())
            } else {
                None
            };
            avcc_sample_to_annex_b(
                sample.bytes.as_ref(),
                self.h264_nal_length_size.unwrap_or(4),
                parameter_sets,
            )?
        } else {
            sample.bytes.to_vec()
        };

        Ok(Some(Packet {
            data,
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

        let sample = self
            .reader
            .read_sample(track_id, self.audio_sample_idx)
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

#[cfg(test)]
mod tests {
    use super::{append_annex_b_nal, avcc_sample_to_annex_b};

    #[test]
    fn avcc_h264_sample_is_converted_to_annex_b() {
        let sample = [
            0x00, 0x00, 0x00, 0x02, 0x67, 0x42, 0x00, 0x00, 0x00, 0x03, 0x68, 0xce, 0x06,
        ];

        let converted = avcc_sample_to_annex_b(&sample, 4, None).unwrap();
        assert_eq!(
            converted,
            vec![0, 0, 0, 1, 0x67, 0x42, 0, 0, 0, 1, 0x68, 0xce, 0x06]
        );
    }

    #[test]
    fn avcc_h264_keyframe_prepends_parameter_sets() {
        let mut parameter_sets = Vec::new();
        append_annex_b_nal(&mut parameter_sets, &[0x67, 0x64, 0x00, 0x1f]);
        append_annex_b_nal(&mut parameter_sets, &[0x68, 0xeb, 0xef, 0x20]);

        let sample = [0x00, 0x00, 0x00, 0x02, 0x65, 0x88];
        let converted = avcc_sample_to_annex_b(&sample, 4, Some(&parameter_sets)).unwrap();

        assert_eq!(
            converted,
            vec![
                0, 0, 0, 1, 0x67, 0x64, 0x00, 0x1f, 0, 0, 0, 1, 0x68, 0xeb, 0xef, 0x20, 0, 0, 0, 1,
                0x65, 0x88
            ]
        );
    }

    #[test]
    fn avcc_h264_conversion_rejects_truncated_sample() {
        let sample = [0x00, 0x00, 0x00, 0x05, 0x65, 0x88];
        assert!(avcc_sample_to_annex_b(&sample, 4, None).is_err());
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
