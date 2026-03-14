//! AVI container demuxer.
//!
//! This demuxer parses RIFF/AVI headers, indexes `movi` packets, and exposes
//! basic packet iteration for `video-info` and frame/audio extraction style
//! workflows.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::video::{AudioCodec, VideoCodec, VideoError};

use super::{AudioStreamInfo, ContainerType, Demuxer, Packet, VideoStreamInfo};

#[derive(Debug, Default, Clone, Copy)]
struct MainAviHeader {
    microsec_per_frame: u32,
    total_frames: u32,
    width: u32,
    height: u32,
}

#[derive(Debug, Clone, Copy)]
struct StreamHeader {
    stream_type: [u8; 4],
    handler: [u8; 4],
    scale: u32,
    rate: u32,
    length: u32,
}

#[derive(Debug, Default, Clone)]
struct AviVideoHeader {
    stream_index: u32,
    handler: [u8; 4],
    compression: [u8; 4],
    width: u32,
    height: u32,
    scale: u32,
    rate: u32,
    length: u32,
    extra_data: Vec<u8>,
}

#[derive(Debug, Default, Clone)]
struct AviAudioHeader {
    stream_index: u32,
    format_tag: u16,
    channels: u16,
    sample_rate: u32,
    avg_bytes_per_sec: u32,
    block_align: Option<u16>,
    bits_per_sample: Option<u16>,
    scale: u32,
    rate: u32,
    length: u32,
    extra_data: Vec<u8>,
}

#[derive(Debug, Clone)]
struct AviPacketIndex {
    offset: u64,
    size: u32,
    pts: i64,
    duration: i64,
    is_keyframe: bool,
    stream_index: u32,
    timestamp_secs: f64,
}

/// AVI demuxer that exposes stream metadata and indexed packet access.
pub struct AviDemuxer {
    reader: BufReader<File>,
    video_info: Option<VideoStreamInfo>,
    audio_info: Option<AudioStreamInfo>,
    video_packets: Vec<AviPacketIndex>,
    audio_packets: Vec<AviPacketIndex>,
    next_video_index: usize,
    next_audio_index: usize,
}

impl AviDemuxer {
    /// Open an AVI file and parse stream metadata.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, VideoError> {
        let file = File::open(path.as_ref()).map_err(|e| VideoError::Io {
            message: format!("Failed to open AVI file: {}", e),
        })?;
        let mut reader = BufReader::new(file);

        let (riff_end, main, video_header, audio_header) = parse_avi_headers(&mut reader)?;

        let video_info = video_header.as_ref().map(|video| {
            let width = if video.width > 0 {
                video.width
            } else {
                main.width
            };
            let height = if video.height > 0 {
                video.height
            } else {
                main.height
            };

            let frame_rate = if video.rate > 0 && video.scale > 0 {
                Some(video.rate as f64 / video.scale as f64)
            } else if main.microsec_per_frame > 0 {
                Some(1_000_000.0 / main.microsec_per_frame as f64)
            } else {
                None
            };

            let frame_count = if main.total_frames > 0 {
                Some(main.total_frames as u64)
            } else if video.length > 0 {
                Some(video.length as u64)
            } else {
                None
            };

            let duration = if let (Some(frames), Some(fps)) = (frame_count, frame_rate) {
                Some(frames as f64 / fps)
            } else if video.length > 0 && video.rate > 0 {
                Some(video.length as f64 * video.scale.max(1) as f64 / video.rate as f64)
            } else {
                None
            };

            let (timebase_num, timebase_den) = if video.scale > 0 && video.rate > 0 {
                (video.scale, video.rate)
            } else if main.microsec_per_frame > 0 {
                (main.microsec_per_frame, 1_000_000)
            } else {
                (1, 1)
            };

            VideoStreamInfo {
                codec: detect_video_codec(video.handler, video.compression),
                width,
                height,
                frame_rate,
                duration,
                frame_count,
                timebase_num,
                timebase_den,
                extra_data: video.extra_data.clone(),
            }
        });

        let audio_info = audio_header.as_ref().map(|audio| {
            let duration = if audio.rate > 0 && audio.scale > 0 && audio.length > 0 {
                Some(audio.length as f64 * audio.scale as f64 / audio.rate as f64)
            } else {
                video_info.as_ref().and_then(|v| v.duration)
            };

            AudioStreamInfo {
                codec: detect_audio_codec(audio.format_tag),
                sample_rate: audio.sample_rate,
                channels: audio.channels,
                bits_per_sample: audio.bits_per_sample,
                duration,
                extra_data: audio.extra_data.clone(),
            }
        });

        let (video_packets, audio_packets) =
            index_avi_packets(&mut reader, riff_end, video_header.as_ref(), audio_header.as_ref())?;

        Ok(Self {
            reader,
            video_info,
            audio_info,
            video_packets,
            audio_packets,
            next_video_index: 0,
            next_audio_index: 0,
        })
    }
}

impl Demuxer for AviDemuxer {
    fn container_type(&self) -> ContainerType {
        ContainerType::Avi
    }

    fn video_info(&self) -> Option<&VideoStreamInfo> {
        self.video_info.as_ref()
    }

    fn audio_info(&self) -> Option<&AudioStreamInfo> {
        self.audio_info.as_ref()
    }

    fn next_video_packet(&mut self) -> Result<Option<Packet>, VideoError> {
        let Some(index) = self.video_packets.get(self.next_video_index) else {
            return Ok(None);
        };
        self.next_video_index += 1;
        read_indexed_packet(&mut self.reader, index)
    }

    fn next_audio_packet(&mut self) -> Result<Option<Packet>, VideoError> {
        let Some(index) = self.audio_packets.get(self.next_audio_index) else {
            return Ok(None);
        };
        self.next_audio_index += 1;
        read_indexed_packet(&mut self.reader, index)
    }

    fn seek(&mut self, timestamp: f64) -> Result<(), VideoError> {
        let timestamp = timestamp.max(0.0);
        self.next_video_index = self
            .video_packets
            .iter()
            .position(|packet| packet.timestamp_secs >= timestamp)
            .unwrap_or(self.video_packets.len());
        self.next_audio_index = self
            .audio_packets
            .iter()
            .position(|packet| packet.timestamp_secs >= timestamp)
            .unwrap_or(self.audio_packets.len());
        Ok(())
    }

    fn reset(&mut self) -> Result<(), VideoError> {
        self.next_video_index = 0;
        self.next_audio_index = 0;
        Ok(())
    }
}

fn parse_avi_headers<R: Read + Seek>(
    reader: &mut R,
) -> Result<(u64, MainAviHeader, Option<AviVideoHeader>, Option<AviAudioHeader>), VideoError> {
    let mut riff_header = [0u8; 12];
    reader.read_exact(&mut riff_header).map_err(|e| VideoError::Io {
        message: format!("Failed to read AVI RIFF header: {}", e),
    })?;

    if &riff_header[0..4] != b"RIFF" || &riff_header[8..12] != b"AVI " {
        return Err(VideoError::Container {
            message: "Invalid AVI file: missing RIFF/AVI signature".to_string(),
        });
    }

    let riff_size = u32::from_le_bytes([
        riff_header[4],
        riff_header[5],
        riff_header[6],
        riff_header[7],
    ]);
    let riff_end = 8u64.checked_add(riff_size as u64).ok_or_else(|| VideoError::Container {
        message: "Invalid AVI file: RIFF size overflow".to_string(),
    })?;

    let mut main = MainAviHeader::default();
    let mut video = None;
    let mut audio = None;

    while reader.stream_position().map_err(io_err)? + 8 <= riff_end {
        let Some((chunk_id, chunk_size)) = read_chunk_header(reader)? else {
            break;
        };

        let chunk_start = reader.stream_position().map_err(io_err)?;
        let chunk_end = checked_chunk_end(chunk_start, chunk_size)?;

        if chunk_id == *b"LIST" {
            if chunk_size < 4 {
                seek_chunk_end(reader, chunk_end, chunk_size)?;
                continue;
            }

            let mut list_type = [0u8; 4];
            reader.read_exact(&mut list_type).map_err(io_err)?;

            if list_type == *b"hdrl" {
                parse_hdrl_list(reader, chunk_end, &mut main, &mut video, &mut audio)?;
            }
        }

        seek_chunk_end(reader, chunk_end, chunk_size)?;
    }

    Ok((riff_end, main, video, audio))
}

fn parse_hdrl_list<R: Read + Seek>(
    reader: &mut R,
    list_end: u64,
    main: &mut MainAviHeader,
    video: &mut Option<AviVideoHeader>,
    audio: &mut Option<AviAudioHeader>,
) -> Result<(), VideoError> {
    let mut stream_index = 0u32;

    while reader.stream_position().map_err(io_err)? + 8 <= list_end {
        let Some((chunk_id, chunk_size)) = read_chunk_header(reader)? else {
            break;
        };

        let chunk_start = reader.stream_position().map_err(io_err)?;
        let chunk_end = checked_chunk_end(chunk_start, chunk_size)?;

        match chunk_id {
            [b'a', b'v', b'i', b'h'] => {
                let data = read_chunk_data(reader, chunk_size)?;
                *main = parse_main_header(&data);
            }
            [b'L', b'I', b'S', b'T'] => {
                if chunk_size >= 4 {
                    let mut list_type = [0u8; 4];
                    reader.read_exact(&mut list_type).map_err(io_err)?;
                    if list_type == *b"strl" {
                        let (stream_video, stream_audio) = parse_stream_list(reader, chunk_end, stream_index)?;
                        if video.is_none() {
                            *video = stream_video;
                        }
                        if audio.is_none() {
                            *audio = stream_audio;
                        }
                        stream_index += 1;
                    }
                }
            }
            _ => {}
        }

        seek_chunk_end(reader, chunk_end, chunk_size)?;
    }

    Ok(())
}

fn parse_stream_list<R: Read + Seek>(
    reader: &mut R,
    list_end: u64,
    stream_index: u32,
) -> Result<(Option<AviVideoHeader>, Option<AviAudioHeader>), VideoError> {
    let mut stream_header = None;
    let mut strf_data = None;

    while reader.stream_position().map_err(io_err)? + 8 <= list_end {
        let Some((chunk_id, chunk_size)) = read_chunk_header(reader)? else {
            break;
        };

        let chunk_start = reader.stream_position().map_err(io_err)?;
        let chunk_end = checked_chunk_end(chunk_start, chunk_size)?;

        match chunk_id {
            [b's', b't', b'r', b'h'] => {
                let data = read_chunk_data(reader, chunk_size)?;
                stream_header = parse_stream_header(&data);
            }
            [b's', b't', b'r', b'f'] => {
                strf_data = Some(read_chunk_data(reader, chunk_size)?);
            }
            _ => {}
        }

        seek_chunk_end(reader, chunk_end, chunk_size)?;
    }

    let Some(header) = stream_header else {
        return Ok((None, None));
    };

    match header.stream_type {
        [b'v', b'i', b'd', b's'] => {
            let mut parsed = AviVideoHeader {
                stream_index,
                handler: header.handler,
                scale: header.scale,
                rate: header.rate,
                length: header.length,
                ..Default::default()
            };

            if let Some(data) = strf_data {
                parse_video_strf(&data, &mut parsed);
            }

            Ok((Some(parsed), None))
        }
        [b'a', b'u', b'd', b's'] => {
            let mut parsed = AviAudioHeader {
                stream_index,
                scale: header.scale,
                rate: header.rate,
                length: header.length,
                ..Default::default()
            };

            if let Some(data) = strf_data {
                parse_audio_strf(&data, &mut parsed);
            }

            Ok((None, Some(parsed)))
        }
        _ => Ok((None, None)),
    }
}

fn parse_main_header(data: &[u8]) -> MainAviHeader {
    if data.len() < 40 {
        return MainAviHeader::default();
    }

    MainAviHeader {
        microsec_per_frame: read_u32_le(data, 0).unwrap_or(0),
        total_frames: read_u32_le(data, 16).unwrap_or(0),
        width: read_u32_le(data, 32).unwrap_or(0),
        height: read_u32_le(data, 36).unwrap_or(0),
    }
}

fn parse_stream_header(data: &[u8]) -> Option<StreamHeader> {
    if data.len() < 36 {
        return None;
    }

    Some(StreamHeader {
        stream_type: [data[0], data[1], data[2], data[3]],
        handler: [data[4], data[5], data[6], data[7]],
        scale: read_u32_le(data, 20).unwrap_or(0),
        rate: read_u32_le(data, 24).unwrap_or(0),
        length: read_u32_le(data, 32).unwrap_or(0),
    })
}

fn parse_video_strf(data: &[u8], out: &mut AviVideoHeader) {
    if data.len() < 20 {
        return;
    }

    out.width = read_i32_le(data, 4).unwrap_or(0).max(0) as u32;
    out.height = read_i32_le(data, 8).unwrap_or(0).unsigned_abs();

    if data.len() >= 20 {
        out.compression = [data[16], data[17], data[18], data[19]];
    }

    if data.len() > 40 {
        out.extra_data = data[40..].to_vec();
    }
}

fn parse_audio_strf(data: &[u8], out: &mut AviAudioHeader) {
    if data.len() < 16 {
        return;
    }

    out.format_tag = read_u16_le(data, 0).unwrap_or(0);
    out.channels = read_u16_le(data, 2).unwrap_or(0);
    out.sample_rate = read_u32_le(data, 4).unwrap_or(0);
    out.avg_bytes_per_sec = read_u32_le(data, 8).unwrap_or(0);
    out.block_align = read_u16_le(data, 12).filter(|value| *value > 0);

    let bits = read_u16_le(data, 14).unwrap_or(0);
    out.bits_per_sample = if bits > 0 { Some(bits) } else { None };

    if data.len() > 18 {
        out.extra_data = data[18..].to_vec();
    }
}

fn index_avi_packets<R: Read + Seek>(
    reader: &mut R,
    riff_end: u64,
    video: Option<&AviVideoHeader>,
    audio: Option<&AviAudioHeader>,
) -> Result<(Vec<AviPacketIndex>, Vec<AviPacketIndex>), VideoError> {
    reader.seek(SeekFrom::Start(12)).map_err(io_err)?;

    let mut video_packets = Vec::new();
    let mut audio_packets = Vec::new();
    let mut video_pts = 0i64;
    let mut audio_pts = 0i64;

    while reader.stream_position().map_err(io_err)? + 8 <= riff_end {
        let Some((chunk_id, chunk_size)) = read_chunk_header(reader)? else {
            break;
        };

        let chunk_start = reader.stream_position().map_err(io_err)?;
        let chunk_end = checked_chunk_end(chunk_start, chunk_size)?;

        if chunk_id == *b"LIST" && chunk_size >= 4 {
            let mut list_type = [0u8; 4];
            reader.read_exact(&mut list_type).map_err(io_err)?;
            if list_type == *b"movi" {
                index_movi_packets(
                    reader,
                    chunk_end,
                    video,
                    audio,
                    &mut video_packets,
                    &mut audio_packets,
                    &mut video_pts,
                    &mut audio_pts,
                )?;
            }
        }

        seek_chunk_end(reader, chunk_end, chunk_size)?;
    }

    Ok((video_packets, audio_packets))
}

fn index_movi_packets<R: Read + Seek>(
    reader: &mut R,
    list_end: u64,
    video: Option<&AviVideoHeader>,
    audio: Option<&AviAudioHeader>,
    video_packets: &mut Vec<AviPacketIndex>,
    audio_packets: &mut Vec<AviPacketIndex>,
    video_pts: &mut i64,
    audio_pts: &mut i64,
) -> Result<(), VideoError> {
    while reader.stream_position().map_err(io_err)? + 8 <= list_end {
        let Some((chunk_id, chunk_size)) = read_chunk_header(reader)? else {
            break;
        };

        let chunk_start = reader.stream_position().map_err(io_err)?;
        let chunk_end = checked_chunk_end(chunk_start, chunk_size)?;

        if chunk_id == *b"LIST" && chunk_size >= 4 {
            let mut list_type = [0u8; 4];
            reader.read_exact(&mut list_type).map_err(io_err)?;
            if list_type == *b"rec " {
                index_movi_packets(
                    reader,
                    chunk_end,
                    video,
                    audio,
                    video_packets,
                    audio_packets,
                    video_pts,
                    audio_pts,
                )?;
            }
            seek_chunk_end(reader, chunk_end, chunk_size)?;
            continue;
        }

        if let Some((stream_index, kind)) = parse_media_chunk_id(chunk_id) {
            match kind {
                AviChunkKind::Video => {
                    if let Some(video_header) = video.filter(|header| header.stream_index == stream_index) {
                        let duration = i64::from(video_header.scale.max(1));
                        let timestamp_secs = if video_header.rate > 0 {
                            *video_pts as f64 / video_header.rate as f64
                        } else {
                            0.0
                        };
                        video_packets.push(AviPacketIndex {
                            offset: chunk_start,
                            size: chunk_size,
                            pts: *video_pts,
                            duration,
                            is_keyframe: video_packets.is_empty() || &chunk_id[2..4] == b"db",
                            stream_index,
                            timestamp_secs,
                        });
                        *video_pts += duration;
                    }
                }
                AviChunkKind::Audio => {
                    if let Some(audio_header) = audio.filter(|header| header.stream_index == stream_index) {
                        let duration = audio_packet_duration(chunk_size, audio_header);
                        let timestamp_secs = if audio_header.rate > 0 {
                            *audio_pts as f64 * audio_header.scale.max(1) as f64 / audio_header.rate as f64
                        } else {
                            0.0
                        };
                        audio_packets.push(AviPacketIndex {
                            offset: chunk_start,
                            size: chunk_size,
                            pts: *audio_pts,
                            duration,
                            is_keyframe: true,
                            stream_index,
                            timestamp_secs,
                        });
                        *audio_pts += duration.max(0);
                    }
                }
            }
        }

        seek_chunk_end(reader, chunk_end, chunk_size)?;
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AviChunkKind {
    Video,
    Audio,
}

fn parse_media_chunk_id(chunk_id: [u8; 4]) -> Option<(u32, AviChunkKind)> {
    let tens = (chunk_id[0] as char).to_digit(10)?;
    let ones = (chunk_id[1] as char).to_digit(10)?;
    let stream_index = tens * 10 + ones;
    let kind = match &chunk_id[2..4] {
        b"dc" | b"db" => AviChunkKind::Video,
        b"wb" => AviChunkKind::Audio,
        _ => return None,
    };
    Some((stream_index, kind))
}

fn audio_packet_duration(chunk_size: u32, header: &AviAudioHeader) -> i64 {
    if header.avg_bytes_per_sec > 0 && header.rate > 0 {
        let numerator = chunk_size as u128 * header.rate as u128;
        let denominator = header.avg_bytes_per_sec as u128 * header.scale.max(1) as u128;
        ((numerator + denominator / 2) / denominator) as i64
    } else if let Some(block_align) = header.block_align.filter(|value| *value > 0) {
        let blocks = chunk_size as u64 / block_align as u64;
        (blocks * header.scale.max(1) as u64) as i64
    } else {
        i64::from(header.scale.max(1))
    }
}

fn read_indexed_packet<R: Read + Seek>(reader: &mut R, index: &AviPacketIndex) -> Result<Option<Packet>, VideoError> {
    reader.seek(SeekFrom::Start(index.offset)).map_err(io_err)?;
    let mut data = vec![0u8; index.size as usize];
    reader.read_exact(&mut data).map_err(io_err)?;
    Ok(Some(Packet {
        data,
        pts: index.pts,
        dts: index.pts,
        duration: index.duration,
        is_keyframe: index.is_keyframe,
        stream_index: index.stream_index,
    }))
}

fn detect_video_codec(handler: [u8; 4], compression: [u8; 4]) -> VideoCodec {
    for fourcc in [handler, compression] {
        let normalized = fourcc_to_ascii(fourcc).to_uppercase();
        if normalized.is_empty() {
            continue;
        }

        match normalized.as_str() {
            "H264" | "X264" | "AVC1" | "DAVC" => return VideoCodec::H264,
            "H265" | "HEVC" | "HEV1" | "HVC1" => return VideoCodec::H265,
            "AV01" => return VideoCodec::Av1,
            "VP80" => return VideoCodec::Vp8,
            "VP90" => return VideoCodec::Vp9,
            _ => {}
        }
    }

    VideoCodec::Unknown(fourcc_to_ascii(handler))
}

fn detect_audio_codec(format_tag: u16) -> AudioCodec {
    match format_tag {
        0x0001 | 0x0003 => AudioCodec::Pcm,
        0x0050 | 0x0055 => AudioCodec::Mp3,
        0x00ff | 0x1600 => AudioCodec::Aac,
        0x674f => AudioCodec::Vorbis,
        0x704f => AudioCodec::Opus,
        _ => AudioCodec::Unknown(format!("WAVE_FORMAT_0x{format_tag:04x}")),
    }
}

fn fourcc_to_ascii(fourcc: [u8; 4]) -> String {
    let mut out = String::new();
    for &b in &fourcc {
        if b == 0 {
            continue;
        }
        if (b as char).is_ascii_alphanumeric() {
            out.push(b as char);
        }
    }
    out
}

fn read_chunk_header<R: Read>(reader: &mut R) -> Result<Option<([u8; 4], u32)>, VideoError> {
    let mut header = [0u8; 8];
    match reader.read_exact(&mut header) {
        Ok(()) => {
            let chunk_id = [header[0], header[1], header[2], header[3]];
            let chunk_size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
            Ok(Some((chunk_id, chunk_size)))
        }
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(io_err(e)),
    }
}

fn read_chunk_data<R: Read>(reader: &mut R, chunk_size: u32) -> Result<Vec<u8>, VideoError> {
    let mut data = vec![0u8; chunk_size as usize];
    reader.read_exact(&mut data).map_err(io_err)?;
    Ok(data)
}

fn checked_chunk_end(chunk_start: u64, chunk_size: u32) -> Result<u64, VideoError> {
    chunk_start
        .checked_add(chunk_size as u64)
        .ok_or_else(|| VideoError::Container {
            message: "Invalid AVI file: chunk size overflow".to_string(),
        })
}

fn seek_chunk_end<R: Seek>(reader: &mut R, chunk_end: u64, chunk_size: u32) -> Result<(), VideoError> {
    let aligned = chunk_end + (chunk_size as u64 & 1);
    reader.seek(SeekFrom::Start(aligned)).map_err(io_err)?;
    Ok(())
}

fn read_u16_le(data: &[u8], offset: usize) -> Option<u16> {
    let end = offset.checked_add(2)?;
    let bytes = data.get(offset..end)?;
    Some(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32_le(data: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let bytes = data.get(offset..end)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_i32_le(data: &[u8], offset: usize) -> Option<i32> {
    let end = offset.checked_add(4)?;
    let bytes = data.get(offset..end)?;
    Some(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn io_err(e: std::io::Error) -> VideoError {
    VideoError::Io {
        message: format!("I/O error while parsing AVI: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn parses_basic_avi_video_and_audio_metadata() {
        let avi = build_test_avi();
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(&avi).expect("write avi bytes");

        let demuxer = AviDemuxer::open(file.path()).expect("open avi");

        let video = demuxer.video_info().expect("video info");
        assert_eq!(video.width, 640);
        assert_eq!(video.height, 480);
        assert_eq!(video.codec, VideoCodec::H264);
        assert_eq!(video.frame_count, Some(2));
        assert_eq!(video.frame_rate.map(|v| v.round() as u32), Some(30));

        let audio = demuxer.audio_info().expect("audio info");
        assert_eq!(audio.codec, AudioCodec::Mp3);
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.sample_rate, 44_100);
        assert_eq!(audio.bits_per_sample, Some(16));
    }

    #[test]
    fn packet_iteration_reads_indexed_media_chunks() {
        let avi = build_test_avi();
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(&avi).expect("write avi bytes");

        let mut demuxer = AviDemuxer::open(file.path()).expect("open avi");
        let video = demuxer
            .next_video_packet()
            .expect("read first video packet")
            .expect("video packet");
        assert_eq!(video.data, vec![1, 2, 3, 4]);
        assert_eq!(video.pts, 0);
        assert!(video.is_keyframe);

        let audio = demuxer
            .next_audio_packet()
            .expect("read first audio packet")
            .expect("audio packet");
        assert_eq!(audio.data, vec![16, 17, 18, 19]);
        assert_eq!(audio.stream_index, 1);
    }

    #[test]
    fn seek_and_reset_reposition_packet_iteration() {
        let avi = build_test_avi();
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(&avi).expect("write avi bytes");

        let mut demuxer = AviDemuxer::open(file.path()).expect("open avi");
        demuxer.seek(1.0 / 60.0).expect("seek");

        let video = demuxer
            .next_video_packet()
            .expect("read second video packet")
            .expect("video packet");
        assert_eq!(video.data, vec![5, 6, 7, 8]);

        demuxer.reset().expect("reset");
        let video = demuxer
            .next_video_packet()
            .expect("read first video packet after reset")
            .expect("video packet");
        assert_eq!(video.data, vec![1, 2, 3, 4]);
    }

    fn build_test_avi() -> Vec<u8> {
        let mut avih = vec![0u8; 56];
        avih[0..4].copy_from_slice(&33_333u32.to_le_bytes()); // ~30 fps
        avih[16..20].copy_from_slice(&2u32.to_le_bytes()); // total frames
        avih[24..28].copy_from_slice(&2u32.to_le_bytes()); // streams
        avih[32..36].copy_from_slice(&640u32.to_le_bytes());
        avih[36..40].copy_from_slice(&480u32.to_le_bytes());

        let mut video_strh = vec![0u8; 56];
        video_strh[0..4].copy_from_slice(b"vids");
        video_strh[4..8].copy_from_slice(b"H264");
        video_strh[20..24].copy_from_slice(&1u32.to_le_bytes()); // scale
        video_strh[24..28].copy_from_slice(&30u32.to_le_bytes()); // rate
        video_strh[32..36].copy_from_slice(&2u32.to_le_bytes()); // length

        let mut video_strf = vec![0u8; 40];
        video_strf[0..4].copy_from_slice(&40u32.to_le_bytes()); // BITMAPINFOHEADER size
        video_strf[4..8].copy_from_slice(&(640i32).to_le_bytes());
        video_strf[8..12].copy_from_slice(&(480i32).to_le_bytes());
        video_strf[12..14].copy_from_slice(&1u16.to_le_bytes()); // planes
        video_strf[14..16].copy_from_slice(&24u16.to_le_bytes()); // bit count
        video_strf[16..20].copy_from_slice(b"H264");

        let mut audio_strh = vec![0u8; 56];
        audio_strh[0..4].copy_from_slice(b"auds");
        audio_strh[20..24].copy_from_slice(&1u32.to_le_bytes()); // scale
        audio_strh[24..28].copy_from_slice(&44_100u32.to_le_bytes()); // rate
        audio_strh[32..36].copy_from_slice(&8u32.to_le_bytes()); // length in bytes for test

        let mut audio_strf = vec![0u8; 16];
        audio_strf[0..2].copy_from_slice(&0x0055u16.to_le_bytes()); // MP3
        audio_strf[2..4].copy_from_slice(&2u16.to_le_bytes());
        audio_strf[4..8].copy_from_slice(&44_100u32.to_le_bytes());
        audio_strf[8..12].copy_from_slice(&176_400u32.to_le_bytes()); // avg bytes/sec
        audio_strf[12..14].copy_from_slice(&4u16.to_le_bytes());
        audio_strf[14..16].copy_from_slice(&16u16.to_le_bytes());

        let video_strl = list_chunk(
            b"strl",
            [
                chunk(b"strh", &video_strh),
                chunk(b"strf", &video_strf),
            ]
            .concat(),
        );

        let audio_strl = list_chunk(
            b"strl",
            [
                chunk(b"strh", &audio_strh),
                chunk(b"strf", &audio_strf),
            ]
            .concat(),
        );

        let hdrl = list_chunk(
            b"hdrl",
            [
                chunk(b"avih", &avih),
                video_strl,
                audio_strl,
            ]
            .concat(),
        );

        let movi = list_chunk(
            b"movi",
            [
                chunk(b"00dc", &[1, 2, 3, 4]),
                chunk(b"01wb", &[16, 17, 18, 19]),
                chunk(b"00dc", &[5, 6, 7, 8]),
                chunk(b"01wb", &[32, 33, 34, 35]),
            ]
            .concat(),
        );

        riff_chunk(b"AVI ", &[hdrl, movi].concat())
    }

    fn chunk(id: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + payload.len() + 1);
        out.extend_from_slice(id);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        if payload.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    fn list_chunk(list_type: &[u8; 4], contents: Vec<u8>) -> Vec<u8> {
        let mut payload = Vec::with_capacity(4 + contents.len());
        payload.extend_from_slice(list_type);
        payload.extend_from_slice(&contents);
        chunk(b"LIST", &payload)
    }

    fn riff_chunk(riff_type: &[u8; 4], contents: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + contents.len());
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(4u32 + contents.len() as u32).to_le_bytes());
        out.extend_from_slice(riff_type);
        out.extend_from_slice(contents);
        out
    }
}
