//! MKV/WebM container demuxer.
//!
//! Stream metadata comes from the `matroska` crate. Packet iteration is
//! implemented locally so MKV/WebM inputs can drive decode and transcode flows.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Duration;

use matroska::{Matroska, Settings};

use crate::video::{AudioCodec, VideoCodec, VideoError};

use super::{AudioStreamInfo, ContainerType, Demuxer, Packet, VideoStreamInfo};

const DEFAULT_TIMECODE_SCALE_NS: u64 = 1_000_000;
const ID_SEGMENT: u32 = 0x1853_8067;
const ID_INFO: u32 = 0x1549_A966;
const ID_TIMECODE_SCALE: u32 = 0x002A_D7B1;
const ID_CLUSTER: u32 = 0x1F43_B675;
const ID_CLUSTER_TIMECODE: u32 = 0xE7;
const ID_SIMPLE_BLOCK: u32 = 0xA3;
const ID_BLOCK_GROUP: u32 = 0xA0;
const ID_BLOCK: u32 = 0xA1;
const ID_BLOCK_DURATION: u32 = 0x9B;
const ID_REFERENCE_BLOCK: u32 = 0xFB;

#[derive(Debug, Clone)]
struct PacketIndex {
    offset: u64,
    size: u32,
    pts: i64,
    duration: i64,
    is_keyframe: bool,
    stream_index: u32,
    timestamp_secs: f64,
}

#[derive(Debug, Clone, Copy)]
struct SelectedTrack {
    number: u64,
    stream_index: u32,
    default_duration_ticks: Option<i64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct SelectedTracks {
    video: Option<SelectedTrack>,
    audio: Option<SelectedTrack>,
}

#[derive(Debug, Clone, Copy)]
struct ElementHeader {
    id: u32,
    data_offset: usize,
    size: u64,
    unknown_size: bool,
}

#[derive(Debug, Clone, Copy)]
struct IndexedFrame {
    offset: usize,
    size: usize,
}

/// MKV/WebM container demuxer.
pub struct MkvDemuxer {
    reader: BufReader<File>,
    video_info: Option<VideoStreamInfo>,
    audio_info: Option<AudioStreamInfo>,
    container_type: ContainerType,
    video_packets: Vec<PacketIndex>,
    audio_packets: Vec<PacketIndex>,
    next_video_index: usize,
    next_audio_index: usize,
}

impl MkvDemuxer {
    /// Open an MKV/WebM file for demuxing.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, VideoError> {
        let path = path.as_ref();
        let container_type = match path.extension().and_then(|e| e.to_str()) {
            Some("webm") => ContainerType::WebM,
            _ => ContainerType::Mkv,
        };

        let metadata_file = File::open(path).map_err(io_err)?;
        let matroska =
            Matroska::open(BufReader::new(metadata_file)).map_err(|e| VideoError::Container {
                message: format!("Failed to parse MKV header: {:?}", e),
            })?;
        let bytes = std::fs::read(path).map_err(io_err)?;
        let (segment_start, segment_end) = find_segment_range(&bytes)?;
        let timecode_scale = parse_timecode_scale(&bytes, segment_start, segment_end)?;
        let (video_info, audio_info, selected_tracks) =
            build_stream_info(&matroska, timecode_scale)?;
        let (video_packets, audio_packets) = index_packets(
            &bytes,
            segment_start,
            segment_end,
            &selected_tracks,
            timecode_scale,
        )?;
        let packet_file = File::open(path).map_err(io_err)?;

        Ok(Self {
            reader: BufReader::new(packet_file),
            video_info,
            audio_info,
            container_type,
            video_packets,
            audio_packets,
            next_video_index: 0,
            next_audio_index: 0,
        })
    }

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

    fn detect_audio_codec(codec_id: &str) -> AudioCodec {
        match codec_id {
            "A_AAC" | "A_AAC/MPEG2/MAIN" | "A_AAC/MPEG2/LC" | "A_AAC/MPEG4/MAIN"
            | "A_AAC/MPEG4/LC" => AudioCodec::Aac,
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
        read_packet(
            &mut self.reader,
            advance_index(&self.video_packets, &mut self.next_video_index),
        )
    }

    fn next_audio_packet(&mut self) -> Result<Option<Packet>, VideoError> {
        read_packet(
            &mut self.reader,
            advance_index(&self.audio_packets, &mut self.next_audio_index),
        )
    }

    fn seek(&mut self, timestamp: f64) -> Result<(), VideoError> {
        let timestamp = timestamp.max(0.0);
        self.next_video_index = seek_index(&self.video_packets, timestamp);
        self.next_audio_index = seek_index(&self.audio_packets, timestamp);
        Ok(())
    }

    fn reset(&mut self) -> Result<(), VideoError> {
        self.next_video_index = 0;
        self.next_audio_index = 0;
        Ok(())
    }
}

fn build_stream_info(
    matroska: &Matroska,
    timecode_scale: u64,
) -> Result<
    (
        Option<VideoStreamInfo>,
        Option<AudioStreamInfo>,
        SelectedTracks,
    ),
    VideoError,
> {
    let timebase_num = u32::try_from(timecode_scale).map_err(|_| VideoError::Container {
        message: format!("Matroska timecode scale {} is too large", timecode_scale),
    })?;
    let mut selected = SelectedTracks::default();
    let mut video_info = None;
    let mut audio_info = None;

    for track in &matroska.tracks {
        match &track.settings {
            Settings::Video(video) if video_info.is_none() => {
                video_info = Some(VideoStreamInfo {
                    codec: MkvDemuxer::detect_video_codec(&track.codec_id),
                    width: video.pixel_width as u32,
                    height: video.pixel_height as u32,
                    frame_rate: track.default_duration.and_then(duration_to_fps),
                    duration: matroska.info.duration.map(|d| d.as_secs_f64()),
                    frame_count: None,
                    timebase_num,
                    timebase_den: 1_000_000_000,
                    extra_data: track.codec_private.clone().unwrap_or_default(),
                });
                selected.video = Some(SelectedTrack {
                    number: track.number,
                    stream_index: track.number as u32,
                    default_duration_ticks: track
                        .default_duration
                        .and_then(|d| duration_to_ticks(d, timecode_scale)),
                });
            }
            Settings::Audio(audio) if audio_info.is_none() => {
                audio_info = Some(AudioStreamInfo {
                    codec: MkvDemuxer::detect_audio_codec(&track.codec_id),
                    sample_rate: audio.sample_rate as u32,
                    channels: audio.channels as u16,
                    bits_per_sample: audio.bit_depth.map(|b| b as u16),
                    duration: matroska.info.duration.map(|d| d.as_secs_f64()),
                    extra_data: track.codec_private.clone().unwrap_or_default(),
                });
                selected.audio = Some(SelectedTrack {
                    number: track.number,
                    stream_index: track.number as u32,
                    default_duration_ticks: track
                        .default_duration
                        .and_then(|d| duration_to_ticks(d, timecode_scale)),
                });
            }
            _ => {}
        }
    }

    Ok((video_info, audio_info, selected))
}

fn duration_to_fps(duration: Duration) -> Option<f64> {
    let seconds = duration.as_secs_f64();
    (seconds > 0.0).then_some(1.0 / seconds)
}

fn duration_to_ticks(duration: Duration, timecode_scale: u64) -> Option<i64> {
    let scale = u128::from(timecode_scale.max(1));
    let ticks = (duration.as_nanos() + scale / 2) / scale;
    i64::try_from(ticks.max(1)).ok()
}

fn index_packets(
    bytes: &[u8],
    segment_start: usize,
    segment_end: usize,
    selected: &SelectedTracks,
    timecode_scale: u64,
) -> Result<(Vec<PacketIndex>, Vec<PacketIndex>), VideoError> {
    let mut video_packets = Vec::new();
    let mut audio_packets = Vec::new();
    let mut offset = segment_start;

    while offset < segment_end {
        let Some(element) = read_element(bytes, offset)? else {
            break;
        };
        let end = element_end(element, segment_end)?;
        if element.id == ID_CLUSTER {
            index_cluster(
                bytes,
                element.data_offset,
                end,
                selected,
                timecode_scale,
                &mut video_packets,
                &mut audio_packets,
            )?;
        }
        offset = end;
    }

    finalize_durations(
        &mut video_packets,
        selected.video.and_then(|t| t.default_duration_ticks),
    );
    finalize_durations(
        &mut audio_packets,
        selected.audio.and_then(|t| t.default_duration_ticks),
    );
    Ok((video_packets, audio_packets))
}

fn index_cluster(
    bytes: &[u8],
    cluster_start: usize,
    cluster_end: usize,
    selected: &SelectedTracks,
    timecode_scale: u64,
    video_packets: &mut Vec<PacketIndex>,
    audio_packets: &mut Vec<PacketIndex>,
) -> Result<(), VideoError> {
    let mut cluster_timecode = 0i64;
    let mut offset = cluster_start;
    while offset < cluster_end {
        let Some(element) = read_element(bytes, offset)? else {
            break;
        };
        let end = element_end(element, cluster_end)?;
        match element.id {
            ID_CLUSTER_TIMECODE => {
                cluster_timecode = i64::try_from(read_uint(bytes, element.data_offset, end)?)
                    .map_err(|_| VideoError::Container {
                        message: "Cluster timecode is too large".to_string(),
                    })?;
            }
            ID_SIMPLE_BLOCK => {
                index_block(
                    bytes,
                    element.data_offset,
                    end,
                    cluster_timecode,
                    selected,
                    timecode_scale,
                    None,
                    None,
                    video_packets,
                    audio_packets,
                )?;
            }
            ID_BLOCK_GROUP => {
                index_block_group(
                    bytes,
                    element.data_offset,
                    end,
                    cluster_timecode,
                    selected,
                    timecode_scale,
                    video_packets,
                    audio_packets,
                )?;
            }
            _ => {}
        }
        offset = end;
    }
    Ok(())
}

fn index_block_group(
    bytes: &[u8],
    group_start: usize,
    group_end: usize,
    cluster_timecode: i64,
    selected: &SelectedTracks,
    timecode_scale: u64,
    video_packets: &mut Vec<PacketIndex>,
    audio_packets: &mut Vec<PacketIndex>,
) -> Result<(), VideoError> {
    let mut block_range = None;
    let mut block_duration = None;
    let mut has_reference = false;
    let mut offset = group_start;
    while offset < group_end {
        let Some(element) = read_element(bytes, offset)? else {
            break;
        };
        let end = element_end(element, group_end)?;
        match element.id {
            ID_BLOCK => block_range = Some((element.data_offset, end)),
            ID_BLOCK_DURATION => {
                block_duration = Some(
                    i64::try_from(read_uint(bytes, element.data_offset, end)?).map_err(|_| {
                        VideoError::Container {
                            message: "Block duration is too large".to_string(),
                        }
                    })?,
                );
            }
            ID_REFERENCE_BLOCK => has_reference = true,
            _ => {}
        }
        offset = end;
    }
    if let Some((start, end)) = block_range {
        index_block(
            bytes,
            start,
            end,
            cluster_timecode,
            selected,
            timecode_scale,
            block_duration,
            Some(!has_reference),
            video_packets,
            audio_packets,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn index_block(
    bytes: &[u8],
    block_start: usize,
    block_end: usize,
    cluster_timecode: i64,
    selected: &SelectedTracks,
    timecode_scale: u64,
    explicit_duration: Option<i64>,
    explicit_keyframe: Option<bool>,
    video_packets: &mut Vec<PacketIndex>,
    audio_packets: &mut Vec<PacketIndex>,
) -> Result<(), VideoError> {
    let (track_number, track_len) = read_vint_value(bytes, block_start, block_end)?;
    let header_end =
        block_start
            .checked_add(track_len + 3)
            .ok_or_else(|| VideoError::Container {
                message: "Matroska block header overflow".to_string(),
            })?;
    if header_end > block_end {
        return Err(VideoError::Container {
            message: "Invalid Matroska block header".to_string(),
        });
    }

    let relative_timecode = i64::from(i16::from_be_bytes([
        bytes[block_start + track_len],
        bytes[block_start + track_len + 1],
    ]));
    let flags = bytes[block_start + track_len + 2];
    let frames = parse_block_frames(bytes, header_end, block_end, flags)?;

    let stream = if selected.video.map(|t| t.number) == Some(track_number) {
        selected.video
    } else if selected.audio.map(|t| t.number) == Some(track_number) {
        selected.audio
    } else {
        None
    };
    let Some(stream) = stream else {
        return Ok(());
    };

    let total_duration = explicit_duration
        .or(stream.default_duration_ticks)
        .unwrap_or(0);
    let per_frame_duration = if !frames.is_empty() && total_duration > 0 {
        (total_duration / frames.len() as i64).max(1)
    } else {
        stream.default_duration_ticks.unwrap_or(0)
    };

    for (frame_index, frame) in frames.iter().enumerate() {
        let pts = if per_frame_duration > 0 {
            cluster_timecode + relative_timecode + per_frame_duration * frame_index as i64
        } else {
            cluster_timecode + relative_timecode
        };
        let packet = PacketIndex {
            offset: frame.offset as u64,
            size: frame.size as u32,
            pts,
            duration: per_frame_duration,
            is_keyframe: explicit_keyframe.unwrap_or(flags & 0x80 != 0),
            stream_index: stream.stream_index,
            timestamp_secs: pts_to_seconds(pts, timecode_scale),
        };

        if selected.video.map(|t| t.number) == Some(track_number) {
            video_packets.push(packet);
        } else {
            audio_packets.push(packet);
        }
    }

    Ok(())
}

fn parse_block_frames(
    bytes: &[u8],
    data_start: usize,
    data_end: usize,
    flags: u8,
) -> Result<Vec<IndexedFrame>, VideoError> {
    let lacing = (flags & 0x06) >> 1;
    if lacing == 0 {
        return Ok(vec![IndexedFrame {
            offset: data_start,
            size: data_end.saturating_sub(data_start),
        }]);
    }

    if data_start >= data_end {
        return Err(VideoError::Container {
            message: "Invalid Matroska lace header".to_string(),
        });
    }

    let frame_count = usize::from(bytes[data_start]) + 1;
    let lace_header_end = data_start + 1;

    match lacing {
        1 => parse_xiph_lacing(bytes, lace_header_end, data_end, frame_count),
        2 => parse_fixed_lacing(lace_header_end, data_end, frame_count),
        3 => parse_ebml_lacing(bytes, lace_header_end, data_end, frame_count),
        _ => Err(VideoError::Container {
            message: format!("Unsupported Matroska lacing mode {}", lacing),
        }),
    }
}

fn parse_xiph_lacing(
    bytes: &[u8],
    mut cursor: usize,
    data_end: usize,
    frame_count: usize,
) -> Result<Vec<IndexedFrame>, VideoError> {
    let mut sizes = Vec::with_capacity(frame_count);
    for _ in 0..frame_count.saturating_sub(1) {
        let mut size = 0usize;
        loop {
            let Some(&value) = bytes.get(cursor) else {
                return Err(VideoError::Container {
                    message: "Truncated Xiph lacing header".to_string(),
                });
            };
            cursor += 1;
            size += value as usize;
            if value != 0xFF {
                break;
            }
        }
        sizes.push(size);
    }
    let consumed = sizes.iter().sum::<usize>();
    let remaining = data_end
        .checked_sub(cursor)
        .ok_or_else(|| VideoError::Container {
            message: "Invalid Xiph lacing payload".to_string(),
        })?;
    if remaining < consumed {
        return Err(VideoError::Container {
            message: "Invalid Xiph lacing sizes".to_string(),
        });
    }
    sizes.push(remaining - consumed);
    frames_from_sizes(cursor, data_end, &sizes)
}

fn parse_fixed_lacing(
    data_start: usize,
    data_end: usize,
    frame_count: usize,
) -> Result<Vec<IndexedFrame>, VideoError> {
    let remaining = data_end
        .checked_sub(data_start)
        .ok_or_else(|| VideoError::Container {
            message: "Invalid fixed lacing payload".to_string(),
        })?;
    if frame_count == 0 || remaining % frame_count != 0 {
        return Err(VideoError::Container {
            message: "Invalid fixed lacing payload".to_string(),
        });
    }
    frames_from_sizes(
        data_start,
        data_end,
        &vec![remaining / frame_count; frame_count],
    )
}

fn parse_ebml_lacing(
    bytes: &[u8],
    mut cursor: usize,
    data_end: usize,
    frame_count: usize,
) -> Result<Vec<IndexedFrame>, VideoError> {
    let (first_size, first_len) = read_vint_value(bytes, cursor, data_end)?;
    cursor += first_len;
    let mut sizes = vec![
        i64::try_from(first_size).map_err(|_| VideoError::Container {
            message: "EBML lace frame size is too large".to_string(),
        })?,
    ];

    for _ in 0..frame_count.saturating_sub(2) {
        let (delta, delta_len) = read_signed_vint(bytes, cursor, data_end)?;
        cursor += delta_len;
        let next_size = sizes.last().copied().unwrap_or(0) + delta;
        if next_size < 0 {
            return Err(VideoError::Container {
                message: "Invalid EBML lacing delta".to_string(),
            });
        }
        sizes.push(next_size);
    }

    let consumed = sizes.iter().try_fold(0usize, |acc, size| {
        usize::try_from(*size)
            .ok()
            .and_then(|size| acc.checked_add(size))
    });
    let Some(consumed) = consumed else {
        return Err(VideoError::Container {
            message: "EBML lace frame size overflow".to_string(),
        });
    };
    let remaining = data_end
        .checked_sub(cursor)
        .ok_or_else(|| VideoError::Container {
            message: "Invalid EBML lacing payload".to_string(),
        })?;
    if remaining < consumed {
        return Err(VideoError::Container {
            message: "Invalid EBML lacing sizes".to_string(),
        });
    }
    sizes.push((remaining - consumed) as i64);

    let mut normalized = Vec::with_capacity(sizes.len());
    for size in sizes {
        normalized.push(usize::try_from(size).map_err(|_| VideoError::Container {
            message: "EBML lace frame size overflow".to_string(),
        })?);
    }
    frames_from_sizes(cursor, data_end, &normalized)
}

fn frames_from_sizes(
    data_start: usize,
    data_end: usize,
    sizes: &[usize],
) -> Result<Vec<IndexedFrame>, VideoError> {
    let mut frames = Vec::with_capacity(sizes.len());
    let mut offset = data_start;
    for &size in sizes {
        let next = offset
            .checked_add(size)
            .ok_or_else(|| VideoError::Container {
                message: "Matroska frame size overflow".to_string(),
            })?;
        if next > data_end {
            return Err(VideoError::Container {
                message: "Matroska frame exceeds block payload".to_string(),
            });
        }
        frames.push(IndexedFrame { offset, size });
        offset = next;
    }
    if offset != data_end {
        return Err(VideoError::Container {
            message: "Matroska lace sizes do not consume full payload".to_string(),
        });
    }
    Ok(frames)
}

fn finalize_durations(packets: &mut [PacketIndex], default_duration: Option<i64>) {
    for index in 0..packets.len().saturating_sub(1) {
        if packets[index].duration <= 0 {
            let delta = packets[index + 1].pts - packets[index].pts;
            if delta > 0 {
                packets[index].duration = delta;
            }
        }
    }
    if let Some(last) = packets.last_mut() {
        if last.duration <= 0 {
            last.duration = default_duration.unwrap_or(0).max(0);
        }
    }
}

fn advance_index<'a>(
    packets: &'a [PacketIndex],
    next_index: &mut usize,
) -> Option<&'a PacketIndex> {
    let packet = packets.get(*next_index)?;
    *next_index += 1;
    Some(packet)
}

fn seek_index(packets: &[PacketIndex], timestamp: f64) -> usize {
    packets
        .iter()
        .position(|packet| packet.timestamp_secs >= timestamp)
        .unwrap_or(packets.len())
}

fn read_packet<R: Read + Seek>(
    reader: &mut R,
    index: Option<&PacketIndex>,
) -> Result<Option<Packet>, VideoError> {
    let Some(index) = index else { return Ok(None) };
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

fn find_segment_range(bytes: &[u8]) -> Result<(usize, usize), VideoError> {
    let mut offset = 0usize;
    while offset < bytes.len() {
        let Some(element) = read_element(bytes, offset)? else {
            break;
        };
        let end = element_end(element, bytes.len())?;
        if element.id == ID_SEGMENT {
            return Ok((element.data_offset, end));
        }
        offset = end;
    }
    Err(VideoError::Container {
        message: "Matroska segment not found".to_string(),
    })
}

fn parse_timecode_scale(
    bytes: &[u8],
    segment_start: usize,
    segment_end: usize,
) -> Result<u64, VideoError> {
    let mut offset = segment_start;
    while offset < segment_end {
        let Some(element) = read_element(bytes, offset)? else {
            break;
        };
        let end = element_end(element, segment_end)?;
        if element.id == ID_INFO {
            let mut info_offset = element.data_offset;
            while info_offset < end {
                let Some(info_element) = read_element(bytes, info_offset)? else {
                    break;
                };
                let info_end = element_end(info_element, end)?;
                if info_element.id == ID_TIMECODE_SCALE {
                    return Ok(read_uint(bytes, info_element.data_offset, info_end)?.max(1));
                }
                info_offset = info_end;
            }
            break;
        }
        offset = end;
    }
    Ok(DEFAULT_TIMECODE_SCALE_NS)
}

fn read_element(bytes: &[u8], offset: usize) -> Result<Option<ElementHeader>, VideoError> {
    if offset >= bytes.len() {
        return Ok(None);
    }
    let (id, id_len) = read_element_id(bytes, offset)?;
    let (size, size_len, unknown_size) = read_element_size(bytes, offset + id_len)?;
    let data_offset =
        offset
            .checked_add(id_len + size_len)
            .ok_or_else(|| VideoError::Container {
                message: "Matroska element header overflow".to_string(),
            })?;
    if data_offset > bytes.len() {
        return Err(VideoError::Container {
            message: "Truncated Matroska element header".to_string(),
        });
    }
    Ok(Some(ElementHeader {
        id,
        data_offset,
        size,
        unknown_size,
    }))
}

fn element_end(element: ElementHeader, parent_end: usize) -> Result<usize, VideoError> {
    if element.unknown_size {
        if element.id == ID_SEGMENT {
            return Ok(parent_end);
        }
        return Err(VideoError::Container {
            message: format!(
                "Unsupported Matroska element with unknown size: 0x{:X}",
                element.id
            ),
        });
    }
    element
        .data_offset
        .checked_add(element.size as usize)
        .filter(|end| *end <= parent_end)
        .ok_or_else(|| VideoError::Container {
            message: format!("Invalid Matroska element size for 0x{:X}", element.id),
        })
}

fn read_element_id(bytes: &[u8], offset: usize) -> Result<(u32, usize), VideoError> {
    let len = vint_len(*bytes.get(offset).ok_or_else(|| VideoError::Container {
        message: "Unexpected end of Matroska data while reading element id".to_string(),
    })?)?;
    let end = offset + len;
    if end > bytes.len() {
        return Err(VideoError::Container {
            message: "Truncated Matroska element id".to_string(),
        });
    }
    let mut id = 0u32;
    for &byte in &bytes[offset..end] {
        id = (id << 8) | u32::from(byte);
    }
    Ok((id, len))
}

fn read_element_size(bytes: &[u8], offset: usize) -> Result<(u64, usize, bool), VideoError> {
    let len = vint_len(*bytes.get(offset).ok_or_else(|| VideoError::Container {
        message: "Unexpected end of Matroska data while reading element size".to_string(),
    })?)?;
    let end = offset + len;
    if end > bytes.len() {
        return Err(VideoError::Container {
            message: "Truncated Matroska element size".to_string(),
        });
    }
    let mut value = u64::from(bytes[offset] & ((1u8 << (8 - len)) - 1));
    for &byte in &bytes[offset + 1..end] {
        value = (value << 8) | u64::from(byte);
    }
    let unknown_size = value == (1u64 << (7 * len)) - 1;
    Ok((value, len, unknown_size))
}

fn read_vint_value(bytes: &[u8], offset: usize, limit: usize) -> Result<(u64, usize), VideoError> {
    let len = vint_len(*bytes.get(offset).ok_or_else(|| VideoError::Container {
        message: "Unexpected end of Matroska data while reading vint".to_string(),
    })?)?;
    let end = offset + len;
    if end > limit || end > bytes.len() {
        return Err(VideoError::Container {
            message: "Truncated Matroska vint".to_string(),
        });
    }
    let mut value = u64::from(bytes[offset] & ((1u8 << (8 - len)) - 1));
    for &byte in &bytes[offset + 1..end] {
        value = (value << 8) | u64::from(byte);
    }
    Ok((value, len))
}

fn read_signed_vint(bytes: &[u8], offset: usize, limit: usize) -> Result<(i64, usize), VideoError> {
    let (raw, len) = read_vint_value(bytes, offset, limit)?;
    let raw = i64::try_from(raw).map_err(|_| VideoError::Container {
        message: "Signed Matroska vint is too large".to_string(),
    })?;
    Ok((raw - ((1i64 << (7 * len - 1)) - 1), len))
}

fn vint_len(first_byte: u8) -> Result<usize, VideoError> {
    if first_byte == 0 {
        return Err(VideoError::Container {
            message: "Invalid Matroska vint prefix".to_string(),
        });
    }
    Ok(first_byte.leading_zeros() as usize + 1)
}

fn read_uint(bytes: &[u8], start: usize, end: usize) -> Result<u64, VideoError> {
    if start >= end || end > bytes.len() || end - start > 8 {
        return Err(VideoError::Container {
            message: "Invalid unsigned Matroska payload".to_string(),
        });
    }
    let mut value = 0u64;
    for &byte in &bytes[start..end] {
        value = (value << 8) | u64::from(byte);
    }
    Ok(value)
}

fn pts_to_seconds(pts: i64, timecode_scale: u64) -> f64 {
    pts as f64 * timecode_scale as f64 / 1_000_000_000.0
}

fn io_err(error: std::io::Error) -> VideoError {
    VideoError::Io {
        message: format!("I/O error while parsing MKV/WebM: {}", error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{Builder, NamedTempFile};

    const ID_SEGMENT_BYTES: &[u8] = &[0x18, 0x53, 0x80, 0x67];
    const ID_INFO_BYTES: &[u8] = &[0x15, 0x49, 0xA9, 0x66];
    const ID_TIMECODE_SCALE_BYTES: &[u8] = &[0x2A, 0xD7, 0xB1];
    const ID_MUXING_APP_BYTES: &[u8] = &[0x4D, 0x80];
    const ID_WRITING_APP_BYTES: &[u8] = &[0x57, 0x41];
    const ID_TRACKS_BYTES: &[u8] = &[0x16, 0x54, 0xAE, 0x6B];
    const ID_TRACK_ENTRY_BYTES: &[u8] = &[0xAE];
    const ID_TRACK_NUMBER_BYTES: &[u8] = &[0xD7];
    const ID_TRACK_UID_BYTES: &[u8] = &[0x73, 0xC5];
    const ID_TRACK_TYPE_BYTES: &[u8] = &[0x83];
    const ID_CODEC_ID_BYTES: &[u8] = &[0x86];
    const ID_DEFAULT_DURATION_BYTES: &[u8] = &[0x23, 0xE3, 0x83];
    const ID_VIDEO_BYTES: &[u8] = &[0xE0];
    const ID_PIXEL_WIDTH_BYTES: &[u8] = &[0xB0];
    const ID_PIXEL_HEIGHT_BYTES: &[u8] = &[0xBA];
    const ID_AUDIO_BYTES: &[u8] = &[0xE1];
    const ID_SAMPLING_FREQ_BYTES: &[u8] = &[0xB5];
    const ID_CHANNELS_BYTES: &[u8] = &[0x9F];
    const ID_CLUSTER_BYTES: &[u8] = &[0x1F, 0x43, 0xB6, 0x75];
    const ID_CLUSTER_TIMECODE_BYTES: &[u8] = &[0xE7];
    const ID_SIMPLE_BLOCK_BYTES: &[u8] = &[0xA3];
    const ID_BLOCK_GROUP_BYTES: &[u8] = &[0xA0];
    const ID_BLOCK_BYTES: &[u8] = &[0xA1];
    const ID_BLOCK_DURATION_BYTES: &[u8] = &[0x9B];

    #[test]
    fn parses_basic_mkv_metadata_and_packets() {
        let mkv = build_test_matroska();
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(&mkv).expect("write mkv bytes");

        let mut demuxer = MkvDemuxer::open(file.path()).expect("open mkv");
        let video = demuxer.video_info().expect("video info");
        assert_eq!(video.codec, VideoCodec::Vp9);
        assert_eq!(video.width, 640);
        assert_eq!(video.height, 360);
        assert_eq!(video.timebase_num, 1_000_000);
        assert_eq!(video.timebase_den, 1_000_000_000);

        let audio = demuxer.audio_info().expect("audio info");
        assert_eq!(audio.codec, AudioCodec::Opus);
        assert_eq!(audio.sample_rate, 48_000);
        assert_eq!(audio.channels, 2);

        let first_video = demuxer
            .next_video_packet()
            .expect("first packet")
            .expect("video");
        assert_eq!(first_video.data, vec![1, 2, 3, 4]);
        assert_eq!(first_video.pts, 0);
        assert!(first_video.is_keyframe);

        let first_audio = demuxer
            .next_audio_packet()
            .expect("audio packet")
            .expect("audio");
        assert_eq!(first_audio.data, vec![16, 17]);
        assert_eq!(first_audio.stream_index, 2);

        let second_audio = demuxer
            .next_audio_packet()
            .expect("audio packet")
            .expect("audio");
        assert_eq!(second_audio.data, vec![18, 19]);

        let second_video = demuxer
            .next_video_packet()
            .expect("second packet")
            .expect("video");
        assert_eq!(second_video.data, vec![5, 6, 7, 8]);
        assert_eq!(second_video.pts, 33);
        assert!(second_video.duration > 0);
    }

    #[test]
    fn supports_webm_seek_and_reset() {
        let webm = build_test_matroska();
        let mut file = Builder::new()
            .suffix(".webm")
            .tempfile()
            .expect("temp webm");
        file.write_all(&webm).expect("write webm bytes");

        let mut demuxer = MkvDemuxer::open(file.path()).expect("open webm");
        assert_eq!(demuxer.container_type(), ContainerType::WebM);

        demuxer.seek(0.02).expect("seek");
        let packet = demuxer
            .next_video_packet()
            .expect("video packet")
            .expect("video");
        assert_eq!(packet.data, vec![5, 6, 7, 8]);

        demuxer.reset().expect("reset");
        let packet = demuxer
            .next_video_packet()
            .expect("video packet")
            .expect("video");
        assert_eq!(packet.data, vec![1, 2, 3, 4]);
    }

    fn build_test_matroska() -> Vec<u8> {
        let info = element(
            ID_INFO_BYTES,
            [
                element(ID_TIMECODE_SCALE_BYTES, encode_uint(1_000_000)),
                element(ID_MUXING_APP_BYTES, b"xeno-test".to_vec()),
                element(ID_WRITING_APP_BYTES, b"xeno-test".to_vec()),
            ]
            .concat(),
        );
        let video_track = element(
            ID_TRACK_ENTRY_BYTES,
            [
                element(ID_TRACK_NUMBER_BYTES, encode_uint(1)),
                element(ID_TRACK_UID_BYTES, encode_uint(1)),
                element(ID_TRACK_TYPE_BYTES, encode_uint(1)),
                element(ID_CODEC_ID_BYTES, b"V_VP9".to_vec()),
                element(ID_DEFAULT_DURATION_BYTES, encode_uint(33_000_000)),
                element(
                    ID_VIDEO_BYTES,
                    [
                        element(ID_PIXEL_WIDTH_BYTES, encode_uint(640)),
                        element(ID_PIXEL_HEIGHT_BYTES, encode_uint(360)),
                    ]
                    .concat(),
                ),
            ]
            .concat(),
        );
        let audio_track = element(
            ID_TRACK_ENTRY_BYTES,
            [
                element(ID_TRACK_NUMBER_BYTES, encode_uint(2)),
                element(ID_TRACK_UID_BYTES, encode_uint(2)),
                element(ID_TRACK_TYPE_BYTES, encode_uint(2)),
                element(ID_CODEC_ID_BYTES, b"A_OPUS".to_vec()),
                element(
                    ID_AUDIO_BYTES,
                    [
                        element(ID_SAMPLING_FREQ_BYTES, 48_000f64.to_be_bytes().to_vec()),
                        element(ID_CHANNELS_BYTES, encode_uint(2)),
                    ]
                    .concat(),
                ),
            ]
            .concat(),
        );
        let tracks = element(ID_TRACKS_BYTES, [video_track, audio_track].concat());
        let cluster = element(
            ID_CLUSTER_BYTES,
            [
                element(ID_CLUSTER_TIMECODE_BYTES, encode_uint(0)),
                simple_block(1, 0, true, &[1, 2, 3, 4]),
                xiph_laced_simple_block(2, 0, true, &[&[16, 17], &[18, 19]]),
                block_group(1, 33, &[5, 6, 7, 8], Some(33)),
            ]
            .concat(),
        );
        element(ID_SEGMENT_BYTES, [info, tracks, cluster].concat())
    }

    fn simple_block(track: u64, timecode: i16, keyframe: bool, data: &[u8]) -> Vec<u8> {
        let mut payload = encode_vint(track);
        payload.extend_from_slice(&timecode.to_be_bytes());
        payload.push(if keyframe { 0x80 } else { 0x00 });
        payload.extend_from_slice(data);
        element(ID_SIMPLE_BLOCK_BYTES, payload)
    }

    fn xiph_laced_simple_block(
        track: u64,
        timecode: i16,
        keyframe: bool,
        frames: &[&[u8]],
    ) -> Vec<u8> {
        let mut payload = encode_vint(track);
        payload.extend_from_slice(&timecode.to_be_bytes());
        payload.push((if keyframe { 0x80 } else { 0x00 }) | 0x02);
        payload.push((frames.len() - 1) as u8);
        for frame in &frames[..frames.len() - 1] {
            let mut remaining = frame.len();
            while remaining >= 0xFF {
                payload.push(0xFF);
                remaining -= 0xFF;
            }
            payload.push(remaining as u8);
        }
        for frame in frames {
            payload.extend_from_slice(frame);
        }
        element(ID_SIMPLE_BLOCK_BYTES, payload)
    }

    fn block_group(track: u64, timecode: i16, data: &[u8], duration: Option<u64>) -> Vec<u8> {
        let mut payload = encode_vint(track);
        payload.extend_from_slice(&timecode.to_be_bytes());
        payload.push(0x00);
        payload.extend_from_slice(data);
        let mut children = vec![element(ID_BLOCK_BYTES, payload)];
        if let Some(duration) = duration {
            children.push(element(ID_BLOCK_DURATION_BYTES, encode_uint(duration)));
        }
        element(ID_BLOCK_GROUP_BYTES, children.concat())
    }

    fn element(id: &[u8], payload: Vec<u8>) -> Vec<u8> {
        let mut out = Vec::with_capacity(id.len() + payload.len() + 8);
        out.extend_from_slice(id);
        out.extend_from_slice(&encode_size(payload.len()));
        out.extend_from_slice(&payload);
        out
    }

    fn encode_uint(value: u64) -> Vec<u8> {
        let bytes = value.to_be_bytes();
        let first = bytes
            .iter()
            .position(|byte| *byte != 0)
            .unwrap_or(bytes.len() - 1);
        bytes[first..].to_vec()
    }

    fn encode_size(size: usize) -> Vec<u8> {
        for len in 1..=8 {
            let max = (1usize << (7 * len)) - 1;
            if size < max {
                let mut out = vec![0u8; len];
                let mut value = size;
                for index in (0..len).rev() {
                    out[index] = (value & 0xFF) as u8;
                    value >>= 8;
                }
                out[0] |= 1 << (8 - len);
                return out;
            }
        }
        panic!("size too large");
    }

    fn encode_vint(value: u64) -> Vec<u8> {
        for len in 1..=8 {
            let max = (1u64 << (7 * len)) - 1;
            if value < max {
                let mut out = vec![0u8; len];
                let mut remaining = value;
                for index in (0..len).rev() {
                    out[index] = (remaining & 0xFF) as u8;
                    remaining >>= 8;
                }
                out[0] |= 1 << (8 - len);
                return out;
            }
        }
        panic!("value too large");
    }
}
