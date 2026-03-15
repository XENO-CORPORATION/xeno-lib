//! IVF container demuxer.
//!
//! IVF (Indeo Video Format) is a simple container for VP8/VP9/AV1 raw bitstreams.
//! It's commonly used for testing and as an intermediate format.
//!
//! # Format Structure
//!
//! - 32-byte file header: signature, version, codec FourCC, dimensions, frame rate, frame count
//! - Frame headers (12 bytes each): frame size, timestamp
//! - Frame data (raw codec bitstream)

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::video::{VideoCodec, VideoError};

use super::{AudioStreamInfo, ContainerType, Demuxer, Packet, VideoStreamInfo};

/// IVF file header (32 bytes).
#[derive(Debug, Clone)]
struct IvfHeader {
    /// FourCC codec identifier
    fourcc: [u8; 4],
    /// Video width
    width: u16,
    /// Video height
    height: u16,
    /// Frame rate numerator
    frame_rate_num: u32,
    /// Frame rate denominator
    frame_rate_den: u32,
    /// Total number of frames
    frame_count: u32,
}

impl IvfHeader {
    fn parse(data: &[u8; 32]) -> Result<Self, VideoError> {
        // Check signature "DKIF"
        if &data[0..4] != b"DKIF" {
            return Err(VideoError::Container {
                message: "Invalid IVF file: bad signature (expected DKIF)".to_string(),
            });
        }

        // Check version (should be 0)
        let version = u16::from_le_bytes([data[4], data[5]]);
        if version != 0 {
            return Err(VideoError::Container {
                message: format!("Unsupported IVF version: {}", version),
            });
        }

        Ok(Self {
            fourcc: [data[8], data[9], data[10], data[11]],
            width: u16::from_le_bytes([data[12], data[13]]),
            height: u16::from_le_bytes([data[14], data[15]]),
            frame_rate_num: u32::from_le_bytes([data[16], data[17], data[18], data[19]]),
            frame_rate_den: u32::from_le_bytes([data[20], data[21], data[22], data[23]]),
            frame_count: u32::from_le_bytes([data[24], data[25], data[26], data[27]]),
        })
    }

    fn codec(&self) -> VideoCodec {
        match &self.fourcc {
            b"AV01" | b"av01" => VideoCodec::Av1,
            b"VP90" | b"vp90" => VideoCodec::Vp9,
            b"VP80" | b"vp80" => VideoCodec::Vp8,
            _ => VideoCodec::Unknown(String::from_utf8_lossy(&self.fourcc).to_string()),
        }
    }
}

/// IVF container demuxer.
pub struct IvfDemuxer {
    reader: BufReader<File>,
    header: IvfHeader,
    video_info: VideoStreamInfo,
    current_frame: u32,
    file_start_pos: u64,
}

impl IvfDemuxer {
    /// Open an IVF file for demuxing.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, VideoError> {
        let file = File::open(path.as_ref()).map_err(|e| VideoError::Io {
            message: format!("Failed to open IVF file: {}", e),
        })?;
        let mut reader = BufReader::new(file);

        // Read and parse header
        let mut header_data = [0u8; 32];
        reader
            .read_exact(&mut header_data)
            .map_err(|e| VideoError::Io {
                message: format!("Failed to read IVF header: {}", e),
            })?;

        let header = IvfHeader::parse(&header_data)?;

        // Calculate frame rate
        let frame_rate = if header.frame_rate_den > 0 {
            Some(header.frame_rate_num as f64 / header.frame_rate_den as f64)
        } else {
            None
        };

        // Calculate duration
        let duration = frame_rate.map(|fps| header.frame_count as f64 / fps);

        let video_info = VideoStreamInfo {
            codec: header.codec(),
            width: header.width as u32,
            height: header.height as u32,
            frame_rate,
            duration,
            frame_count: Some(header.frame_count as u64),
            timebase_num: header.frame_rate_den,
            timebase_den: header.frame_rate_num,
            extra_data: Vec::new(),
        };

        Ok(Self {
            reader,
            header,
            video_info,
            current_frame: 0,
            file_start_pos: 32, // After header
        })
    }

    /// Read the next frame from the IVF file.
    fn read_frame(&mut self) -> Result<Option<Packet>, VideoError> {
        if self.current_frame >= self.header.frame_count {
            return Ok(None);
        }

        // Read frame header (12 bytes)
        let mut frame_header = [0u8; 12];
        match self.reader.read_exact(&mut frame_header) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => {
                return Err(VideoError::Io {
                    message: format!("Failed to read frame header: {}", e),
                })
            }
        }

        let frame_size = u32::from_le_bytes([
            frame_header[0],
            frame_header[1],
            frame_header[2],
            frame_header[3],
        ]) as usize;

        let timestamp = u64::from_le_bytes([
            frame_header[4],
            frame_header[5],
            frame_header[6],
            frame_header[7],
            frame_header[8],
            frame_header[9],
            frame_header[10],
            frame_header[11],
        ]);

        // Read frame data
        let mut data = vec![0u8; frame_size];
        self.reader
            .read_exact(&mut data)
            .map_err(|e| VideoError::Io {
                message: format!("Failed to read frame data: {}", e),
            })?;

        // Check if this is a keyframe (first byte analysis for AV1/VP9)
        let is_keyframe = self.detect_keyframe(&data);

        self.current_frame += 1;

        Ok(Some(Packet {
            data,
            pts: timestamp as i64,
            dts: timestamp as i64, // IVF doesn't have B-frames reordering
            duration: 1,           // One frame duration in timebase units
            is_keyframe,
            stream_index: 0,
        }))
    }

    /// Detect if a frame is a keyframe based on codec-specific markers.
    fn detect_keyframe(&self, data: &[u8]) -> bool {
        if data.is_empty() {
            return false;
        }

        match self.header.codec() {
            VideoCodec::Av1 => {
                // Parse AV1 OBU frame headers and inspect frame_type.
                // If parsing fails, fall back to first-frame heuristic.
                detect_av1_keyframe(data).unwrap_or(self.current_frame == 0)
            }
            VideoCodec::Vp9 => {
                // VP9: Bit 0 of first byte is frame_marker, bit 1 is profile
                // Keyframe if show_frame is set and frame is not an inter frame
                (data[0] & 0x02) == 0
            }
            VideoCodec::Vp8 => {
                // VP8: First bit indicates keyframe (0 = key, 1 = inter)
                (data[0] & 0x01) == 0
            }
            _ => self.current_frame == 0,
        }
    }
}

/// Parse AV1 OBUs and detect whether the first frame header indicates a random-access frame.
///
/// Returns:
/// - `Some(true)` for key/switch frame
/// - `Some(false)` for non-key frame
/// - `None` if parsing is inconclusive
fn detect_av1_keyframe(data: &[u8]) -> Option<bool> {
    // AV1 OBU types
    const OBU_FRAME_HEADER: u8 = 3;
    const OBU_FRAME: u8 = 6;

    let mut pos = 0usize;
    while pos < data.len() {
        let header = *data.get(pos)?;
        pos += 1;

        let obu_type = (header >> 3) & 0x0F;
        let has_extension = (header & 0x04) != 0;
        let has_size_field = (header & 0x02) != 0;

        if has_extension {
            // extension_header (temporal_id/spatial_id)
            pos = pos.checked_add(1)?;
            if pos > data.len() {
                return None;
            }
        }

        let payload: &[u8];
        if has_size_field {
            let (payload_len, leb_len) = parse_leb128(data.get(pos..)?)?;
            pos = pos.checked_add(leb_len)?;
            let end = pos.checked_add(payload_len)?;
            if end > data.len() {
                return None;
            }
            payload = &data[pos..end];
            pos = end;
        } else {
            // Without explicit size we can only parse the remaining bytes as one OBU.
            payload = data.get(pos..)?;
            pos = data.len();
        }

        if obu_type == OBU_FRAME_HEADER || obu_type == OBU_FRAME {
            return parse_av1_frame_header_keyframe(payload);
        }

        if !has_size_field {
            break;
        }
    }

    None
}

/// Parse unsigned LEB128 integer.
fn parse_leb128(data: &[u8]) -> Option<(usize, usize)> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;

    for (idx, &byte) in data.iter().enumerate() {
        let part = (byte & 0x7F) as u64;
        value |= part.checked_shl(shift)?;

        if (byte & 0x80) == 0 {
            let v = usize::try_from(value).ok()?;
            return Some((v, idx + 1));
        }

        shift += 7;
        if shift >= 64 {
            return None;
        }
    }

    None
}

/// Parse enough of AV1 uncompressed frame header to determine frame_type.
fn parse_av1_frame_header_keyframe(payload: &[u8]) -> Option<bool> {
    let mut bits = BitReader::new(payload);

    // uncompressed_header(): show_existing_frame (1 bit)
    if bits.read_bit()? {
        // Re-showing an existing frame is not a random-access frame.
        return Some(false);
    }

    // frame_type (2 bits): 0=KEY, 1=INTER, 2=INTRA_ONLY, 3=SWITCH
    let frame_type = bits.read_bits(2)? as u8;

    // Treat SWITCH as keyframe-equivalent for seeking.
    Some(matches!(frame_type, 0 | 3))
}

struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    fn read_bit(&mut self) -> Option<bool> {
        let byte_idx = self.bit_pos / 8;
        let bit_idx = self.bit_pos % 8;
        let byte = *self.data.get(byte_idx)?;
        self.bit_pos += 1;
        Some(((byte >> (7 - bit_idx)) & 1) != 0)
    }

    fn read_bits(&mut self, n: u8) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | (self.read_bit()? as u32);
        }
        Some(v)
    }
}

impl Demuxer for IvfDemuxer {
    fn container_type(&self) -> ContainerType {
        ContainerType::Ivf
    }

    fn video_info(&self) -> Option<&VideoStreamInfo> {
        Some(&self.video_info)
    }

    fn audio_info(&self) -> Option<&AudioStreamInfo> {
        None // IVF doesn't contain audio
    }

    fn next_video_packet(&mut self) -> Result<Option<Packet>, VideoError> {
        self.read_frame()
    }

    fn next_audio_packet(&mut self) -> Result<Option<Packet>, VideoError> {
        Ok(None) // IVF doesn't contain audio
    }

    fn seek(&mut self, timestamp: f64) -> Result<(), VideoError> {
        // IVF seeking is complex because frames have variable size
        // For simplicity, we'll seek to the beginning and skip frames
        self.reset()?;

        let target_frame = if let Some(fps) = self.video_info.frame_rate {
            (timestamp * fps) as u32
        } else {
            return Err(VideoError::Container {
                message: "Cannot seek: frame rate unknown".to_string(),
            });
        };

        // Skip frames until we reach target
        while self.current_frame < target_frame {
            if self.read_frame()?.is_none() {
                break;
            }
        }

        Ok(())
    }

    fn reset(&mut self) -> Result<(), VideoError> {
        self.reader
            .seek(SeekFrom::Start(self.file_start_pos))
            .map_err(|e| VideoError::Io {
                message: format!("Failed to seek to start: {}", e),
            })?;
        self.current_frame = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn av1_keyframe_detection_key_frame() {
        // OBU_FRAME with size=1 and payload bits:
        // show_existing_frame=0, frame_type=00 (KEY_FRAME)
        let data = [0x32, 0x01, 0x00];
        assert_eq!(detect_av1_keyframe(&data), Some(true));
    }

    #[test]
    fn av1_keyframe_detection_inter_frame() {
        // show_existing_frame=0, frame_type=01 (INTER_FRAME)
        let data = [0x32, 0x01, 0x20];
        assert_eq!(detect_av1_keyframe(&data), Some(false));
    }

    #[test]
    fn av1_keyframe_detection_switch_frame() {
        // show_existing_frame=0, frame_type=11 (SWITCH_FRAME)
        let data = [0x32, 0x01, 0x60];
        assert_eq!(detect_av1_keyframe(&data), Some(true));
    }

    #[test]
    fn leb128_parses_multi_byte_value() {
        // 300 => 0b1010_1100 0b0000_0010 in LEB128
        let data = [0xAC, 0x02];
        assert_eq!(parse_leb128(&data), Some((300, 2)));
    }
}
