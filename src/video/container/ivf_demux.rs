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
        reader.read_exact(&mut header_data).map_err(|e| VideoError::Io {
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
        self.reader.read_exact(&mut data).map_err(|e| VideoError::Io {
            message: format!("Failed to read frame data: {}", e),
        })?;

        // Check if this is a keyframe (first byte analysis for AV1/VP9)
        let is_keyframe = self.detect_keyframe(&data);

        self.current_frame += 1;

        Ok(Some(Packet {
            data,
            pts: timestamp as i64,
            dts: timestamp as i64, // IVF doesn't have B-frames reordering
            duration: 1, // One frame duration in timebase units
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
                // AV1: Check OBU header for key frame
                // First byte: OBU type in bits 3-6
                let _obu_type = (data[0] >> 3) & 0x0F;
                // OBU_FRAME = 6, check if it's a key frame
                // TODO: Implement proper AV1 keyframe detection
                // For simplicity, assume first frame is keyframe
                self.current_frame == 0
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
