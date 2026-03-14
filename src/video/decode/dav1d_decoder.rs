//! Software AV1 decoder using dav1d.
//!
//! This module provides a software fallback for AV1 decoding when
//! NVDEC is not available or not suitable.

use std::io::{BufReader, Read};
use std::path::Path;

use dav1d::{Decoder, Settings};

use crate::video::VideoError;

use super::{DecodedFrame, DecoderCapabilities, OutputFormat};

/// Software AV1 decoder using dav1d.
pub struct Dav1dDecoder {
    decoder: Decoder,
    width: u32,
    height: u32,
    frame_count: u64,
}

impl Dav1dDecoder {
    /// Create a new dav1d decoder.
    pub fn new() -> Result<Self, VideoError> {
        let settings = Settings::new();
        let decoder = Decoder::with_settings(&settings).map_err(|e| VideoError::Decoding {
            message: format!("Failed to create dav1d decoder: {:?}", e),
        })?;

        Ok(Self {
            decoder,
            width: 0,
            height: 0,
            frame_count: 0,
        })
    }

    /// Check if dav1d is available (always true if compiled with feature).
    pub fn is_available() -> bool {
        true
    }

    /// Get capabilities for software decoding.
    pub fn get_capabilities() -> DecoderCapabilities {
        DecoderCapabilities {
            supported: true,
            max_width: 8192,  // Software can handle any resolution
            max_height: 8192,
            max_bit_depth: 12,
            num_engines: 1,  // Software = single "engine"
        }
    }

    /// Decode an IVF file containing AV1 video.
    pub fn decode_ivf_file<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<DecodedFrame>, VideoError> {
        use std::fs::File;

        let file = File::open(path.as_ref()).map_err(|e| VideoError::Io {
            message: format!("Failed to open file: {}", e),
        })?;
        let mut reader = BufReader::new(file);

        // Read IVF header (32 bytes)
        let mut header = [0u8; 32];
        reader.read_exact(&mut header).map_err(|e| VideoError::Io {
            message: format!("Failed to read IVF header: {}", e),
        })?;

        // Validate signature
        if &header[0..4] != b"DKIF" {
            return Err(VideoError::Decoding {
                message: "Invalid IVF file: bad signature".to_string(),
            });
        }

        // Parse header
        let fourcc = std::str::from_utf8(&header[8..12]).unwrap_or("????");
        if fourcc != "AV01" {
            return Err(VideoError::Decoding {
                message: format!("dav1d only supports AV1, got: {}", fourcc),
            });
        }

        let width = u16::from_le_bytes([header[12], header[13]]) as u32;
        let height = u16::from_le_bytes([header[14], header[15]]) as u32;
        let frame_count = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);

        self.width = width;
        self.height = height;

        let mut frames = Vec::new();

        // Read and decode frames
        for _ in 0..frame_count {
            // Read frame header (12 bytes)
            let mut frame_header = [0u8; 12];
            if reader.read_exact(&mut frame_header).is_err() {
                break; // End of file
            }

            let frame_size = u32::from_le_bytes([
                frame_header[0],
                frame_header[1],
                frame_header[2],
                frame_header[3],
            ]) as usize;

            let frame_pts = u64::from_le_bytes([
                frame_header[4],
                frame_header[5],
                frame_header[6],
                frame_header[7],
                frame_header[8],
                frame_header[9],
                frame_header[10],
                frame_header[11],
            ]) as i64;

            // Read frame data
            let mut frame_data = vec![0u8; frame_size];
            reader.read_exact(&mut frame_data).map_err(|e| VideoError::Io {
                message: format!("Failed to read frame data: {}", e),
            })?;

            // Send data to decoder
            self.decoder.send_data(frame_data, None, None, None).map_err(|e| VideoError::Decoding {
                message: format!("Failed to send data to decoder: {:?}", e),
            })?;

            // Try to get decoded frames
            loop {
                match self.decoder.get_picture() {
                    Ok(picture) => {
                        let decoded = self.picture_to_frame(&picture, frame_pts)?;
                        frames.push(decoded);
                        self.frame_count += 1;
                    }
                    Err(dav1d::Error::Again) => break, // Need more data
                    Err(e) => {
                        return Err(VideoError::Decoding {
                            message: format!("Decode error: {:?}", e),
                        });
                    }
                }
            }
        }

        // Flush decoder
        self.decoder.flush();
        loop {
            match self.decoder.get_picture() {
                Ok(picture) => {
                    let decoded = self.picture_to_frame(&picture, 0)?;
                    frames.push(decoded);
                    self.frame_count += 1;
                }
                Err(dav1d::Error::Again) => break,
                Err(_) => break,
            }
        }

        Ok(frames)
    }

    /// Convert a dav1d Picture to our DecodedFrame format.
    fn picture_to_frame(&self, picture: &dav1d::Picture, pts: i64) -> Result<DecodedFrame, VideoError> {
        let width = picture.width() as u32;
        let height = picture.height() as u32;

        // Get Y, U, V planes
        let y_plane = picture.plane(dav1d::PlanarImageComponent::Y);
        let u_plane = picture.plane(dav1d::PlanarImageComponent::U);
        let v_plane = picture.plane(dav1d::PlanarImageComponent::V);

        let y_stride = picture.stride(dav1d::PlanarImageComponent::Y) as usize;
        let uv_stride = picture.stride(dav1d::PlanarImageComponent::U) as usize;

        // Convert to NV12 format (Y plane + interleaved UV)
        let y_size = (width * height) as usize;
        let uv_height = height as usize / 2;
        let uv_width = width as usize / 2;
        let uv_size = uv_width * uv_height * 2;

        let mut data = Vec::with_capacity(y_size + uv_size);

        // Copy Y plane (remove stride padding)
        for row in 0..height as usize {
            let start = row * y_stride;
            let end = start + width as usize;
            data.extend_from_slice(&y_plane[start..end]);
        }

        // Interleave U and V planes to create NV12 UV plane
        for row in 0..uv_height {
            for col in 0..uv_width {
                let u_idx = row * uv_stride + col;
                let v_idx = row * uv_stride + col;
                data.push(u_plane[u_idx]);
                data.push(v_plane[v_idx]);
            }
        }

        Ok(DecodedFrame {
            width,
            height,
            pts,
            decode_index: self.frame_count,
            format: OutputFormat::Nv12,
            data,
            strides: vec![width as usize, width as usize],
        })
    }

    /// Decode a single OBU (Open Bitstream Unit) packet.
    pub fn decode_packet(&mut self, data: &[u8], pts: i64) -> Result<Option<DecodedFrame>, VideoError> {
        self.decoder.send_data(data.to_vec(), None, None, None).map_err(|e| VideoError::Decoding {
            message: format!("Failed to send data: {:?}", e),
        })?;

        match self.decoder.get_picture() {
            Ok(picture) => {
                let frame = self.picture_to_frame(&picture, pts)?;
                self.frame_count += 1;
                Ok(Some(frame))
            }
            Err(dav1d::Error::Again) => Ok(None),
            Err(e) => Err(VideoError::Decoding {
                message: format!("Decode error: {:?}", e),
            }),
        }
    }
}

impl Default for Dav1dDecoder {
    fn default() -> Self {
        Self::new().expect("Failed to create dav1d decoder")
    }
}
