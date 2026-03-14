//! Software H.264 decoder using OpenH264.
//!
//! This provides a CPU fallback for H.264 streams when NVDEC is unavailable.

use std::collections::VecDeque;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use openh264::decoder::{Decoder, DecoderConfig as OpenH264DecoderConfig, Flush};
use openh264::formats::YUVSource;
use openh264::{OpenH264API, nal_units};

use crate::video::VideoError;

use super::{DecodeCodec, DecodedFrame, DecoderCapabilities, OutputFormat, VideoDecoder};

/// Software H.264 decoder using OpenH264.
pub struct OpenH264Decoder {
    decoder: Decoder,
    frames: VecDeque<DecodedFrame>,
    pending_pts: VecDeque<i64>,
    frame_count: u64,
}

impl OpenH264Decoder {
    /// Create a new OpenH264 decoder.
    pub fn new() -> Result<Self, VideoError> {
        let decoder_config = OpenH264DecoderConfig::new().flush_after_decode(Flush::NoFlush);
        let decoder =
            Decoder::with_api_config(OpenH264API::from_source(), decoder_config).map_err(|e| {
                VideoError::Decoding {
                    message: format!("Failed to create OpenH264 decoder: {:?}", e),
                }
            })?;

        Ok(Self {
            decoder,
            frames: VecDeque::new(),
            pending_pts: VecDeque::new(),
            frame_count: 0,
        })
    }

    /// Check if OpenH264 is available (always true if compiled with feature).
    pub fn is_available() -> bool {
        true
    }

    /// Get capabilities for software H.264 decoding.
    pub fn capabilities() -> DecoderCapabilities {
        DecoderCapabilities {
            supported: true,
            max_width: 8192,
            max_height: 8192,
            max_bit_depth: 8,
            num_engines: 1,
        }
    }

    fn decoded_picture_to_frame(
        picture: &openh264::decoder::DecodedYUV<'_>,
        pts: i64,
        decode_index: u64,
    ) -> Result<DecodedFrame, VideoError> {
        let (width, height) = picture.dimensions();
        let mut rgba = vec![0u8; width * height * 4];
        picture.write_rgba8(&mut rgba);

        Ok(DecodedFrame {
            width: width as u32,
            height: height as u32,
            pts,
            decode_index,
            format: OutputFormat::Rgba,
            data: rgba,
            strides: vec![width * 4],
        })
    }

    /// Decode a raw Annex B H.264 file into the internal frame queue.
    pub fn decode_h264_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), VideoError> {
        let mut data = Vec::new();
        File::open(path.as_ref())
            .and_then(|mut file| file.read_to_end(&mut data))
            .map_err(|e| VideoError::Io {
                message: format!("Failed to read H.264 file: {}", e),
            })?;

        let mut pts = 0i64;
        for packet in nal_units(&data) {
            self.decode_packet(packet, pts)?;
            pts += 1;
        }
        self.flush()?;
        Ok(())
    }
}

impl VideoDecoder for OpenH264Decoder {
    fn decode_file(&mut self, path: &str) -> Result<(), VideoError> {
        self.decode_h264_file(path)
    }

    fn decode_packet(&mut self, data: &[u8], pts: i64) -> Result<(), VideoError> {
        if data.is_empty() {
            return Ok(());
        }

        self.pending_pts.push_back(pts);
        let next_pts = self.pending_pts.front().copied().unwrap_or(pts);
        if let Some(decoded) = self.decoder.decode(data).map_err(|e| VideoError::Decoding {
            message: format!("OpenH264 decode failed: {:?}", e),
        })? {
            let frame = Self::decoded_picture_to_frame(&decoded, next_pts, self.frame_count)?;
            self.frames.push_back(frame);
            self.frame_count += 1;
            self.pending_pts.pop_front();
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), VideoError> {
        let mut pending_pts = std::mem::take(&mut self.pending_pts);
        let pictures = self
            .decoder
            .flush_remaining()
            .map_err(|e| VideoError::Decoding {
                message: format!("OpenH264 flush failed: {:?}", e),
            })?;

        for picture in pictures {
            let pts = pending_pts.pop_front().unwrap_or(self.frame_count as i64);
            let frame = Self::decoded_picture_to_frame(&picture, pts, self.frame_count)?;
            self.frames.push_back(frame);
            self.frame_count += 1;
        }

        self.pending_pts = pending_pts;

        Ok(())
    }

    fn next_frame(&mut self) -> Result<Option<DecodedFrame>, VideoError> {
        Ok(self.frames.pop_front())
    }

    fn get_capabilities(&self, codec: DecodeCodec) -> Result<DecoderCapabilities, VideoError> {
        Ok(if codec == DecodeCodec::H264 {
            Self::capabilities()
        } else {
            DecoderCapabilities {
                supported: false,
                max_width: 0,
                max_height: 0,
                max_bit_depth: 0,
                num_engines: 0,
            }
        })
    }
}

impl Default for OpenH264Decoder {
    fn default() -> Self {
        Self::new().expect("Failed to create OpenH264 decoder")
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgba, RgbaImage};
    use crate::video::encode::{H264Encoder, H264EncoderConfig};

    use super::*;

    #[test]
    fn openh264_round_trip_decodes_encoded_frame() {
        let mut pixels = RgbaImage::new(16, 16);
        for (x, y, pixel) in pixels.enumerate_pixels_mut() {
            *pixel = Rgba([(x * 8) as u8, (y * 8) as u8, 128, 255]);
        }

        let input = DynamicImage::ImageRgba8(pixels);
        let mut encoder = H264Encoder::create(H264EncoderConfig::new(16, 16)).unwrap();
        let encoded = encoder.encode_frame(&input).unwrap().unwrap();

        let mut decoder = OpenH264Decoder::new().unwrap();
        decoder.decode_packet(&encoded.data, 123).unwrap();
        decoder.flush().unwrap();

        let frame = decoder.next_frame().unwrap().expect("expected decoded frame");
        assert_eq!(frame.width, 16);
        assert_eq!(frame.height, 16);
        assert_eq!(frame.pts, 123);
        assert_eq!(frame.format, OutputFormat::Rgba);
        assert_eq!(frame.data.len(), 16 * 16 * 4);
    }
}
