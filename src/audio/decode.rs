//! Audio decoding using Symphonia.

use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{Decoder, DecoderOptions};
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::error::{AudioError, AudioResult};

/// Decoded audio data.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    /// Interleaved samples (f32, normalized to -1.0 to 1.0)
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
}

impl DecodedAudio {
    /// Get duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate > 0 && self.channels > 0 {
            self.samples.len() as f64 / (self.sample_rate as f64 * self.channels as f64)
        } else {
            0.0
        }
    }

    /// Get number of frames (samples per channel).
    pub fn num_frames(&self) -> usize {
        if self.channels > 0 {
            self.samples.len() / self.channels as usize
        } else {
            0
        }
    }

    /// Convert to mono by averaging channels.
    pub fn to_mono(&self) -> DecodedAudio {
        if self.channels == 1 {
            return self.clone();
        }

        let num_frames = self.num_frames();
        let mut mono = Vec::with_capacity(num_frames);

        for frame in 0..num_frames {
            let mut sum = 0.0f32;
            for ch in 0..self.channels as usize {
                sum += self.samples[frame * self.channels as usize + ch];
            }
            mono.push(sum / self.channels as f32);
        }

        DecodedAudio {
            samples: mono,
            sample_rate: self.sample_rate,
            channels: 1,
        }
    }

    /// Get samples for a specific channel.
    pub fn channel(&self, channel: usize) -> Vec<f32> {
        if channel >= self.channels as usize {
            return Vec::new();
        }

        let num_frames = self.num_frames();
        let mut ch_samples = Vec::with_capacity(num_frames);

        for frame in 0..num_frames {
            ch_samples.push(self.samples[frame * self.channels as usize + channel]);
        }

        ch_samples
    }

    /// Convert to WAV bytes (for saving).
    pub fn to_wav_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // RIFF header
        bytes.extend_from_slice(b"RIFF");
        let data_size = self.samples.len() * 2; // 16-bit samples
        let file_size = (36 + data_size) as u32;
        bytes.extend_from_slice(&file_size.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");

        // fmt chunk
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        bytes.extend_from_slice(&(self.channels as u16).to_le_bytes());
        bytes.extend_from_slice(&self.sample_rate.to_le_bytes());
        let byte_rate = self.sample_rate * self.channels * 2;
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        let block_align = self.channels * 2;
        bytes.extend_from_slice(&(block_align as u16).to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

        // data chunk
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data_size as u32).to_le_bytes());

        // Convert f32 samples to i16
        for sample in &self.samples {
            let s = (*sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            bytes.extend_from_slice(&s.to_le_bytes());
        }

        bytes
    }

    /// Save as WAV file.
    pub fn save_wav<P: AsRef<Path>>(&self, path: P) -> AudioResult<()> {
        std::fs::write(path, self.to_wav_bytes())?;
        Ok(())
    }
}

/// Audio samples iterator chunk.
#[derive(Debug, Clone)]
pub struct AudioSamples {
    /// Sample data (interleaved f32)
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
}

/// Audio decoder for streaming decode.
pub struct AudioDecoder {
    format_reader: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    track_id: u32,
    sample_rate: u32,
    channels: u32,
    sample_buf: Option<SampleBuffer<f32>>,
}

impl AudioDecoder {
    /// Open an audio file for decoding.
    pub fn open<P: AsRef<Path>>(path: P) -> AudioResult<Self> {
        let path = path.as_ref();

        // Open file
        let file = std::fs::File::open(path).map_err(|e| AudioError::OpenFailed {
            path: path.to_path_buf(),
            source: e,
        })?;

        // Create media source
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Create hint
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        // Probe format
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
            .map_err(|e| AudioError::Symphonia(e.to_string()))?;

        let format_reader = probed.format;

        // Find audio track
        let track = format_reader
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
            .ok_or(AudioError::NoAudioTrack)?;

        let track_id = track.id;
        let codec_params = &track.codec_params;

        // Get audio parameters
        let sample_rate = codec_params.sample_rate.unwrap_or(44100);
        let channels = codec_params.channels.map(|c| c.count() as u32).unwrap_or(2);

        // Create decoder
        let decoder = symphonia::default::get_codecs()
            .make(codec_params, &DecoderOptions::default())
            .map_err(|e| AudioError::Symphonia(e.to_string()))?;

        Ok(AudioDecoder {
            format_reader,
            decoder,
            track_id,
            sample_rate,
            channels,
            sample_buf: None,
        })
    }

    /// Get sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get number of channels.
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Decode entire file to memory.
    pub fn decode_all(mut self) -> AudioResult<DecodedAudio> {
        let mut all_samples = Vec::new();

        while let Some(chunk) = self.next_chunk()? {
            all_samples.extend(chunk.samples);
        }

        Ok(DecodedAudio {
            samples: all_samples,
            sample_rate: self.sample_rate,
            channels: self.channels,
        })
    }

    /// Get next chunk of decoded samples.
    pub fn next_chunk(&mut self) -> AudioResult<Option<AudioSamples>> {
        loop {
            // Read next packet
            let packet = match self.format_reader.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(None); // End of stream
                }
                Err(e) => return Err(AudioError::Symphonia(e.to_string())),
            };

            // Skip packets from other tracks
            if packet.track_id() != self.track_id {
                continue;
            }

            // Decode packet
            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(symphonia::core::errors::Error::DecodeError(_)) => continue, // Skip errors
                Err(e) => return Err(AudioError::Symphonia(e.to_string())),
            };

            // Convert to f32 samples
            let spec = *decoded.spec();
            let duration = decoded.capacity() as u64;

            // Ensure sample buffer is properly sized
            if self.sample_buf.is_none() || self.sample_buf.as_ref().unwrap().capacity() < duration as usize {
                self.sample_buf = Some(SampleBuffer::new(duration, spec));
            }

            let sample_buf = self.sample_buf.as_mut().unwrap();
            sample_buf.copy_interleaved_ref(decoded);

            let samples = sample_buf.samples().to_vec();

            return Ok(Some(AudioSamples {
                samples,
                sample_rate: spec.rate,
                channels: spec.channels.count() as u32,
            }));
        }
    }
}

/// Decode an audio file completely.
pub fn decode_file<P: AsRef<Path>>(path: P) -> AudioResult<DecodedAudio> {
    AudioDecoder::open(path)?.decode_all()
}

/// Extract audio from a video file.
pub fn extract_audio_from_video<P: AsRef<Path>>(path: P) -> AudioResult<DecodedAudio> {
    // Symphonia handles containers like MP4/MKV that contain video+audio
    // It will automatically find the audio track
    decode_file(path)
}
