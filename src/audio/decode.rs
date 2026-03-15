//! Audio decoding using Symphonia.

use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{Decoder, DecoderOptions};
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::error::{AudioError, AudioResult};

#[cfg(feature = "audio-encode-opus")]
use audiopus::{
    coder::Decoder as OpusDecoder, packet as opus_packet, Channels as OpusChannels,
    SampleRate as OpusSampleRate,
};

#[cfg(feature = "audio-encode-opus")]
const OGG_CAPTURE_PATTERN: &[u8; 4] = b"OggS";
#[cfg(feature = "audio-encode-opus")]
const OPUS_HEAD_MAGIC: &[u8; 8] = b"OpusHead";
#[cfg(feature = "audio-encode-opus")]
const OPUS_TAGS_MAGIC: &[u8; 8] = b"OpusTags";
#[cfg(feature = "audio-encode-opus")]
const OPUS_OUTPUT_RATE: u32 = 48_000;

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
        let data_size = self.samples.len().saturating_mul(2); // 16-bit samples
        // Cap at u32::MAX to avoid overflow (WAV format uses u32 sizes)
        let data_size_u32 = (data_size as u64).min(u32::MAX as u64) as u32;
        let file_size = data_size_u32.saturating_add(36);
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
        bytes.extend_from_slice(&data_size_u32.to_le_bytes());

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

#[cfg(feature = "audio-encode-opus")]
#[derive(Debug)]
struct OggOpusStream {
    packets: Vec<Vec<u8>>,
    channels: u32,
    input_sample_rate: u32,
    pre_skip_48k: usize,
    final_granule_48k: Option<usize>,
}

#[cfg(feature = "audio-encode-opus")]
fn map_opus_decode_error<E: std::fmt::Display>(err: E) -> AudioError {
    AudioError::DecodeFailed {
        message: err.to_string(),
    }
}

#[cfg(feature = "audio-encode-opus")]
fn resample_interleaved_linear(samples: &[f32], channels: usize, source_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || channels == 0 || source_rate == 0 || target_rate == 0 || source_rate == target_rate {
        return samples.to_vec();
    }

    let input_frames = samples.len() / channels;
    if input_frames == 0 {
        return Vec::new();
    }

    let output_frames = ((input_frames as u64 * target_rate as u64) / source_rate as u64).max(1) as usize;
    let mut output = vec![0.0f32; output_frames * channels];

    for out_frame in 0..output_frames {
        let src_position = out_frame as f64 * source_rate as f64 / target_rate as f64;
        let src_index = src_position.floor() as usize;
        let next_index = (src_index + 1).min(input_frames.saturating_sub(1));
        let fraction = (src_position - src_index as f64) as f32;

        for channel in 0..channels {
            let a = samples[src_index * channels + channel];
            let b = samples[next_index * channels + channel];
            output[out_frame * channels + channel] = a + (b - a) * fraction;
        }
    }

    output
}

#[cfg(feature = "audio-encode-opus")]
fn parse_ogg_opus_stream(bytes: &[u8]) -> AudioResult<Option<OggOpusStream>> {
    if bytes.len() < 27 || &bytes[..4] != OGG_CAPTURE_PATTERN {
        return Ok(None);
    }

    let mut offset = 0usize;
    let mut serial = None;
    let mut packet = Vec::new();
    let mut saw_head = false;
    let mut saw_tags = false;
    let mut packets = Vec::new();
    let mut channels = 0u32;
    let mut input_sample_rate = OPUS_OUTPUT_RATE;
    let mut pre_skip_48k = 0usize;
    let mut final_granule_48k = None;

    while offset < bytes.len() {
        if bytes.len() - offset < 27 {
            return Err(AudioError::DecodeFailed {
                message: "truncated Ogg page header".to_string(),
            });
        }
        if &bytes[offset..offset + 4] != OGG_CAPTURE_PATTERN {
            return Ok(None);
        }

        let page_segments = bytes[offset + 26] as usize;
        let segment_table_start = offset + 27;
        let segment_table_end = segment_table_start + page_segments;
        if segment_table_end > bytes.len() {
            return Err(AudioError::DecodeFailed {
                message: "truncated Ogg lacing table".to_string(),
            });
        }

        let granule_position = u64::from_le_bytes(
            bytes[offset + 6..offset + 14]
                .try_into()
                .expect("Ogg granule position slice is exactly 8 bytes"),
        );
        let page_serial = u32::from_le_bytes(
            bytes[offset + 14..offset + 18]
                .try_into()
                .expect("Ogg page serial slice is exactly 4 bytes"),
        );
        if let Some(expected_serial) = serial {
            if page_serial != expected_serial {
                return Err(AudioError::DecodeFailed {
                    message: "multiple Ogg logical streams are not supported".to_string(),
                });
            }
        } else {
            serial = Some(page_serial);
        }

        let mut data_offset = segment_table_end;
        for &segment_len in &bytes[segment_table_start..segment_table_end] {
            let segment_len = segment_len as usize;
            let data_end = data_offset + segment_len;
            if data_end > bytes.len() {
                return Err(AudioError::DecodeFailed {
                    message: "truncated Ogg packet payload".to_string(),
                });
            }

            packet.extend_from_slice(&bytes[data_offset..data_end]);
            data_offset = data_end;

            if segment_len == 255 {
                continue;
            }

            if !saw_head {
                if packet.len() < 19 || &packet[..8] != OPUS_HEAD_MAGIC {
                    return Ok(None);
                }
                let channel_count = packet[9];
                if !(1..=2).contains(&channel_count) {
                    return Err(AudioError::InvalidChannels {
                        count: channel_count as u32,
                    });
                }
                if packet[18] != 0 {
                    return Err(AudioError::UnsupportedFormat {
                        format: "Ogg Opus channel mapping family > 0".to_string(),
                    });
                }

                channels = channel_count as u32;
                pre_skip_48k = u16::from_le_bytes([packet[10], packet[11]]) as usize;
                input_sample_rate =
                    u32::from_le_bytes([packet[12], packet[13], packet[14], packet[15]]).max(1);
                saw_head = true;
            } else if !saw_tags {
                if packet.len() < 8 || &packet[..8] != OPUS_TAGS_MAGIC {
                    return Err(AudioError::DecodeFailed {
                        message: "missing OpusTags packet".to_string(),
                    });
                }
                saw_tags = true;
            } else {
                packets.push(std::mem::take(&mut packet));
                if granule_position != u64::MAX {
                    final_granule_48k = Some(granule_position as usize);
                }
                continue;
            }

            packet.clear();
        }

        offset = data_offset;
    }

    if !packet.is_empty() {
        return Err(AudioError::DecodeFailed {
            message: "unterminated Ogg packet".to_string(),
        });
    }

    if !saw_head || !saw_tags {
        return Ok(None);
    }

    Ok(Some(OggOpusStream {
        packets,
        channels,
        input_sample_rate,
        pre_skip_48k,
        final_granule_48k,
    }))
}

#[cfg(feature = "audio-encode-opus")]
fn decode_ogg_opus_file<P: AsRef<Path>>(path: P) -> AudioResult<Option<DecodedAudio>> {
    let bytes = std::fs::read(path)?;
    let Some(stream) = parse_ogg_opus_stream(&bytes)? else {
        return Ok(None);
    };

    let channels = match stream.channels {
        1 => OpusChannels::Mono,
        2 => OpusChannels::Stereo,
        other => return Err(AudioError::InvalidChannels { count: other }),
    };

    let mut decoder = OpusDecoder::new(OpusSampleRate::Hz48000, channels).map_err(map_opus_decode_error)?;
    let mut samples_48k = Vec::new();
    let channel_count = stream.channels as usize;

    for packet in &stream.packets {
        let frame_count = opus_packet::nb_samples(packet.as_slice(), OpusSampleRate::Hz48000)
            .map_err(map_opus_decode_error)?;
        let mut decoded = vec![0.0f32; frame_count * channel_count];
        let written = decoder
            .decode_float(Some(packet.as_slice()), decoded.as_mut_slice(), false)
            .map_err(map_opus_decode_error)?;
        decoded.truncate(written * channel_count);
        samples_48k.extend(decoded);
    }

    if stream.pre_skip_48k > 0 {
        let skip = stream.pre_skip_48k.saturating_mul(channel_count);
        if skip >= samples_48k.len() {
            return Err(AudioError::DecodeFailed {
                message: "Opus pre-skip exceeds decoded sample count".to_string(),
            });
        }
        samples_48k.drain(..skip);
    }

    if let Some(final_granule_48k) = stream.final_granule_48k {
        let wanted_frames = final_granule_48k.saturating_sub(stream.pre_skip_48k);
        let wanted_samples = wanted_frames.saturating_mul(channel_count);
        if wanted_samples < samples_48k.len() {
            samples_48k.truncate(wanted_samples);
        }
    }

    let output_sample_rate = if matches!(stream.input_sample_rate, 8_000 | 12_000 | 16_000 | 24_000 | 48_000) {
        stream.input_sample_rate
    } else {
        OPUS_OUTPUT_RATE
    };
    let samples = if output_sample_rate == OPUS_OUTPUT_RATE {
        samples_48k
    } else {
        resample_interleaved_linear(&samples_48k, channel_count, OPUS_OUTPUT_RATE, output_sample_rate)
    };

    Ok(Some(DecodedAudio {
        samples,
        sample_rate: output_sample_rate,
        channels: stream.channels,
    }))
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
            if self.sample_buf.is_none() || self.sample_buf.as_ref().expect("checked is_none above").capacity() < duration as usize {
                self.sample_buf = Some(SampleBuffer::new(duration, spec));
            }

            let sample_buf = self.sample_buf.as_mut().expect("sample_buf guaranteed Some by preceding assignment");
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
    let path = path.as_ref();

    #[cfg(feature = "audio-encode-opus")]
    if matches!(
        path.extension().and_then(|ext| ext.to_str()).map(|ext| ext.to_ascii_lowercase()),
        Some(ext) if ext == "opus" || ext == "ogg"
    ) {
        if let Some(decoded) = decode_ogg_opus_file(path)? {
            return Ok(decoded);
        }
    }

    AudioDecoder::open(path)?.decode_all()
}

/// Extract audio from a video file.
pub fn extract_audio_from_video<P: AsRef<Path>>(path: P) -> AudioResult<DecodedAudio> {
    // Symphonia handles containers like MP4/MKV that contain video+audio
    // It will automatically find the audio track
    decode_file(path)
}
