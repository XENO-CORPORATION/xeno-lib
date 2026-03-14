//! Audio metadata types.

use std::path::Path;
use std::time::Duration;

use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::error::{AudioError, AudioResult};

/// Audio codec types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec {
    /// MP3 (MPEG Audio Layer III)
    Mp3,
    /// AAC (Advanced Audio Coding)
    Aac,
    /// FLAC (Free Lossless Audio Codec)
    Flac,
    /// Vorbis (OGG Vorbis)
    Vorbis,
    /// Opus
    Opus,
    /// ALAC (Apple Lossless Audio Codec)
    Alac,
    /// PCM (Uncompressed)
    Pcm,
    /// Unknown codec
    Unknown,
}

impl AudioCodec {
    /// Get codec name as string.
    pub fn name(&self) -> &'static str {
        match self {
            AudioCodec::Mp3 => "MP3",
            AudioCodec::Aac => "AAC",
            AudioCodec::Flac => "FLAC",
            AudioCodec::Vorbis => "Vorbis",
            AudioCodec::Opus => "Opus",
            AudioCodec::Alac => "ALAC",
            AudioCodec::Pcm => "PCM",
            AudioCodec::Unknown => "Unknown",
        }
    }

    /// Check if codec is lossless.
    pub fn is_lossless(&self) -> bool {
        matches!(self, AudioCodec::Flac | AudioCodec::Alac | AudioCodec::Pcm)
    }
}

/// Audio container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// MP3 file
    Mp3,
    /// MP4/M4A container
    Mp4,
    /// OGG container
    Ogg,
    /// FLAC file
    Flac,
    /// WAV file
    Wav,
    /// AIFF file
    Aiff,
    /// Matroska/WebM container
    Matroska,
    /// Unknown format
    Unknown,
}

impl AudioFormat {
    /// Detect format from file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "mp3" => AudioFormat::Mp3,
            "mp4" | "m4a" | "m4b" | "m4p" | "aac" => AudioFormat::Mp4,
            "ogg" | "oga" | "opus" => AudioFormat::Ogg,
            "flac" => AudioFormat::Flac,
            "wav" | "wave" => AudioFormat::Wav,
            "aiff" | "aif" => AudioFormat::Aiff,
            "mkv" | "mka" | "webm" => AudioFormat::Matroska,
            _ => AudioFormat::Unknown,
        }
    }

    /// Get format name.
    pub fn name(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "MP3",
            AudioFormat::Mp4 => "MP4/M4A",
            AudioFormat::Ogg => "OGG",
            AudioFormat::Flac => "FLAC",
            AudioFormat::Wav => "WAV",
            AudioFormat::Aiff => "AIFF",
            AudioFormat::Matroska => "Matroska",
            AudioFormat::Unknown => "Unknown",
        }
    }
}

/// Audio file information.
#[derive(Debug, Clone)]
pub struct AudioInfo {
    /// Audio codec
    pub codec: AudioCodec,
    /// Container format
    pub format: AudioFormat,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
    /// Bits per sample (if known)
    pub bits_per_sample: Option<u32>,
    /// Duration in seconds
    pub duration_secs: f64,
    /// Total number of samples
    pub total_samples: Option<u64>,
    /// Bitrate in kbps (if known)
    pub bitrate_kbps: Option<u32>,
    /// File size in bytes
    pub file_size: u64,
    /// Title metadata
    pub title: Option<String>,
    /// Artist metadata
    pub artist: Option<String>,
    /// Album metadata
    pub album: Option<String>,
}

impl AudioInfo {
    /// Get audio info from a file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> AudioResult<Self> {
        let path = path.as_ref();

        // Get file size
        let file_size = std::fs::metadata(path)
            .map_err(|e| AudioError::OpenFailed {
                path: path.to_path_buf(),
                source: e,
            })?
            .len();

        // Detect format from extension
        let format = path
            .extension()
            .and_then(|e| e.to_str())
            .map(AudioFormat::from_extension)
            .unwrap_or(AudioFormat::Unknown);

        // Open file
        let file = std::fs::File::open(path).map_err(|e| AudioError::OpenFailed {
            path: path.to_path_buf(),
            source: e,
        })?;

        // Create media source
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Create hint from extension
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        // Probe the format
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
            .map_err(|e| AudioError::Symphonia(e.to_string()))?;

        let mut format_reader = probed.format;

        // Find the first audio track
        let track = format_reader
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
            .ok_or(AudioError::NoAudioTrack)?;

        let codec_params = &track.codec_params;

        // Determine codec
        let codec = match codec_params.codec {
            symphonia::core::codecs::CODEC_TYPE_MP3 => AudioCodec::Mp3,
            symphonia::core::codecs::CODEC_TYPE_AAC => AudioCodec::Aac,
            symphonia::core::codecs::CODEC_TYPE_FLAC => AudioCodec::Flac,
            symphonia::core::codecs::CODEC_TYPE_VORBIS => AudioCodec::Vorbis,
            symphonia::core::codecs::CODEC_TYPE_OPUS => AudioCodec::Opus,
            symphonia::core::codecs::CODEC_TYPE_ALAC => AudioCodec::Alac,
            symphonia::core::codecs::CODEC_TYPE_PCM_S16LE
            | symphonia::core::codecs::CODEC_TYPE_PCM_S16BE
            | symphonia::core::codecs::CODEC_TYPE_PCM_S24LE
            | symphonia::core::codecs::CODEC_TYPE_PCM_S24BE
            | symphonia::core::codecs::CODEC_TYPE_PCM_S32LE
            | symphonia::core::codecs::CODEC_TYPE_PCM_S32BE
            | symphonia::core::codecs::CODEC_TYPE_PCM_F32LE
            | symphonia::core::codecs::CODEC_TYPE_PCM_F32BE => AudioCodec::Pcm,
            _ => AudioCodec::Unknown,
        };

        // Get sample rate
        let sample_rate = codec_params.sample_rate.unwrap_or(44100);

        // Get channels
        let channels = codec_params
            .channels
            .map(|c| c.count() as u32)
            .unwrap_or(2);

        // Get bits per sample
        let bits_per_sample = codec_params.bits_per_sample;

        // Get total samples and duration
        let total_samples = codec_params.n_frames;
        let duration_secs = if let Some(n_frames) = total_samples {
            n_frames as f64 / sample_rate as f64
        } else if let Some(tb) = track.codec_params.time_base {
            if let Some(dur) = format_reader.tracks()[0].codec_params.n_frames {
                dur as f64 * tb.numer as f64 / tb.denom as f64
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Estimate bitrate
        let bitrate_kbps = if duration_secs > 0.0 {
            Some((file_size as f64 * 8.0 / duration_secs / 1000.0) as u32)
        } else {
            None
        };

        // Get metadata
        let mut title = None;
        let mut artist = None;
        let mut album = None;

        if let Some(metadata) = format_reader.metadata().current() {
            for tag in metadata.tags() {
                match tag.std_key {
                    Some(symphonia::core::meta::StandardTagKey::TrackTitle) => {
                        title = Some(tag.value.to_string());
                    }
                    Some(symphonia::core::meta::StandardTagKey::Artist) => {
                        artist = Some(tag.value.to_string());
                    }
                    Some(symphonia::core::meta::StandardTagKey::Album) => {
                        album = Some(tag.value.to_string());
                    }
                    _ => {}
                }
            }
        }

        Ok(AudioInfo {
            codec,
            format,
            sample_rate,
            channels,
            bits_per_sample,
            duration_secs,
            total_samples,
            bitrate_kbps,
            file_size,
            title,
            artist,
            album,
        })
    }

    /// Get duration as Duration type.
    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(self.duration_secs)
    }

    /// Format duration as string (MM:SS or HH:MM:SS).
    pub fn duration_string(&self) -> String {
        let total_secs = self.duration_secs as u64;
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        let secs = total_secs % 60;

        if hours > 0 {
            format!("{}:{:02}:{:02}", hours, mins, secs)
        } else {
            format!("{}:{:02}", mins, secs)
        }
    }
}
