//! Audio processing module for xeno-lib.
//!
//! This module provides pure Rust audio decoding, encoding, filtering, and processing.
//!
//! # Decoding (via Symphonia)
//!
//! - **Codecs**: MP3, AAC, FLAC, Vorbis, Opus, ALAC, PCM/WAV, AIFF
//! - **Containers**: MP4/M4A, MKV/WebM, OGG, WAV, AIFF, MP3
//!
//! # Encoding (pure Rust)
//!
//! - **WAV**: via hound (lossless, uncompressed)
//! - **FLAC**: via flacenc (lossless, ~60% compression)
//! - **Opus**: via audiopus (high-quality lossy, best for streaming)
//!
//! # Filters (pure Rust, FFmpeg-equivalent)
//!
//! - **Volume**: Adjust gain in dB (like `-af volume`)
//! - **Fade**: Fade in/out effects (like `-af afade`)
//! - **Normalize**: Peak and RMS normalization (like `-af loudnorm`)
//! - **Compress**: Dynamic range compression (like `-af acompressor`)
//! - **Mix**: Mix and add audio streams (like `-af amix`)
//!
//! # Features
//!
//! - `audio` - Basic audio decoding
//! - `audio-resample` - Audio resampling support
//! - `audio-encode` - WAV encoding
//! - `audio-encode-flac` - FLAC encoding
//! - `audio-encode-opus` - Opus encoding (high-quality lossy)
//!
//! # Example: Decode audio
//!
//! ```ignore
//! use xeno_lib::audio::{AudioDecoder, AudioInfo};
//!
//! // Get audio info
//! let info = AudioInfo::from_file("song.mp3")?;
//! println!("Duration: {}s, Sample rate: {}Hz", info.duration_secs, info.sample_rate);
//!
//! // Decode audio
//! let decoder = AudioDecoder::open("song.mp3")?;
//! let audio = decoder.decode_all()?;
//! ```
//!
//! # Example: Apply audio filters
//!
//! ```ignore
//! use xeno_lib::audio::filters::{adjust_volume, fade_in, normalize_peak, FadeCurve};
//!
//! // Increase volume by 6dB
//! let louder = adjust_volume(&samples, 6.0);
//!
//! // Apply fade-in over 2 seconds
//! let faded = fade_in(&samples, 44100, 2.0, FadeCurve::SCurve);
//!
//! // Normalize to -1dB peak
//! let normalized = normalize_peak(&samples, -1.0);
//! ```
//!
//! # Example: Encode audio to WAV
//!
//! ```ignore
//! use xeno_lib::audio::encode::{encode_wav, WavConfig};
//!
//! let config = WavConfig::new(44100, 2).with_bits(24);
//! encode_wav(&samples, "output.wav", config)?;
//! ```

#[cfg(feature = "audio")]
mod decode;
#[cfg(feature = "audio")]
mod error;
#[cfg(feature = "audio")]
mod metadata;

#[cfg(feature = "audio-encode")]
pub mod encode;

// Audio filters - always available with audio feature
#[cfg(feature = "audio")]
pub mod filters;

#[cfg(feature = "audio")]
pub use decode::{decode_file, extract_audio_from_video, AudioDecoder, AudioSamples, DecodedAudio};
#[cfg(feature = "audio")]
pub use error::{AudioError, AudioResult};
#[cfg(feature = "audio")]
pub use metadata::{AudioCodec, AudioInfo, AudioFormat};

// Re-export common filter types at module level for convenience
#[cfg(feature = "audio")]
pub use filters::{
    adjust_volume, adjust_volume_inplace, apply_gain,
    fade_in, fade_out, fade_in_out,
    normalize_peak, normalize_rms,
    compress, limit,
    mix, add, mono_to_stereo, stereo_to_mono,
    trim_silence, detect_silence, remove_dc_offset,
    invert_phase,
    db_to_linear, linear_to_db, calculate_rms, calculate_peak_db, calculate_rms_db,
    FadeCurve,
};

// Re-export common encode types at module level for convenience
#[cfg(feature = "audio-encode")]
pub use encode::{encode_wav, encode_wav_to_bytes, WavConfig, AudioOutputFormat, AudioEncodeError, AudioEncodeResult};

#[cfg(feature = "audio-encode-flac")]
pub use encode::{encode_flac, encode_flac_to_bytes, FlacConfig};

#[cfg(feature = "audio-encode-opus")]
pub use encode::{encode_opus, OpusApplication, OpusEncoder, OpusEncoderConfig, OpusError, OpusResult};

// Audio effects module (Phase 3 - Reverb, EQ, Pitch, etc.)
#[cfg(feature = "audio")]
pub mod effects;

// Audio visualization module (Phase 3 - Waveform, Spectrum, Spectrogram)
#[cfg(feature = "audio")]
pub mod visualization;
