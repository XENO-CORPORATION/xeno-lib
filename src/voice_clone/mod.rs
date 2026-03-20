// TODO: MIGRATE TO XENO-RT — this module belongs in the inference runtime, not the processing library.
// All ONNX Runtime inference (model loading, voice embedding extraction, speech synthesis) should
// move to xeno-rt. The VoiceEmbedding and SynthesizedAudio structs are data types that should be
// shared.
//!
//! AI-powered voice cloning and text-to-speech synthesis.
//!
//! Converts text to speech using reference audio for voice cloning.
//! Supports XTTS v2 (real-time), Bark (creative), and Tortoise TTS (quality).
//!
//! # Output Contract
//!
//! All audio outputs are f32 PCM samples in range [-1.0, 1.0] at the configured
//! sample rate (default 24000 Hz for XTTS).
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::voice_clone::{synthesize_speech, load_voice_model, VoiceCloneConfig};
//!
//! let config = VoiceCloneConfig::default();
//! let mut session = load_voice_model(&config)?;
//! let audio = synthesize_speech("Hello, world!", None, &config)?;
//! println!("Generated {} samples at {} Hz", audio.samples.len(), audio.sample_rate);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod config;
pub mod model;
pub mod processor;

pub use config::{VoiceCloneConfig, VoiceCloneModel};
pub use model::{load_voice_model, VoiceCloneSession};

use crate::error::TransformError;

/// Speaker voice embedding extracted from reference audio.
#[derive(Debug, Clone)]
pub struct VoiceEmbedding {
    /// Embedding vector (typically 256-512 dimensions).
    pub data: Vec<f32>,
    /// Sample rate of the reference audio.
    pub sample_rate: u32,
    /// Language detected in reference.
    pub language: String,
}

/// Synthesized audio output.
#[derive(Debug, Clone)]
pub struct SynthesizedAudio {
    /// f32 PCM samples in [-1.0, 1.0].
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Duration in seconds.
    pub duration_secs: f32,
    /// Language used for synthesis.
    pub language: String,
}

/// Extracts a voice embedding from reference audio for cloning.
///
/// Requires at least 3 seconds of clean speech for quality results.
///
/// # Arguments
///
/// * `reference_audio` - f32 PCM samples of the reference speaker.
/// * `config` - Voice cloning configuration.
pub fn extract_embedding(
    reference_audio: &[f32],
    config: &VoiceCloneConfig,
) -> Result<VoiceEmbedding, TransformError> {
    processor::extract_voice_embedding(reference_audio, config)
}

/// Synthesizes speech from text, optionally cloning a voice.
///
/// # Arguments
///
/// * `text` - Text to speak.
/// * `voice` - Optional voice embedding for cloning. None = default voice.
/// * `config` - Synthesis configuration.
pub fn synthesize_speech(
    text: &str,
    voice: Option<&VoiceEmbedding>,
    config: &VoiceCloneConfig,
) -> Result<SynthesizedAudio, TransformError> {
    processor::synthesize(text, voice, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesize_speech() {
        let config = VoiceCloneConfig::default();
        let audio = synthesize_speech("Test speech synthesis.", None, &config).unwrap();
        assert!(!audio.samples.is_empty());
        assert_eq!(audio.sample_rate, 24000);
    }
}
