// TODO: MIGRATE TO XENO-RT — this module belongs in the inference runtime, not the processing library.
// All ONNX Runtime inference (model loading, MusicGen/Riffusion inference, audio generation)
// should move to xeno-rt. The GeneratedMusic struct is a data type that should be shared.
//!
//! AI-powered music generation from text descriptions.
//!
//! Supports MusicGen (Meta) for high-quality text-to-music and Riffusion
//! for creative spectogram-based generation.
//!
//! # Output Contract
//!
//! Audio outputs are f32 PCM samples in [-1.0, 1.0] at the model's native
//! sample rate (32000 Hz for MusicGen, 44100 Hz for Riffusion).
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::music_gen::{generate_music, MusicGenConfig};
//!
//! let config = MusicGenConfig::default().with_duration(15.0);
//! let music = generate_music("upbeat electronic dance music with synths", &config)?;
//! println!("Generated {} seconds of music", music.duration_secs);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod config;
pub mod model;
pub mod processor;

pub use config::{MusicGenConfig, MusicGenModel};
pub use model::{load_music_model, MusicGenSession};

use crate::error::TransformError;

/// Generated music output.
#[derive(Debug, Clone)]
pub struct GeneratedMusic {
    /// f32 PCM samples in [-1.0, 1.0].
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Duration in seconds.
    pub duration_secs: f32,
    /// The text prompt used for generation.
    pub prompt: String,
    /// Number of audio channels (1 = mono, 2 = stereo).
    pub channels: u32,
}

/// Generates music from a text description.
///
/// # Arguments
///
/// * `prompt` - Natural language description of desired music.
/// * `config` - Generation configuration.
pub fn generate_music(
    prompt: &str,
    config: &MusicGenConfig,
) -> Result<GeneratedMusic, TransformError> {
    processor::generate_from_text(prompt, config)
}

/// Generates music conditioned on a melody input.
///
/// Only supported by `MusicGenMelody` model.
pub fn generate_music_with_melody(
    prompt: &str,
    melody: &[f32],
    config: &MusicGenConfig,
) -> Result<GeneratedMusic, TransformError> {
    processor::generate_with_melody(prompt, melody, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_music() {
        let config = MusicGenConfig::default();
        let result = generate_music("chill lofi beats", &config).unwrap();
        assert!(!result.samples.is_empty());
        assert_eq!(result.prompt, "chill lofi beats");
    }
}
