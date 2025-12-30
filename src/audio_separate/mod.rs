//! AI-powered audio source separation using Demucs.
//!
//! This module provides music source separation for isolating
//! vocals, instruments, drums, and bass from mixed audio.
//!
//! # Features
//!
//! - Separate vocals from music (karaoke generation)
//! - Extract individual instrument stems
//! - GPU acceleration via CUDA
//! - High-quality overlap-add processing
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::audio_separate::{separate, load_separator, SeparationConfig, AudioStem};
//!
//! let config = SeparationConfig::default().vocals_only();
//! let mut separator = load_separator(&config)?;
//!
//! // Load stereo audio at 44.1kHz
//! let audio = StereoAudio::from_interleaved(&samples);
//! let separated = separate(&audio, 44100, &mut separator)?;
//!
//! let vocals = separated.stems.get(&AudioStem::Vocals).unwrap();
//! // Save or process vocals...
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Model Download
//!
//! Download Demucs ONNX model:
//! - Demucs: [GitHub](https://github.com/facebookresearch/demucs)
//!
//! Default path: `~/.xeno-lib/models/demucs_hybrid.onnx`

mod config;
mod model;
mod processor;

pub use config::{AudioStem, SeparationConfig, SeparationModel};
pub use model::{load_separator, SeparatorSession};
pub use processor::{SeparatedAudio, StereoAudio};

use crate::error::TransformError;

/// Separates audio into individual stems.
///
/// # Arguments
///
/// * `audio` - Stereo audio input
/// * `sample_rate` - Sample rate of the audio (will be resampled if needed)
/// * `session` - A loaded separator model session
///
/// # Returns
///
/// Separated audio with individual stems.
pub fn separate(
    audio: &StereoAudio,
    sample_rate: u32,
    session: &mut SeparatorSession,
) -> Result<SeparatedAudio, TransformError> {
    processor::separate_impl(audio, sample_rate, session)
}

/// Quick separation that loads model and processes in one call.
pub fn separate_quick(
    audio: &StereoAudio,
    sample_rate: u32,
) -> Result<SeparatedAudio, TransformError> {
    let config = SeparationConfig::default();
    let mut session = load_separator(&config)?;
    separate(audio, sample_rate, &mut session)
}

/// Isolates vocals from a mix (convenience function).
pub fn isolate_vocals(
    audio: &StereoAudio,
    sample_rate: u32,
) -> Result<StereoAudio, TransformError> {
    let config = SeparationConfig::default().vocals_only();
    let mut session = load_separator(&config)?;
    let separated = separate(audio, sample_rate, &mut session)?;

    separated
        .stems
        .get(&AudioStem::Vocals)
        .cloned()
        .ok_or_else(|| TransformError::InferenceFailed {
            message: "vocals stem not found in output".to_string(),
        })
}

/// Removes vocals from a mix (creates instrumental).
pub fn remove_vocals(
    audio: &StereoAudio,
    sample_rate: u32,
) -> Result<StereoAudio, TransformError> {
    let config = SeparationConfig::default().instrumental_only();
    let mut session = load_separator(&config)?;
    let separated = separate(audio, sample_rate, &mut session)?;

    separated
        .stems
        .get(&AudioStem::Instrumental)
        .cloned()
        .ok_or_else(|| TransformError::InferenceFailed {
            message: "instrumental stem not found in output".to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SeparationConfig::default();
        assert_eq!(config.model, SeparationModel::DemucsHybrid);
        assert!(config.use_gpu);
    }

    #[test]
    fn test_config_vocals_only() {
        let config = SeparationConfig::default().vocals_only();
        assert_eq!(config.stems, vec![AudioStem::Vocals]);
    }

    #[test]
    fn test_config_instrumental_only() {
        let config = SeparationConfig::default().instrumental_only();
        assert_eq!(config.stems, vec![AudioStem::Instrumental]);
    }

    #[test]
    fn test_audio_stem_name() {
        assert_eq!(AudioStem::Vocals.name(), "vocals");
        assert_eq!(AudioStem::Drums.name(), "drums");
        assert_eq!(AudioStem::Bass.name(), "bass");
    }
}
