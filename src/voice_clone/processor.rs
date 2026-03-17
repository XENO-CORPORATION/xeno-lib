//! Audio processing for voice cloning and TTS synthesis.

use crate::error::TransformError;
use super::config::VoiceCloneConfig;
use super::{SynthesizedAudio, VoiceEmbedding};

/// Extracts a voice embedding from reference audio.
///
/// The embedding captures speaker characteristics (timbre, pitch, speaking style)
/// and can be used to synthesize new speech in that voice.
///
/// # Arguments
///
/// * `reference_audio` - f32 PCM samples at the model's expected sample rate.
/// * `_config` - Voice cloning configuration.
///
/// # Returns
///
/// A `VoiceEmbedding` that can be passed to `synthesize_speech`.
pub fn extract_voice_embedding(
    reference_audio: &[f32],
    _config: &VoiceCloneConfig,
) -> Result<VoiceEmbedding, TransformError> {
    if reference_audio.is_empty() {
        return Err(TransformError::InvalidParameter {
            name: "reference_audio",
            value: 0.0,
        });
    }

    // Minimum 3 seconds of reference audio for quality cloning
    let min_samples = _config.sample_rate as usize * 3;
    if reference_audio.len() < min_samples {
        return Err(TransformError::InvalidParameter {
            name: "reference_audio_length",
            value: reference_audio.len() as f32,
        });
    }

    // Stub: in production, this runs the encoder portion of the model
    // to extract speaker embeddings (typically 256-512 dimensional vector).
    let embedding_dim = 512;
    let embedding = vec![0.0f32; embedding_dim];

    Ok(VoiceEmbedding {
        data: embedding,
        sample_rate: _config.sample_rate,
        language: _config.language.clone(),
    })
}

/// Synthesizes speech from text using a voice embedding.
///
/// # Arguments
///
/// * `text` - Text to synthesize.
/// * `_embedding` - Optional voice embedding for cloning. None = default voice.
/// * `config` - Synthesis configuration.
///
/// # Returns
///
/// `SynthesizedAudio` with f32 PCM samples.
pub fn synthesize(
    text: &str,
    _embedding: Option<&VoiceEmbedding>,
    config: &VoiceCloneConfig,
) -> Result<SynthesizedAudio, TransformError> {
    if text.is_empty() {
        return Err(TransformError::InvalidParameter {
            name: "text",
            value: 0.0,
        });
    }

    // Stub: in production, this runs the decoder with text tokens + voice embedding.
    // Estimate output length: ~150 words/minute, ~5 chars/word
    let chars = text.len() as f32;
    let words = chars / 5.0;
    let duration_secs = (words / 150.0) * 60.0 / config.speed;
    let num_samples = (duration_secs * config.sample_rate as f32) as usize;

    // Generate silence as placeholder
    let samples = vec![0.0f32; num_samples.max(1)];

    Ok(SynthesizedAudio {
        samples,
        sample_rate: config.sample_rate,
        duration_secs,
        language: config.language.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_embedding_empty_audio() {
        let config = VoiceCloneConfig::default();
        let result = extract_voice_embedding(&[], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_embedding_too_short() {
        let config = VoiceCloneConfig::default();
        let short_audio = vec![0.0f32; 100];
        let result = extract_voice_embedding(&short_audio, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_synthesize_empty_text() {
        let config = VoiceCloneConfig::default();
        let result = synthesize("", None, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_synthesize_produces_audio() {
        let config = VoiceCloneConfig::default();
        let result = synthesize("Hello world, this is a test.", None, &config).unwrap();
        assert!(!result.samples.is_empty());
        assert_eq!(result.sample_rate, 24000);
        assert!(result.duration_secs > 0.0);
    }
}
