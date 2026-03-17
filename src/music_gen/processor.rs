//! Music generation processing.

use crate::error::TransformError;
use super::config::MusicGenConfig;
use super::GeneratedMusic;

/// Generates music from a text description.
///
/// # Arguments
///
/// * `prompt` - Text description of desired music (e.g., "upbeat jazz piano").
/// * `config` - Generation configuration.
///
/// # Returns
///
/// `GeneratedMusic` with f32 PCM samples.
pub fn generate_from_text(
    prompt: &str,
    config: &MusicGenConfig,
) -> Result<GeneratedMusic, TransformError> {
    if prompt.is_empty() {
        return Err(TransformError::InvalidParameter {
            name: "prompt",
            value: 0.0,
        });
    }

    // Stub: in production, tokenize prompt, run through MusicGen decoder
    let sample_rate = config.model.sample_rate();
    let num_samples = (config.duration_secs * sample_rate as f32) as usize;
    let samples = vec![0.0f32; num_samples.max(1)];

    Ok(GeneratedMusic {
        samples,
        sample_rate,
        duration_secs: config.duration_secs,
        prompt: prompt.to_string(),
        channels: 1,
    })
}

/// Generates music conditioned on a melody.
///
/// # Arguments
///
/// * `prompt` - Text description of desired style.
/// * `melody` - f32 PCM melody samples for conditioning.
/// * `config` - Generation configuration.
pub fn generate_with_melody(
    prompt: &str,
    melody: &[f32],
    config: &MusicGenConfig,
) -> Result<GeneratedMusic, TransformError> {
    if prompt.is_empty() {
        return Err(TransformError::InvalidParameter {
            name: "prompt",
            value: 0.0,
        });
    }

    if melody.is_empty() {
        return Err(TransformError::InvalidParameter {
            name: "melody",
            value: 0.0,
        });
    }

    if !config.model.supports_melody() {
        return Err(TransformError::InferenceFailed {
            message: format!(
                "Model {} does not support melody conditioning",
                config.model.display_name()
            ),
        });
    }

    let sample_rate = config.model.sample_rate();
    let num_samples = (config.duration_secs * sample_rate as f32) as usize;
    let samples = vec![0.0f32; num_samples.max(1)];

    Ok(GeneratedMusic {
        samples,
        sample_rate,
        duration_secs: config.duration_secs,
        prompt: prompt.to_string(),
        channels: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_from_text() {
        let config = MusicGenConfig::default();
        let result = generate_from_text("upbeat jazz piano", &config).unwrap();
        assert!(!result.samples.is_empty());
        assert_eq!(result.sample_rate, 32000);
    }

    #[test]
    fn test_generate_empty_prompt() {
        let config = MusicGenConfig::default();
        let result = generate_from_text("", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_melody_unsupported_model() {
        let config = MusicGenConfig::default(); // MusicGenSmall doesn't support melody
        let melody = vec![0.0f32; 1000];
        let result = generate_with_melody("jazz", &melody, &config);
        assert!(result.is_err());
    }
}
