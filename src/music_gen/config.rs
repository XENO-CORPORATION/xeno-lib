//! Configuration for AI music generation.

use std::path::PathBuf;

/// Available music generation models.
///
/// # Model Comparison (2025-2026 Research)
///
/// - **MusicGen** (Meta): Text-to-music, high quality, multiple sizes.
///   Supports melody conditioning. ONNX export via Optimum.
/// - **Riffusion** (Community): Spectogram-based music via Stable Diffusion.
///   Creative but lower quality. ONNX via diffusers export.
///
/// Chosen: MusicGen for quality, Riffusion for creative experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MusicGenModel {
    /// MusicGen Small — 300M params, fast generation.
    #[default]
    MusicGenSmall,
    /// MusicGen Medium — 1.5B params, better quality.
    MusicGenMedium,
    /// MusicGen Large — 3.3B params, highest quality.
    MusicGenLarge,
    /// MusicGen Melody — conditioned on melody input.
    MusicGenMelody,
    /// Riffusion — spectogram-based music via diffusion.
    Riffusion,
}

impl MusicGenModel {
    /// Default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            Self::MusicGenSmall => "musicgen_small.onnx",
            Self::MusicGenMedium => "musicgen_medium.onnx",
            Self::MusicGenLarge => "musicgen_large.onnx",
            Self::MusicGenMelody => "musicgen_melody.onnx",
            Self::Riffusion => "riffusion.onnx",
        }
    }

    /// Display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::MusicGenSmall => "MusicGen Small (Fast)",
            Self::MusicGenMedium => "MusicGen Medium",
            Self::MusicGenLarge => "MusicGen Large (Quality)",
            Self::MusicGenMelody => "MusicGen Melody",
            Self::Riffusion => "Riffusion (Creative)",
        }
    }

    /// Output sample rate.
    pub fn sample_rate(&self) -> u32 {
        match self {
            Self::Riffusion => 44100,
            _ => 32000,
        }
    }

    /// Whether melody conditioning is supported.
    pub fn supports_melody(&self) -> bool {
        matches!(self, Self::MusicGenMelody)
    }
}

/// Configuration for music generation.
#[derive(Debug, Clone)]
pub struct MusicGenConfig {
    /// Model to use.
    pub model: MusicGenModel,
    /// Custom model path.
    pub model_path: Option<PathBuf>,
    /// Use GPU acceleration.
    pub use_gpu: bool,
    /// GPU device ID.
    pub gpu_device_id: i32,
    /// Duration of generated music in seconds. Default: 10.0.
    pub duration_secs: f32,
    /// Temperature for generation. Default: 1.0.
    pub temperature: f32,
    /// Top-k sampling. Default: 250.
    pub top_k: u32,
    /// Top-p (nucleus) sampling. Default: 0.0 (disabled).
    pub top_p: f32,
    /// Classifier-free guidance scale. Default: 3.0.
    pub guidance_scale: f32,
}

impl Default for MusicGenConfig {
    fn default() -> Self {
        Self {
            model: MusicGenModel::default(),
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            duration_secs: 10.0,
            temperature: 1.0,
            top_k: 250,
            top_p: 0.0,
            guidance_scale: 3.0,
        }
    }
}

impl MusicGenConfig {
    /// Create config with specified model.
    pub fn new(model: MusicGenModel) -> Self {
        Self { model, ..Default::default() }
    }

    /// Set model path.
    pub fn with_model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Enable/disable GPU.
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set duration.
    pub fn with_duration(mut self, secs: f32) -> Self {
        self.duration_secs = secs.clamp(1.0, 300.0);
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.clamp(0.0, 2.0);
        self
    }

    /// Get effective model path.
    pub fn effective_model_path(&self) -> PathBuf {
        if let Some(ref path) = self.model_path {
            path.clone()
        } else {
            crate::model_utils::default_model_path(self.model.default_filename())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MusicGenConfig::default();
        assert_eq!(config.model, MusicGenModel::MusicGenSmall);
        assert!((config.duration_secs - 10.0).abs() < 0.01);
        assert!(config.use_gpu);
    }

    #[test]
    fn test_model_sample_rates() {
        assert_eq!(MusicGenModel::MusicGenSmall.sample_rate(), 32000);
        assert_eq!(MusicGenModel::Riffusion.sample_rate(), 44100);
    }

    #[test]
    fn test_duration_clamping() {
        let config = MusicGenConfig::default().with_duration(500.0);
        assert!((config.duration_secs - 300.0).abs() < 0.01);
    }
}
