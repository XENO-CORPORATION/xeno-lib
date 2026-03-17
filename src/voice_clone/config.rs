//! Configuration for voice cloning / text-to-speech synthesis.

use std::path::PathBuf;

/// Available voice cloning / TTS models.
///
/// # Model Comparison (2025-2026 Research)
///
/// - **XTTS v2** (Coqui): Best quality voice cloning from ~6s reference audio.
///   Multilingual, real-time capable. ONNX export available.
/// - **Bark** (Suno): Text-to-audio including music and effects.
///   Good for creative sound design. ONNX export supported.
/// - **Tortoise TTS**: Highest quality but very slow (~30s per sentence).
///   Best for offline batch processing. ONNX export experimental.
///
/// Chosen approach: XTTS for real-time voice cloning, Bark for creative audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VoiceCloneModel {
    /// XTTS v2 — real-time voice cloning from reference audio.
    #[default]
    XttsV2,

    /// Bark — text-to-audio with voice presets, music, effects.
    Bark,

    /// Tortoise TTS — highest quality, slow generation.
    TortoiseTts,
}

impl VoiceCloneModel {
    /// Returns the default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            Self::XttsV2 => "xtts_v2.onnx",
            Self::Bark => "bark.onnx",
            Self::TortoiseTts => "tortoise_tts.onnx",
        }
    }

    /// Returns human-readable name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::XttsV2 => "XTTS v2 (Real-time)",
            Self::Bark => "Bark (Creative Audio)",
            Self::TortoiseTts => "Tortoise TTS (Quality)",
        }
    }

    /// Whether this model supports voice cloning from reference.
    pub fn supports_cloning(&self) -> bool {
        match self {
            Self::XttsV2 => true,
            Self::Bark => true,
            Self::TortoiseTts => true,
        }
    }

    /// Approximate real-time factor (1.0 = real-time).
    pub fn realtime_factor(&self) -> f32 {
        match self {
            Self::XttsV2 => 1.0,
            Self::Bark => 2.0,
            Self::TortoiseTts => 15.0,
        }
    }
}

/// Configuration for voice synthesis.
#[derive(Debug, Clone)]
pub struct VoiceCloneConfig {
    /// Model to use.
    pub model: VoiceCloneModel,
    /// Custom model path.
    pub model_path: Option<PathBuf>,
    /// Use GPU acceleration.
    pub use_gpu: bool,
    /// GPU device ID.
    pub gpu_device_id: i32,
    /// Output sample rate in Hz. Default: 24000 (XTTS native).
    pub sample_rate: u32,
    /// Language code (e.g., "en", "es", "fr", "de", "ja").
    pub language: String,
    /// Speech speed multiplier (0.5 = half speed, 2.0 = double). Default: 1.0.
    pub speed: f32,
    /// Temperature for generation randomness. Default: 0.7.
    pub temperature: f32,
    /// Top-k sampling parameter. Default: 50.
    pub top_k: u32,
    /// Repetition penalty. Default: 2.0.
    pub repetition_penalty: f32,
}

impl Default for VoiceCloneConfig {
    fn default() -> Self {
        Self {
            model: VoiceCloneModel::default(),
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            sample_rate: 24000,
            language: "en".to_string(),
            speed: 1.0,
            temperature: 0.7,
            top_k: 50,
            repetition_penalty: 2.0,
        }
    }
}

impl VoiceCloneConfig {
    /// Create config with specified model.
    pub fn new(model: VoiceCloneModel) -> Self {
        Self {
            model,
            ..Default::default()
        }
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

    /// Set language.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    /// Set speech speed.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed.clamp(0.25, 4.0);
        self
    }

    /// Set generation temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.clamp(0.0, 2.0);
        self
    }

    /// Set output sample rate.
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
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
        let config = VoiceCloneConfig::default();
        assert_eq!(config.model, VoiceCloneModel::XttsV2);
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.language, "en");
        assert!(config.use_gpu);
    }

    #[test]
    fn test_model_properties() {
        assert!(VoiceCloneModel::XttsV2.supports_cloning());
        assert!((VoiceCloneModel::XttsV2.realtime_factor() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_speed_clamping() {
        let config = VoiceCloneConfig::default().with_speed(10.0);
        assert!((config.speed - 4.0).abs() < 0.01);

        let config = VoiceCloneConfig::default().with_speed(0.1);
        assert!((config.speed - 0.25).abs() < 0.01);
    }
}
