//! Configuration for speech-to-text transcription.

use std::path::PathBuf;

/// Available transcription models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TranscribeModel {
    /// Whisper Tiny - Fastest, lower quality.
    WhisperTiny,

    /// Whisper Base - Good balance.
    #[default]
    WhisperBase,

    /// Whisper Small - Better quality.
    WhisperSmall,

    /// Whisper Medium - High quality.
    WhisperMedium,

    /// Whisper Large - Best quality.
    WhisperLarge,
}

impl TranscribeModel {
    /// Returns the default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            TranscribeModel::WhisperTiny => "whisper-tiny.onnx",
            TranscribeModel::WhisperBase => "whisper-base.onnx",
            TranscribeModel::WhisperSmall => "whisper-small.onnx",
            TranscribeModel::WhisperMedium => "whisper-medium.onnx",
            TranscribeModel::WhisperLarge => "whisper-large.onnx",
        }
    }

    /// Returns the sample rate expected by the model.
    pub fn sample_rate(&self) -> u32 {
        16000 // All Whisper models use 16kHz
    }

    /// Returns the context window size in samples.
    pub fn context_size(&self) -> usize {
        480000 // 30 seconds at 16kHz
    }
}

/// Supported languages for transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Language {
    /// Auto-detect language.
    #[default]
    Auto,
    English,
    Spanish,
    French,
    German,
    Italian,
    Portuguese,
    Japanese,
    Chinese,
    Korean,
    Russian,
    Arabic,
}

impl Language {
    /// Returns the language code.
    pub fn code(&self) -> Option<&'static str> {
        match self {
            Language::Auto => None,
            Language::English => Some("en"),
            Language::Spanish => Some("es"),
            Language::French => Some("fr"),
            Language::German => Some("de"),
            Language::Italian => Some("it"),
            Language::Portuguese => Some("pt"),
            Language::Japanese => Some("ja"),
            Language::Chinese => Some("zh"),
            Language::Korean => Some("ko"),
            Language::Russian => Some("ru"),
            Language::Arabic => Some("ar"),
        }
    }
}

/// Configuration for transcription.
#[derive(Debug, Clone)]
pub struct TranscribeConfig {
    /// The transcription model to use.
    pub model: TranscribeModel,

    /// Path to the ONNX model file.
    pub model_path: Option<PathBuf>,

    /// Whether to use GPU acceleration.
    pub use_gpu: bool,

    /// CUDA device ID.
    pub gpu_device_id: i32,

    /// Target language for transcription.
    pub language: Language,

    /// Whether to translate to English.
    pub translate: bool,

    /// Whether to include timestamps.
    pub include_timestamps: bool,

    /// Temperature for sampling (0.0 = deterministic).
    pub temperature: f32,
}

impl Default for TranscribeConfig {
    fn default() -> Self {
        Self {
            model: TranscribeModel::default(),
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            language: Language::Auto,
            translate: false,
            include_timestamps: true,
            temperature: 0.0,
        }
    }
}

impl TranscribeConfig {
    /// Create a new configuration with the specified model.
    pub fn new(model: TranscribeModel) -> Self {
        Self {
            model,
            ..Default::default()
        }
    }

    /// Set the model path.
    pub fn with_model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Enable or disable GPU.
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set target language.
    pub fn with_language(mut self, language: Language) -> Self {
        self.language = language;
        self
    }

    /// Enable translation to English.
    pub fn with_translation(mut self, translate: bool) -> Self {
        self.translate = translate;
        self
    }

    /// Get effective model path.
    pub fn effective_model_path(&self) -> PathBuf {
        if let Some(ref path) = self.model_path {
            path.clone()
        } else {
            default_model_path(self.model.default_filename())
        }
    }
}

fn default_model_path(filename: &str) -> PathBuf {
    let home = dirs_next().unwrap_or_else(|| PathBuf::from("."));
    home.join(".xeno-lib").join("models").join(filename)
}

fn dirs_next() -> Option<PathBuf> {
    #[cfg(windows)]
    { std::env::var("USERPROFILE").ok().map(PathBuf::from) }
    #[cfg(not(windows))]
    { std::env::var("HOME").ok().map(PathBuf::from) }
}
