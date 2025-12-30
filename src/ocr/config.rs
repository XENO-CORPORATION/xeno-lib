//! OCR configuration.

use std::path::PathBuf;

/// OCR model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OcrModel {
    /// PaddleOCR - high accuracy, multilingual.
    #[default]
    PaddleOcr,
    /// CRNN - lightweight scene text recognition.
    Crnn,
    /// EasyOCR-style model.
    EasyOcr,
}

impl OcrModel {
    /// Get model filenames (detection + recognition).
    pub fn model_filenames(&self) -> (&'static str, &'static str) {
        match self {
            Self::PaddleOcr => ("paddle_det.onnx", "paddle_rec.onnx"),
            Self::Crnn => ("crnn_det.onnx", "crnn_rec.onnx"),
            Self::EasyOcr => ("easyocr_det.onnx", "easyocr_rec.onnx"),
        }
    }
}

/// Supported languages for OCR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OcrLanguage {
    #[default]
    English,
    Chinese,
    Japanese,
    Korean,
    Arabic,
    Hindi,
    Spanish,
    French,
    German,
    Russian,
    /// Multiple languages (slower).
    Multilingual,
}

impl OcrLanguage {
    /// Get language code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::English => "en",
            Self::Chinese => "ch",
            Self::Japanese => "ja",
            Self::Korean => "ko",
            Self::Arabic => "ar",
            Self::Hindi => "hi",
            Self::Spanish => "es",
            Self::French => "fr",
            Self::German => "de",
            Self::Russian => "ru",
            Self::Multilingual => "multi",
        }
    }
}

/// OCR configuration.
#[derive(Debug, Clone)]
pub struct OcrConfig {
    /// OCR model to use.
    pub model: OcrModel,
    /// Primary language.
    pub language: OcrLanguage,
    /// Custom model directory.
    pub model_dir: Option<PathBuf>,
    /// Detection confidence threshold.
    pub det_threshold: f32,
    /// Recognition confidence threshold.
    pub rec_threshold: f32,
    /// Use GPU acceleration.
    pub use_gpu: bool,
    /// GPU device ID.
    pub gpu_device_id: i32,
    /// Enable text angle correction.
    pub angle_correction: bool,
    /// Maximum image dimension for processing.
    pub max_dimension: u32,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            model: OcrModel::default(),
            language: OcrLanguage::default(),
            model_dir: None,
            det_threshold: 0.5,
            rec_threshold: 0.5,
            use_gpu: true,
            gpu_device_id: 0,
            angle_correction: true,
            max_dimension: 2048,
        }
    }
}

impl OcrConfig {
    /// Create a new OCR configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model type.
    pub fn with_model(mut self, model: OcrModel) -> Self {
        self.model = model;
        self
    }

    /// Set the language.
    pub fn with_language(mut self, language: OcrLanguage) -> Self {
        self.language = language;
        self
    }

    /// Set detection confidence threshold.
    pub fn with_det_threshold(mut self, threshold: f32) -> Self {
        self.det_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set recognition confidence threshold.
    pub fn with_rec_threshold(mut self, threshold: f32) -> Self {
        self.rec_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Use CPU only.
    pub fn cpu_only(mut self) -> Self {
        self.use_gpu = false;
        self
    }

    /// Enable angle correction.
    pub fn with_angle_correction(mut self, enable: bool) -> Self {
        self.angle_correction = enable;
        self
    }

    /// Get detection model path.
    pub fn det_model_path(&self) -> PathBuf {
        let (det_file, _) = self.model.model_filenames();
        self.model_dir
            .clone()
            .unwrap_or_else(default_model_dir)
            .join(det_file)
    }

    /// Get recognition model path.
    pub fn rec_model_path(&self) -> PathBuf {
        let (_, rec_file) = self.model.model_filenames();
        self.model_dir
            .clone()
            .unwrap_or_else(default_model_dir)
            .join(rec_file)
    }
}

fn default_model_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    home.join(".xeno-lib").join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OcrConfig::default();
        assert_eq!(config.model, OcrModel::PaddleOcr);
        assert_eq!(config.language, OcrLanguage::English);
        assert!(config.use_gpu);
    }

    #[test]
    fn test_config_builder() {
        let config = OcrConfig::new()
            .with_model(OcrModel::Crnn)
            .with_language(OcrLanguage::Japanese)
            .with_det_threshold(0.7)
            .cpu_only();

        assert_eq!(config.model, OcrModel::Crnn);
        assert_eq!(config.language, OcrLanguage::Japanese);
        assert!((config.det_threshold - 0.7).abs() < 0.001);
        assert!(!config.use_gpu);
    }

    #[test]
    fn test_language_codes() {
        assert_eq!(OcrLanguage::English.code(), "en");
        assert_eq!(OcrLanguage::Chinese.code(), "ch");
        assert_eq!(OcrLanguage::Japanese.code(), "ja");
    }
}
