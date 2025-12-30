//! Face analysis configuration.

use std::path::PathBuf;

/// Face analysis configuration.
#[derive(Debug, Clone)]
pub struct FaceAnalysisConfig {
    /// Path to age estimation model.
    pub age_model_path: Option<PathBuf>,
    /// Path to gender classification model.
    pub gender_model_path: Option<PathBuf>,
    /// Path to emotion recognition model.
    pub emotion_model_path: Option<PathBuf>,
    /// Enable age estimation.
    pub estimate_age: bool,
    /// Enable gender classification.
    pub classify_gender: bool,
    /// Enable emotion recognition.
    pub recognize_emotion: bool,
    /// Minimum face size for analysis.
    pub min_face_size: u32,
    /// Use GPU acceleration.
    pub use_gpu: bool,
    /// GPU device ID.
    pub gpu_device_id: i32,
}

impl Default for FaceAnalysisConfig {
    fn default() -> Self {
        Self {
            age_model_path: None,
            gender_model_path: None,
            emotion_model_path: None,
            estimate_age: true,
            classify_gender: true,
            recognize_emotion: true,
            min_face_size: 48,
            use_gpu: true,
            gpu_device_id: 0,
        }
    }
}

impl FaceAnalysisConfig {
    /// Create a new configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable only age estimation.
    pub fn age_only() -> Self {
        Self {
            estimate_age: true,
            classify_gender: false,
            recognize_emotion: false,
            ..Default::default()
        }
    }

    /// Enable only gender classification.
    pub fn gender_only() -> Self {
        Self {
            estimate_age: false,
            classify_gender: true,
            recognize_emotion: false,
            ..Default::default()
        }
    }

    /// Enable only emotion recognition.
    pub fn emotion_only() -> Self {
        Self {
            estimate_age: false,
            classify_gender: false,
            recognize_emotion: true,
            ..Default::default()
        }
    }

    /// Set minimum face size.
    pub fn with_min_face_size(mut self, size: u32) -> Self {
        self.min_face_size = size.max(16);
        self
    }

    /// Use CPU only.
    pub fn cpu_only(mut self) -> Self {
        self.use_gpu = false;
        self
    }

    /// Get age model path.
    pub fn age_model_path(&self) -> PathBuf {
        self.age_model_path
            .clone()
            .unwrap_or_else(|| default_model_dir().join("age_estimation.onnx"))
    }

    /// Get gender model path.
    pub fn gender_model_path(&self) -> PathBuf {
        self.gender_model_path
            .clone()
            .unwrap_or_else(|| default_model_dir().join("gender_classification.onnx"))
    }

    /// Get emotion model path.
    pub fn emotion_model_path(&self) -> PathBuf {
        self.emotion_model_path
            .clone()
            .unwrap_or_else(|| default_model_dir().join("emotion_recognition.onnx"))
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
        let config = FaceAnalysisConfig::default();
        assert!(config.estimate_age);
        assert!(config.classify_gender);
        assert!(config.recognize_emotion);
    }

    #[test]
    fn test_presets() {
        let age = FaceAnalysisConfig::age_only();
        assert!(age.estimate_age);
        assert!(!age.classify_gender);
        assert!(!age.recognize_emotion);

        let gender = FaceAnalysisConfig::gender_only();
        assert!(!gender.estimate_age);
        assert!(gender.classify_gender);

        let emotion = FaceAnalysisConfig::emotion_only();
        assert!(emotion.recognize_emotion);
        assert!(!emotion.estimate_age);
    }
}
