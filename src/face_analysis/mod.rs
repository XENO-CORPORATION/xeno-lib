//! AI-powered face analysis: age, gender, and emotion detection.
//!
//! Analyze faces to estimate age, determine gender, and recognize emotions.
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::face_analysis::{analyze_face, load_analyzer, FaceAnalysisConfig};
//!
//! let config = FaceAnalysisConfig::default();
//! let mut analyzer = load_analyzer(&config)?;
//!
//! let image = image::open("photo.jpg")?;
//! let results = analyze_face(&image, &mut analyzer)?;
//!
//! for result in &results {
//!     println!("Age: ~{}, Gender: {:?}, Emotion: {:?}",
//!         result.age, result.gender, result.emotion);
//! }
//! ```

mod config;
mod model;
mod processor;

pub use config::*;
pub use model::*;
pub use processor::*;

/// Detected gender.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gender {
    Male,
    Female,
    Unknown,
}

impl Gender {
    pub fn from_score(male_score: f32) -> Self {
        if male_score > 0.6 {
            Self::Male
        } else if male_score < 0.4 {
            Self::Female
        } else {
            Self::Unknown
        }
    }
}

/// Detected emotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emotion {
    Neutral,
    Happy,
    Sad,
    Angry,
    Fearful,
    Disgusted,
    Surprised,
    Unknown,
}

impl Emotion {
    pub fn all() -> &'static [Emotion] {
        &[
            Self::Neutral,
            Self::Happy,
            Self::Sad,
            Self::Angry,
            Self::Fearful,
            Self::Disgusted,
            Self::Surprised,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Neutral => "neutral",
            Self::Happy => "happy",
            Self::Sad => "sad",
            Self::Angry => "angry",
            Self::Fearful => "fearful",
            Self::Disgusted => "disgusted",
            Self::Surprised => "surprised",
            Self::Unknown => "unknown",
        }
    }

    pub fn from_index(idx: usize) -> Self {
        Self::all().get(idx).copied().unwrap_or(Self::Unknown)
    }
}

/// Face analysis result.
#[derive(Debug, Clone)]
pub struct FaceAnalysisResult {
    /// Estimated age.
    pub age: f32,
    /// Age confidence.
    pub age_confidence: f32,
    /// Detected gender.
    pub gender: Gender,
    /// Gender confidence (0-1).
    pub gender_confidence: f32,
    /// Primary emotion.
    pub emotion: Emotion,
    /// Emotion confidence.
    pub emotion_confidence: f32,
    /// All emotion scores.
    pub emotion_scores: Vec<(Emotion, f32)>,
    /// Face bounding box (x, y, width, height).
    pub bbox: (u32, u32, u32, u32),
}

impl FaceAnalysisResult {
    /// Get age range string.
    pub fn age_range(&self) -> String {
        let lower = (self.age.max(0.0) / 5.0).floor() * 5.0;
        let upper = lower + 5.0;
        format!("{}-{}", lower as u32, upper as u32)
    }

    /// Get top N emotions.
    pub fn top_emotions(&self, n: usize) -> Vec<(Emotion, f32)> {
        let mut sorted = self.emotion_scores.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(n).collect()
    }
}
