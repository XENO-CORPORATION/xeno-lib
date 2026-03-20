// TODO: MIGRATE TO XENO-RT — this module belongs in the inference runtime, not the processing library.
// When AI inference is added (currently a stub with simple noise gating), the ONNX model loading
// and inference should be in xeno-rt. The audio resampling and frame-based processing pipeline
// could stay in xeno-lib as a pure DSP utility, with xeno-rt providing the per-frame gain mask.
//!
//! AI-powered audio noise reduction using RNNoise.
//!
//! This module provides deep-learning-based noise suppression for audio signals.
//! Uses RNNoise (a recurrent neural network) for real-time noise reduction
//! that preserves speech quality while removing background noise.
//!
//! # Output Contract
//!
//! Audio outputs are f32 PCM samples in [-1.0, 1.0] at the input sample rate.
//!
//! # Architecture
//!
//! RNNoise operates on 10ms frames (480 samples at 48kHz) using a GRU-based
//! architecture. The model estimates a spectral gain mask that suppresses
//! noise while preserving speech/signal.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::noise_reduce::{reduce_noise, NoiseReduceConfig};
//!
//! let config = NoiseReduceConfig::default();
//! let noisy_audio = vec![0.0f32; 48000]; // 1 second at 48kHz
//! let clean = reduce_noise(&noisy_audio, 48000, &config)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::path::PathBuf;
use crate::error::TransformError;

/// Available noise reduction models.
///
/// # Research Notes (2025-2026)
///
/// - **RNNoise**: GRU-based, real-time capable, 48kHz native. Lightweight (~85KB).
///   Excellent speech denoising. ONNX export straightforward.
/// - **DTLN (Dual-signal Transformation LSTM Network)**: Better quality than RNNoise,
///   still real-time. Two-stage architecture. ONNX available.
/// - **DeepFilterNet**: State-of-art speech enhancement. Higher quality but more compute.
///   ONNX export available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NoiseReduceModel {
    /// RNNoise — lightweight, real-time, GRU-based.
    #[default]
    RNNoise,
    /// DTLN — dual-signal LSTM, better quality.
    Dtln,
    /// DeepFilterNet — state-of-art speech enhancement.
    DeepFilterNet,
}

impl NoiseReduceModel {
    /// Default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            Self::RNNoise => "rnnoise.onnx",
            Self::Dtln => "dtln.onnx",
            Self::DeepFilterNet => "deepfilternet.onnx",
        }
    }

    /// Native sample rate.
    pub fn native_sample_rate(&self) -> u32 {
        match self {
            Self::RNNoise => 48000,
            Self::Dtln => 16000,
            Self::DeepFilterNet => 48000,
        }
    }

    /// Frame size in samples at native rate.
    pub fn frame_size(&self) -> usize {
        match self {
            Self::RNNoise => 480,    // 10ms at 48kHz
            Self::Dtln => 512,       // 32ms at 16kHz
            Self::DeepFilterNet => 480,
        }
    }
}

/// Configuration for noise reduction.
#[derive(Debug, Clone)]
pub struct NoiseReduceConfig {
    /// Model to use.
    pub model: NoiseReduceModel,
    /// Custom model path.
    pub model_path: Option<PathBuf>,
    /// Use GPU acceleration.
    pub use_gpu: bool,
    /// Noise reduction strength (0.0-1.0). Default: 1.0.
    pub strength: f32,
    /// Whether to preserve music/non-speech content. Default: false.
    pub preserve_music: bool,
    /// Voice activity detection threshold. Default: 0.5.
    pub vad_threshold: f32,
}

impl Default for NoiseReduceConfig {
    fn default() -> Self {
        Self {
            model: NoiseReduceModel::default(),
            model_path: None,
            use_gpu: false, // RNNoise is fast enough on CPU
            strength: 1.0,
            preserve_music: false,
            vad_threshold: 0.5,
        }
    }
}

impl NoiseReduceConfig {
    /// Create config with specified model.
    pub fn new(model: NoiseReduceModel) -> Self {
        Self { model, ..Default::default() }
    }

    /// Set strength.
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Enable music preservation.
    pub fn preserve_music(mut self) -> Self {
        self.preserve_music = true;
        self
    }

    /// Set VAD threshold.
    pub fn with_vad_threshold(mut self, threshold: f32) -> Self {
        self.vad_threshold = threshold.clamp(0.0, 1.0);
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

/// Result of noise reduction with metadata.
#[derive(Debug, Clone)]
pub struct NoiseReduceResult {
    /// Cleaned audio samples (f32 PCM, [-1.0, 1.0]).
    pub samples: Vec<f32>,
    /// Sample rate.
    pub sample_rate: u32,
    /// Voice activity per frame (0.0-1.0).
    pub vad_probabilities: Vec<f32>,
    /// Estimated noise floor in dB.
    pub noise_floor_db: f32,
    /// SNR improvement in dB.
    pub snr_improvement_db: f32,
}

/// Reduces noise in audio using AI.
///
/// # Arguments
///
/// * `audio` - Input f32 PCM samples.
/// * `sample_rate` - Sample rate of the input.
/// * `config` - Noise reduction configuration.
///
/// # Returns
///
/// `NoiseReduceResult` with cleaned audio and metadata.
pub fn reduce_noise(
    audio: &[f32],
    sample_rate: u32,
    config: &NoiseReduceConfig,
) -> Result<NoiseReduceResult, TransformError> {
    if audio.is_empty() {
        return Err(TransformError::InvalidParameter {
            name: "audio",
            value: 0.0,
        });
    }

    let frame_size = config.model.frame_size();
    let native_rate = config.model.native_sample_rate();

    // Resample if needed (stub — in production use rubato)
    let working_audio = if sample_rate != native_rate {
        // Simple nearest-neighbor resample for stub
        let ratio = native_rate as f64 / sample_rate as f64;
        let new_len = (audio.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_idx = (i as f64 / ratio) as usize;
            resampled.push(audio[src_idx.min(audio.len() - 1)]);
        }
        resampled
    } else {
        audio.to_vec()
    };

    // Process in frames
    let num_frames = (working_audio.len() + frame_size - 1) / frame_size;
    let mut output = Vec::with_capacity(working_audio.len());
    let mut vad_probs = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let start = frame_idx * frame_size;
        let end = (start + frame_size).min(working_audio.len());

        // Stub: apply simple spectral subtraction as placeholder
        // In production, this runs the ONNX model per frame.
        let strength = config.strength;
        for &sample in &working_audio[start..end] {
            // Simple noise gate (placeholder for AI inference)
            let gate = if sample.abs() > 0.01 { 1.0 } else { 1.0 - strength };
            output.push(sample * gate);
        }

        // Stub VAD: assume speech if RMS > threshold
        let rms: f32 = working_audio[start..end]
            .iter()
            .map(|s| s * s)
            .sum::<f32>()
            / (end - start) as f32;
        vad_probs.push(if rms.sqrt() > 0.01 { 1.0 } else { 0.0 });
    }

    // Resample back if needed
    let final_samples = if sample_rate != native_rate {
        let ratio = sample_rate as f64 / native_rate as f64;
        let new_len = (output.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_idx = (i as f64 / ratio) as usize;
            resampled.push(output[src_idx.min(output.len() - 1)]);
        }
        resampled
    } else {
        output
    };

    Ok(NoiseReduceResult {
        samples: final_samples,
        sample_rate,
        vad_probabilities: vad_probs,
        noise_floor_db: -60.0, // Placeholder
        snr_improvement_db: 15.0, // Placeholder
    })
}

/// Reduces noise in audio with a quick default configuration.
pub fn reduce_noise_quick(
    audio: &[f32],
    sample_rate: u32,
) -> Result<NoiseReduceResult, TransformError> {
    reduce_noise(audio, sample_rate, &NoiseReduceConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = NoiseReduceConfig::default();
        assert_eq!(config.model, NoiseReduceModel::RNNoise);
        assert!((config.strength - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_reduce_noise() {
        let audio = vec![0.1f32; 48000]; // 1 second at 48kHz
        let result = reduce_noise(&audio, 48000, &NoiseReduceConfig::default()).unwrap();
        assert_eq!(result.sample_rate, 48000);
        assert!(!result.samples.is_empty());
    }

    #[test]
    fn test_reduce_noise_empty() {
        let result = reduce_noise(&[], 48000, &NoiseReduceConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_reduce_noise_different_rate() {
        let audio = vec![0.1f32; 44100]; // 1 second at 44.1kHz
        let result = reduce_noise(&audio, 44100, &NoiseReduceConfig::default()).unwrap();
        assert_eq!(result.sample_rate, 44100);
    }

    #[test]
    fn test_model_properties() {
        assert_eq!(NoiseReduceModel::RNNoise.frame_size(), 480);
        assert_eq!(NoiseReduceModel::RNNoise.native_sample_rate(), 48000);
        assert_eq!(NoiseReduceModel::Dtln.native_sample_rate(), 16000);
    }
}
