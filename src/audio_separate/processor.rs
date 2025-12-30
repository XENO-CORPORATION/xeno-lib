//! Audio separation processing logic.

use std::collections::HashMap;
use ndarray::Array3;

use crate::error::TransformError;
use super::config::AudioStem;
use super::model::SeparatorSession;

/// Separated audio tracks.
#[derive(Debug, Clone)]
pub struct SeparatedAudio {
    /// Separated stems by type.
    pub stems: HashMap<AudioStem, StereoAudio>,
    /// Original sample rate.
    pub sample_rate: u32,
}

/// Stereo audio data.
#[derive(Debug, Clone)]
pub struct StereoAudio {
    /// Left channel samples.
    pub left: Vec<f32>,
    /// Right channel samples.
    pub right: Vec<f32>,
}

impl StereoAudio {
    /// Creates mono audio by averaging channels.
    pub fn to_mono(&self) -> Vec<f32> {
        self.left
            .iter()
            .zip(self.right.iter())
            .map(|(l, r)| (l + r) / 2.0)
            .collect()
    }

    /// Interleaves left and right channels.
    pub fn to_interleaved(&self) -> Vec<f32> {
        let mut interleaved = Vec::with_capacity(self.left.len() * 2);
        for (l, r) in self.left.iter().zip(self.right.iter()) {
            interleaved.push(*l);
            interleaved.push(*r);
        }
        interleaved
    }

    /// Creates from interleaved samples.
    pub fn from_interleaved(samples: &[f32]) -> Self {
        let mut left = Vec::with_capacity(samples.len() / 2);
        let mut right = Vec::with_capacity(samples.len() / 2);

        for chunk in samples.chunks(2) {
            left.push(chunk[0]);
            right.push(chunk.get(1).copied().unwrap_or(chunk[0]));
        }

        Self { left, right }
    }

    /// Creates from mono samples.
    pub fn from_mono(samples: &[f32]) -> Self {
        Self {
            left: samples.to_vec(),
            right: samples.to_vec(),
        }
    }
}

/// Separates audio into stems.
pub fn separate_impl(
    audio: &StereoAudio,
    sample_rate: u32,
    session: &mut SeparatorSession,
) -> Result<SeparatedAudio, TransformError> {
    let target_rate = session.sample_rate();
    let config = session.config().clone();

    // Resample if necessary
    let resampled = if sample_rate != target_rate {
        StereoAudio {
            left: resample(&audio.left, sample_rate, target_rate),
            right: resample(&audio.right, sample_rate, target_rate),
        }
    } else {
        audio.clone()
    };

    let chunk_size = config.chunk_size;
    let overlap_samples = (chunk_size as f32 * config.overlap) as usize;
    let hop_size = chunk_size - overlap_samples;

    let total_samples = resampled.left.len();
    let num_stems = config.model.stems().len();

    // Initialize output stems
    let mut output_stems: Vec<StereoAudio> = (0..num_stems)
        .map(|_| StereoAudio {
            left: vec![0.0; total_samples],
            right: vec![0.0; total_samples],
        })
        .collect();

    let mut weight_sum = vec![0.0f32; total_samples];

    // Process in chunks with overlap-add
    let mut pos = 0;
    while pos < total_samples {
        let end = (pos + chunk_size).min(total_samples);
        let chunk_len = end - pos;

        // Extract chunk
        let left_chunk: Vec<f32> = resampled.left[pos..end].to_vec();
        let right_chunk: Vec<f32> = resampled.right[pos..end].to_vec();

        // Pad if necessary
        let mut left_padded = left_chunk.clone();
        let mut right_padded = right_chunk.clone();
        if chunk_len < chunk_size {
            left_padded.resize(chunk_size, 0.0);
            right_padded.resize(chunk_size, 0.0);
        }

        // Create input tensor [1, 2, chunk_size]
        let mut input = Array3::<f32>::zeros((1, 2, chunk_size));
        for i in 0..chunk_size {
            input[[0, 0, i]] = left_padded[i];
            input[[0, 1, i]] = right_padded[i];
        }

        // Run inference
        let output = session.run(&input)?;

        // Create window for overlap-add
        let window = create_window(chunk_len);

        // Extract stems from output [1, stems, 2, samples]
        for stem_idx in 0..num_stems.min(output.shape()[1]) {
            for i in 0..chunk_len {
                let w = window[i];
                output_stems[stem_idx].left[pos + i] += output[[0, stem_idx, 0, i]] * w;
                output_stems[stem_idx].right[pos + i] += output[[0, stem_idx, 1, i]] * w;
                if stem_idx == 0 {
                    weight_sum[pos + i] += w;
                }
            }
        }

        pos += hop_size;
    }

    // Normalize by window sum
    for stem in &mut output_stems {
        for i in 0..total_samples {
            if weight_sum[i] > 0.0 {
                stem.left[i] /= weight_sum[i];
                stem.right[i] /= weight_sum[i];
            }
        }
    }

    // Resample back if necessary
    let final_stems: Vec<StereoAudio> = if sample_rate != target_rate {
        output_stems
            .into_iter()
            .map(|s| StereoAudio {
                left: resample(&s.left, target_rate, sample_rate),
                right: resample(&s.right, target_rate, sample_rate),
            })
            .collect()
    } else {
        output_stems
    };

    // Map stems to AudioStem enum
    let model_stems = config.model.stems();
    let mut stems_map = HashMap::new();

    for (i, stem_type) in model_stems.iter().enumerate() {
        if i < final_stems.len() && config.stems.contains(stem_type) {
            stems_map.insert(*stem_type, final_stems[i].clone());
        }
    }

    Ok(SeparatedAudio {
        stems: stems_map,
        sample_rate,
    })
}

/// Creates a Hann window.
fn create_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let t = i as f32 / (size - 1).max(1) as f32;
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * t).cos())
        })
        .collect()
}

/// Resamples audio using linear interpolation.
fn resample(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate {
        return samples.to_vec();
    }

    let ratio = target_rate as f32 / source_rate as f32;
    let new_len = (samples.len() as f32 * ratio) as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f32 / ratio;
        let idx0 = (src_idx.floor() as usize).min(samples.len() - 1);
        let idx1 = (idx0 + 1).min(samples.len() - 1);
        let frac = src_idx - idx0 as f32;

        let sample = samples[idx0] * (1.0 - frac) + samples[idx1] * frac;
        resampled.push(sample);
    }

    resampled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stereo_to_mono() {
        let stereo = StereoAudio {
            left: vec![0.0, 1.0, 0.5],
            right: vec![1.0, 0.0, 0.5],
        };
        let mono = stereo.to_mono();
        assert_eq!(mono, vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_stereo_interleaved_roundtrip() {
        let stereo = StereoAudio {
            left: vec![1.0, 2.0, 3.0],
            right: vec![4.0, 5.0, 6.0],
        };
        let interleaved = stereo.to_interleaved();
        let recovered = StereoAudio::from_interleaved(&interleaved);

        assert_eq!(stereo.left, recovered.left);
        assert_eq!(stereo.right, recovered.right);
    }

    #[test]
    fn test_from_mono() {
        let mono = vec![1.0, 2.0, 3.0];
        let stereo = StereoAudio::from_mono(&mono);
        assert_eq!(stereo.left, mono);
        assert_eq!(stereo.right, mono);
    }

    #[test]
    fn test_create_window() {
        let window = create_window(100);
        assert_eq!(window.len(), 100);

        // Hann window should be 0 at edges and 1 at center
        assert!(window[0] < 0.01);
        assert!(window[99] < 0.01);
        assert!(window[50] > 0.99);
    }

    #[test]
    fn test_resample() {
        let samples = vec![0.0, 1.0, 0.0];
        let upsampled = resample(&samples, 1, 2);
        assert_eq!(upsampled.len(), 6);

        let downsampled = resample(&samples, 2, 1);
        assert_eq!(downsampled.len(), 1);
    }
}
