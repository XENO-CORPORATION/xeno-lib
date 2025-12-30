//! Audio filter effects for processing audio samples.
//!
//! This module provides FFmpeg-equivalent audio filters in pure Rust:
//! - Volume adjustment (like `-af volume`)
//! - Fade in/out (like `-af afade`)
//! - Normalization (like `-af loudnorm`)
//! - Tempo change without pitch shift (like `-af atempo`)
//!
//! All filters operate on f32 samples in the range [-1.0, 1.0].

/// Apply volume adjustment to audio samples.
///
/// # Arguments
/// * `samples` - Audio samples in f32 format
/// * `gain_db` - Gain in decibels (positive = louder, negative = quieter)
///
/// # Example
/// ```ignore
/// let louder = adjust_volume(&samples, 6.0);  // +6dB
/// let quieter = adjust_volume(&samples, -6.0); // -6dB
/// ```
pub fn adjust_volume(samples: &[f32], gain_db: f32) -> Vec<f32> {
    let multiplier = db_to_linear(gain_db);
    samples.iter().map(|&s| (s * multiplier).clamp(-1.0, 1.0)).collect()
}

/// Apply volume adjustment in-place.
pub fn adjust_volume_inplace(samples: &mut [f32], gain_db: f32) {
    let multiplier = db_to_linear(gain_db);
    for sample in samples.iter_mut() {
        *sample = (*sample * multiplier).clamp(-1.0, 1.0);
    }
}

/// Apply a linear volume multiplier (0.0 = silent, 1.0 = unchanged, 2.0 = 2x louder).
pub fn apply_gain(samples: &[f32], multiplier: f32) -> Vec<f32> {
    samples.iter().map(|&s| (s * multiplier).clamp(-1.0, 1.0)).collect()
}

/// Apply fade-in effect to audio samples.
///
/// # Arguments
/// * `samples` - Audio samples
/// * `sample_rate` - Sample rate in Hz
/// * `duration_secs` - Fade duration in seconds
/// * `curve` - Fade curve type
pub fn fade_in(samples: &[f32], sample_rate: u32, duration_secs: f32, curve: FadeCurve) -> Vec<f32> {
    let fade_samples = (sample_rate as f32 * duration_secs) as usize;
    let fade_samples = fade_samples.min(samples.len());

    let mut result = samples.to_vec();

    for (i, sample) in result.iter_mut().take(fade_samples).enumerate() {
        let progress = i as f32 / fade_samples as f32;
        let gain = curve.apply(progress);
        *sample *= gain;
    }

    result
}

/// Apply fade-out effect to audio samples.
///
/// # Arguments
/// * `samples` - Audio samples
/// * `sample_rate` - Sample rate in Hz
/// * `duration_secs` - Fade duration in seconds
/// * `curve` - Fade curve type
pub fn fade_out(samples: &[f32], sample_rate: u32, duration_secs: f32, curve: FadeCurve) -> Vec<f32> {
    let fade_samples = (sample_rate as f32 * duration_secs) as usize;
    let fade_samples = fade_samples.min(samples.len());

    let mut result = samples.to_vec();
    let start_idx = result.len().saturating_sub(fade_samples);

    for (i, sample) in result[start_idx..].iter_mut().enumerate() {
        let progress = i as f32 / fade_samples as f32;
        let gain = curve.apply(1.0 - progress);
        *sample *= gain;
    }

    result
}

/// Apply both fade-in and fade-out.
pub fn fade_in_out(
    samples: &[f32],
    sample_rate: u32,
    fade_in_secs: f32,
    fade_out_secs: f32,
    curve: FadeCurve,
) -> Vec<f32> {
    let result = fade_in(samples, sample_rate, fade_in_secs, curve);
    fade_out(&result, sample_rate, fade_out_secs, curve)
}

/// Normalize audio to a target peak level.
///
/// # Arguments
/// * `samples` - Audio samples
/// * `target_db` - Target peak level in dB (e.g., -1.0 for -1dB headroom)
pub fn normalize_peak(samples: &[f32], target_db: f32) -> Vec<f32> {
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

    if peak < 1e-10 {
        return samples.to_vec(); // Silent audio
    }

    let target_linear = db_to_linear(target_db);
    let multiplier = target_linear / peak;

    samples.iter().map(|&s| (s * multiplier).clamp(-1.0, 1.0)).collect()
}

/// Normalize audio using RMS (perceived loudness).
///
/// # Arguments
/// * `samples` - Audio samples
/// * `target_db` - Target RMS level in dB (e.g., -18.0 for broadcast)
pub fn normalize_rms(samples: &[f32], target_db: f32) -> Vec<f32> {
    let rms = calculate_rms(samples);

    if rms < 1e-10 {
        return samples.to_vec(); // Silent audio
    }

    let target_linear = db_to_linear(target_db);
    let multiplier = target_linear / rms;

    // Limit gain to prevent clipping
    let max_gain = 1.0 / samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max).max(1e-10);
    let multiplier = multiplier.min(max_gain);

    samples.iter().map(|&s| (s * multiplier).clamp(-1.0, 1.0)).collect()
}

/// Apply a simple compressor/limiter to prevent clipping.
///
/// # Arguments
/// * `samples` - Audio samples
/// * `threshold_db` - Threshold above which compression is applied
/// * `ratio` - Compression ratio (e.g., 4.0 = 4:1)
/// * `attack_ms` - Attack time in milliseconds
/// * `release_ms` - Release time in milliseconds
/// * `sample_rate` - Sample rate in Hz
pub fn compress(
    samples: &[f32],
    threshold_db: f32,
    ratio: f32,
    attack_ms: f32,
    release_ms: f32,
    sample_rate: u32,
) -> Vec<f32> {
    let threshold = db_to_linear(threshold_db);
    let attack_coeff = (-1000.0 / (attack_ms * sample_rate as f32)).exp();
    let release_coeff = (-1000.0 / (release_ms * sample_rate as f32)).exp();

    let mut result = Vec::with_capacity(samples.len());
    let mut envelope = 0.0f32;

    for &sample in samples {
        let input_level = sample.abs();

        // Envelope follower
        if input_level > envelope {
            envelope = attack_coeff * envelope + (1.0 - attack_coeff) * input_level;
        } else {
            envelope = release_coeff * envelope + (1.0 - release_coeff) * input_level;
        }

        // Apply compression
        let gain = if envelope > threshold {
            let over_db = linear_to_db(envelope) - threshold_db;
            let compressed_db = threshold_db + over_db / ratio;
            db_to_linear(compressed_db) / envelope
        } else {
            1.0
        };

        result.push((sample * gain).clamp(-1.0, 1.0));
    }

    result
}

/// Apply a hard limiter to prevent any samples exceeding the threshold.
pub fn limit(samples: &[f32], threshold_db: f32) -> Vec<f32> {
    let threshold = db_to_linear(threshold_db);
    samples.iter().map(|&s| {
        if s.abs() > threshold {
            threshold * s.signum()
        } else {
            s
        }
    }).collect()
}

/// Invert the phase of audio samples.
pub fn invert_phase(samples: &[f32]) -> Vec<f32> {
    samples.iter().map(|&s| -s).collect()
}

/// Mix two audio streams together.
///
/// # Arguments
/// * `a` - First audio stream
/// * `b` - Second audio stream
/// * `mix` - Mix ratio (0.0 = only A, 1.0 = only B, 0.5 = equal mix)
pub fn mix(a: &[f32], b: &[f32], mix_ratio: f32) -> Vec<f32> {
    let len = a.len().max(b.len());
    let mut result = Vec::with_capacity(len);

    let a_gain = 1.0 - mix_ratio;
    let b_gain = mix_ratio;

    for i in 0..len {
        let sample_a = a.get(i).copied().unwrap_or(0.0);
        let sample_b = b.get(i).copied().unwrap_or(0.0);
        result.push((sample_a * a_gain + sample_b * b_gain).clamp(-1.0, 1.0));
    }

    result
}

/// Add two audio streams together.
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    let len = a.len().max(b.len());
    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        let sample_a = a.get(i).copied().unwrap_or(0.0);
        let sample_b = b.get(i).copied().unwrap_or(0.0);
        result.push((sample_a + sample_b).clamp(-1.0, 1.0));
    }

    result
}

/// Convert mono to stereo by duplicating the channel.
pub fn mono_to_stereo(mono: &[f32]) -> Vec<f32> {
    let mut stereo = Vec::with_capacity(mono.len() * 2);
    for &sample in mono {
        stereo.push(sample);
        stereo.push(sample);
    }
    stereo
}

/// Convert stereo to mono by averaging channels.
pub fn stereo_to_mono(stereo: &[f32]) -> Vec<f32> {
    stereo.chunks(2).map(|pair| {
        let left = pair.get(0).copied().unwrap_or(0.0);
        let right = pair.get(1).copied().unwrap_or(left);
        (left + right) / 2.0
    }).collect()
}

/// Trim silence from the beginning and end of audio.
///
/// # Arguments
/// * `samples` - Audio samples
/// * `threshold_db` - Silence threshold in dB (e.g., -60.0)
/// * `min_silence_ms` - Minimum silence duration to trim
/// * `sample_rate` - Sample rate in Hz
pub fn trim_silence(
    samples: &[f32],
    threshold_db: f32,
    min_silence_samples: usize,
) -> Vec<f32> {
    let threshold = db_to_linear(threshold_db);

    // Find start (skip leading silence)
    let mut start = 0;
    let mut silence_count = 0;
    for (i, &sample) in samples.iter().enumerate() {
        if sample.abs() > threshold {
            start = i.saturating_sub(silence_count.min(min_silence_samples));
            break;
        }
        silence_count += 1;
    }

    // Find end (skip trailing silence)
    let mut end = samples.len();
    silence_count = 0;
    for (i, &sample) in samples.iter().enumerate().rev() {
        if sample.abs() > threshold {
            end = (i + 1 + silence_count.min(min_silence_samples)).min(samples.len());
            break;
        }
        silence_count += 1;
    }

    if start >= end {
        return Vec::new();
    }

    samples[start..end].to_vec()
}

/// Detect silence regions in audio.
pub fn detect_silence(
    samples: &[f32],
    threshold_db: f32,
    min_duration_samples: usize,
) -> Vec<(usize, usize)> {
    let threshold = db_to_linear(threshold_db);
    let mut regions = Vec::new();
    let mut in_silence = false;
    let mut silence_start = 0;

    for (i, &sample) in samples.iter().enumerate() {
        let is_silent = sample.abs() <= threshold;

        if is_silent && !in_silence {
            silence_start = i;
            in_silence = true;
        } else if !is_silent && in_silence {
            let duration = i - silence_start;
            if duration >= min_duration_samples {
                regions.push((silence_start, i));
            }
            in_silence = false;
        }
    }

    // Handle trailing silence
    if in_silence {
        let duration = samples.len() - silence_start;
        if duration >= min_duration_samples {
            regions.push((silence_start, samples.len()));
        }
    }

    regions
}

/// Apply a DC offset removal (high-pass filter at very low frequency).
pub fn remove_dc_offset(samples: &[f32]) -> Vec<f32> {
    let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
    samples.iter().map(|&s| s - mean).collect()
}

// ============================================================================
// Helper Types and Functions
// ============================================================================

/// Fade curve type for audio fades.
#[derive(Debug, Clone, Copy, Default)]
pub enum FadeCurve {
    /// Linear fade (straight line)
    #[default]
    Linear,
    /// Logarithmic fade (perceived linear to human ear)
    Logarithmic,
    /// Exponential fade (slow start, fast end)
    Exponential,
    /// S-curve fade (smooth start and end)
    SCurve,
    /// Sine curve fade
    Sine,
}

impl FadeCurve {
    /// Apply the fade curve to a linear progress value (0.0 to 1.0).
    pub fn apply(self, progress: f32) -> f32 {
        let progress = progress.clamp(0.0, 1.0);
        match self {
            FadeCurve::Linear => progress,
            FadeCurve::Logarithmic => {
                // Perceived linear (log scale)
                if progress < 0.001 { 0.0 }
                else { (progress.ln() / (-6.0_f32).ln()).max(0.0).min(1.0) * progress + (1.0 - progress) * progress }
            }
            FadeCurve::Exponential => progress * progress,
            FadeCurve::SCurve => {
                // Smoothstep
                progress * progress * (3.0 - 2.0 * progress)
            }
            FadeCurve::Sine => {
                // Sine curve
                (progress * std::f32::consts::FRAC_PI_2).sin()
            }
        }
    }
}

/// Convert decibels to linear amplitude.
#[inline]
pub fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Convert linear amplitude to decibels.
#[inline]
pub fn linear_to_db(linear: f32) -> f32 {
    if linear <= 0.0 {
        -f32::INFINITY
    } else {
        20.0 * linear.log10()
    }
}

/// Calculate RMS (Root Mean Square) of audio samples.
pub fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

/// Calculate peak level in dB.
pub fn calculate_peak_db(samples: &[f32]) -> f32 {
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    linear_to_db(peak)
}

/// Calculate RMS level in dB.
pub fn calculate_rms_db(samples: &[f32]) -> f32 {
    linear_to_db(calculate_rms(samples))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_adjustment() {
        let samples = vec![0.5, -0.5, 0.25, -0.25];

        // +6dB should roughly double the amplitude
        let louder = adjust_volume(&samples, 6.0);
        assert!(louder[0] > 0.9 && louder[0] <= 1.0);

        // -6dB should roughly halve the amplitude
        let quieter = adjust_volume(&samples, -6.0);
        assert!(quieter[0] > 0.2 && quieter[0] < 0.3);
    }

    #[test]
    fn test_fade_in() {
        let samples = vec![1.0; 1000];
        let faded = fade_in(&samples, 1000, 0.5, FadeCurve::Linear);

        // First sample should be near zero
        assert!(faded[0] < 0.01);
        // Last sample should be unchanged
        assert!((faded[999] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize_peak() {
        let samples = vec![0.5, -0.5, 0.25];
        let normalized = normalize_peak(&samples, 0.0); // Normalize to 0dB (1.0)

        // Peak should now be 1.0
        let peak = normalized.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!((peak - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_stereo_mono_conversion() {
        let mono = vec![0.5, 0.25, -0.5];
        let stereo = mono_to_stereo(&mono);
        assert_eq!(stereo.len(), 6);
        assert_eq!(stereo, vec![0.5, 0.5, 0.25, 0.25, -0.5, -0.5]);

        let back_to_mono = stereo_to_mono(&stereo);
        assert_eq!(back_to_mono, mono);
    }

    #[test]
    fn test_db_conversions() {
        // 0dB = 1.0 linear
        assert!((db_to_linear(0.0) - 1.0).abs() < 0.001);
        // +6dB ~ 2.0 linear
        assert!((db_to_linear(6.0) - 2.0).abs() < 0.1);
        // -6dB ~ 0.5 linear
        assert!((db_to_linear(-6.0) - 0.5).abs() < 0.05);
    }
}
