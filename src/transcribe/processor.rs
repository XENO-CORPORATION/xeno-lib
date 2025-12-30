//! Transcription processing logic.

use ndarray::Array3;

use crate::error::TransformError;
use super::model::TranscriberSession;

/// A transcription segment with timestamp.
#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
    /// Transcribed text.
    pub text: String,
}

/// Complete transcription result.
#[derive(Debug, Clone)]
pub struct Transcript {
    /// Full transcribed text.
    pub text: String,
    /// Individual segments with timestamps.
    pub segments: Vec<TranscriptSegment>,
    /// Detected language (if auto-detect was used).
    pub language: Option<String>,
}

/// Transcribes audio samples.
pub fn transcribe_impl(
    samples: &[f32],
    sample_rate: u32,
    session: &mut TranscriberSession,
) -> Result<Transcript, TransformError> {
    // Resample if necessary
    let target_rate = session.sample_rate();
    let resampled = if sample_rate != target_rate {
        resample_audio(samples, sample_rate, target_rate)
    } else {
        samples.to_vec()
    };

    // Convert to mel spectrogram
    let mel = audio_to_mel(&resampled, target_rate)?;

    // Run transcription
    let token_ids = session.run(&mel)?;

    // Decode tokens to text
    let (text, segments) = decode_tokens(&token_ids, session.config().include_timestamps);

    Ok(Transcript {
        text,
        segments,
        language: None, // Would be detected by model
    })
}

/// Converts audio samples to mel spectrogram.
fn audio_to_mel(samples: &[f32], sample_rate: u32) -> Result<Array3<f32>, TransformError> {
    // Whisper uses 80 mel bins and 30-second windows
    let n_mels = 80;
    let n_fft = 400;
    let hop_length = 160;
    let _sample_rate = sample_rate as usize;

    // Pad or trim to 30 seconds
    let target_length = 30 * sample_rate as usize;
    let padded: Vec<f32> = if samples.len() >= target_length {
        samples[..target_length].to_vec()
    } else {
        let mut p = samples.to_vec();
        p.resize(target_length, 0.0);
        p
    };

    // Calculate number of frames
    let n_frames = (target_length - n_fft) / hop_length + 1;

    // Create mel spectrogram (simplified version)
    // In production, you'd use a proper FFT library
    let mut mel = Array3::<f32>::zeros((1, n_mels, n_frames));

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        let end = (start + n_fft).min(padded.len());

        // Calculate frame energy (simplified - real implementation uses FFT)
        let frame_energy: f32 = padded[start..end]
            .iter()
            .map(|s| s * s)
            .sum::<f32>()
            .sqrt();

        // Distribute across mel bins (simplified)
        for mel_idx in 0..n_mels {
            let mel_weight = ((mel_idx as f32 / n_mels as f32) * std::f32::consts::PI).sin();
            mel[[0, mel_idx, frame_idx]] = (frame_energy * mel_weight + 1e-10).log10();
        }
    }

    // Normalize
    let mean = mel.mean().unwrap_or(0.0);
    let std = mel.std(0.0);
    if std > 0.0 {
        mel.mapv_inplace(|v| (v - mean) / std);
    }

    Ok(mel)
}

/// Resamples audio to target sample rate.
fn resample_audio(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
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

/// Decodes token IDs to text.
fn decode_tokens(tokens: &[i64], _include_timestamps: bool) -> (String, Vec<TranscriptSegment>) {
    // Simplified token decoding - real implementation needs vocabulary
    // This is a placeholder that would be replaced with actual vocabulary lookup

    let mut text = String::new();
    let mut segments = Vec::new();
    let mut current_segment = String::new();
    let mut segment_start = 0.0f32;

    // Special tokens in Whisper vocabulary
    const TOKEN_EOT: i64 = 50257; // End of text
    const TOKEN_TIMESTAMP_BASE: i64 = 50364; // Timestamp tokens start here

    for &token in tokens {
        if token == TOKEN_EOT {
            break;
        }

        if token >= TOKEN_TIMESTAMP_BASE {
            // This is a timestamp token
            let time = (token - TOKEN_TIMESTAMP_BASE) as f32 * 0.02; // 20ms per timestamp

            if !current_segment.is_empty() {
                segments.push(TranscriptSegment {
                    start: segment_start,
                    end: time,
                    text: current_segment.trim().to_string(),
                });
                text.push_str(&current_segment);
                current_segment.clear();
            }
            segment_start = time;
        } else if token > 0 && token < 50257 {
            // Regular token - would look up in vocabulary
            // Placeholder: just note the token ID
            current_segment.push_str(&format!("[{}]", token));
        }
    }

    if !current_segment.is_empty() {
        text.push_str(&current_segment);
        segments.push(TranscriptSegment {
            start: segment_start,
            end: segment_start + 1.0,
            text: current_segment.trim().to_string(),
        });
    }

    (text.trim().to_string(), segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_same_rate() {
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let resampled = resample_audio(&samples, 16000, 16000);
        assert_eq!(resampled.len(), samples.len());
    }

    #[test]
    fn test_resample_upsample() {
        let samples = vec![0.0, 1.0];
        let resampled = resample_audio(&samples, 8000, 16000);
        assert_eq!(resampled.len(), 4);
    }

    #[test]
    fn test_resample_downsample() {
        let samples = vec![0.0, 0.5, 1.0, 0.5];
        let resampled = resample_audio(&samples, 16000, 8000);
        assert_eq!(resampled.len(), 2);
    }

    #[test]
    fn test_audio_to_mel_shape() {
        let samples = vec![0.0; 16000 * 30]; // 30 seconds at 16kHz
        let mel = audio_to_mel(&samples, 16000).unwrap();
        assert_eq!(mel.shape()[0], 1); // Batch size
        assert_eq!(mel.shape()[1], 80); // Mel bins
    }
}
