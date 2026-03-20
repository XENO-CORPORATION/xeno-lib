// TODO: MIGRATE TO XENO-RT — this module belongs in the inference runtime, not the processing library.
// All ONNX Runtime inference (model loading, Whisper inference, audio preprocessing) should move
// to xeno-rt. The to_srt() and to_vtt() formatting functions are pure text processing and STAY
// in xeno-lib (or move to a shared subtitle utility). The Transcript/TranscriptSegment structs
// should be shared types.
//!
//! AI-powered speech-to-text transcription using Whisper.
//!
//! This module provides automatic speech recognition (ASR) for
//! converting audio to text with optional timestamps.
//!
//! # Features
//!
//! - Transcribe audio to text
//! - Multiple languages supported
//! - Optional translation to English
//! - Timestamps for subtitles
//! - GPU acceleration via CUDA
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::transcribe::{transcribe, load_transcriber, TranscribeConfig};
//!
//! let config = TranscribeConfig::default();
//! let mut transcriber = load_transcriber(&config)?;
//!
//! // Assuming you have audio samples at 16kHz
//! let samples: Vec<f32> = vec![]; // Load your audio
//! let transcript = transcribe(&samples, 16000, &mut transcriber)?;
//!
//! println!("Text: {}", transcript.text);
//! for seg in &transcript.segments {
//!     println!("[{:.2} - {:.2}] {}", seg.start, seg.end, seg.text);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Model Download
//!
//! Download Whisper ONNX model:
//! - Whisper: [HuggingFace](https://huggingface.co/openai/whisper-base)
//!
//! Default path: `~/.xeno-lib/models/whisper-base.onnx`

mod config;
mod model;
mod processor;

pub use config::{Language, TranscribeConfig, TranscribeModel};
pub use model::{load_transcriber, TranscriberSession};
pub use processor::{Transcript, TranscriptSegment};

use crate::error::TransformError;

/// Transcribes audio samples to text.
///
/// # Arguments
///
/// * `samples` - Audio samples as f32 values (-1.0 to 1.0)
/// * `sample_rate` - Sample rate of the audio (will be resampled to 16kHz)
/// * `session` - A loaded transcriber model session
///
/// # Returns
///
/// A transcript with text and optional timestamps.
pub fn transcribe(
    samples: &[f32],
    sample_rate: u32,
    session: &mut TranscriberSession,
) -> Result<Transcript, TransformError> {
    processor::transcribe_impl(samples, sample_rate, session)
}

/// Quick transcription that loads model and processes in one call.
pub fn transcribe_quick(samples: &[f32], sample_rate: u32) -> Result<Transcript, TransformError> {
    let config = TranscribeConfig::default();
    let mut session = load_transcriber(&config)?;
    transcribe(samples, sample_rate, &mut session)
}

/// Generates SRT subtitle format from transcript.
pub fn to_srt(transcript: &Transcript) -> String {
    let mut srt = String::new();

    for (i, segment) in transcript.segments.iter().enumerate() {
        srt.push_str(&format!("{}\n", i + 1));
        srt.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(segment.start),
            format_srt_time(segment.end)
        ));
        srt.push_str(&segment.text);
        srt.push_str("\n\n");
    }

    srt
}

/// Generates VTT subtitle format from transcript.
pub fn to_vtt(transcript: &Transcript) -> String {
    let mut vtt = String::from("WEBVTT\n\n");

    for segment in &transcript.segments {
        vtt.push_str(&format!(
            "{} --> {}\n",
            format_vtt_time(segment.start),
            format_vtt_time(segment.end)
        ));
        vtt.push_str(&segment.text);
        vtt.push_str("\n\n");
    }

    vtt
}

fn format_srt_time(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0) as u32;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;

    format!("{:02}:{:02}:{:02},{:03}", hours, mins, secs, ms)
}

fn format_vtt_time(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0) as u32;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;

    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = TranscribeConfig::default();
        assert_eq!(config.model, TranscribeModel::WhisperBase);
        assert!(config.use_gpu);
        assert_eq!(config.language, Language::Auto);
    }

    #[test]
    fn test_format_srt_time() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(61.5), "00:01:01,500");
        assert_eq!(format_srt_time(3661.123), "01:01:01,123");
    }

    #[test]
    fn test_format_vtt_time() {
        assert_eq!(format_vtt_time(0.0), "00:00:00.000");
        assert_eq!(format_vtt_time(61.5), "00:01:01.500");
    }

    #[test]
    fn test_to_srt() {
        let transcript = Transcript {
            text: "Hello world".to_string(),
            segments: vec![
                TranscriptSegment { start: 0.0, end: 1.0, text: "Hello".to_string() },
                TranscriptSegment { start: 1.0, end: 2.0, text: "world".to_string() },
            ],
            language: None,
        };

        let srt = to_srt(&transcript);
        assert!(srt.contains("1\n00:00:00,000 --> 00:00:01,000\nHello"));
        assert!(srt.contains("2\n00:00:01,000 --> 00:00:02,000\nworld"));
    }

    #[test]
    fn test_to_vtt() {
        let transcript = Transcript {
            text: "Hello".to_string(),
            segments: vec![
                TranscriptSegment { start: 0.0, end: 1.0, text: "Hello".to_string() },
            ],
            language: None,
        };

        let vtt = to_vtt(&transcript);
        assert!(vtt.starts_with("WEBVTT"));
        assert!(vtt.contains("00:00:00.000 --> 00:00:01.000"));
    }
}
