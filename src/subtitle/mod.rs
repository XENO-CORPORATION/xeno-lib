//! Subtitle parsing, generation, and burn-in functionality.
//!
//! Supports SRT, VTT, and ASS/SSA subtitle formats with rendering
//! capabilities for hardcoding subtitles into video frames.
//!
//! # Features
//!
//! - **SRT parsing/generation**: Standard SubRip format
//! - **VTT parsing/generation**: WebVTT format
//! - **ASS/SSA support**: Advanced SubStation Alpha with styling
//! - **Subtitle burn-in**: Render subtitles onto video frames
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::subtitle::{parse_srt, render_subtitle, SubtitleStyle};
//!
//! let subs = parse_srt(include_str!("subtitles.srt"))?;
//! let style = SubtitleStyle::default();
//!
//! // Get subtitle at specific time
//! if let Some(text) = subs.text_at(10.5) {
//!     let frame = render_subtitle(&frame, &text, &style)?;
//! }
//! ```

mod parser;
mod render;

pub use parser::*;
pub use render::*;

use std::fmt;
use std::path::Path;
use std::fs;

use crate::error::TransformError;

/// A timed subtitle cue.
#[derive(Debug, Clone)]
pub struct SubtitleCue {
    /// Cue index (1-based for SRT).
    pub index: u32,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Subtitle text (may contain newlines).
    pub text: String,
    /// Optional styling (for ASS).
    pub style: Option<String>,
    /// Optional position override.
    pub position: Option<SubtitlePosition>,
}

impl SubtitleCue {
    /// Create a new subtitle cue.
    pub fn new(index: u32, start: f64, end: f64, text: impl Into<String>) -> Self {
        Self {
            index,
            start,
            end,
            text: text.into(),
            style: None,
            position: None,
        }
    }

    /// Check if this cue is active at the given time.
    pub fn is_active_at(&self, time: f64) -> bool {
        time >= self.start && time < self.end
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

/// Subtitle position.
#[derive(Debug, Clone, Copy)]
pub struct SubtitlePosition {
    /// X position (0.0 = left, 1.0 = right).
    pub x: f32,
    /// Y position (0.0 = top, 1.0 = bottom).
    pub y: f32,
    /// Alignment.
    pub align: SubtitleAlign,
}

impl Default for SubtitlePosition {
    fn default() -> Self {
        Self {
            x: 0.5,
            y: 0.9, // Near bottom
            align: SubtitleAlign::Center,
        }
    }
}

/// Subtitle text alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleAlign {
    Left,
    Center,
    Right,
}

/// A collection of subtitle cues.
#[derive(Debug, Clone, Default)]
pub struct Subtitles {
    /// The subtitle cues.
    pub cues: Vec<SubtitleCue>,
    /// Subtitle format.
    pub format: SubtitleFormat,
    /// Optional title/header.
    pub title: Option<String>,
}

impl Subtitles {
    /// Create an empty subtitle collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a cue.
    pub fn add_cue(&mut self, cue: SubtitleCue) {
        self.cues.push(cue);
    }

    /// Get all cues active at a specific time.
    pub fn cues_at(&self, time: f64) -> Vec<&SubtitleCue> {
        self.cues.iter().filter(|c| c.is_active_at(time)).collect()
    }

    /// Get the combined text at a specific time.
    pub fn text_at(&self, time: f64) -> Option<String> {
        let active: Vec<_> = self.cues_at(time);
        if active.is_empty() {
            None
        } else {
            Some(active.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n"))
        }
    }

    /// Total duration (end of last cue).
    pub fn duration(&self) -> f64 {
        self.cues.iter().map(|c| c.end).fold(0.0, f64::max)
    }

    /// Number of cues.
    pub fn len(&self) -> usize {
        self.cues.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.cues.is_empty()
    }

    /// Sort cues by start time.
    pub fn sort(&mut self) {
        self.cues.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
    }

    /// Shift all timings by an offset (can be negative).
    pub fn shift(&mut self, offset: f64) {
        for cue in &mut self.cues {
            cue.start = (cue.start + offset).max(0.0);
            cue.end = (cue.end + offset).max(0.0);
        }
    }

    /// Scale all timings by a factor.
    pub fn scale(&mut self, factor: f64) {
        for cue in &mut self.cues {
            cue.start *= factor;
            cue.end *= factor;
        }
    }

    /// Load from file (auto-detects format).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, TransformError> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|e| TransformError::FileReadFailed {
            message: format!("{}: {}", path.display(), e),
        })?;

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext.to_lowercase().as_str() {
            "srt" => parse_srt(&content),
            "vtt" => parse_vtt(&content),
            "ass" | "ssa" => parse_ass(&content),
            _ => {
                // Try to auto-detect
                if content.starts_with("WEBVTT") {
                    parse_vtt(&content)
                } else if content.contains("[Script Info]") {
                    parse_ass(&content)
                } else {
                    parse_srt(&content)
                }
            }
        }
    }

    /// Save to file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), TransformError> {
        let path = path.as_ref();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let content = match ext.to_lowercase().as_str() {
            "vtt" => self.to_vtt(),
            "ass" | "ssa" => self.to_ass(),
            _ => self.to_srt(),
        };

        fs::write(path, content).map_err(|e| TransformError::FileWriteFailed {
            message: format!("{}: {}", path.display(), e),
        })
    }

    /// Convert to SRT format string.
    pub fn to_srt(&self) -> String {
        let mut output = String::new();
        for (i, cue) in self.cues.iter().enumerate() {
            output.push_str(&format!("{}\n", i + 1));
            output.push_str(&format!(
                "{} --> {}\n",
                format_srt_time(cue.start),
                format_srt_time(cue.end)
            ));
            output.push_str(&cue.text);
            output.push_str("\n\n");
        }
        output
    }

    /// Convert to VTT format string.
    pub fn to_vtt(&self) -> String {
        let mut output = String::from("WEBVTT\n\n");
        for (i, cue) in self.cues.iter().enumerate() {
            output.push_str(&format!("{}\n", i + 1));
            output.push_str(&format!(
                "{} --> {}\n",
                format_vtt_time(cue.start),
                format_vtt_time(cue.end)
            ));
            output.push_str(&cue.text);
            output.push_str("\n\n");
        }
        output
    }

    /// Convert to ASS format string.
    pub fn to_ass(&self) -> String {
        let mut output = String::new();

        // Script info
        output.push_str("[Script Info]\n");
        output.push_str(&format!("Title: {}\n", self.title.as_deref().unwrap_or("Untitled")));
        output.push_str("ScriptType: v4.00+\n");
        output.push_str("WrapStyle: 0\n");
        output.push_str("ScaledBorderAndShadow: yes\n\n");

        // Styles
        output.push_str("[V4+ Styles]\n");
        output.push_str("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n");
        output.push_str("Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1\n\n");

        // Events
        output.push_str("[Events]\n");
        output.push_str("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n");
        for cue in &self.cues {
            let style = cue.style.as_deref().unwrap_or("Default");
            let text = cue.text.replace('\n', "\\N");
            output.push_str(&format!(
                "Dialogue: 0,{},{},{},,,0,0,0,,{}\n",
                format_ass_time(cue.start),
                format_ass_time(cue.end),
                style,
                text
            ));
        }

        output
    }
}

/// Subtitle file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubtitleFormat {
    #[default]
    Srt,
    Vtt,
    Ass,
}

impl fmt::Display for SubtitleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Srt => write!(f, "SRT"),
            Self::Vtt => write!(f, "VTT"),
            Self::Ass => write!(f, "ASS"),
        }
    }
}

/// Format time for SRT (HH:MM:SS,mmm).
pub fn format_srt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;
    format!("{:02}:{:02}:{:02},{:03}", hours, mins, secs, ms)
}

/// Format time for VTT (HH:MM:SS.mmm).
pub fn format_vtt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;
    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, ms)
}

/// Format time for ASS (H:MM:SS.cc).
pub fn format_ass_time(seconds: f64) -> String {
    let total_cs = (seconds * 100.0) as u64;
    let cs = total_cs % 100;
    let total_secs = total_cs / 100;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;
    format!("{}:{:02}:{:02}.{:02}", hours, mins, secs, cs)
}

/// Parse SRT timestamp.
pub fn parse_srt_time(s: &str) -> Option<f64> {
    // Format: HH:MM:SS,mmm or HH:MM:SS.mmm
    let s = s.trim().replace(',', ".");
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return None;
    }

    let hours: f64 = parts[0].parse().ok()?;
    let mins: f64 = parts[1].parse().ok()?;
    let secs: f64 = parts[2].parse().ok()?;

    Some(hours * 3600.0 + mins * 60.0 + secs)
}

/// Parse ASS timestamp.
pub fn parse_ass_time(s: &str) -> Option<f64> {
    // Format: H:MM:SS.cc
    let parts: Vec<&str> = s.trim().split(':').collect();
    if parts.len() != 3 {
        return None;
    }

    let hours: f64 = parts[0].parse().ok()?;
    let mins: f64 = parts[1].parse().ok()?;
    let secs: f64 = parts[2].parse().ok()?;

    Some(hours * 3600.0 + mins * 60.0 + secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subtitle_cue() {
        let cue = SubtitleCue::new(1, 1.0, 5.0, "Hello, World!");
        assert!(cue.is_active_at(2.0));
        assert!(!cue.is_active_at(0.5));
        assert!(!cue.is_active_at(5.5));
        assert_eq!(cue.duration(), 4.0);
    }

    #[test]
    fn test_subtitles_text_at() {
        let mut subs = Subtitles::new();
        subs.add_cue(SubtitleCue::new(1, 1.0, 5.0, "First"));
        subs.add_cue(SubtitleCue::new(2, 10.0, 15.0, "Second"));

        assert_eq!(subs.text_at(2.0), Some("First".to_string()));
        assert_eq!(subs.text_at(12.0), Some("Second".to_string()));
        assert_eq!(subs.text_at(7.0), None);
    }

    #[test]
    fn test_format_srt_time() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(1.5), "00:00:01,500");
        assert_eq!(format_srt_time(3661.123), "01:01:01,123");
    }

    #[test]
    fn test_parse_srt_time() {
        assert!((parse_srt_time("00:00:01,500").unwrap() - 1.5).abs() < 0.001);
        assert!((parse_srt_time("01:01:01,123").unwrap() - 3661.123).abs() < 0.001);
    }

    #[test]
    fn test_format_vtt_time() {
        assert_eq!(format_vtt_time(1.5), "00:00:01.500");
    }

    #[test]
    fn test_format_ass_time() {
        assert_eq!(format_ass_time(1.55), "0:00:01.55");
    }

    #[test]
    fn test_shift_scale() {
        let mut subs = Subtitles::new();
        subs.add_cue(SubtitleCue::new(1, 1.0, 5.0, "Test"));

        subs.shift(1.0);
        assert_eq!(subs.cues[0].start, 2.0);

        subs.scale(2.0);
        assert_eq!(subs.cues[0].start, 4.0);
    }

    #[test]
    fn test_to_srt() {
        let mut subs = Subtitles::new();
        subs.add_cue(SubtitleCue::new(1, 1.0, 5.0, "Hello"));
        let srt = subs.to_srt();
        assert!(srt.contains("00:00:01,000 --> 00:00:05,000"));
        assert!(srt.contains("Hello"));
    }
}
