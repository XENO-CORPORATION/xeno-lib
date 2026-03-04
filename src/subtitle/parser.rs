//! Subtitle format parsers.

use super::{parse_ass_time, parse_srt_time, SubtitleCue, SubtitleFormat, Subtitles};
use crate::error::TransformError;

/// Parse SRT subtitle file content.
pub fn parse_srt(content: &str) -> Result<Subtitles, TransformError> {
    let mut subtitles = Subtitles::new();
    subtitles.format = SubtitleFormat::Srt;

    let content = content.trim_start_matches('\u{feff}'); // Remove BOM
    let blocks: Vec<&str> = content
        .split("\n\n")
        .filter(|b| !b.trim().is_empty())
        .collect();

    for block in blocks {
        let lines: Vec<&str> = block.lines().collect();
        if lines.len() < 3 {
            continue;
        }

        // Line 0: index
        let index: u32 = lines[0].trim().parse().unwrap_or(0);

        // Line 1: timestamps
        let times: Vec<&str> = lines[1].split("-->").collect();
        if times.len() != 2 {
            continue;
        }

        let start = parse_srt_time(times[0].trim()).ok_or_else(|| TransformError::ParseFailed {
            message: format!("Invalid SRT start time: {}", times[0]),
        })?;
        let end = parse_srt_time(times[1].trim()).ok_or_else(|| TransformError::ParseFailed {
            message: format!("Invalid SRT end time: {}", times[1]),
        })?;

        // Remaining lines: text
        let text = lines[2..].join("\n");

        subtitles.add_cue(SubtitleCue::new(index, start, end, strip_html_tags(&text)));
    }

    Ok(subtitles)
}

/// Parse VTT subtitle file content.
pub fn parse_vtt(content: &str) -> Result<Subtitles, TransformError> {
    let mut subtitles = Subtitles::new();
    subtitles.format = SubtitleFormat::Vtt;

    let content = content.trim_start_matches('\u{feff}'); // Remove BOM

    // Skip WEBVTT header
    let content = if content.starts_with("WEBVTT") {
        content.lines().skip(1).collect::<Vec<_>>().join("\n")
    } else {
        content.to_string()
    };

    let blocks: Vec<&str> = content
        .split("\n\n")
        .filter(|b| !b.trim().is_empty())
        .collect();

    let mut index = 0u32;
    for block in blocks {
        let lines: Vec<&str> = block.lines().collect();
        if lines.is_empty() {
            continue;
        }

        // Find the line with timestamps
        let mut time_line_idx = 0;
        for (i, line) in lines.iter().enumerate() {
            if line.contains("-->") {
                time_line_idx = i;
                break;
            }
        }

        if !lines[time_line_idx].contains("-->") {
            continue;
        }

        // Parse optional cue identifier
        if time_line_idx > 0 {
            // First line is cue ID, ignore it
        }

        // Parse timestamps (may have position settings after)
        let time_parts: Vec<&str> = lines[time_line_idx].split("-->").collect();
        if time_parts.len() != 2 {
            continue;
        }

        // VTT timestamps can have positioning after the end time
        let start_str = time_parts[0].trim();
        let end_and_settings: Vec<&str> = time_parts[1].trim().split_whitespace().collect();
        let end_str = end_and_settings.first().unwrap_or(&"");

        let start = parse_srt_time(start_str).ok_or_else(|| TransformError::ParseFailed {
            message: format!("Invalid VTT start time: {}", start_str),
        })?;
        let end = parse_srt_time(end_str).ok_or_else(|| TransformError::ParseFailed {
            message: format!("Invalid VTT end time: {}", end_str),
        })?;

        // Text is remaining lines
        let text = lines[time_line_idx + 1..].join("\n");

        index += 1;
        subtitles.add_cue(SubtitleCue::new(index, start, end, strip_vtt_tags(&text)));
    }

    Ok(subtitles)
}

/// Parse ASS/SSA subtitle file content.
pub fn parse_ass(content: &str) -> Result<Subtitles, TransformError> {
    let mut subtitles = Subtitles::new();
    subtitles.format = SubtitleFormat::Ass;

    let content = content.trim_start_matches('\u{feff}'); // Remove BOM

    let mut in_events = false;
    let mut format_fields: Vec<&str> = Vec::new();
    let mut index = 0u32;

    for line in content.lines() {
        let line = line.trim();

        // Check for section headers
        if line.starts_with('[') {
            in_events = line.eq_ignore_ascii_case("[Events]");
            continue;
        }

        // Parse title
        if line.to_lowercase().starts_with("title:") {
            subtitles.title = Some(line[6..].trim().to_string());
            continue;
        }

        if !in_events {
            continue;
        }

        // Parse format line
        if line.to_lowercase().starts_with("format:") {
            format_fields = line[7..].split(',').map(|s| s.trim()).collect();
            continue;
        }

        // Parse dialogue lines
        if line.to_lowercase().starts_with("dialogue:") {
            let values_str = &line[9..];
            let values: Vec<&str> = split_ass_values(values_str, format_fields.len());

            if values.len() < format_fields.len() {
                continue;
            }

            // Find field indices
            let start_idx = format_fields
                .iter()
                .position(|&f| f.eq_ignore_ascii_case("Start"));
            let end_idx = format_fields
                .iter()
                .position(|&f| f.eq_ignore_ascii_case("End"));
            let text_idx = format_fields
                .iter()
                .position(|&f| f.eq_ignore_ascii_case("Text"));
            let style_idx = format_fields
                .iter()
                .position(|&f| f.eq_ignore_ascii_case("Style"));

            let start = start_idx
                .and_then(|i| values.get(i))
                .and_then(|s| parse_ass_time(s))
                .unwrap_or(0.0);
            let end = end_idx
                .and_then(|i| values.get(i))
                .and_then(|s| parse_ass_time(s))
                .unwrap_or(0.0);
            let text = text_idx
                .and_then(|i| values.get(i))
                .map(|s| strip_ass_tags(s))
                .unwrap_or_default();
            let style = style_idx.and_then(|i| values.get(i)).map(|s| s.to_string());

            index += 1;
            let mut cue = SubtitleCue::new(index, start, end, text);
            cue.style = style;
            subtitles.add_cue(cue);
        }
    }

    Ok(subtitles)
}

/// Split ASS values (handles commas in text field).
fn split_ass_values(s: &str, num_fields: usize) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut count = 0;

    for (i, c) in s.char_indices() {
        if c == ',' && count < num_fields - 1 {
            result.push(&s[start..i]);
            start = i + 1;
            count += 1;
        }
    }

    // Last field (Text) contains the rest
    if start < s.len() {
        result.push(&s[start..]);
    }

    result
}

/// Strip HTML tags from SRT text.
fn strip_html_tags(s: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;

    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(c),
            _ => {}
        }
    }

    result
}

/// Strip VTT voice/class tags.
fn strip_vtt_tags(s: &str) -> String {
    let result = strip_html_tags(s);

    // Remove voice spans like <v Speaker>
    // Already handled by strip_html_tags

    result
}

/// Strip ASS style tags.
fn strip_ass_tags(s: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if c == '{' {
            in_tag = true;
        } else if c == '}' {
            in_tag = false;
        } else if !in_tag {
            // Handle \N (newline)
            if c == '\\' && i + 1 < chars.len() && (chars[i + 1] == 'N' || chars[i + 1] == 'n') {
                result.push('\n');
                i += 1;
            } else {
                result.push(c);
            }
        }

        i += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_srt() {
        let srt = r#"1
00:00:01,000 --> 00:00:05,000
Hello, World!

2
00:00:06,000 --> 00:00:10,000
Second line
with newline
"#;
        let subs = parse_srt(srt).unwrap();
        assert_eq!(subs.len(), 2);
        assert_eq!(subs.cues[0].text, "Hello, World!");
        assert!((subs.cues[0].start - 1.0).abs() < 0.001);
        assert!(subs.cues[1].text.contains("newline"));
    }

    #[test]
    fn test_parse_vtt() {
        let vtt = r#"WEBVTT

1
00:00:01.000 --> 00:00:05.000
Hello, World!

2
00:00:06.000 --> 00:00:10.000 align:middle
Second cue
"#;
        let subs = parse_vtt(vtt).unwrap();
        assert_eq!(subs.len(), 2);
        assert_eq!(subs.cues[0].text, "Hello, World!");
    }

    #[test]
    fn test_parse_ass() {
        let ass = r#"[Script Info]
Title: Test

[V4+ Styles]
Format: Name, Fontname, Fontsize
Style: Default,Arial,48

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,Hello\NWorld
"#;
        let subs = parse_ass(ass).unwrap();
        assert_eq!(subs.len(), 1);
        assert!(subs.cues[0].text.contains('\n'));
    }

    #[test]
    fn test_strip_html_tags() {
        assert_eq!(strip_html_tags("<b>bold</b>"), "bold");
        assert_eq!(strip_html_tags("<i>italic</i>"), "italic");
        assert_eq!(strip_html_tags("no tags"), "no tags");
    }

    #[test]
    fn test_strip_ass_tags() {
        assert_eq!(strip_ass_tags("{\\b1}bold{\\b0}"), "bold");
        assert_eq!(strip_ass_tags("Line1\\NLine2"), "Line1\nLine2");
    }
}
