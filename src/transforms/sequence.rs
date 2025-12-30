use crate::error::TransformError;
use image::DynamicImage;
use std::path::Path;

/// Information about a frame sequence.
#[derive(Debug, Clone)]
pub struct SequenceInfo {
    /// Number of frames in the sequence
    pub frame_count: usize,
    /// Width of frames (assumes all frames are same size)
    pub width: u32,
    /// Height of frames (assumes all frames are same size)
    pub height: u32,
    /// Image format detected
    pub format: String,
    /// First frame number
    pub start_frame: usize,
    /// Last frame number
    pub end_frame: usize,
}

/// Loads a sequence of frames from disk.
/// Pattern example: "frame_%04d.jpg" will load frame_0001.jpg, frame_0002.jpg, etc.
pub fn load_sequence<P: AsRef<Path>>(
    pattern: P,
    start: usize,
    end: usize,
) -> Result<Vec<DynamicImage>, TransformError> {
    (start..=end)
        .map(|i| {
            let path = format_path_pattern(pattern.as_ref(), i);
            image::open(&path).map_err(|_| {
                TransformError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Failed to load frame: {}", path),
                ))
            })
        })
        .collect()
}

/// Saves a sequence of images to disk.
/// Pattern example: "output_%04d.png" will save output_0001.png, output_0002.png, etc.
pub fn save_sequence<P: AsRef<Path>>(
    images: &[DynamicImage],
    pattern: P,
    start_index: usize,
) -> Result<(), TransformError> {
    images
        .iter()
        .enumerate()
        .try_for_each(|(i, img)| {
            let path = format_path_pattern(pattern.as_ref(), start_index + i);
            img.save(&path).map_err(|_| {
                TransformError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to save frame: {}", path),
                ))
            })
        })
}

/// Gets information about a frame sequence.
pub fn sequence_info<P: AsRef<Path>>(
    pattern: P,
    start: usize,
    end: usize,
) -> Result<SequenceInfo, TransformError> {
    let mut frame_count = 0;
    let mut width = 0;
    let mut height = 0;
    let mut format = String::new();
    let mut actual_start = start;
    let mut actual_end = start;

    for i in start..=end {
        let path = format_path_pattern(pattern.as_ref(), i);
        if let Ok(img) = image::open(&path) {
            if frame_count == 0 {
                width = img.width();
                height = img.height();
                format = detect_format(&path);
                actual_start = i;
            }
            actual_end = i;
            frame_count += 1;
        }
    }

    if frame_count == 0 {
        return Err(TransformError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No frames found in sequence",
        )));
    }

    Ok(SequenceInfo {
        frame_count,
        width,
        height,
        format,
        start_frame: actual_start,
        end_frame: actual_end,
    })
}

fn format_path_pattern(pattern: &Path, index: usize) -> String {
    let pattern_str = pattern.to_string_lossy();

    // Support %04d, %05d, etc. format specifiers
    if let Some(pos) = pattern_str.find('%') {
        if let Some(end) = pattern_str[pos..].find('d') {
            let format_spec = &pattern_str[pos..pos + end + 1];
            let width = format_spec
                .chars()
                .filter(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<usize>()
                .unwrap_or(4);

            let formatted = format!("{:0width$}", index, width = width);
            return pattern_str.replace(format_spec, &formatted);
        }
    }

    // Fallback: simple index replacement
    pattern_str.replace("{}", &index.to_string())
}

fn detect_format(path: &str) -> String {
    Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_uppercase())
        .unwrap_or_else(|| "UNKNOWN".to_string())
}

/// Validates a frame sequence for integrity.
/// Checks that all frames exist, have consistent dimensions, and can be loaded.
pub fn validate_sequence<P: AsRef<Path>>(
    pattern: P,
    start: usize,
    end: usize,
) -> Result<(), TransformError> {
    let info = sequence_info(&pattern, start, end)?;

    // Verify all frames are present and valid
    let mut missing_frames = Vec::new();
    let mut dimension_mismatches = Vec::new();

    for i in start..=end {
        let path = format_path_pattern(pattern.as_ref(), i);
        match image::open(&path) {
            Ok(img) => {
                if img.width() != info.width || img.height() != info.height {
                    dimension_mismatches.push((i, img.width(), img.height()));
                }
            }
            Err(_) => {
                missing_frames.push(i);
            }
        }
    }

    if !missing_frames.is_empty() {
        return Err(TransformError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Missing frames: {:?}", missing_frames),
        )));
    }

    if !dimension_mismatches.is_empty() {
        return Err(TransformError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Dimension mismatches (expected {}x{}): {:?}",
                info.width, info.height, dimension_mismatches
            ),
        )));
    }

    Ok(())
}
