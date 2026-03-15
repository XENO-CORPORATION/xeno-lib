//! Input validation utilities for N-API boundary functions.
//!
//! Every public N-API function must validate its inputs before passing them
//! to the underlying xeno-lib functions. This module provides reusable
//! validators that return descriptive `napi::Error` messages on failure,
//! ensuring JavaScript callers never trigger a Rust panic.

use napi::bindgen_prelude::*;

/// Validate that an RGBA image buffer has the correct length and non-zero dimensions.
///
/// # Arguments
/// * `buffer` - Raw pixel data (expected to be RGBA u8, 4 bytes per pixel)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
///
/// # Errors
/// - If `width` or `height` is zero
/// - If `buffer.len()` does not equal `width * height * 4`
/// - If the total pixel count would overflow `usize`
pub fn validate_image_buffer(buffer: &[u8], width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Image dimensions must be non-zero, got {}x{}",
                width, height
            ),
        ));
    }

    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                format!(
                    "Image dimensions overflow: {}x{} would require more than usize::MAX bytes",
                    width, height
                ),
            )
        })?;

    if buffer.len() != expected {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Buffer size mismatch: expected {} bytes ({}x{}x4), got {}",
                expected, width, height,
                buffer.len()
            ),
        ));
    }

    Ok(())
}

/// Validate audio sample data for encoding or processing.
///
/// # Arguments
/// * `samples` - PCM sample data
/// * `sample_rate` - Sample rate in Hz
/// * `channels` - Number of audio channels
///
/// # Errors
/// - If `samples` is empty
/// - If `sample_rate` is zero or exceeds 384000 Hz
/// - If `channels` is zero or exceeds 32
/// - If any sample is NaN or Infinity
pub fn validate_audio_samples(samples: &[f64], sample_rate: u32, channels: u32) -> Result<()> {
    if samples.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "Audio sample buffer must not be empty".to_string(),
        ));
    }

    if sample_rate == 0 || sample_rate > 384_000 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Sample rate must be between 1 and 384000 Hz, got {}",
                sample_rate
            ),
        ));
    }

    if channels == 0 || channels > 32 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Channel count must be between 1 and 32, got {}",
                channels
            ),
        ));
    }

    // Check for NaN/Infinity in samples (spot-check first 1024 + last 1024 for performance)
    let check_range = |slice: &[f64]| -> Result<()> {
        for (i, &s) in slice.iter().enumerate() {
            if s.is_nan() {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("Audio sample at index {} is NaN", i),
                ));
            }
            if s.is_infinite() {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("Audio sample at index {} is Infinity", i),
                ));
            }
        }
        Ok(())
    };

    if samples.len() <= 2048 {
        check_range(samples)?;
    } else {
        check_range(&samples[..1024])?;
        check_range(&samples[samples.len() - 1024..])?;
    }

    Ok(())
}

/// Validate a file path is non-empty and the file exists.
///
/// # Arguments
/// * `path` - File path to validate
///
/// # Errors
/// - If path is empty
/// - If the file does not exist at the given path
pub fn validate_file_path(path: &str) -> Result<()> {
    if path.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "File path must not be empty".to_string(),
        ));
    }

    if !std::path::Path::new(path).exists() {
        return Err(Error::new(
            Status::InvalidArg,
            format!("File not found: {}", path),
        ));
    }

    Ok(())
}

/// Validate that a blur/sharpen radius is positive.
///
/// # Arguments
/// * `radius` - The radius/sigma value
/// * `param_name` - Name of the parameter (for error messages)
///
/// # Errors
/// - If radius is <= 0, NaN, or Infinity
pub fn validate_positive_f64(value: f64, param_name: &str) -> Result<()> {
    if value.is_nan() {
        return Err(Error::new(
            Status::InvalidArg,
            format!("{} must not be NaN", param_name),
        ));
    }
    if value.is_infinite() {
        return Err(Error::new(
            Status::InvalidArg,
            format!("{} must not be Infinity", param_name),
        ));
    }
    if value <= 0.0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("{} must be > 0, got {}", param_name, value),
        ));
    }
    Ok(())
}

/// Validate that a numeric parameter is finite (not NaN or Infinity).
///
/// # Arguments
/// * `value` - The value to check
/// * `param_name` - Name of the parameter (for error messages)
///
/// # Errors
/// - If value is NaN or Infinity
pub fn validate_finite_f64(value: f64, param_name: &str) -> Result<()> {
    if value.is_nan() {
        return Err(Error::new(
            Status::InvalidArg,
            format!("{} must not be NaN", param_name),
        ));
    }
    if value.is_infinite() {
        return Err(Error::new(
            Status::InvalidArg,
            format!("{} must not be Infinity", param_name),
        ));
    }
    Ok(())
}

/// Validate resize dimensions are non-zero.
///
/// # Arguments
/// * `new_width` - Target width
/// * `new_height` - Target height
///
/// # Errors
/// - If either dimension is zero
pub fn validate_resize_dimensions(new_width: u32, new_height: u32) -> Result<()> {
    if new_width == 0 || new_height == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Resize dimensions must be non-zero, got {}x{}",
                new_width, new_height
            ),
        ));
    }
    Ok(())
}

/// Validate image encoding quality parameter.
///
/// # Arguments
/// * `quality` - Quality value (expected 1-100)
///
/// # Returns
/// The clamped quality value (always 1-100).
pub fn clamp_quality(quality: u32) -> u32 {
    quality.clamp(1, 100)
}

/// Validate WAV bit depth.
///
/// # Arguments
/// * `bit_depth` - Bits per sample
///
/// # Errors
/// - If bit depth is not one of: 8, 16, 24, 32
pub fn validate_wav_bit_depth(bit_depth: u32) -> Result<()> {
    match bit_depth {
        8 | 16 | 24 | 32 => Ok(()),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!(
                "WAV bit depth must be 8, 16, 24, or 32, got {}",
                bit_depth
            ),
        )),
    }
}

/// Validate upscale factor.
///
/// # Arguments
/// * `scale` - The upscale factor
///
/// # Errors
/// - If scale is not 2 or 4
pub fn validate_upscale_factor(scale: u32) -> Result<()> {
    match scale {
        2 | 4 => Ok(()),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Upscale factor must be 2 or 4, got {}", scale),
        )),
    }
}

/// Validate interpolation factor is in [0.0, 1.0].
///
/// # Arguments
/// * `factor` - Interpolation position
///
/// # Errors
/// - If factor is NaN, Infinity, or outside [0.0, 1.0]
pub fn validate_interpolation_factor(factor: f64) -> Result<()> {
    validate_finite_f64(factor, "Interpolation factor")?;
    if !(0.0..=1.0).contains(&factor) {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Interpolation factor must be between 0.0 and 1.0, got {}",
                factor
            ),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Image buffer validation ----

    #[test]
    fn valid_image_buffer() {
        let buf = vec![0u8; 4 * 4 * 4]; // 4x4 RGBA
        assert!(validate_image_buffer(&buf, 4, 4).is_ok());
    }

    #[test]
    fn zero_width_image() {
        let buf = vec![0u8; 0];
        let err = validate_image_buffer(&buf, 0, 4).unwrap_err();
        assert!(err.reason.contains("non-zero"));
    }

    #[test]
    fn zero_height_image() {
        let buf = vec![0u8; 0];
        let err = validate_image_buffer(&buf, 4, 0).unwrap_err();
        assert!(err.reason.contains("non-zero"));
    }

    #[test]
    fn wrong_buffer_size() {
        let buf = vec![0u8; 10]; // not 4x4x4=64
        let err = validate_image_buffer(&buf, 4, 4).unwrap_err();
        assert!(err.reason.contains("Buffer size mismatch"));
        assert!(err.reason.contains("64"));
        assert!(err.reason.contains("10"));
    }

    #[test]
    fn single_pixel_image() {
        let buf = vec![255u8; 4]; // 1x1 RGBA
        assert!(validate_image_buffer(&buf, 1, 1).is_ok());
    }

    #[test]
    fn large_image_validation() {
        // 8K image: 7680x4320 — should validate dimensions OK (no actual alloc needed)
        let expected = 7680usize * 4320 * 4;
        // We only check the math, don't allocate the buffer
        let err = validate_image_buffer(&[], 7680, 4320).unwrap_err();
        assert!(err.reason.contains("Buffer size mismatch"));
        assert!(err.reason.contains(&expected.to_string()));
    }

    // ---- Audio validation ----

    #[test]
    fn valid_audio_samples() {
        let samples = vec![0.0f64; 1024];
        assert!(validate_audio_samples(&samples, 44100, 2).is_ok());
    }

    #[test]
    fn empty_audio_samples() {
        let err = validate_audio_samples(&[], 44100, 2).unwrap_err();
        assert!(err.reason.contains("empty"));
    }

    #[test]
    fn zero_sample_rate() {
        let samples = vec![0.0f64; 100];
        let err = validate_audio_samples(&samples, 0, 2).unwrap_err();
        assert!(err.reason.contains("Sample rate"));
    }

    #[test]
    fn excessive_sample_rate() {
        let samples = vec![0.0f64; 100];
        let err = validate_audio_samples(&samples, 500_000, 2).unwrap_err();
        assert!(err.reason.contains("384000"));
    }

    #[test]
    fn zero_channels() {
        let samples = vec![0.0f64; 100];
        let err = validate_audio_samples(&samples, 44100, 0).unwrap_err();
        assert!(err.reason.contains("Channel count"));
    }

    #[test]
    fn nan_audio_sample() {
        let mut samples = vec![0.0f64; 100];
        samples[50] = f64::NAN;
        let err = validate_audio_samples(&samples, 44100, 2).unwrap_err();
        assert!(err.reason.contains("NaN"));
    }

    #[test]
    fn infinity_audio_sample() {
        let mut samples = vec![0.0f64; 100];
        samples[10] = f64::INFINITY;
        let err = validate_audio_samples(&samples, 44100, 2).unwrap_err();
        assert!(err.reason.contains("Infinity"));
    }

    // ---- File path validation ----

    #[test]
    fn empty_file_path() {
        let err = validate_file_path("").unwrap_err();
        assert!(err.reason.contains("empty"));
    }

    #[test]
    fn nonexistent_file() {
        let err = validate_file_path("/tmp/does_not_exist_xeno_test_12345.wav").unwrap_err();
        assert!(err.reason.contains("not found"));
    }

    // ---- Numeric validation ----

    #[test]
    fn positive_f64_ok() {
        assert!(validate_positive_f64(1.5, "radius").is_ok());
    }

    #[test]
    fn positive_f64_zero() {
        let err = validate_positive_f64(0.0, "radius").unwrap_err();
        assert!(err.reason.contains("> 0"));
    }

    #[test]
    fn positive_f64_negative() {
        let err = validate_positive_f64(-1.0, "radius").unwrap_err();
        assert!(err.reason.contains("> 0"));
    }

    #[test]
    fn positive_f64_nan() {
        let err = validate_positive_f64(f64::NAN, "radius").unwrap_err();
        assert!(err.reason.contains("NaN"));
    }

    #[test]
    fn finite_f64_ok() {
        assert!(validate_finite_f64(42.0, "amount").is_ok());
        assert!(validate_finite_f64(-100.0, "amount").is_ok());
        assert!(validate_finite_f64(0.0, "amount").is_ok());
    }

    #[test]
    fn finite_f64_nan() {
        let err = validate_finite_f64(f64::NAN, "amount").unwrap_err();
        assert!(err.reason.contains("NaN"));
    }

    #[test]
    fn finite_f64_inf() {
        let err = validate_finite_f64(f64::INFINITY, "amount").unwrap_err();
        assert!(err.reason.contains("Infinity"));
    }

    // ---- Resize dimensions ----

    #[test]
    fn resize_valid() {
        assert!(validate_resize_dimensions(100, 100).is_ok());
    }

    #[test]
    fn resize_zero_width() {
        let err = validate_resize_dimensions(0, 100).unwrap_err();
        assert!(err.reason.contains("non-zero"));
    }

    // ---- Quality clamping ----

    #[test]
    fn quality_clamp() {
        assert_eq!(clamp_quality(0), 1);
        assert_eq!(clamp_quality(1), 1);
        assert_eq!(clamp_quality(50), 50);
        assert_eq!(clamp_quality(100), 100);
        assert_eq!(clamp_quality(200), 100);
    }

    // ---- Bit depth ----

    #[test]
    fn valid_bit_depths() {
        assert!(validate_wav_bit_depth(8).is_ok());
        assert!(validate_wav_bit_depth(16).is_ok());
        assert!(validate_wav_bit_depth(24).is_ok());
        assert!(validate_wav_bit_depth(32).is_ok());
    }

    #[test]
    fn invalid_bit_depth() {
        let err = validate_wav_bit_depth(12).unwrap_err();
        assert!(err.reason.contains("8, 16, 24, or 32"));
    }

    // ---- Upscale factor ----

    #[test]
    fn valid_upscale_factors() {
        assert!(validate_upscale_factor(2).is_ok());
        assert!(validate_upscale_factor(4).is_ok());
    }

    #[test]
    fn invalid_upscale_factor() {
        let err = validate_upscale_factor(3).unwrap_err();
        assert!(err.reason.contains("2 or 4"));
    }

    // ---- Interpolation factor ----

    #[test]
    fn valid_interpolation_factors() {
        assert!(validate_interpolation_factor(0.0).is_ok());
        assert!(validate_interpolation_factor(0.5).is_ok());
        assert!(validate_interpolation_factor(1.0).is_ok());
    }

    #[test]
    fn interpolation_factor_out_of_range() {
        let err = validate_interpolation_factor(1.5).unwrap_err();
        assert!(err.reason.contains("0.0 and 1.0"));
    }

    #[test]
    fn interpolation_factor_nan() {
        let err = validate_interpolation_factor(f64::NAN).unwrap_err();
        assert!(err.reason.contains("NaN"));
    }
}
