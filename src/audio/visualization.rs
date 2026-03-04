//! Audio visualization: waveform and spectrum displays.
//!
//! Generate visual representations of audio data for UI display.
//!
//! # Features
//!
//! - **Waveform**: Time-domain amplitude display
//! - **Spectrum**: Frequency-domain visualization
//! - **Spectrogram**: Time-frequency heatmap
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::audio::visualization::{render_waveform, WaveformConfig};
//!
//! let samples: Vec<f32> = load_audio("song.mp3")?;
//! let config = WaveformConfig::default();
//! let waveform = render_waveform(&samples, &config);
//! waveform.save("waveform.png")?;
//! ```

use image::{DynamicImage, Rgba, RgbaImage};
use std::f32::consts::PI;

/// Waveform visualization configuration.
#[derive(Debug, Clone)]
pub struct WaveformConfig {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Waveform color.
    pub color: [u8; 4],
    /// Background color.
    pub background: [u8; 4],
    /// Draw as filled waveform.
    pub filled: bool,
    /// Center line color (if shown).
    pub center_line_color: Option<[u8; 4]>,
    /// Samples per pixel (for downsampling).
    pub samples_per_pixel: Option<usize>,
}

impl Default for WaveformConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 200,
            color: [0, 150, 255, 255],     // Blue
            background: [30, 30, 30, 255], // Dark gray
            filled: true,
            center_line_color: Some([100, 100, 100, 255]),
            samples_per_pixel: None,
        }
    }
}

impl WaveformConfig {
    /// Create a minimal waveform style.
    pub fn minimal() -> Self {
        Self {
            color: [255, 255, 255, 255],
            background: [0, 0, 0, 255],
            filled: false,
            center_line_color: None,
            ..Default::default()
        }
    }

    /// Create a SoundCloud-like style.
    pub fn soundcloud() -> Self {
        Self {
            color: [255, 85, 0, 255],         // Orange
            background: [255, 255, 255, 255], // White
            filled: true,
            center_line_color: None,
            ..Default::default()
        }
    }
}

/// Spectrum visualization configuration.
#[derive(Debug, Clone)]
pub struct SpectrumConfig {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of frequency bins.
    pub bins: usize,
    /// Use logarithmic frequency scale.
    pub log_scale: bool,
    /// Color map.
    pub color_map: ColorMap,
    /// Background color.
    pub background: [u8; 4],
}

impl Default for SpectrumConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 300,
            bins: 64,
            log_scale: true,
            color_map: ColorMap::Viridis,
            background: [0, 0, 0, 255],
        }
    }
}

/// Color map for spectrum visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMap {
    /// Purple to yellow.
    #[default]
    Viridis,
    /// Blue to red through white.
    Plasma,
    /// Black to white.
    Grayscale,
    /// Blue to red.
    Jet,
    /// Green monochrome.
    Green,
}

impl ColorMap {
    /// Get color for normalized value (0-1).
    pub fn get(&self, t: f32) -> [u8; 4] {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Viridis => viridis(t),
            Self::Plasma => plasma(t),
            Self::Grayscale => {
                let v = (t * 255.0) as u8;
                [v, v, v, 255]
            }
            Self::Jet => jet(t),
            Self::Green => {
                let v = (t * 255.0) as u8;
                [0, v, 0, 255]
            }
        }
    }
}

/// Render audio waveform.
pub fn render_waveform(samples: &[f32], config: &WaveformConfig) -> DynamicImage {
    let mut img = RgbaImage::from_pixel(config.width, config.height, Rgba(config.background));

    if samples.is_empty() {
        return DynamicImage::ImageRgba8(img);
    }

    let center = config.height / 2;
    let amplitude = (config.height / 2) as f32 * 0.9;

    // Draw center line
    if let Some(line_color) = config.center_line_color {
        for x in 0..config.width {
            img.put_pixel(x, center, Rgba(line_color));
        }
    }

    // Render waveform
    for x in 0..config.width {
        let start = (x as usize * samples.len()) / config.width as usize;
        let end = ((x as usize + 1) * samples.len()) / config.width as usize;

        if start >= samples.len() {
            break;
        }

        let chunk = &samples[start..end.min(samples.len())];
        if chunk.is_empty() {
            continue;
        }

        // Get min and max in chunk (for filled waveform)
        let (min_val, max_val) = chunk
            .iter()
            .fold((0.0f32, 0.0f32), |(min, max), &v| (min.min(v), max.max(v)));

        let y_min = (center as f32 - max_val * amplitude).round() as u32;
        let y_max = (center as f32 - min_val * amplitude).round() as u32;

        if config.filled {
            // Draw filled waveform
            for y in y_min.min(config.height - 1)..=y_max.min(config.height - 1) {
                img.put_pixel(x, y, Rgba(config.color));
            }
        } else {
            // Draw line waveform
            let y = (center as f32 - chunk[0] * amplitude).round() as u32;
            if y < config.height {
                img.put_pixel(x, y, Rgba(config.color));
            }
        }
    }

    DynamicImage::ImageRgba8(img)
}

/// Render audio spectrum (single frame).
pub fn render_spectrum(
    samples: &[f32],
    _sample_rate: u32,
    config: &SpectrumConfig,
) -> DynamicImage {
    let mut img = RgbaImage::from_pixel(config.width, config.height, Rgba(config.background));

    if samples.is_empty() {
        return DynamicImage::ImageRgba8(img);
    }

    // Compute FFT magnitudes
    let fft_size = 2048.min(samples.len()).next_power_of_two();
    let magnitudes = compute_spectrum(samples, fft_size);

    // Bin the frequencies
    let bins = bin_spectrum(&magnitudes, config.bins, config.log_scale);

    // Render bars
    let bar_width = config.width / config.bins as u32;
    let max_mag = bins.iter().cloned().fold(0.0f32, f32::max).max(0.001);

    for (i, &mag) in bins.iter().enumerate() {
        let normalized = (mag / max_mag).sqrt(); // Square root for better visibility
        let bar_height = (normalized * config.height as f32) as u32;

        let x_start = i as u32 * bar_width;

        for x in x_start..x_start + bar_width.saturating_sub(1) {
            for y in (config.height - bar_height)..config.height {
                if x < config.width && y < config.height {
                    let t = (config.height - y) as f32 / config.height as f32;
                    img.put_pixel(x, y, Rgba(config.color_map.get(t)));
                }
            }
        }
    }

    DynamicImage::ImageRgba8(img)
}

/// Render spectrogram (time-frequency visualization).
pub fn render_spectrogram(
    samples: &[f32],
    _sample_rate: u32,
    width: u32,
    height: u32,
    color_map: ColorMap,
) -> DynamicImage {
    let mut img = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 255]));

    if samples.is_empty() {
        return DynamicImage::ImageRgba8(img);
    }

    let hop_size = samples.len() / width as usize;
    let fft_size = 1024.min(samples.len()).next_power_of_two();
    let num_bins = height as usize;

    // Find max magnitude for normalization
    let mut max_mag = 0.0f32;
    let mut all_spectra = Vec::new();

    for x in 0..width {
        let start = (x as usize * hop_size).min(samples.len().saturating_sub(fft_size));
        let chunk = &samples[start
            ..start
                .min(samples.len())
                .saturating_add(fft_size)
                .min(samples.len())];

        let magnitudes = compute_spectrum(chunk, fft_size);
        let bins = bin_spectrum(&magnitudes, num_bins, true);

        max_mag = bins.iter().fold(max_mag, |m, &v| m.max(v));
        all_spectra.push(bins);
    }

    max_mag = max_mag.max(0.001);

    // Render
    for (x, spectrum) in all_spectra.iter().enumerate() {
        for (bin_idx, &mag) in spectrum.iter().enumerate() {
            let y = height - 1 - bin_idx as u32;
            let normalized = (mag / max_mag).sqrt();
            if x < width as usize && y < height {
                img.put_pixel(x as u32, y, Rgba(color_map.get(normalized)));
            }
        }
    }

    DynamicImage::ImageRgba8(img)
}

/// Get waveform data as points.
pub fn get_waveform_points(samples: &[f32], num_points: usize) -> Vec<(f32, f32)> {
    if samples.is_empty() || num_points == 0 {
        return Vec::new();
    }

    let spp = (samples.len() / num_points).max(1);
    let mut points = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let start = i * spp;
        let end = ((i + 1) * spp).min(samples.len());

        if start >= samples.len() {
            break;
        }

        let chunk = &samples[start..end];
        let (min, max) = chunk
            .iter()
            .fold((0.0f32, 0.0f32), |(min, max), &v| (min.min(v), max.max(v)));

        points.push((min, max));
    }

    points
}

/// Get spectrum data as frequency bins.
pub fn get_spectrum_data(samples: &[f32], num_bins: usize) -> Vec<f32> {
    if samples.is_empty() {
        return vec![0.0; num_bins];
    }

    let fft_size = 2048.min(samples.len()).next_power_of_two();
    let magnitudes = compute_spectrum(samples, fft_size);
    bin_spectrum(&magnitudes, num_bins, true)
}

// Internal FFT computation

fn compute_spectrum(samples: &[f32], fft_size: usize) -> Vec<f32> {
    // Simple DFT (for small sizes) or placeholder
    // Real implementation would use rustfft crate
    let n = fft_size.min(samples.len());

    // Apply Hann window and compute DFT
    let mut magnitudes = vec![0.0f32; n / 2];

    for k in 0..n / 2 {
        let mut re = 0.0f32;
        let mut im = 0.0f32;

        for (i, &sample) in samples.iter().take(n).enumerate() {
            let window = 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos());
            let angle = -2.0 * PI * k as f32 * i as f32 / n as f32;
            re += sample * window * angle.cos();
            im += sample * window * angle.sin();
        }

        magnitudes[k] = (re * re + im * im).sqrt();
    }

    magnitudes
}

fn bin_spectrum(magnitudes: &[f32], num_bins: usize, log_scale: bool) -> Vec<f32> {
    let mut bins = vec![0.0f32; num_bins];

    if magnitudes.is_empty() || num_bins == 0 {
        return bins;
    }

    for (bin_idx, bin) in bins.iter_mut().enumerate() {
        let (start, end) = if log_scale {
            // Logarithmic binning
            let log_start =
                (bin_idx as f32 / num_bins as f32 * (magnitudes.len() as f32).ln()).exp();
            let log_end =
                ((bin_idx + 1) as f32 / num_bins as f32 * (magnitudes.len() as f32).ln()).exp();
            (log_start as usize, log_end as usize)
        } else {
            // Linear binning
            let start = bin_idx * magnitudes.len() / num_bins;
            let end = (bin_idx + 1) * magnitudes.len() / num_bins;
            (start, end)
        };

        let start = start.min(magnitudes.len());
        let end = end.max(start + 1).min(magnitudes.len());

        if start < end {
            *bin = magnitudes[start..end].iter().sum::<f32>() / (end - start) as f32;
        }
    }

    bins
}

// Color maps

fn viridis(t: f32) -> [u8; 4] {
    // Simplified viridis approximation
    let r = (0.267 + 0.003 * t + 1.169 * t * t - 0.433 * t * t * t).clamp(0.0, 1.0);
    let g = (0.004 + 0.874 * t - 0.195 * t * t + 0.114 * t * t * t).clamp(0.0, 1.0);
    let b = (0.329 + 0.675 * t - 0.756 * t * t + 0.346 * t * t * t).clamp(0.0, 1.0);

    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255]
}

fn plasma(t: f32) -> [u8; 4] {
    let r = (0.050 + 2.740 * t - 3.030 * t * t + 1.250 * t * t * t).clamp(0.0, 1.0);
    let g = (0.030 + 0.500 * t + 0.450 * t * t).clamp(0.0, 1.0);
    let b = (0.530 + 0.480 * t - 1.450 * t * t + 0.640 * t * t * t).clamp(0.0, 1.0);

    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255]
}

fn jet(t: f32) -> [u8; 4] {
    let r = if t < 0.375 {
        0.0
    } else if t < 0.625 {
        (t - 0.375) * 4.0
    } else if t < 0.875 {
        1.0
    } else {
        1.0 - (t - 0.875) * 4.0
    };

    let g = if t < 0.125 {
        0.0
    } else if t < 0.375 {
        (t - 0.125) * 4.0
    } else if t < 0.625 {
        1.0
    } else if t < 0.875 {
        1.0 - (t - 0.625) * 4.0
    } else {
        0.0
    };

    let b = if t < 0.125 {
        0.5 + t * 4.0
    } else if t < 0.375 {
        1.0
    } else if t < 0.625 {
        1.0 - (t - 0.375) * 4.0
    } else {
        0.0
    };

    [
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (b.clamp(0.0, 1.0) * 255.0) as u8,
        255,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_sine(freq: f32, sample_rate: u32, duration: f32) -> Vec<f32> {
        let num_samples = (sample_rate as f32 * duration) as usize;
        (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_render_waveform() {
        let samples = generate_sine(440.0, 44100, 0.1);
        let config = WaveformConfig::default();
        let img = render_waveform(&samples, &config);
        assert_eq!(img.width(), config.width);
        assert_eq!(img.height(), config.height);
    }

    #[test]
    fn test_render_spectrum() {
        let samples = generate_sine(440.0, 44100, 0.1);
        let config = SpectrumConfig::default();
        let img = render_spectrum(&samples, 44100, &config);
        assert_eq!(img.width(), config.width);
        assert_eq!(img.height(), config.height);
    }

    #[test]
    fn test_render_spectrogram() {
        let samples = generate_sine(440.0, 44100, 0.5);
        let img = render_spectrogram(&samples, 44100, 400, 200, ColorMap::Viridis);
        assert_eq!(img.width(), 400);
        assert_eq!(img.height(), 200);
    }

    #[test]
    fn test_get_waveform_points() {
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
        let points = get_waveform_points(&samples, 4);
        assert_eq!(points.len(), 4);
    }

    #[test]
    fn test_color_maps() {
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let _ = ColorMap::Viridis.get(t);
            let _ = ColorMap::Plasma.get(t);
            let _ = ColorMap::Jet.get(t);
            let _ = ColorMap::Grayscale.get(t);
        }
    }
}
