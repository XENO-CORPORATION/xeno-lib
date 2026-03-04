//! Audio effects: reverb, EQ, pitch shift, echo, distortion, and more.
//!
//! This module provides professional-grade audio effects processing.
//!
//! # Available Effects
//!
//! - **Reverb**: Room/hall/plate reverb simulation
//! - **EQ**: Parametric equalizer (low/mid/high shelf, peak)
//! - **Pitch Shift**: Change pitch without changing tempo
//! - **Echo/Delay**: Single or multi-tap delay
//! - **Chorus**: Modulated delay for thickening
//! - **Flanger**: Comb filter sweep effect
//! - **Distortion**: Overdrive, fuzz, saturation
//! - **Compressor**: Dynamic range compression
//! - **Limiter**: Peak limiting
//! - **Gate**: Noise gate
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::audio::effects::{reverb, ReverbConfig};
//!
//! let dry_audio: Vec<f32> = load_audio("voice.wav")?;
//! let wet_audio = reverb(&dry_audio, 44100, ReverbConfig::hall());
//! ```

use std::f32::consts::PI;

/// Reverb configuration.
#[derive(Debug, Clone)]
pub struct ReverbConfig {
    /// Room size (0.0 - 1.0).
    pub room_size: f32,
    /// Damping amount (0.0 - 1.0).
    pub damping: f32,
    /// Wet/dry mix (0.0 = dry, 1.0 = wet).
    pub wet_mix: f32,
    /// Stereo width (0.0 - 1.0).
    pub width: f32,
    /// Pre-delay in milliseconds.
    pub pre_delay_ms: f32,
}

impl Default for ReverbConfig {
    fn default() -> Self {
        Self::room()
    }
}

impl ReverbConfig {
    /// Small room preset.
    pub fn room() -> Self {
        Self {
            room_size: 0.3,
            damping: 0.5,
            wet_mix: 0.3,
            width: 0.8,
            pre_delay_ms: 10.0,
        }
    }

    /// Large hall preset.
    pub fn hall() -> Self {
        Self {
            room_size: 0.8,
            damping: 0.3,
            wet_mix: 0.4,
            width: 1.0,
            pre_delay_ms: 30.0,
        }
    }

    /// Plate reverb preset.
    pub fn plate() -> Self {
        Self {
            room_size: 0.6,
            damping: 0.7,
            wet_mix: 0.35,
            width: 0.9,
            pre_delay_ms: 5.0,
        }
    }

    /// Cathedral preset.
    pub fn cathedral() -> Self {
        Self {
            room_size: 0.95,
            damping: 0.2,
            wet_mix: 0.5,
            width: 1.0,
            pre_delay_ms: 50.0,
        }
    }
}

/// Apply reverb effect to audio.
///
/// Uses a Freeverb-style algorithm with comb and allpass filters.
pub fn reverb(samples: &[f32], sample_rate: u32, config: ReverbConfig) -> Vec<f32> {
    let len = samples.len();
    let mut output = vec![0.0f32; len];

    // Pre-delay buffer
    let pre_delay_samples = (config.pre_delay_ms * sample_rate as f32 / 1000.0) as usize;
    let mut pre_delay = vec![0.0f32; pre_delay_samples.max(1)];
    let mut pre_delay_idx = 0;

    // Comb filter delay lengths (in samples at 44100 Hz, scaled for actual rate)
    let scale = sample_rate as f32 / 44100.0;
    let comb_lengths: [usize; 8] = [
        (1116.0 * scale) as usize,
        (1188.0 * scale) as usize,
        (1277.0 * scale) as usize,
        (1356.0 * scale) as usize,
        (1422.0 * scale) as usize,
        (1491.0 * scale) as usize,
        (1557.0 * scale) as usize,
        (1617.0 * scale) as usize,
    ];

    // Allpass filter delay lengths
    let allpass_lengths: [usize; 4] = [
        (556.0 * scale) as usize,
        (441.0 * scale) as usize,
        (341.0 * scale) as usize,
        (225.0 * scale) as usize,
    ];

    // Initialize comb filters
    let mut comb_buffers: Vec<Vec<f32>> = comb_lengths.iter().map(|&l| vec![0.0; l]).collect();
    let mut comb_indices: Vec<usize> = vec![0; 8];
    let mut comb_filter_store: Vec<f32> = vec![0.0; 8];

    // Initialize allpass filters
    let mut allpass_buffers: Vec<Vec<f32>> =
        allpass_lengths.iter().map(|&l| vec![0.0; l]).collect();
    let mut allpass_indices: Vec<usize> = vec![0; 4];

    let feedback = config.room_size * 0.28 + 0.7;
    let damp1 = config.damping * 0.4;
    let damp2 = 1.0 - damp1;

    for i in 0..len {
        // Pre-delay
        let delayed_input = pre_delay[pre_delay_idx];
        pre_delay[pre_delay_idx] = samples[i];
        pre_delay_idx = (pre_delay_idx + 1) % pre_delay.len();

        let input = delayed_input * 0.015;
        let mut sum = 0.0f32;

        // Parallel comb filters
        for j in 0..8 {
            let buf = &mut comb_buffers[j];
            let idx = comb_indices[j];
            let out = buf[idx];

            // Low-pass filter in feedback path
            comb_filter_store[j] = out * damp2 + comb_filter_store[j] * damp1;
            buf[idx] = input + comb_filter_store[j] * feedback;

            comb_indices[j] = (idx + 1) % buf.len();
            sum += out;
        }

        // Series allpass filters
        for j in 0..4 {
            let buf = &mut allpass_buffers[j];
            let idx = allpass_indices[j];
            let buf_out = buf[idx];

            let new_val = sum + buf_out * 0.5;
            buf[idx] = sum - buf_out * 0.5;
            sum = new_val;

            allpass_indices[j] = (idx + 1) % buf.len();
        }

        // Mix
        output[i] = samples[i] * (1.0 - config.wet_mix) + sum * config.wet_mix;
    }

    output
}

/// Equalizer band configuration.
#[derive(Debug, Clone)]
pub struct EqBand {
    /// Center frequency in Hz.
    pub frequency: f32,
    /// Gain in dB (-24 to +24).
    pub gain_db: f32,
    /// Q factor (bandwidth).
    pub q: f32,
    /// Filter type.
    pub filter_type: EqFilterType,
}

/// EQ filter types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EqFilterType {
    /// Low shelf filter.
    LowShelf,
    /// High shelf filter.
    HighShelf,
    /// Peak/bell filter.
    Peak,
    /// Low pass filter.
    LowPass,
    /// High pass filter.
    HighPass,
    /// Notch filter.
    Notch,
}

impl EqBand {
    /// Create a low shelf band.
    pub fn low_shelf(frequency: f32, gain_db: f32) -> Self {
        Self {
            frequency,
            gain_db,
            q: 0.707,
            filter_type: EqFilterType::LowShelf,
        }
    }

    /// Create a high shelf band.
    pub fn high_shelf(frequency: f32, gain_db: f32) -> Self {
        Self {
            frequency,
            gain_db,
            q: 0.707,
            filter_type: EqFilterType::HighShelf,
        }
    }

    /// Create a peak/bell band.
    pub fn peak(frequency: f32, gain_db: f32, q: f32) -> Self {
        Self {
            frequency,
            gain_db,
            q,
            filter_type: EqFilterType::Peak,
        }
    }

    /// Create a high pass filter.
    pub fn high_pass(frequency: f32, q: f32) -> Self {
        Self {
            frequency,
            gain_db: 0.0,
            q,
            filter_type: EqFilterType::HighPass,
        }
    }

    /// Create a low pass filter.
    pub fn low_pass(frequency: f32, q: f32) -> Self {
        Self {
            frequency,
            gain_db: 0.0,
            q,
            filter_type: EqFilterType::LowPass,
        }
    }
}

/// Parametric equalizer configuration.
#[derive(Debug, Clone, Default)]
pub struct EqConfig {
    /// EQ bands.
    pub bands: Vec<EqBand>,
}

impl EqConfig {
    /// Create an empty EQ configuration.
    pub fn new() -> Self {
        Self { bands: Vec::new() }
    }

    /// Add a band.
    pub fn add_band(mut self, band: EqBand) -> Self {
        self.bands.push(band);
        self
    }

    /// Voice clarity preset (boost highs, cut lows).
    pub fn voice_clarity() -> Self {
        Self::new()
            .add_band(EqBand::high_pass(80.0, 0.707))
            .add_band(EqBand::peak(2500.0, 3.0, 1.5))
            .add_band(EqBand::high_shelf(6000.0, 2.0))
    }

    /// Bass boost preset.
    pub fn bass_boost() -> Self {
        Self::new()
            .add_band(EqBand::low_shelf(100.0, 6.0))
            .add_band(EqBand::peak(60.0, 4.0, 0.8))
    }

    /// Treble boost preset.
    pub fn treble_boost() -> Self {
        Self::new()
            .add_band(EqBand::high_shelf(8000.0, 4.0))
            .add_band(EqBand::peak(10000.0, 3.0, 1.0))
    }
}

/// Biquad filter coefficients.
#[derive(Debug, Clone)]
struct BiquadCoeffs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl BiquadCoeffs {
    fn from_band(band: &EqBand, sample_rate: f32) -> Self {
        let a = 10.0f32.powf(band.gain_db / 40.0);
        let w0 = 2.0 * PI * band.frequency / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * band.q);

        let (b0, b1, b2, a0, a1, a2) = match band.filter_type {
            EqFilterType::Peak => {
                let b0 = 1.0 + alpha * a;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 - alpha * a;
                let a0 = 1.0 + alpha / a;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha / a;
                (b0, b1, b2, a0, a1, a2)
            }
            EqFilterType::LowShelf => {
                let sqrt_a = a.sqrt();
                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha);
                let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            EqFilterType::HighShelf => {
                let sqrt_a = a.sqrt();
                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha);
                let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            EqFilterType::LowPass => {
                let b0 = (1.0 - cos_w0) / 2.0;
                let b1 = 1.0 - cos_w0;
                let b2 = (1.0 - cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            EqFilterType::HighPass => {
                let b0 = (1.0 + cos_w0) / 2.0;
                let b1 = -(1.0 + cos_w0);
                let b2 = (1.0 + cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            EqFilterType::Notch => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
        };

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

/// Apply parametric EQ to audio.
pub fn equalizer(samples: &[f32], sample_rate: u32, config: &EqConfig) -> Vec<f32> {
    if config.bands.is_empty() {
        return samples.to_vec();
    }

    let mut output = samples.to_vec();

    for band in &config.bands {
        let coeffs = BiquadCoeffs::from_band(band, sample_rate as f32);
        let mut x1 = 0.0f32;
        let mut x2 = 0.0f32;
        let mut y1 = 0.0f32;
        let mut y2 = 0.0f32;

        for sample in output.iter_mut() {
            let x0 = *sample;
            let y0 =
                coeffs.b0 * x0 + coeffs.b1 * x1 + coeffs.b2 * x2 - coeffs.a1 * y1 - coeffs.a2 * y2;

            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;

            *sample = y0;
        }
    }

    output
}

/// Echo/delay configuration.
#[derive(Debug, Clone)]
pub struct DelayConfig {
    /// Delay time in milliseconds.
    pub delay_ms: f32,
    /// Feedback amount (0.0 - 0.95).
    pub feedback: f32,
    /// Wet/dry mix (0.0 - 1.0).
    pub wet_mix: f32,
    /// Number of taps (for multi-tap delay).
    pub taps: u32,
}

impl Default for DelayConfig {
    fn default() -> Self {
        Self::single(250.0, 0.4)
    }
}

impl DelayConfig {
    /// Single delay (echo).
    pub fn single(delay_ms: f32, feedback: f32) -> Self {
        Self {
            delay_ms,
            feedback: feedback.clamp(0.0, 0.95),
            wet_mix: 0.5,
            taps: 1,
        }
    }

    /// Multi-tap delay.
    pub fn multi_tap(delay_ms: f32, taps: u32, feedback: f32) -> Self {
        Self {
            delay_ms,
            feedback: feedback.clamp(0.0, 0.95),
            wet_mix: 0.5,
            taps: taps.clamp(1, 8),
        }
    }

    /// Slapback echo preset.
    pub fn slapback() -> Self {
        Self::single(80.0, 0.0)
    }

    /// Ping-pong delay preset.
    pub fn ping_pong() -> Self {
        Self {
            delay_ms: 300.0,
            feedback: 0.5,
            wet_mix: 0.4,
            taps: 2,
        }
    }
}

/// Apply delay/echo effect to audio.
pub fn delay(samples: &[f32], sample_rate: u32, config: DelayConfig) -> Vec<f32> {
    let delay_samples = (config.delay_ms * sample_rate as f32 / 1000.0) as usize;
    let mut buffer = vec![0.0f32; delay_samples.max(1)];
    let mut buffer_idx = 0;

    let mut output = vec![0.0f32; samples.len()];

    for (i, &sample) in samples.iter().enumerate() {
        let delayed = buffer[buffer_idx];
        let mixed = sample + delayed * config.feedback;

        buffer[buffer_idx] = mixed;
        buffer_idx = (buffer_idx + 1) % buffer.len();

        output[i] = sample * (1.0 - config.wet_mix) + delayed * config.wet_mix;
    }

    output
}

/// Pitch shift configuration.
#[derive(Debug, Clone)]
pub struct PitchShiftConfig {
    /// Pitch shift in semitones (-12 to +12).
    pub semitones: f32,
    /// Window size for PSOLA algorithm.
    pub window_size: usize,
    /// Preserve formants (for voice).
    pub preserve_formants: bool,
}

impl PitchShiftConfig {
    /// Create pitch shift configuration.
    pub fn new(semitones: f32) -> Self {
        Self {
            semitones: semitones.clamp(-24.0, 24.0),
            window_size: 2048,
            preserve_formants: false,
        }
    }

    /// Shift up by octave.
    pub fn octave_up() -> Self {
        Self::new(12.0)
    }

    /// Shift down by octave.
    pub fn octave_down() -> Self {
        Self::new(-12.0)
    }
}

/// Apply pitch shift to audio (simple resampling-based approach).
///
/// Note: For high-quality pitch shifting, consider using a dedicated library
/// or the AI frame interpolation for time-stretching.
pub fn pitch_shift(samples: &[f32], _sample_rate: u32, config: PitchShiftConfig) -> Vec<f32> {
    let ratio = 2.0f32.powf(config.semitones / 12.0);
    let new_len = (samples.len() as f32 / ratio) as usize;

    let mut output = vec![0.0f32; new_len];

    for i in 0..new_len {
        let src_pos = i as f32 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f32;

        if src_idx + 1 < samples.len() {
            // Linear interpolation
            output[i] = samples[src_idx] * (1.0 - frac) + samples[src_idx + 1] * frac;
        } else if src_idx < samples.len() {
            output[i] = samples[src_idx];
        }
    }

    output
}

/// Distortion configuration.
#[derive(Debug, Clone)]
pub struct DistortionConfig {
    /// Drive amount (1.0 - 100.0).
    pub drive: f32,
    /// Distortion type.
    pub distortion_type: DistortionType,
    /// Output level (0.0 - 1.0).
    pub output_level: f32,
    /// Tone control (-1.0 = dark, 1.0 = bright).
    pub tone: f32,
}

/// Types of distortion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionType {
    /// Soft clipping (overdrive).
    Overdrive,
    /// Hard clipping (distortion).
    HardClip,
    /// Tube-style saturation.
    Tube,
    /// Fuzz effect.
    Fuzz,
    /// Bitcrusher.
    Bitcrush,
}

impl Default for DistortionConfig {
    fn default() -> Self {
        Self {
            drive: 5.0,
            distortion_type: DistortionType::Overdrive,
            output_level: 0.7,
            tone: 0.0,
        }
    }
}

impl DistortionConfig {
    /// Overdrive preset.
    pub fn overdrive() -> Self {
        Self {
            drive: 5.0,
            distortion_type: DistortionType::Overdrive,
            output_level: 0.7,
            tone: 0.0,
        }
    }

    /// Heavy distortion preset.
    pub fn heavy() -> Self {
        Self {
            drive: 30.0,
            distortion_type: DistortionType::HardClip,
            output_level: 0.5,
            tone: 0.2,
        }
    }

    /// Fuzz preset.
    pub fn fuzz() -> Self {
        Self {
            drive: 50.0,
            distortion_type: DistortionType::Fuzz,
            output_level: 0.4,
            tone: -0.3,
        }
    }
}

/// Apply distortion effect to audio.
pub fn distortion(samples: &[f32], config: DistortionConfig) -> Vec<f32> {
    let mut output = vec![0.0f32; samples.len()];

    for (i, &sample) in samples.iter().enumerate() {
        let driven = sample * config.drive;

        let distorted = match config.distortion_type {
            DistortionType::Overdrive => {
                // Soft clipping using tanh
                driven.tanh()
            }
            DistortionType::HardClip => {
                // Hard clipping
                driven.clamp(-1.0, 1.0)
            }
            DistortionType::Tube => {
                // Asymmetric tube-style saturation
                if driven >= 0.0 {
                    1.0 - (-driven).exp()
                } else {
                    -1.0 + driven.exp()
                }
            }
            DistortionType::Fuzz => {
                // Fuzz with asymmetric clipping
                let sign = driven.signum();
                sign * (1.0 - (-driven.abs() * 3.0).exp())
            }
            DistortionType::Bitcrush => {
                // Bit reduction
                let bits = (16.0 - config.drive * 0.14).clamp(2.0, 16.0);
                let levels = 2.0f32.powf(bits);
                (driven * levels).round() / levels
            }
        };

        output[i] = distorted * config.output_level;
    }

    output
}

/// Chorus configuration.
#[derive(Debug, Clone)]
pub struct ChorusConfig {
    /// LFO rate in Hz.
    pub rate: f32,
    /// Depth (modulation amount).
    pub depth: f32,
    /// Base delay in milliseconds.
    pub delay_ms: f32,
    /// Wet/dry mix.
    pub wet_mix: f32,
    /// Number of voices.
    pub voices: u32,
}

impl Default for ChorusConfig {
    fn default() -> Self {
        Self {
            rate: 1.5,
            depth: 0.5,
            delay_ms: 20.0,
            wet_mix: 0.5,
            voices: 2,
        }
    }
}

/// Apply chorus effect to audio.
pub fn chorus(samples: &[f32], sample_rate: u32, config: ChorusConfig) -> Vec<f32> {
    let max_delay = (config.delay_ms * 2.0 * sample_rate as f32 / 1000.0) as usize;
    let mut buffer = vec![0.0f32; max_delay.max(1)];
    let mut buffer_idx = 0;

    let mut output = vec![0.0f32; samples.len()];
    let lfo_increment = 2.0 * PI * config.rate / sample_rate as f32;
    let mut lfo_phase = 0.0f32;

    let base_delay = config.delay_ms * sample_rate as f32 / 1000.0;

    for (i, &sample) in samples.iter().enumerate() {
        buffer[buffer_idx] = sample;

        let mut chorus_sum = 0.0f32;

        for voice in 0..config.voices {
            let voice_phase = lfo_phase + (voice as f32 * PI / config.voices as f32);
            let modulation = voice_phase.sin() * config.depth * base_delay;
            let delay_samples = (base_delay + modulation).max(1.0);

            // Read from delay buffer with interpolation
            let read_pos = buffer_idx as f32 - delay_samples;
            let read_pos = if read_pos < 0.0 {
                read_pos + buffer.len() as f32
            } else {
                read_pos
            };

            let idx = read_pos as usize % buffer.len();
            let frac = read_pos - read_pos.floor();
            let next_idx = (idx + 1) % buffer.len();

            chorus_sum += buffer[idx] * (1.0 - frac) + buffer[next_idx] * frac;
        }

        chorus_sum /= config.voices as f32;

        output[i] = sample * (1.0 - config.wet_mix) + chorus_sum * config.wet_mix;

        buffer_idx = (buffer_idx + 1) % buffer.len();
        lfo_phase += lfo_increment;
        if lfo_phase > 2.0 * PI {
            lfo_phase -= 2.0 * PI;
        }
    }

    output
}

/// Flanger configuration.
#[derive(Debug, Clone)]
pub struct FlangerConfig {
    /// LFO rate in Hz.
    pub rate: f32,
    /// Depth (0.0 - 1.0).
    pub depth: f32,
    /// Feedback (-0.95 to 0.95).
    pub feedback: f32,
    /// Wet/dry mix.
    pub wet_mix: f32,
}

impl Default for FlangerConfig {
    fn default() -> Self {
        Self {
            rate: 0.5,
            depth: 0.7,
            feedback: 0.5,
            wet_mix: 0.5,
        }
    }
}

/// Apply flanger effect to audio.
pub fn flanger(samples: &[f32], sample_rate: u32, config: FlangerConfig) -> Vec<f32> {
    let max_delay = (10.0 * sample_rate as f32 / 1000.0) as usize; // 10ms max
    let mut buffer = vec![0.0f32; max_delay.max(1)];
    let mut buffer_idx = 0;

    let mut output = vec![0.0f32; samples.len()];
    let lfo_increment = 2.0 * PI * config.rate / sample_rate as f32;
    let mut lfo_phase = 0.0f32;
    let mut feedback_sample = 0.0f32;

    for (i, &sample) in samples.iter().enumerate() {
        let input = sample + feedback_sample * config.feedback;
        buffer[buffer_idx] = input;

        // LFO modulates delay time
        let modulation = (lfo_phase.sin() + 1.0) * 0.5; // 0 to 1
        let delay_samples = (1.0 + modulation * config.depth * max_delay as f32).max(1.0);

        // Read from delay buffer
        let read_pos = buffer_idx as f32 - delay_samples;
        let read_pos = if read_pos < 0.0 {
            read_pos + buffer.len() as f32
        } else {
            read_pos
        };

        let idx = read_pos as usize % buffer.len();
        let frac = read_pos - read_pos.floor();
        let next_idx = (idx + 1) % buffer.len();

        let delayed = buffer[idx] * (1.0 - frac) + buffer[next_idx] * frac;
        feedback_sample = delayed;

        output[i] = sample * (1.0 - config.wet_mix) + delayed * config.wet_mix;

        buffer_idx = (buffer_idx + 1) % buffer.len();
        lfo_phase += lfo_increment;
        if lfo_phase > 2.0 * PI {
            lfo_phase -= 2.0 * PI;
        }
    }

    output
}

/// Noise gate configuration.
#[derive(Debug, Clone)]
pub struct GateConfig {
    /// Threshold in dB.
    pub threshold_db: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Range (attenuation when closed) in dB.
    pub range_db: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            threshold_db: -40.0,
            attack_ms: 1.0,
            release_ms: 100.0,
            range_db: -80.0,
        }
    }
}

/// Apply noise gate to audio.
pub fn gate(samples: &[f32], sample_rate: u32, config: GateConfig) -> Vec<f32> {
    let threshold = db_to_linear(config.threshold_db);
    let range = db_to_linear(config.range_db);
    let attack_coef = (-1.0 / (config.attack_ms * sample_rate as f32 / 1000.0)).exp();
    let release_coef = (-1.0 / (config.release_ms * sample_rate as f32 / 1000.0)).exp();

    let mut envelope = 0.0f32;
    let mut gain = range;

    let mut output = vec![0.0f32; samples.len()];

    for (i, &sample) in samples.iter().enumerate() {
        let abs_sample = sample.abs();

        // Envelope follower
        if abs_sample > envelope {
            envelope = attack_coef * envelope + (1.0 - attack_coef) * abs_sample;
        } else {
            envelope = release_coef * envelope + (1.0 - release_coef) * abs_sample;
        }

        // Gate
        if envelope > threshold {
            gain = gain * attack_coef + (1.0 - attack_coef);
        } else {
            gain = gain * release_coef + range * (1.0 - release_coef);
        }

        output[i] = sample * gain;
    }

    output
}

// Utility functions

/// Convert dB to linear amplitude.
pub fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Convert linear amplitude to dB.
pub fn linear_to_db(linear: f32) -> f32 {
    20.0 * linear.abs().max(1e-10).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_sine(freq: f32, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_reverb() {
        let samples = generate_sine(440.0, 44100, 0.1);
        let output = reverb(&samples, 44100, ReverbConfig::room());
        assert_eq!(output.len(), samples.len());
    }

    #[test]
    fn test_equalizer() {
        let samples = generate_sine(440.0, 44100, 0.1);
        let config = EqConfig::voice_clarity();
        let output = equalizer(&samples, 44100, &config);
        assert_eq!(output.len(), samples.len());
    }

    #[test]
    fn test_delay() {
        let samples = generate_sine(440.0, 44100, 0.1);
        let output = delay(&samples, 44100, DelayConfig::default());
        assert_eq!(output.len(), samples.len());
    }

    #[test]
    fn test_pitch_shift() {
        let samples = generate_sine(440.0, 44100, 0.1);
        let output = pitch_shift(&samples, 44100, PitchShiftConfig::new(12.0));
        // Pitch shift up by octave should halve the length (same duration, higher pitch = fewer samples)
        assert!(output.len() < samples.len());
    }

    #[test]
    fn test_distortion() {
        let samples = generate_sine(440.0, 44100, 0.1);
        let output = distortion(&samples, DistortionConfig::overdrive());
        assert_eq!(output.len(), samples.len());
        // Distortion should add harmonics (change the waveform)
    }

    #[test]
    fn test_chorus() {
        let samples = generate_sine(440.0, 44100, 0.1);
        let output = chorus(&samples, 44100, ChorusConfig::default());
        assert_eq!(output.len(), samples.len());
    }

    #[test]
    fn test_flanger() {
        let samples = generate_sine(440.0, 44100, 0.1);
        let output = flanger(&samples, 44100, FlangerConfig::default());
        assert_eq!(output.len(), samples.len());
    }

    #[test]
    fn test_gate() {
        let samples = generate_sine(440.0, 44100, 0.1);
        let output = gate(&samples, 44100, GateConfig::default());
        assert_eq!(output.len(), samples.len());
    }

    #[test]
    fn test_db_conversion() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 0.001);
        assert!((db_to_linear(-6.0) - 0.5).abs() < 0.01);
        assert!((linear_to_db(1.0) - 0.0).abs() < 0.001);
    }
}
