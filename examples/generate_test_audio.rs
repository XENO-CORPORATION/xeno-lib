//! Generate a test audio file for testing audio encoding.

use std::f32::consts::PI;

fn main() {
    // Generate 1 second of 440 Hz sine wave at 44100 Hz, mono
    let sample_rate = 44100u32;
    let duration_secs = 1.0f32;
    let frequency = 440.0f32;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    
    let mut samples: Vec<f32> = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (t * frequency * 2.0 * PI).sin() * 0.5; // 50% amplitude
        samples.push(sample);
    }
    
    // Write using hound
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    
    let mut writer = hound::WavWriter::create("test_audio.wav", spec).unwrap();
    for sample in &samples {
        let s = (*sample * 32767.0) as i16;
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
    
    println!("Created test_audio.wav (1 second, 440 Hz sine, 44100 Hz, mono)");
}
