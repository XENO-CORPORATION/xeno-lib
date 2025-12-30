//! Configuration for audio source separation.

use std::path::PathBuf;

/// Available audio separation models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeparationModel {
    /// Demucs v4 Hybrid - Best quality for music.
    #[default]
    DemucsHybrid,

    /// Demucs v4 MDX - Optimized for vocals.
    DemucsMdx,

    /// UVR MDX-Net - Specialized for vocals.
    UvrMdx,
}

impl SeparationModel {
    /// Returns the default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            SeparationModel::DemucsHybrid => "demucs_hybrid.onnx",
            SeparationModel::DemucsMdx => "demucs_mdx.onnx",
            SeparationModel::UvrMdx => "uvr_mdx.onnx",
        }
    }

    /// Returns expected sample rate.
    pub fn sample_rate(&self) -> u32 {
        match self {
            SeparationModel::DemucsHybrid => 44100,
            SeparationModel::DemucsMdx => 44100,
            SeparationModel::UvrMdx => 44100,
        }
    }

    /// Returns the stems this model can separate.
    pub fn stems(&self) -> &'static [AudioStem] {
        match self {
            SeparationModel::DemucsHybrid => &[
                AudioStem::Vocals,
                AudioStem::Drums,
                AudioStem::Bass,
                AudioStem::Other,
            ],
            SeparationModel::DemucsMdx => &[
                AudioStem::Vocals,
                AudioStem::Instrumental,
            ],
            SeparationModel::UvrMdx => &[
                AudioStem::Vocals,
                AudioStem::Instrumental,
            ],
        }
    }
}

/// Audio stems that can be separated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioStem {
    /// Vocal track.
    Vocals,
    /// Instrumental (non-vocal).
    Instrumental,
    /// Drums/percussion.
    Drums,
    /// Bass instruments.
    Bass,
    /// Other instruments.
    Other,
}

impl AudioStem {
    /// Returns a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            AudioStem::Vocals => "vocals",
            AudioStem::Instrumental => "instrumental",
            AudioStem::Drums => "drums",
            AudioStem::Bass => "bass",
            AudioStem::Other => "other",
        }
    }
}

/// Configuration for audio separation.
#[derive(Debug, Clone)]
pub struct SeparationConfig {
    /// The separation model to use.
    pub model: SeparationModel,

    /// Path to the ONNX model file.
    pub model_path: Option<PathBuf>,

    /// Whether to use GPU acceleration.
    pub use_gpu: bool,

    /// CUDA device ID.
    pub gpu_device_id: i32,

    /// Which stems to extract.
    pub stems: Vec<AudioStem>,

    /// Chunk size for processing (in samples).
    pub chunk_size: usize,

    /// Overlap between chunks (0.0 - 0.5).
    pub overlap: f32,
}

impl Default for SeparationConfig {
    fn default() -> Self {
        Self {
            model: SeparationModel::default(),
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            stems: vec![AudioStem::Vocals, AudioStem::Instrumental],
            chunk_size: 44100 * 10, // 10 seconds
            overlap: 0.25,
        }
    }
}

impl SeparationConfig {
    /// Create a new configuration with the specified model.
    pub fn new(model: SeparationModel) -> Self {
        Self {
            model,
            stems: model.stems().to_vec(),
            ..Default::default()
        }
    }

    /// Set the model path.
    pub fn with_model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Enable or disable GPU.
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set which stems to extract.
    pub fn with_stems(mut self, stems: Vec<AudioStem>) -> Self {
        self.stems = stems;
        self
    }

    /// Extract only vocals.
    pub fn vocals_only(mut self) -> Self {
        self.stems = vec![AudioStem::Vocals];
        self
    }

    /// Extract only instrumental.
    pub fn instrumental_only(mut self) -> Self {
        self.stems = vec![AudioStem::Instrumental];
        self
    }

    /// Get effective model path.
    pub fn effective_model_path(&self) -> PathBuf {
        if let Some(ref path) = self.model_path {
            path.clone()
        } else {
            default_model_path(self.model.default_filename())
        }
    }
}

fn default_model_path(filename: &str) -> PathBuf {
    let home = dirs_next().unwrap_or_else(|| PathBuf::from("."));
    home.join(".xeno-lib").join("models").join(filename)
}

fn dirs_next() -> Option<PathBuf> {
    #[cfg(windows)]
    { std::env::var("USERPROFILE").ok().map(PathBuf::from) }
    #[cfg(not(windows))]
    { std::env::var("HOME").ok().map(PathBuf::from) }
}
