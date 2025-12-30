//! Configuration types for AI upscaling.

use std::path::PathBuf;

/// Available Real-ESRGAN upscaling models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpscaleModel {
    /// Real-ESRGAN x2 - 2x upscaling, general purpose.
    /// Good for mild enhancement without extreme magnification.
    RealEsrganX2,

    /// Real-ESRGAN x4plus - 4x upscaling, optimized for photos.
    /// Best quality for real-world photographs.
    #[default]
    RealEsrganX4Plus,

    /// Real-ESRGAN x4plus anime - 4x upscaling, optimized for anime/artwork.
    /// Produces cleaner lines and flatter colors suitable for illustrations.
    RealEsrganX4Anime,

    /// Real-ESRGAN x8 - 8x upscaling, extreme magnification.
    /// Use when you need maximum enlargement (e.g., 240p -> 4K).
    RealEsrganX8,
}

impl UpscaleModel {
    /// Returns the upscaling factor for this model.
    pub fn scale_factor(&self) -> u32 {
        match self {
            UpscaleModel::RealEsrganX2 => 2,
            UpscaleModel::RealEsrganX4Plus => 4,
            UpscaleModel::RealEsrganX4Anime => 4,
            UpscaleModel::RealEsrganX8 => 8,
        }
    }

    /// Returns the default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            UpscaleModel::RealEsrganX2 => "realesrgan_x2.onnx",
            UpscaleModel::RealEsrganX4Plus => "realesrgan_x4plus.onnx",
            UpscaleModel::RealEsrganX4Anime => "realesrgan_x4plus_anime.onnx",
            UpscaleModel::RealEsrganX8 => "realesrgan_x8.onnx",
        }
    }

    /// Returns a human-readable name for this model.
    pub fn display_name(&self) -> &'static str {
        match self {
            UpscaleModel::RealEsrganX2 => "Real-ESRGAN x2",
            UpscaleModel::RealEsrganX4Plus => "Real-ESRGAN x4+ (Photo)",
            UpscaleModel::RealEsrganX4Anime => "Real-ESRGAN x4+ (Anime)",
            UpscaleModel::RealEsrganX8 => "Real-ESRGAN x8",
        }
    }
}

/// Configuration for AI upscaling.
#[derive(Debug, Clone)]
pub struct UpscaleConfig {
    /// The upscaling model to use.
    pub model: UpscaleModel,

    /// Path to the ONNX model file.
    /// If None, uses default path: `~/.xeno-lib/models/{model_filename}`
    pub model_path: Option<PathBuf>,

    /// Whether to attempt GPU (CUDA) acceleration.
    /// Falls back to CPU if CUDA is unavailable.
    pub use_gpu: bool,

    /// CUDA device ID when using GPU acceleration.
    pub gpu_device_id: i32,

    /// Tile size for processing large images.
    /// Images larger than this will be split into tiles.
    /// Smaller = less VRAM, but slower. Default: 256
    pub tile_size: u32,

    /// Overlap between tiles to avoid seam artifacts.
    /// Should be at least 8-16 pixels. Default: 16
    pub tile_overlap: u32,

    /// Denoise strength (0.0 to 1.0).
    /// Higher values reduce noise but may lose detail. Default: 0.0
    pub denoise_strength: f32,
}

impl Default for UpscaleConfig {
    fn default() -> Self {
        Self {
            model: UpscaleModel::default(),
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            tile_size: 256,
            tile_overlap: 16,
            denoise_strength: 0.0,
        }
    }
}

impl UpscaleConfig {
    /// Create a new configuration with the specified model.
    pub fn new(model: UpscaleModel) -> Self {
        Self {
            model,
            ..Default::default()
        }
    }

    /// Set the model path explicitly.
    pub fn with_model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Enable or disable GPU acceleration.
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set the CUDA device ID.
    pub fn with_device_id(mut self, device_id: i32) -> Self {
        self.gpu_device_id = device_id;
        self
    }

    /// Set the tile size for processing large images.
    pub fn with_tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = tile_size.max(64); // Minimum 64px
        self
    }

    /// Set the tile overlap.
    pub fn with_tile_overlap(mut self, overlap: u32) -> Self {
        self.tile_overlap = overlap;
        self
    }

    /// Set the denoise strength.
    pub fn with_denoise(mut self, strength: f32) -> Self {
        self.denoise_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Get the effective model path.
    pub fn effective_model_path(&self) -> PathBuf {
        if let Some(ref path) = self.model_path {
            path.clone()
        } else {
            default_model_path(self.model.default_filename())
        }
    }

    /// Get the scale factor for the current model.
    pub fn scale_factor(&self) -> u32 {
        self.model.scale_factor()
    }
}

/// Returns the default model path based on the user's home directory.
fn default_model_path(filename: &str) -> PathBuf {
    let home = dirs_next().unwrap_or_else(|| PathBuf::from("."));
    home.join(".xeno-lib").join("models").join(filename)
}

/// Cross-platform home directory detection.
fn dirs_next() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE").ok().map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_factors() {
        assert_eq!(UpscaleModel::RealEsrganX2.scale_factor(), 2);
        assert_eq!(UpscaleModel::RealEsrganX4Plus.scale_factor(), 4);
        assert_eq!(UpscaleModel::RealEsrganX4Anime.scale_factor(), 4);
        assert_eq!(UpscaleModel::RealEsrganX8.scale_factor(), 8);
    }

    #[test]
    fn test_default_config() {
        let config = UpscaleConfig::default();
        assert_eq!(config.model, UpscaleModel::RealEsrganX4Plus);
        assert!(config.use_gpu);
        assert_eq!(config.tile_size, 256);
        assert_eq!(config.tile_overlap, 16);
    }

    #[test]
    fn test_config_builder() {
        let config = UpscaleConfig::new(UpscaleModel::RealEsrganX8)
            .with_gpu(false)
            .with_tile_size(512)
            .with_denoise(0.5);

        assert_eq!(config.model, UpscaleModel::RealEsrganX8);
        assert!(!config.use_gpu);
        assert_eq!(config.tile_size, 512);
        assert!((config.denoise_strength - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tile_size_minimum() {
        let config = UpscaleConfig::default().with_tile_size(10);
        assert_eq!(config.tile_size, 64); // Clamped to minimum
    }
}
