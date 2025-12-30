//! Style transfer configuration.

use std::path::PathBuf;

/// Pre-trained style models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PretrainedStyle {
    /// Mosaic tile pattern.
    Mosaic,
    /// Candy/pop art style.
    Candy,
    /// Udnie abstract style.
    Udnie,
    /// Starry Night (Van Gogh).
    StarryNight,
    /// The Scream (Munch).
    TheScream,
    /// Composition VII (Kandinsky).
    Kandinsky,
    /// Rain Princess.
    RainPrincess,
    /// Feathers texture.
    Feathers,
    /// La Muse (Picasso-inspired).
    LaMuse,
    /// Pointillism style.
    Pointillism,
}

impl PretrainedStyle {
    /// Get the model filename for this style.
    pub fn model_filename(&self) -> &'static str {
        match self {
            Self::Mosaic => "style_mosaic.onnx",
            Self::Candy => "style_candy.onnx",
            Self::Udnie => "style_udnie.onnx",
            Self::StarryNight => "style_starry_night.onnx",
            Self::TheScream => "style_the_scream.onnx",
            Self::Kandinsky => "style_kandinsky.onnx",
            Self::RainPrincess => "style_rain_princess.onnx",
            Self::Feathers => "style_feathers.onnx",
            Self::LaMuse => "style_la_muse.onnx",
            Self::Pointillism => "style_pointillism.onnx",
        }
    }

    /// Get display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Mosaic => "Mosaic",
            Self::Candy => "Candy",
            Self::Udnie => "Udnie",
            Self::StarryNight => "Starry Night",
            Self::TheScream => "The Scream",
            Self::Kandinsky => "Kandinsky",
            Self::RainPrincess => "Rain Princess",
            Self::Feathers => "Feathers",
            Self::LaMuse => "La Muse",
            Self::Pointillism => "Pointillism",
        }
    }

    /// List all available styles.
    pub fn all() -> &'static [PretrainedStyle] {
        &[
            Self::Mosaic,
            Self::Candy,
            Self::Udnie,
            Self::StarryNight,
            Self::TheScream,
            Self::Kandinsky,
            Self::RainPrincess,
            Self::Feathers,
            Self::LaMuse,
            Self::Pointillism,
        ]
    }
}

/// Style transfer configuration.
#[derive(Debug, Clone)]
pub struct StyleConfig {
    /// Pre-trained style (mutually exclusive with custom_style_path).
    pub pretrained_style: Option<PretrainedStyle>,
    /// Custom style image path (for arbitrary style transfer).
    pub custom_style_path: Option<PathBuf>,
    /// Custom model path (overrides default).
    pub model_path: Option<PathBuf>,
    /// Style strength (0.0 - 1.0).
    pub strength: f32,
    /// Preserve original colors.
    pub preserve_colors: bool,
    /// Use GPU acceleration.
    pub use_gpu: bool,
    /// GPU device ID.
    pub gpu_device_id: i32,
    /// Maximum output dimension (for memory management).
    pub max_dimension: u32,
}

impl Default for StyleConfig {
    fn default() -> Self {
        Self {
            pretrained_style: Some(PretrainedStyle::Mosaic),
            custom_style_path: None,
            model_path: None,
            strength: 1.0,
            preserve_colors: false,
            use_gpu: true,
            gpu_device_id: 0,
            max_dimension: 1024,
        }
    }
}

impl StyleConfig {
    /// Create configuration for a pre-trained style.
    pub fn new(style: PretrainedStyle) -> Self {
        Self {
            pretrained_style: Some(style),
            ..Default::default()
        }
    }

    /// Create configuration for arbitrary style transfer.
    pub fn arbitrary<P: Into<PathBuf>>(style_image: P) -> Self {
        Self {
            pretrained_style: None,
            custom_style_path: Some(style_image.into()),
            ..Default::default()
        }
    }

    /// Set style strength.
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Preserve original colors.
    pub fn with_preserve_colors(mut self, preserve: bool) -> Self {
        self.preserve_colors = preserve;
        self
    }

    /// Set custom model path.
    pub fn with_model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Use CPU only.
    pub fn cpu_only(mut self) -> Self {
        self.use_gpu = false;
        self
    }

    /// Set maximum dimension.
    pub fn with_max_dimension(mut self, max_dim: u32) -> Self {
        self.max_dimension = max_dim;
        self
    }

    /// Get effective model path.
    pub fn effective_model_path(&self) -> PathBuf {
        if let Some(ref path) = self.model_path {
            path.clone()
        } else {
            let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            let filename = self.pretrained_style
                .map(|s| s.model_filename())
                .unwrap_or("style_arbitrary.onnx");
            home.join(".xeno-lib").join("models").join(filename)
        }
    }
}

fn dirs_home_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE").ok().map(PathBuf::from)
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}

mod dirs {
    use std::path::PathBuf;

    pub fn home_dir() -> Option<PathBuf> {
        super::dirs_home_dir()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pretrained_styles() {
        let styles = PretrainedStyle::all();
        assert!(styles.len() >= 10);

        for style in styles {
            assert!(!style.model_filename().is_empty());
            assert!(!style.display_name().is_empty());
        }
    }

    #[test]
    fn test_config_builder() {
        let config = StyleConfig::new(PretrainedStyle::StarryNight)
            .with_strength(0.8)
            .with_preserve_colors(true)
            .cpu_only();

        assert_eq!(config.pretrained_style, Some(PretrainedStyle::StarryNight));
        assert!((config.strength - 0.8).abs() < 0.001);
        assert!(config.preserve_colors);
        assert!(!config.use_gpu);
    }

    #[test]
    fn test_arbitrary_config() {
        let config = StyleConfig::arbitrary("custom_style.jpg");
        assert!(config.pretrained_style.is_none());
        assert!(config.custom_style_path.is_some());
    }
}
