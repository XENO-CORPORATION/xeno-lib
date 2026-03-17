//! Configuration for AI video generation.

use std::path::PathBuf;

/// Available video generation models.
///
/// # Model Comparison (2025-2026 Research)
///
/// - **Stable Video Diffusion (SVD)**: Image-to-video, 14-25 frames, good quality.
///   ONNX export via Optimum. Best for short clips from still images.
/// - **AnimateDiff**: Text-to-video via LoRA on Stable Diffusion. Creative control.
///   ONNX export supported via diffusers. Good for stylized animation.
///
/// Both require significant GPU memory (~8GB VRAM minimum).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoGenModel {
    /// Stable Video Diffusion — image-to-video generation.
    #[default]
    StableVideoDiffusion,
    /// SVD XT — extended to 25 frames.
    StableVideoDiffusionXt,
    /// AnimateDiff — text/image to animated video.
    AnimateDiff,
}

impl VideoGenModel {
    /// Default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            Self::StableVideoDiffusion => "svd.onnx",
            Self::StableVideoDiffusionXt => "svd_xt.onnx",
            Self::AnimateDiff => "animatediff.onnx",
        }
    }

    /// Display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::StableVideoDiffusion => "Stable Video Diffusion (14 frames)",
            Self::StableVideoDiffusionXt => "SVD-XT (25 frames)",
            Self::AnimateDiff => "AnimateDiff (Text-to-Video)",
        }
    }

    /// Default number of output frames.
    pub fn default_frames(&self) -> u32 {
        match self {
            Self::StableVideoDiffusion => 14,
            Self::StableVideoDiffusionXt => 25,
            Self::AnimateDiff => 16,
        }
    }

    /// Default output resolution.
    pub fn default_resolution(&self) -> (u32, u32) {
        match self {
            Self::StableVideoDiffusion | Self::StableVideoDiffusionXt => (1024, 576),
            Self::AnimateDiff => (512, 512),
        }
    }
}

/// Configuration for video generation.
#[derive(Debug, Clone)]
pub struct VideoGenConfig {
    /// Model to use.
    pub model: VideoGenModel,
    /// Custom model path.
    pub model_path: Option<PathBuf>,
    /// Use GPU acceleration.
    pub use_gpu: bool,
    /// GPU device ID.
    pub gpu_device_id: i32,
    /// Number of frames to generate.
    pub num_frames: u32,
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
    /// Frames per second for output. Default: 7.
    pub fps: f32,
    /// Number of diffusion steps. Default: 25.
    pub num_steps: u32,
    /// Guidance scale. Default: 3.0.
    pub guidance_scale: f32,
    /// Motion bucket ID (SVD only). Controls amount of motion. Default: 127.
    pub motion_bucket_id: u32,
    /// Noise augmentation level (SVD). Default: 0.02.
    pub noise_aug_strength: f32,
    /// Random seed for reproducibility. None = random.
    pub seed: Option<u64>,
}

impl Default for VideoGenConfig {
    fn default() -> Self {
        let model = VideoGenModel::default();
        let (w, h) = model.default_resolution();
        Self {
            model,
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            num_frames: model.default_frames(),
            width: w,
            height: h,
            fps: 7.0,
            num_steps: 25,
            guidance_scale: 3.0,
            motion_bucket_id: 127,
            noise_aug_strength: 0.02,
            seed: None,
        }
    }
}

impl VideoGenConfig {
    /// Create config with specified model.
    pub fn new(model: VideoGenModel) -> Self {
        let (w, h) = model.default_resolution();
        Self {
            model,
            num_frames: model.default_frames(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    /// Set model path.
    pub fn with_model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Enable/disable GPU.
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set number of frames.
    pub fn with_frames(mut self, frames: u32) -> Self {
        self.num_frames = frames.clamp(1, 100);
        self
    }

    /// Set resolution.
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set FPS.
    pub fn with_fps(mut self, fps: f32) -> Self {
        self.fps = fps.clamp(1.0, 60.0);
        self
    }

    /// Set seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get effective model path.
    pub fn effective_model_path(&self) -> PathBuf {
        if let Some(ref path) = self.model_path {
            path.clone()
        } else {
            crate::model_utils::default_model_path(self.model.default_filename())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = VideoGenConfig::default();
        assert_eq!(config.model, VideoGenModel::StableVideoDiffusion);
        assert_eq!(config.num_frames, 14);
        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 576);
    }

    #[test]
    fn test_model_properties() {
        assert_eq!(VideoGenModel::StableVideoDiffusionXt.default_frames(), 25);
        assert_eq!(VideoGenModel::AnimateDiff.default_resolution(), (512, 512));
    }

    #[test]
    fn test_frame_clamping() {
        let config = VideoGenConfig::default().with_frames(200);
        assert_eq!(config.num_frames, 100);
    }
}
