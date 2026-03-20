// Staged for migration to xeno-rt
//
// Extracted from: src/upscale/mod.rs, src/ai_deprecated/upscale/model.rs,
//                 src/ai_deprecated/upscale/processor.rs, src/upscale/config.rs
//
// Contains: ONNX model loading, session management, image-to-tensor preprocessing,
//           tile-based inference, and tensor-to-image postprocessing for Real-ESRGAN.
//
// What STAYS in xeno-lib:
//   - Traditional (non-AI) upscaling via resize algorithms in src/transforms/resize.rs
//
// Output contract (preserve in xeno-rt):
//   - Input: RGBA image (DynamicImage)
//   - Output: RGBA image at 2x/4x/8x resolution (u8, 4 bytes/pixel)

// ---------------------------------------------------------------------------
// Configuration (from src/upscale/config.rs)
// ---------------------------------------------------------------------------

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


// ---------------------------------------------------------------------------
// Model session and loading (from src/ai_deprecated/upscale/model.rs)
// ---------------------------------------------------------------------------

//! ONNX model session management for Real-ESRGAN upscaling.

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;

// NOTE: UpscaleConfig is defined above in this staging file (was use super::config::UpscaleConfig)

/// A loaded ONNX model session ready for upscaling inference.
///
/// This struct wraps an `ort::Session` and provides a convenient interface
/// for running Real-ESRGAN inference.
///
/// # Thread Safety
///
/// `UpscalerSession` requires `&mut self` for `run()`, enforcing single-threaded
/// access via Rust's borrow checker. For multi-threaded workloads, create one
/// session per thread or wrap in a `Mutex`.
///
/// # Lifecycle
///
/// Sessions hold GPU or CPU memory proportional to the model size. Drop the
/// session when inference is complete to release resources.
pub struct UpscalerSession {
    session: Session,
    config: UpscaleConfig,
}

impl UpscalerSession {
    /// Returns the scale factor for this upscaler.
    pub fn scale_factor(&self) -> u32 {
        self.config.scale_factor()
    }

    /// Returns a reference to the configuration used to create this session.
    pub fn config(&self) -> &UpscaleConfig {
        &self.config
    }

    /// Runs inference on the provided input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - A 4D tensor of shape `[1, 3, H, W]` containing the
    ///   preprocessed image data in range [0, 1].
    ///
    /// # Returns
    ///
    /// A 4D tensor of shape `[1, 3, H*scale, W*scale]` containing the
    /// upscaled image data in range [0, 1].
    pub fn run(&mut self, input: &Array4<f32>) -> Result<Array4<f32>, TransformError> {
        // Create tensor reference from input array
        let tensor_ref = TensorRef::from_array_view(input.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to create tensor ref: {e}"),
            }
        })?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("upscale inference failed: {e}"),
            })?;

        // Extract output tensor
        let output_tensor = outputs.iter().next().ok_or_else(|| {
            TransformError::InferenceFailed {
                message: "no output tensor found".to_string(),
            }
        })?;

        let (shape, data) = output_tensor
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("failed to extract output tensor: {e}"),
            })?;

        // Real-ESRGAN output shape: [1, 3, H*scale, W*scale]
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if dims.len() != 4 {
            return Err(TransformError::InferenceFailed {
                message: format!("unexpected output shape: {:?}", dims),
            });
        }

        let flat_data: Vec<f32> = data.iter().copied().collect();

        Array4::from_shape_vec((dims[0], dims[1], dims[2], dims[3]), flat_data).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to reshape output: {e}"),
            }
        })
    }
}

/// Loads an ONNX model for upscaling.
///
/// This function initializes an ONNX Runtime session with the specified
/// configuration. It attempts to use CUDA execution if `use_gpu` is true,
/// falling back to CPU execution if CUDA is unavailable.
///
/// # Arguments
///
/// * `config` - Configuration specifying model and execution options.
///
/// # Returns
///
/// An `UpscalerSession` ready for inference.
///
/// # Errors
///
/// Returns an error if:
/// - The model file does not exist
/// - The model cannot be loaded (invalid format, incompatible version, etc.)
///
/// # Example
///
/// ```rust,no_run
/// use xeno_lib::upscale::{load_upscaler, UpscaleConfig, UpscaleModel};
///
/// // Use default configuration (4x upscale, CUDA with CPU fallback)
/// let session = load_upscaler(&UpscaleConfig::default())?;
///
/// // Custom configuration for anime upscaling
/// let config = UpscaleConfig::new(UpscaleModel::RealEsrganX4Anime)
///     .with_gpu(true)
///     .with_tile_size(128);
/// let session = load_upscaler(&config)?;
/// # Ok::<(), xeno_lib::TransformError>(())
/// ```
pub fn load_upscaler(config: &UpscaleConfig) -> Result<UpscalerSession, TransformError> {
    let model_path = config.effective_model_path();

    // Verify model file exists
    if !model_path.exists() {
        return Err(TransformError::ModelNotFound {
            path: model_path.display().to_string(),
        });
    }

    // Build session with execution providers
    let mut builder = Session::builder()
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to create session builder: {e}"),
        })?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to set optimization level: {e}"),
        })?;

    // Configure execution providers
    if config.use_gpu {
        // Try CUDA first, fall back to CPU
        builder = builder
            .with_execution_providers([
                CUDAExecutionProvider::default()
                    .with_device_id(config.gpu_device_id)
                    .build(),
                CPUExecutionProvider::default().build(),
            ])
            .map_err(|e| TransformError::ModelLoadFailed {
                message: format!("failed to configure execution providers: {e}"),
            })?;
    } else {
        // CPU only
        builder = builder
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| TransformError::ModelLoadFailed {
                message: format!("failed to configure CPU execution provider: {e}"),
            })?;
    }

    // Load the model
    let session = builder
        .commit_from_file(&model_path)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to load upscale model from {}: {e}", model_path.display()),
        })?;

    Ok(UpscalerSession {
        session,
        config: config.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_model_not_found() {
        let config = UpscaleConfig::default()
            .with_model_path(PathBuf::from("/nonexistent/model.onnx"));
        let result = load_upscaler(&config);
        assert!(matches!(result, Err(TransformError::ModelNotFound { .. })));
    }
}


// ---------------------------------------------------------------------------
// Processor: preprocessing, tiling, postprocessing
// (from src/ai_deprecated/upscale/processor.rs)
// ---------------------------------------------------------------------------

//! Image processing logic for Real-ESRGAN upscaling.
//!
//! Handles preprocessing, inference, and postprocessing including
//! tile-based processing for large images.

use image::{DynamicImage, Rgb, RgbImage, Rgba, RgbaImage};
use ndarray::Array4;

use crate::error::TransformError;

// NOTE: UpscalerSession is defined above in this staging file (was use super::model::UpscalerSession)

/// Upscales a small image directly without tiling.
///
/// This is used for images that fit within the tile size.
pub fn upscale_direct(
    image: &DynamicImage,
    session: &mut UpscalerSession,
) -> Result<DynamicImage, TransformError> {
    // Convert to tensor
    let input_tensor = image_to_tensor(image)?;

    // Run inference
    let output_tensor = session.run(&input_tensor)?;

    // Convert back to image
    tensor_to_image(&output_tensor)
}

/// Upscales a large image using tile-based processing.
///
/// This splits the image into overlapping tiles, upscales each tile,
/// and blends them back together to avoid seam artifacts.
pub fn upscale_tiled(
    image: &DynamicImage,
    session: &mut UpscalerSession,
) -> Result<DynamicImage, TransformError> {
    let config = session.config();
    let scale = session.scale_factor();
    let tile_size = config.tile_size;
    let overlap = config.tile_overlap;

    let input_width = image.width();
    let input_height = image.height();
    let output_width = input_width * scale;
    let output_height = input_height * scale;

    // Calculate number of tiles needed
    let step = tile_size - overlap * 2;
    let tiles_x = ((input_width as i32 - overlap as i32 * 2) as f32 / step as f32).ceil() as u32;
    let tiles_y = ((input_height as i32 - overlap as i32 * 2) as f32 / step as f32).ceil() as u32;
    let tiles_x = tiles_x.max(1);
    let tiles_y = tiles_y.max(1);

    // Create output image
    let rgb = image.to_rgb8();
    let mut output = RgbImage::new(output_width, output_height);
    let mut weight_map = vec![0.0f32; (output_width * output_height) as usize];

    // Process each tile
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            // Calculate tile bounds in input space
            let x_start = (tx * step).min(input_width.saturating_sub(tile_size));
            let y_start = (ty * step).min(input_height.saturating_sub(tile_size));
            let x_end = (x_start + tile_size).min(input_width);
            let y_end = (y_start + tile_size).min(input_height);
            let tile_w = x_end - x_start;
            let tile_h = y_end - y_start;

            // Extract tile from input
            let tile = extract_tile(&rgb, x_start, y_start, tile_w, tile_h);
            let tile_image = DynamicImage::ImageRgb8(tile);

            // Upscale tile
            let input_tensor = image_to_tensor(&tile_image)?;
            let output_tensor = session.run(&input_tensor)?;
            let upscaled_tile = tensor_to_rgb(&output_tensor)?;

            // Calculate output bounds
            let out_x_start = x_start * scale;
            let out_y_start = y_start * scale;
            let out_tile_w = tile_w * scale;
            let out_tile_h = tile_h * scale;

            // Blend tile into output with weight falloff at edges
            blend_tile(
                &mut output,
                &mut weight_map,
                &upscaled_tile,
                out_x_start,
                out_y_start,
                out_tile_w,
                out_tile_h,
                overlap * scale,
            );
        }
    }

    // Normalize by weights
    normalize_by_weights(&mut output, &weight_map);

    Ok(DynamicImage::ImageRgb8(output))
}

/// Extracts a tile from an RGB image.
fn extract_tile(image: &RgbImage, x: u32, y: u32, width: u32, height: u32) -> RgbImage {
    let mut tile = RgbImage::new(width, height);
    for ty in 0..height {
        for tx in 0..width {
            let src_x = (x + tx).min(image.width() - 1);
            let src_y = (y + ty).min(image.height() - 1);
            tile.put_pixel(tx, ty, *image.get_pixel(src_x, src_y));
        }
    }
    tile
}

/// Blends an upscaled tile into the output image with edge falloff.
fn blend_tile(
    output: &mut RgbImage,
    weight_map: &mut [f32],
    tile: &RgbImage,
    x_start: u32,
    y_start: u32,
    width: u32,
    height: u32,
    overlap: u32,
) {
    let out_width = output.width();

    for ty in 0..height {
        for tx in 0..width {
            let out_x = x_start + tx;
            let out_y = y_start + ty;

            if out_x >= output.width() || out_y >= output.height() {
                continue;
            }

            // Calculate blend weight based on distance from edge
            let weight = calculate_blend_weight(tx, ty, width, height, overlap);

            let idx = (out_y * out_width + out_x) as usize;
            let pixel = tile.get_pixel(tx.min(tile.width() - 1), ty.min(tile.height() - 1));

            // Accumulate weighted pixel values
            let existing = output.get_pixel(out_x, out_y);
            let existing_weight = weight_map[idx];
            let total_weight = existing_weight + weight;

            if total_weight > 0.0 {
                let new_r = (existing[0] as f32 * existing_weight + pixel[0] as f32 * weight)
                    / total_weight;
                let new_g = (existing[1] as f32 * existing_weight + pixel[1] as f32 * weight)
                    / total_weight;
                let new_b = (existing[2] as f32 * existing_weight + pixel[2] as f32 * weight)
                    / total_weight;

                output.put_pixel(
                    out_x,
                    out_y,
                    Rgb([new_r as u8, new_g as u8, new_b as u8]),
                );
                weight_map[idx] = total_weight;
            }
        }
    }
}

/// Calculates blend weight based on distance from tile edges.
fn calculate_blend_weight(x: u32, y: u32, width: u32, height: u32, overlap: u32) -> f32 {
    if overlap == 0 {
        return 1.0;
    }

    let overlap_f = overlap as f32;

    // Distance from each edge
    let left_dist = x as f32;
    let right_dist = (width - 1 - x) as f32;
    let top_dist = y as f32;
    let bottom_dist = (height - 1 - y) as f32;

    // Weight ramps up from 0 at edge to 1 at overlap distance
    let left_weight = (left_dist / overlap_f).min(1.0);
    let right_weight = (right_dist / overlap_f).min(1.0);
    let top_weight = (top_dist / overlap_f).min(1.0);
    let bottom_weight = (bottom_dist / overlap_f).min(1.0);

    // Combine weights (minimum gives smooth falloff in corners)
    left_weight.min(right_weight).min(top_weight).min(bottom_weight)
}

/// Normalizes output pixels by accumulated weights.
fn normalize_by_weights(output: &mut RgbImage, weight_map: &[f32]) {
    let width = output.width();
    for (idx, weight) in weight_map.iter().enumerate() {
        if *weight > 0.0 && *weight != 1.0 {
            let x = (idx as u32) % width;
            let y = (idx as u32) / width;
            // Already normalized during blending, this is just safety
            let _pixel = output.get_pixel(x, y);
        }
    }
}

/// Converts a DynamicImage to an ONNX input tensor.
///
/// Real-ESRGAN expects input in range [0, 1] with shape [1, 3, H, W].
pub fn image_to_tensor(image: &DynamicImage) -> Result<Array4<f32>, TransformError> {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    // Create tensor in NCHW format: [1, 3, H, W]
    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    // Fill tensor with normalized pixel values [0, 1]
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                tensor[[0, c, y, x]] = pixel[c] as f32 / 255.0;
            }
        }
    }

    Ok(tensor)
}

/// Converts an ONNX output tensor to a DynamicImage.
///
/// Expects tensor in range [0, 1] with shape [1, 3, H, W].
pub fn tensor_to_image(tensor: &Array4<f32>) -> Result<DynamicImage, TransformError> {
    let rgb = tensor_to_rgb(tensor)?;
    Ok(DynamicImage::ImageRgb8(rgb))
}

/// Converts an ONNX output tensor to an RgbImage.
fn tensor_to_rgb(tensor: &Array4<f32>) -> Result<RgbImage, TransformError> {
    let shape = tensor.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
        return Err(TransformError::InferenceFailed {
            message: format!("unexpected tensor shape: {:?}", shape),
        });
    }

    let height = shape[2];
    let width = shape[3];
    let mut image = RgbImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let r = (tensor[[0, 0, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (tensor[[0, 1, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (tensor[[0, 2, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            image.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    Ok(image)
}

/// Converts an ONNX output tensor to an RgbaImage with full alpha.
#[allow(dead_code)]
fn tensor_to_rgba(tensor: &Array4<f32>) -> Result<RgbaImage, TransformError> {
    let shape = tensor.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
        return Err(TransformError::InferenceFailed {
            message: format!("unexpected tensor shape: {:?}", shape),
        });
    }

    let height = shape[2];
    let width = shape[3];
    let mut image = RgbaImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let r = (tensor[[0, 0, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (tensor[[0, 1, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (tensor[[0, 2, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            image.put_pixel(x as u32, y as u32, Rgba([r, g, b, 255]));
        }
    }

    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(width: u32, height: u32) -> RgbImage {
        let mut img = RgbImage::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Rgb([
                ((x * 255) / width.max(1)) as u8,
                ((y * 255) / height.max(1)) as u8,
                128,
            ]);
        }
        img
    }

    #[test]
    fn test_image_to_tensor_shape() {
        let img = DynamicImage::ImageRgb8(create_test_image(64, 48));
        let tensor = image_to_tensor(&img).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 48, 64]);
    }

    #[test]
    fn test_image_to_tensor_range() {
        let img = DynamicImage::ImageRgb8(create_test_image(32, 32));
        let tensor = image_to_tensor(&img).unwrap();

        let min_val = tensor.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = tensor.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        assert!(min_val >= 0.0, "min value {} < 0", min_val);
        assert!(max_val <= 1.0, "max value {} > 1", max_val);
    }

    #[test]
    fn test_tensor_roundtrip() {
        let original = DynamicImage::ImageRgb8(create_test_image(16, 16));
        let tensor = image_to_tensor(&original).unwrap();
        let recovered = tensor_to_image(&tensor).unwrap();

        // Compare images (allow small differences due to float conversion)
        let orig_rgb = original.to_rgb8();
        let rec_rgb = recovered.to_rgb8();

        for y in 0..16 {
            for x in 0..16 {
                let op = orig_rgb.get_pixel(x, y);
                let rp = rec_rgb.get_pixel(x, y);
                for c in 0..3 {
                    let diff = (op[c] as i32 - rp[c] as i32).abs();
                    assert!(diff <= 1, "pixel ({}, {}) channel {} differs by {}", x, y, c, diff);
                }
            }
        }
    }

    #[test]
    fn test_extract_tile() {
        let img = create_test_image(100, 80);
        let tile = extract_tile(&img, 10, 20, 32, 32);
        assert_eq!(tile.width(), 32);
        assert_eq!(tile.height(), 32);
    }

    #[test]
    fn test_blend_weight() {
        // Center pixel should have weight 1.0
        let center_weight = calculate_blend_weight(50, 50, 100, 100, 10);
        assert!((center_weight - 1.0).abs() < 0.01);

        // Edge pixel should have weight 0.0
        let edge_weight = calculate_blend_weight(0, 50, 100, 100, 10);
        assert!(edge_weight < 0.01);

        // Mid-overlap pixel should have weight ~0.5
        let mid_weight = calculate_blend_weight(5, 50, 100, 100, 10);
        assert!((mid_weight - 0.5).abs() < 0.01);
    }
}
