// Staged for migration to xeno-rt
//
// Extracted from: src/background/mod.rs, src/ai_deprecated/background/model.rs,
//                 src/ai_deprecated/background/preprocess.rs
//
// Contains: ONNX model loading, session management, image-to-tensor preprocessing,
//           inference execution, and tensor output extraction for BiRefNet.
//
// What STAYS in xeno-lib:
//   - postprocess::apply_mask() -- pure image processing (guided filter, morphological ops)
//
// Output contract (preserve in xeno-rt):
//   - Input: RGBA image (DynamicImage)
//   - Output: RGBA image with transparent background (u8, 4 bytes/pixel)
//   - Mask intermediate: single-channel f32 Array2 (0.0=bg, 1.0=fg)

// ---------------------------------------------------------------------------
// Model session and loading (from src/ai_deprecated/background/model.rs)
// ---------------------------------------------------------------------------

//! ONNX model session management for background removal.

use std::path::PathBuf;

use ndarray::{Array2, Array4};
use ort::{
    execution_providers::{CUDAExecutionProvider, CPUExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;

/// Configuration for background removal model loading and inference.
#[derive(Debug, Clone)]
pub struct BackgroundRemovalConfig {
    /// Path to the ONNX model file.
    /// Default: `~/.xeno-lib/models/birefnet-general.onnx`
    pub model_path: PathBuf,

    /// Whether to attempt GPU (CUDA) acceleration.
    /// Falls back to CPU if CUDA is unavailable.
    pub use_gpu: bool,

    /// CUDA device ID when using GPU acceleration.
    pub gpu_device_id: i32,

    /// Confidence threshold for foreground detection (0.0 - 1.0).
    /// Pixels with confidence above this threshold are kept as foreground.
    pub confidence_threshold: f32,
}

impl Default for BackgroundRemovalConfig {
    fn default() -> Self {
        Self {
            model_path: default_model_path(),
            use_gpu: true,
            gpu_device_id: 0,
            confidence_threshold: 0.1,
        }
    }
}

/// Returns the default model path based on the user's home directory.
fn default_model_path() -> PathBuf {
    let home = dirs_next().unwrap_or_else(|| PathBuf::from("."));
    home.join(".xeno-lib").join("models").join("birefnet-general.onnx")
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

/// A loaded ONNX model session ready for inference.
///
/// This struct wraps an `ort::Session` and provides a convenient interface
/// for running background removal inference.
///
/// # Thread Safety
///
/// `ModelSession` requires `&mut self` for `run()`, enforcing single-threaded
/// access via Rust's borrow checker. While the underlying `ort::Session` is
/// thread-safe for concurrent inference, this wrapper uses exclusive access
/// to simplify lifetime management. For multi-threaded workloads, create one
/// `ModelSession` per thread or wrap in a `Mutex`.
///
/// # Lifecycle
///
/// Sessions hold GPU or CPU memory proportional to the model size. Drop the
/// session when inference is complete to release resources. There is no
/// built-in session caching; callers should reuse sessions across multiple
/// images and drop them explicitly when no longer needed.
pub struct ModelSession {
    session: Session,
    input_size: (u32, u32),
    config: BackgroundRemovalConfig,
}

impl ModelSession {
    /// Returns the expected input dimensions (width, height).
    pub fn input_size(&self) -> (u32, u32) {
        self.input_size
    }

    /// Returns a reference to the configuration used to create this session.
    pub fn config(&self) -> &BackgroundRemovalConfig {
        &self.config
    }

    /// Runs inference on the provided input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - A 4D tensor of shape `[1, 3, H, W]` containing the
    ///   preprocessed image data.
    ///
    /// # Returns
    ///
    /// A 2D array of shape `[H, W]` containing the predicted alpha mask.
    pub fn run(&mut self, input: &Array4<f32>) -> Result<Array2<f32>, TransformError> {
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
                message: format!("inference failed: {e}"),
            })?;

        // Extract output mask
        // BiRefNet output shape: [1, 1, H, W]
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

        // Convert from [1, 1, H, W] to [H, W]
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let (h, w) = if dims.len() == 4 {
            (dims[2], dims[3])
        } else if dims.len() == 3 {
            (dims[1], dims[2])
        } else {
            (dims[0], dims[1])
        };

        let flat_data: Vec<f32> = data.iter().copied().collect();
        Array2::from_shape_vec((h, w), flat_data).map_err(|e| TransformError::InferenceFailed {
            message: format!("failed to reshape output: {e}"),
        })
    }
}

/// Loads an ONNX model for background removal.
///
/// This function initializes an ONNX Runtime session with the specified
/// configuration. It attempts to use CUDA execution if `use_gpu` is true,
/// falling back to CPU execution if CUDA is unavailable.
///
/// # Arguments
///
/// * `config` - Configuration specifying model path and execution options.
///
/// # Returns
///
/// A `ModelSession` ready for inference.
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
/// use xeno_lib::background::{load_model, BackgroundRemovalConfig};
///
/// // Use default configuration (CUDA with CPU fallback)
/// let session = load_model(&BackgroundRemovalConfig::default())?;
///
/// // Custom configuration
/// let config = BackgroundRemovalConfig {
///     model_path: "/path/to/custom/model.onnx".into(),
///     use_gpu: false,  // CPU only
///     ..Default::default()
/// };
/// let session = load_model(&config)?;
/// # Ok::<(), xeno_lib::TransformError>(())
/// ```
pub fn load_model(config: &BackgroundRemovalConfig) -> Result<ModelSession, TransformError> {
    // Verify model file exists
    if !config.model_path.exists() {
        return Err(TransformError::ModelNotFound {
            path: config.model_path.display().to_string(),
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
        })?
        .with_memory_pattern(true)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to enable memory pattern: {e}"),
        })?
        .with_intra_threads(4)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to set intra threads: {e}"),
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
        .commit_from_file(&config.model_path)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to load model: {e}"),
        })?;

    // BiRefNet uses 1024x1024 input
    let input_size = (1024, 1024);

    Ok(ModelSession {
        session,
        input_size,
        config: config.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BackgroundRemovalConfig::default();
        assert!(config.use_gpu);
        assert_eq!(config.gpu_device_id, 0);
        assert!((config.confidence_threshold - 0.1).abs() < f32::EPSILON);
        assert!(config.model_path.to_string_lossy().contains("birefnet-general.onnx"));
    }

    #[test]
    fn test_model_not_found() {
        let config = BackgroundRemovalConfig {
            model_path: PathBuf::from("/nonexistent/model.onnx"),
            ..Default::default()
        };
        let result = load_model(&config);
        assert!(matches!(result, Err(TransformError::ModelNotFound { .. })));
    }
}


// ---------------------------------------------------------------------------
// Preprocessing (from src/ai_deprecated/background/preprocess.rs)
// ---------------------------------------------------------------------------

//! Image preprocessing for ONNX inference.
//!
//! Converts `DynamicImage` to normalized tensors in the format expected by
//! the BiRefNet model.

use image::{imageops::FilterType, DynamicImage};
use ndarray::Array4;

use crate::error::TransformError;

/// BiRefNet normalization parameters (ImageNet standard).
/// The model expects input normalized with ImageNet mean/std.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Converts a `DynamicImage` to a normalized ONNX input tensor.
///
/// # Processing Steps
///
/// 1. Resize to target dimensions (bilinear interpolation)
/// 2. Convert to RGB f32 in range [0, 1]
/// 3. Normalize using BiRefNet/ImageNet mean/std values
/// 4. Transpose from HWC to CHW format
/// 5. Add batch dimension
///
/// # Arguments
///
/// * `image` - The input image to preprocess
/// * `target_size` - The target dimensions (width, height) for the model
///
/// # Returns
///
/// A 4D tensor of shape `[1, 3, H, W]` ready for inference.
pub fn image_to_tensor(
    image: &DynamicImage,
    target_size: (u32, u32),
) -> Result<Array4<f32>, TransformError> {
    let (target_width, target_height) = target_size;

    // Resize image to model input size
    let resized = image.resize_exact(target_width, target_height, FilterType::Lanczos3);

    // Convert to RGB8
    let rgb = resized.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    // Create tensor in CHW format with batch dimension: [1, 3, H, W]
    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    // Fill tensor with normalized pixel values
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);

            // Normalize: (pixel / 255.0 - mean) / std
            for c in 0..3 {
                let value = pixel[c] as f32 / 255.0;
                let normalized = (value - MEAN[c]) / STD[c];
                tensor[[0, c, y, x]] = normalized;
            }
        }
    }

    Ok(tensor)
}

/// Converts a `DynamicImage` to a tensor with parallel processing.
///
/// This is an optimized version that uses rayon for parallel pixel processing,
/// beneficial for large images.
#[allow(dead_code)]
pub fn image_to_tensor_parallel(
    image: &DynamicImage,
    target_size: (u32, u32),
) -> Result<Array4<f32>, TransformError> {
    use rayon::prelude::*;

    let (target_width, target_height) = target_size;

    // Resize image to model input size
    let resized = image.resize_exact(target_width, target_height, FilterType::Lanczos3);
    let rgb = resized.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    // Get raw pixel data
    let raw_pixels = rgb.as_raw();

    // Create flattened arrays for each channel
    let pixels_per_channel = height * width;
    let mut r_channel = vec![0.0f32; pixels_per_channel];
    let mut g_channel = vec![0.0f32; pixels_per_channel];
    let mut b_channel = vec![0.0f32; pixels_per_channel];

    // Parallel processing of pixels
    r_channel
        .par_iter_mut()
        .zip(g_channel.par_iter_mut())
        .zip(b_channel.par_iter_mut())
        .enumerate()
        .for_each(|(idx, ((r, g), b))| {
            let pixel_offset = idx * 3;
            *r = (raw_pixels[pixel_offset] as f32 / 255.0 - MEAN[0]) / STD[0];
            *g = (raw_pixels[pixel_offset + 1] as f32 / 255.0 - MEAN[1]) / STD[1];
            *b = (raw_pixels[pixel_offset + 2] as f32 / 255.0 - MEAN[2]) / STD[2];
        });

    // Combine channels into tensor
    let mut data = Vec::with_capacity(3 * pixels_per_channel);
    data.extend(r_channel);
    data.extend(g_channel);
    data.extend(b_channel);

    // Reshape to [1, 3, H, W]
    Array4::from_shape_vec((1, 3, height, width), data).map_err(|_| {
        TransformError::AllocationFailed {
            width: width as u32,
            height: height as u32,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn create_test_rgb_image(width: u32, height: u32) -> DynamicImage {
        let mut img = RgbImage::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Rgb([
                ((x * 255) / width) as u8,
                ((y * 255) / height) as u8,
                128,
            ]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_tensor_shape() {
        let img = create_test_rgb_image(100, 80);
        let tensor = image_to_tensor(&img, (1024, 1024)).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 1024, 1024]);
    }

    #[test]
    fn test_tensor_range() {
        let img = create_test_rgb_image(64, 64);
        let tensor = image_to_tensor(&img, (64, 64)).unwrap();

        // After ImageNet normalization, values are expected outside [0, 1].
        let min_val = tensor.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = tensor.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        assert!(min_val >= -2.2, "min value {} is out of range", min_val);
        assert!(max_val <= 2.7, "max value {} is out of range", max_val);
    }

    #[test]
    fn test_parallel_produces_same_result() {
        let img = create_test_rgb_image(128, 128);
        let tensor1 = image_to_tensor(&img, (128, 128)).unwrap();
        let tensor2 = image_to_tensor_parallel(&img, (128, 128)).unwrap();

        // Compare tensors (allow small floating point differences)
        for (a, b) in tensor1.iter().zip(tensor2.iter()) {
            assert!((a - b).abs() < 1e-5, "tensors differ: {} vs {}", a, b);
        }
    }
}
