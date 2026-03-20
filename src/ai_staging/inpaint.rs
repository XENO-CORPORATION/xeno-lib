// Staged for migration to xeno-rt
//
// Extracted from: src/inpaint/mod.rs, src/ai_deprecated/inpaint/model.rs,
//                 src/ai_deprecated/inpaint/processor.rs, src/inpaint/config.rs
//
// Contains: ONNX model loading, session management, image+mask tensor preprocessing,
//           inpainting inference, and tensor-to-image postprocessing for LaMa.
//
// What STAYS in xeno-lib:
//   - create_mask(), MaskRegion enum (pure geometry utilities in src/inpaint/mod.rs)
//
// Output contract (preserve in xeno-rt):
//   - Input: RGBA image + binary mask (DynamicImage pair)
//   - Output: RGBA image with masked regions filled (u8, 4 bytes/pixel)

// ---------------------------------------------------------------------------
// Configuration (from src/inpaint/config.rs)
// ---------------------------------------------------------------------------

//! Configuration for image inpainting/object removal.

use std::path::PathBuf;

/// Available inpainting models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InpaintModel {
    /// LaMa - Large Mask Inpainting with Fourier Convolutions.
    #[default]
    LaMa,

    /// MAT - Mask-Aware Transformer for Large Hole Inpainting.
    Mat,
}

impl InpaintModel {
    /// Returns the default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            InpaintModel::LaMa => "lama.onnx",
            InpaintModel::Mat => "mat.onnx",
        }
    }

    /// Returns expected input size (0 = dynamic).
    pub fn input_size(&self) -> (u32, u32) {
        match self {
            InpaintModel::LaMa => (512, 512),
            InpaintModel::Mat => (512, 512),
        }
    }
}

/// Configuration for inpainting.
#[derive(Debug, Clone)]
pub struct InpaintConfig {
    /// The inpainting model to use.
    pub model: InpaintModel,

    /// Path to the ONNX model file.
    pub model_path: Option<PathBuf>,

    /// Whether to use GPU acceleration.
    pub use_gpu: bool,

    /// CUDA device ID.
    pub gpu_device_id: i32,

    /// Whether to dilate the mask slightly for better results.
    pub dilate_mask: bool,

    /// Mask dilation radius in pixels.
    pub dilation_radius: u32,
}

impl Default for InpaintConfig {
    fn default() -> Self {
        Self {
            model: InpaintModel::default(),
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            dilate_mask: true,
            dilation_radius: 3,
        }
    }
}

impl InpaintConfig {
    /// Create a new configuration with the specified model.
    pub fn new(model: InpaintModel) -> Self {
        Self {
            model,
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

    /// Enable or disable mask dilation.
    pub fn with_mask_dilation(mut self, dilate: bool, radius: u32) -> Self {
        self.dilate_mask = dilate;
        self.dilation_radius = radius;
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


// ---------------------------------------------------------------------------
// Model session and loading (from src/ai_deprecated/inpaint/model.rs)
// ---------------------------------------------------------------------------

//! ONNX model session for image inpainting.

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;
// NOTE: InpaintConfig is defined above in this staging file (was use super::config::InpaintConfig)

/// A loaded inpainting model session.
pub struct InpainterSession {
    session: Session,
    config: InpaintConfig,
}

impl InpainterSession {
    pub fn config(&self) -> &InpaintConfig {
        &self.config
    }

    pub fn input_size(&self) -> (u32, u32) {
        self.config.model.input_size()
    }

    /// Runs inpainting inference.
    /// Takes an image and mask, returns the inpainted image.
    pub fn run(
        &mut self,
        image: &Array4<f32>,
        mask: &Array4<f32>,
    ) -> Result<Array4<f32>, TransformError> {
        let image_tensor = TensorRef::from_array_view(image.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to create image tensor ref: {e}"),
            }
        })?;

        let mask_tensor = TensorRef::from_array_view(mask.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to create mask tensor ref: {e}"),
            }
        })?;

        let outputs = self
            .session
            .run(ort::inputs![image_tensor, mask_tensor])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("inpainting inference failed: {e}"),
            })?;

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

        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let flat_data: Vec<f32> = data.iter().copied().collect();

        Array4::from_shape_vec((dims[0], dims[1], dims[2], dims[3]), flat_data).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to reshape output: {e}"),
            }
        })
    }
}

/// Loads an inpainting model.
pub fn load_inpainter(config: &InpaintConfig) -> Result<InpainterSession, TransformError> {
    let model_path = config.effective_model_path();

    if !model_path.exists() {
        return Err(TransformError::ModelNotFound {
            path: model_path.display().to_string(),
        });
    }

    let mut builder = Session::builder()
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to create session builder: {e}"),
        })?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to set optimization level: {e}"),
        })?;

    if config.use_gpu {
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
        builder = builder
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| TransformError::ModelLoadFailed {
                message: format!("failed to configure CPU: {e}"),
            })?;
    }

    let session = builder
        .commit_from_file(&model_path)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("failed to load inpainting model: {e}"),
        })?;

    Ok(InpainterSession {
        session,
        config: config.clone(),
    })
}


// ---------------------------------------------------------------------------
// Processor: preprocessing, mask tensor creation, inference, postprocessing
// (from src/ai_deprecated/inpaint/processor.rs)
// ---------------------------------------------------------------------------

//! Inpainting processing logic.

use image::{DynamicImage, Rgb, RgbImage, GrayImage, Luma, imageops::FilterType};
use ndarray::Array4;

use crate::error::TransformError;
// NOTE: InpainterSession is defined above in this staging file (was use super::model::InpainterSession)

/// Inpaints an image using a binary mask.
pub fn inpaint_impl(
    image: &DynamicImage,
    mask: &DynamicImage,
    session: &mut InpainterSession,
) -> Result<DynamicImage, TransformError> {
    let (input_w, input_h) = session.input_size();
    let original_width = image.width();
    let original_height = image.height();

    // Resize image to model input size
    let resized_img = image.resize_exact(input_w, input_h, FilterType::Lanczos3);
    let resized_mask = mask.resize_exact(input_w, input_h, FilterType::Nearest);

    // Optionally dilate mask
    let processed_mask = if session.config().dilate_mask {
        dilate_mask(&resized_mask.to_luma8(), session.config().dilation_radius)
    } else {
        resized_mask.to_luma8()
    };

    // Convert to tensors
    let image_tensor = image_to_tensor(&resized_img)?;
    let mask_tensor = mask_to_tensor(&processed_mask)?;

    // Run inpainting
    let output_tensor = session.run(&image_tensor, &mask_tensor)?;

    // Convert back to image
    let inpainted = tensor_to_image(&output_tensor)?;

    // Resize back to original dimensions
    let final_image = inpainted.resize_exact(original_width, original_height, FilterType::Lanczos3);

    Ok(final_image)
}

/// Dilates a binary mask by the specified radius.
fn dilate_mask(mask: &GrayImage, radius: u32) -> GrayImage {
    let width = mask.width();
    let height = mask.height();
    let mut dilated = GrayImage::new(width, height);
    let r = radius as i32;

    for y in 0..height {
        for x in 0..width {
            let mut max_val = 0u8;

            for dy in -r..=r {
                for dx in -r..=r {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;

                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let val = mask.get_pixel(nx as u32, ny as u32)[0];
                        max_val = max_val.max(val);
                    }
                }
            }

            dilated.put_pixel(x, y, Luma([max_val]));
        }
    }

    dilated
}

/// Converts image to tensor normalized to [0, 1].
fn image_to_tensor(image: &DynamicImage) -> Result<Array4<f32>, TransformError> {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

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

/// Converts grayscale mask to tensor (1 channel).
fn mask_to_tensor(mask: &GrayImage) -> Result<Array4<f32>, TransformError> {
    let (width, height) = (mask.width() as usize, mask.height() as usize);

    let mut tensor = Array4::<f32>::zeros((1, 1, height, width));

    for y in 0..height {
        for x in 0..width {
            let pixel = mask.get_pixel(x as u32, y as u32);
            // Binary mask: >127 = 1.0, else = 0.0
            tensor[[0, 0, y, x]] = if pixel[0] > 127 { 1.0 } else { 0.0 };
        }
    }

    Ok(tensor)
}

/// Converts tensor from [0, 1] range to image.
fn tensor_to_image(tensor: &Array4<f32>) -> Result<DynamicImage, TransformError> {
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

    Ok(DynamicImage::ImageRgb8(image))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(width: u32, height: u32) -> DynamicImage {
        let mut img = RgbImage::new(width, height);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Rgb([
                ((x * 255) / width.max(1)) as u8,
                ((y * 255) / height.max(1)) as u8,
                128,
            ]);
        }
        DynamicImage::ImageRgb8(img)
    }

    fn create_test_mask(width: u32, height: u32) -> GrayImage {
        let mut mask = GrayImage::new(width, height);
        // Create a circular mask in the center
        let cx = width / 2;
        let cy = height / 2;
        let radius = width.min(height) / 4;

        for y in 0..height {
            for x in 0..width {
                let dx = x as i32 - cx as i32;
                let dy = y as i32 - cy as i32;
                let dist = ((dx * dx + dy * dy) as f32).sqrt();

                if dist < radius as f32 {
                    mask.put_pixel(x, y, Luma([255]));
                } else {
                    mask.put_pixel(x, y, Luma([0]));
                }
            }
        }
        mask
    }

    #[test]
    fn test_image_tensor_roundtrip() {
        let img = create_test_image(64, 64);
        let tensor = image_to_tensor(&img).unwrap();
        let recovered = tensor_to_image(&tensor).unwrap();

        assert_eq!(recovered.width(), 64);
        assert_eq!(recovered.height(), 64);
    }

    #[test]
    fn test_mask_tensor_binary() {
        let mask = create_test_mask(64, 64);
        let tensor = mask_to_tensor(&mask).unwrap();

        // Check all values are 0 or 1
        for &val in tensor.iter() {
            assert!(val == 0.0 || val == 1.0, "mask values should be binary");
        }
    }

    #[test]
    fn test_mask_dilation() {
        let mask = create_test_mask(64, 64);
        let dilated = dilate_mask(&mask, 3);

        // Dilated mask should have more white pixels
        let original_white: u32 = mask.pixels().map(|p| if p[0] > 127 { 1 } else { 0 }).sum();
        let dilated_white: u32 = dilated.pixels().map(|p| if p[0] > 127 { 1 } else { 0 }).sum();

        assert!(dilated_white >= original_white, "dilation should not reduce mask area");
    }
}
