// Staged for migration to xeno-rt
//
// Extracted from: src/colorize/mod.rs, src/ai_deprecated/colorize/model.rs,
//                 src/ai_deprecated/colorize/processor.rs, src/colorize/config.rs
//
// Contains: ONNX model loading, session management, grayscale-to-tensor preprocessing,
//           colorization inference, LAB-to-RGB conversion, and tensor postprocessing
//           for DDColor.
//
// What STAYS in xeno-lib:
//   - Nothing (all colorization code is inference-dependent)
//
// Output contract (preserve in xeno-rt):
//   - Input: Grayscale or B&W image (DynamicImage)
//   - Output: Colorized RGB image (u8, 4 bytes/pixel when converted to RGBA)

// ---------------------------------------------------------------------------
// Configuration (from src/colorize/config.rs)
// ---------------------------------------------------------------------------

//! Configuration for image colorization.

use std::path::PathBuf;

/// Available colorization models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorizeModel {
    /// DDColor - State-of-the-art dual decoder colorization.
    #[default]
    DDColor,

    /// DeOldify - Classic colorization model.
    DeOldify,
}

impl ColorizeModel {
    /// Returns the default model filename.
    pub fn default_filename(&self) -> &'static str {
        match self {
            ColorizeModel::DDColor => "ddcolor.onnx",
            ColorizeModel::DeOldify => "deoldify.onnx",
        }
    }

    /// Returns expected input size.
    pub fn input_size(&self) -> (u32, u32) {
        match self {
            ColorizeModel::DDColor => (512, 512),
            ColorizeModel::DeOldify => (256, 256),
        }
    }
}

/// Configuration for colorization.
#[derive(Debug, Clone)]
pub struct ColorizeConfig {
    /// The colorization model to use.
    pub model: ColorizeModel,

    /// Path to the ONNX model file.
    pub model_path: Option<PathBuf>,

    /// Whether to use GPU acceleration.
    pub use_gpu: bool,

    /// CUDA device ID.
    pub gpu_device_id: i32,

    /// Color saturation adjustment (0.0 - 2.0, 1.0 = no change).
    pub saturation: f32,

    /// Whether to preserve original luminance.
    pub preserve_luminance: bool,
}

impl Default for ColorizeConfig {
    fn default() -> Self {
        Self {
            model: ColorizeModel::default(),
            model_path: None,
            use_gpu: true,
            gpu_device_id: 0,
            saturation: 1.0,
            preserve_luminance: true,
        }
    }
}

impl ColorizeConfig {
    /// Create a new configuration with the specified model.
    pub fn new(model: ColorizeModel) -> Self {
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

    /// Set saturation adjustment.
    pub fn with_saturation(mut self, saturation: f32) -> Self {
        self.saturation = saturation.clamp(0.0, 2.0);
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
// Model session and loading (from src/ai_deprecated/colorize/model.rs)
// ---------------------------------------------------------------------------

//! ONNX model session for colorization.

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;
// NOTE: ColorizeConfig is defined above in this staging file (was use super::config::ColorizeConfig)

/// A loaded colorization model session.
pub struct ColorizerSession {
    session: Session,
    config: ColorizeConfig,
}

impl ColorizerSession {
    pub fn config(&self) -> &ColorizeConfig {
        &self.config
    }

    pub fn input_size(&self) -> (u32, u32) {
        self.config.model.input_size()
    }

    /// Runs colorization inference.
    pub fn run(&mut self, input: &Array4<f32>) -> Result<Array4<f32>, TransformError> {
        let tensor_ref = TensorRef::from_array_view(input.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("failed to create tensor ref: {e}"),
            }
        })?;

        let outputs = self
            .session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("colorization inference failed: {e}"),
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

/// Loads a colorization model.
pub fn load_colorizer(config: &ColorizeConfig) -> Result<ColorizerSession, TransformError> {
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
            message: format!("failed to load colorization model: {e}"),
        })?;

    Ok(ColorizerSession {
        session,
        config: config.clone(),
    })
}


// ---------------------------------------------------------------------------
// Processor: preprocessing, inference, LAB-to-RGB postprocessing
// (from src/ai_deprecated/colorize/processor.rs)
// ---------------------------------------------------------------------------

//! Colorization processing logic.

use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};
use ndarray::Array4;

use crate::error::TransformError;
// NOTE: ColorizerSession is defined above in this staging file (was use super::model::ColorizerSession)

/// Colorizes a grayscale image.
pub fn colorize_impl(
    image: &DynamicImage,
    session: &mut ColorizerSession,
) -> Result<DynamicImage, TransformError> {
    let (input_w, input_h) = session.input_size();
    let original_width = image.width();
    let original_height = image.height();

    // Convert to grayscale luminance
    let gray = image.to_luma8();

    // Resize to model input size
    let resized_gray = image::imageops::resize(&gray, input_w, input_h, FilterType::Lanczos3);

    // Convert grayscale to tensor (L channel in LAB-like format)
    let input_tensor = grayscale_to_tensor(&resized_gray)?;

    // Run colorization (model outputs AB channels)
    let output_tensor = session.run(&input_tensor)?;

    // Combine L with predicted AB and convert to RGB
    let colorized = tensor_to_rgb(&output_tensor, &resized_gray, session.config().saturation)?;

    // Resize back to original dimensions
    let final_image = colorized.resize_exact(original_width, original_height, FilterType::Lanczos3);

    Ok(final_image)
}

/// Converts grayscale image to tensor for colorization model.
fn grayscale_to_tensor(gray: &image::GrayImage) -> Result<Array4<f32>, TransformError> {
    let (width, height) = (gray.width() as usize, gray.height() as usize);

    // DDColor expects L channel normalized to [0, 1]
    let mut tensor = Array4::<f32>::zeros((1, 1, height, width));

    for y in 0..height {
        for x in 0..width {
            let pixel = gray.get_pixel(x as u32, y as u32);
            tensor[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
        }
    }

    Ok(tensor)
}

/// Converts model output (AB channels) combined with L to RGB image.
fn tensor_to_rgb(
    ab_tensor: &Array4<f32>,
    gray: &image::GrayImage,
    saturation: f32,
) -> Result<DynamicImage, TransformError> {
    let shape = ab_tensor.shape();

    // Handle different output shapes
    let (height, width) = if shape.len() == 4 {
        (shape[2], shape[3])
    } else {
        return Err(TransformError::InferenceFailed {
            message: format!("unexpected tensor shape: {:?}", shape),
        });
    };

    let mut image = RgbImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            // Get original luminance
            let l = gray.get_pixel(
                (x as u32).min(gray.width() - 1),
                (y as u32).min(gray.height() - 1),
            )[0] as f32 / 255.0;

            // Get predicted color channels
            // Models typically output RGB directly or AB channels
            let (r, g, b) = if shape[1] >= 3 {
                // RGB output
                let r = ab_tensor[[0, 0, y, x]];
                let g = ab_tensor[[0, 1, y, x]];
                let b = ab_tensor[[0, 2, y, x]];
                (r, g, b)
            } else if shape[1] == 2 {
                // AB output - convert LAB to RGB
                let a = ab_tensor[[0, 0, y, x]] * saturation;
                let b_ch = ab_tensor[[0, 1, y, x]] * saturation;
                lab_to_rgb(l, a, b_ch)
            } else {
                // Single channel - treat as grayscale
                (l, l, l)
            };

            // Clamp and convert to u8
            let r_u8 = (r * 255.0).clamp(0.0, 255.0) as u8;
            let g_u8 = (g * 255.0).clamp(0.0, 255.0) as u8;
            let b_u8 = (b * 255.0).clamp(0.0, 255.0) as u8;

            image.put_pixel(x as u32, y as u32, Rgb([r_u8, g_u8, b_u8]));
        }
    }

    Ok(DynamicImage::ImageRgb8(image))
}

/// Converts LAB color to RGB.
fn lab_to_rgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    // Convert LAB to XYZ
    let l_scaled = l * 100.0;
    let a_scaled = a * 128.0;
    let b_scaled = b * 128.0;

    let y = (l_scaled + 16.0) / 116.0;
    let x = a_scaled / 500.0 + y;
    let z = y - b_scaled / 200.0;

    let x = if x.powi(3) > 0.008856 {
        x.powi(3)
    } else {
        (x - 16.0 / 116.0) / 7.787
    };
    let y = if y.powi(3) > 0.008856 {
        y.powi(3)
    } else {
        (y - 16.0 / 116.0) / 7.787
    };
    let z = if z.powi(3) > 0.008856 {
        z.powi(3)
    } else {
        (z - 16.0 / 116.0) / 7.787
    };

    // Reference white D65
    let x = x * 0.95047;
    let z = z * 1.08883;

    // XYZ to RGB (sRGB)
    let r = x * 3.2406 - y * 1.5372 - z * 0.4986;
    let g = -x * 0.9689 + y * 1.8758 + z * 0.0415;
    let b = x * 0.0557 - y * 0.2040 + z * 1.0570;

    // Gamma correction
    let r = if r > 0.0031308 {
        1.055 * r.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * r
    };
    let g = if g > 0.0031308 {
        1.055 * g.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * g
    };
    let b = if b > 0.0031308 {
        1.055 * b.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * b
    };

    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grayscale_tensor_shape() {
        let gray = image::GrayImage::new(64, 64);
        let tensor = grayscale_to_tensor(&gray).unwrap();
        assert_eq!(tensor.shape(), &[1, 1, 64, 64]);
    }

    #[test]
    fn test_lab_to_rgb_white() {
        let (r, g, b) = lab_to_rgb(1.0, 0.0, 0.0);
        assert!(r > 0.9 && g > 0.9 && b > 0.9);
    }

    #[test]
    fn test_lab_to_rgb_black() {
        let (r, g, b) = lab_to_rgb(0.0, 0.0, 0.0);
        assert!(r < 0.1 && g < 0.1 && b < 0.1);
    }
}
