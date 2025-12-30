//! ONNX model session for image inpainting.

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;
use super::config::InpaintConfig;

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
