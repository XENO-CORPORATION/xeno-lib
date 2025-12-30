//! Style transfer ONNX model session.

use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

use crate::error::TransformError;
use super::config::StyleConfig;

/// A loaded style transfer model session.
pub struct StyleSession {
    session: Session,
    config: StyleConfig,
}

impl StyleSession {
    /// Get the configuration.
    pub fn config(&self) -> &StyleConfig {
        &self.config
    }

    /// Run style transfer inference.
    pub fn run(&mut self, input: &Array4<f32>) -> Result<Array4<f32>, TransformError> {
        let tensor_ref = TensorRef::from_array_view(input.view()).map_err(|e| {
            TransformError::InferenceFailed {
                message: format!("Failed to create tensor ref: {e}"),
            }
        })?;

        let outputs = self
            .session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("Style transfer inference failed: {e}"),
            })?;

        let output_tensor = outputs.iter().next().ok_or_else(|| {
            TransformError::InferenceFailed {
                message: "No output tensor found".to_string(),
            }
        })?;

        let (shape, data) = output_tensor
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| TransformError::InferenceFailed {
                message: format!("Failed to extract output tensor: {e}"),
            })?;

        // Reconstruct array from output
        let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        if shape_vec.len() != 4 {
            return Err(TransformError::InferenceFailed {
                message: format!("Expected 4D output, got {}D", shape_vec.len()),
            });
        }

        let output_data: Vec<f32> = data.iter().copied().collect();
        Array4::from_shape_vec(
            (shape_vec[0], shape_vec[1], shape_vec[2], shape_vec[3]),
            output_data,
        )
        .map_err(|e| TransformError::InferenceFailed {
            message: format!("Failed to reshape output: {e}"),
        })
    }
}

/// Load a style transfer model.
pub fn load_style_model(config: &StyleConfig) -> Result<StyleSession, TransformError> {
    let model_path = config.effective_model_path();

    if !model_path.exists() {
        return Err(TransformError::ModelNotFound {
            path: model_path.display().to_string(),
        });
    }

    let mut builder = Session::builder()
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("Failed to create session builder: {e}"),
        })?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("Failed to set optimization level: {e}"),
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
                message: format!("Failed to configure execution providers: {e}"),
            })?;
    } else {
        builder = builder
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| TransformError::ModelLoadFailed {
                message: format!("Failed to configure CPU: {e}"),
            })?;
    }

    let session = builder
        .commit_from_file(&model_path)
        .map_err(|e| TransformError::ModelLoadFailed {
            message: format!("Failed to load style transfer model: {e}"),
        })?;

    Ok(StyleSession {
        session,
        config: config.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_not_found() {
        let config = StyleConfig::new(super::super::config::PretrainedStyle::Mosaic)
            .with_model_path("/nonexistent/model.onnx");
        let result = load_style_model(&config);
        assert!(result.is_err());
    }
}
