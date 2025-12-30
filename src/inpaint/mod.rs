//! AI-powered image inpainting using LaMa.
//!
//! This module provides object removal and content-aware fill
//! using deep learning models.
//!
//! # Features
//!
//! - Remove unwanted objects from images
//! - Fill in missing/damaged regions
//! - Content-aware background reconstruction
//! - GPU acceleration via CUDA
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::inpaint::{inpaint, load_inpainter, InpaintConfig};
//!
//! let config = InpaintConfig::default();
//! let mut inpainter = load_inpainter(&config)?;
//!
//! let image = image::open("photo.jpg")?;
//! let mask = image::open("mask.png")?; // White = area to remove
//!
//! let result = inpaint(&image, &mask, &mut inpainter)?;
//! result.save("cleaned.png")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Mask Format
//!
//! The mask should be a grayscale image where:
//! - White (255) = areas to inpaint/remove
//! - Black (0) = areas to preserve
//!
//! # Model Download
//!
//! Download LaMa ONNX model:
//! - LaMa: [GitHub](https://github.com/advimman/lama)
//!
//! Default path: `~/.xeno-lib/models/lama.onnx`

mod config;
mod model;
mod processor;

use image::DynamicImage;

pub use config::{InpaintConfig, InpaintModel};
pub use model::{load_inpainter, InpainterSession};

use crate::error::TransformError;

/// Inpaints an image using a binary mask.
///
/// # Arguments
///
/// * `image` - The input image
/// * `mask` - Binary mask (white = inpaint, black = keep)
/// * `session` - A loaded inpainter model session
///
/// # Returns
///
/// The inpainted image with masked regions filled.
pub fn inpaint(
    image: &DynamicImage,
    mask: &DynamicImage,
    session: &mut InpainterSession,
) -> Result<DynamicImage, TransformError> {
    processor::inpaint_impl(image, mask, session)
}

/// Quick inpaint function that loads the model and processes in one call.
pub fn inpaint_quick(
    image: &DynamicImage,
    mask: &DynamicImage,
) -> Result<DynamicImage, TransformError> {
    let config = InpaintConfig::default();
    let mut session = load_inpainter(&config)?;
    inpaint(image, mask, &mut session)
}

/// Creates a mask from user-defined regions.
///
/// Helper function to create a mask programmatically.
pub fn create_mask(width: u32, height: u32, regions: &[MaskRegion]) -> DynamicImage {
    use image::{GrayImage, Luma};

    let mut mask = GrayImage::new(width, height);

    for region in regions {
        match region {
            MaskRegion::Rectangle { x, y, w, h } => {
                for py in *y..(*y + *h).min(height) {
                    for px in *x..(*x + *w).min(width) {
                        mask.put_pixel(px, py, Luma([255]));
                    }
                }
            }
            MaskRegion::Circle { cx, cy, radius } => {
                let r2 = (*radius as i32) * (*radius as i32);
                for py in 0..height {
                    for px in 0..width {
                        let dx = px as i32 - *cx as i32;
                        let dy = py as i32 - *cy as i32;
                        if dx * dx + dy * dy <= r2 {
                            mask.put_pixel(px, py, Luma([255]));
                        }
                    }
                }
            }
        }
    }

    DynamicImage::ImageLuma8(mask)
}

/// A region to mask for inpainting.
#[derive(Debug, Clone)]
pub enum MaskRegion {
    /// Rectangular region.
    Rectangle { x: u32, y: u32, w: u32, h: u32 },
    /// Circular region.
    Circle { cx: u32, cy: u32, radius: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = InpaintConfig::default();
        assert_eq!(config.model, InpaintModel::LaMa);
        assert!(config.use_gpu);
    }

    #[test]
    fn test_create_rect_mask() {
        let mask = create_mask(100, 100, &[
            MaskRegion::Rectangle { x: 10, y: 10, w: 20, h: 20 }
        ]);

        let gray = mask.to_luma8();
        assert_eq!(gray.get_pixel(15, 15)[0], 255);
        assert_eq!(gray.get_pixel(0, 0)[0], 0);
    }

    #[test]
    fn test_create_circle_mask() {
        let mask = create_mask(100, 100, &[
            MaskRegion::Circle { cx: 50, cy: 50, radius: 10 }
        ]);

        let gray = mask.to_luma8();
        assert_eq!(gray.get_pixel(50, 50)[0], 255);
        assert_eq!(gray.get_pixel(0, 0)[0], 0);
    }
}
