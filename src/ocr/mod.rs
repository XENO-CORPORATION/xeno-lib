//! AI-powered Optical Character Recognition (OCR).
//!
//! Extract text from images using deep learning models.
//!
//! # Models
//!
//! - **PaddleOCR**: High-accuracy multilingual OCR
//! - **CRNN**: Convolutional Recurrent Neural Network for scene text
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::ocr::{extract_text, load_ocr_model, OcrConfig};
//!
//! let config = OcrConfig::default();
//! let mut model = load_ocr_model(&config)?;
//!
//! let image = image::open("document.jpg")?;
//! let result = extract_text(&image, &mut model)?;
//!
//! println!("Text: {}", result.text);
//! for block in &result.blocks {
//!     println!("Block at {:?}: {}", block.bbox, block.text);
//! }
//! ```

mod config;
mod model;
mod processor;

pub use config::*;
pub use model::*;
pub use processor::*;
