//! AI-powered neural style transfer.
//!
//! Transform images into artistic styles using neural networks.
//!
//! # Models
//!
//! - **Fast Neural Style**: Pre-trained models for specific styles (Mosaic, Candy, Udnie, etc.)
//! - **Arbitrary Style Transfer**: Apply any image as a style reference
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::style_transfer::{stylize, load_style_model, StyleConfig, PretrainedStyle};
//!
//! let config = StyleConfig::new(PretrainedStyle::Mosaic);
//! let mut model = load_style_model(&config)?;
//!
//! let content = image::open("photo.jpg")?;
//! let styled = stylize(&content, &mut model)?;
//! styled.save("styled.png")?;
//! ```

mod config;
mod model;
mod processor;

pub use config::*;
pub use model::*;
pub use processor::*;
