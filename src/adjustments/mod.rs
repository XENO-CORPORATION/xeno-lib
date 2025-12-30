//! Image color and tone adjustment utilities.

mod color;

pub use color::{
    adjust_brightness, adjust_contrast, adjust_exposure, adjust_gamma, adjust_hue,
    adjust_saturation, grayscale, invert,
};

#[cfg(test)]
mod tests;
