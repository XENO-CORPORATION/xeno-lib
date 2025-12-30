//! Image analysis utilities: metadata inspection and histogram generation.

use crate::error::TransformError;
use image::{DynamicImage, ImageFormat};

pub mod comparison;
pub mod exif;
pub mod statistics;

pub use comparison::{ComparisonMetrics, compare};
pub use exif::{read_exif_from_path, read_exif_from_reader};
pub use statistics::{Histogram, HistogramChannel, ImageInfo};

/// Return basic metadata about the provided image buffer.
pub fn image_info(image: &DynamicImage) -> ImageInfo {
    ImageInfo::from_dynamic_image(image)
}

/// Compute a per-channel histogram (0..=255 bins) for the supplied image.
pub fn histogram(image: &DynamicImage) -> Histogram {
    Histogram::from_dynamic_image(image)
}

/// Identify the image format from raw bytes if possible.
pub fn sniff_format(bytes: &[u8]) -> Result<ImageFormat, TransformError> {
    image::guess_format(bytes).map_err(|_| TransformError::UnsupportedFormat)
}

#[cfg(test)]
mod tests;
