use crate::error::TransformError;
use image::{DynamicImage, RgbaImage};

/// Alignment position for placing images on canvas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Alignment {
    /// Top-left corner
    TopLeft,
    /// Top-center
    TopCenter,
    /// Top-right corner
    TopRight,
    /// Middle-left
    MiddleLeft,
    /// Center
    Center,
    /// Middle-right
    MiddleRight,
    /// Bottom-left corner
    BottomLeft,
    /// Bottom-center
    BottomCenter,
    /// Bottom-right corner
    BottomRight,
}

/// Centers an image on a larger canvas of the specified dimensions.
/// The canvas is filled with the specified color.
pub fn center_on_canvas(
    image: &DynamicImage,
    canvas_width: u32,
    canvas_height: u32,
    color: [u8; 4],
) -> Result<DynamicImage, TransformError> {
    align(image, canvas_width, canvas_height, Alignment::Center, color)
}

/// Aligns an image on a larger canvas of the specified dimensions.
/// The canvas is filled with the specified color.
pub fn align(
    image: &DynamicImage,
    canvas_width: u32,
    canvas_height: u32,
    alignment: Alignment,
    color: [u8; 4],
) -> Result<DynamicImage, TransformError> {
    if canvas_width < image.width() || canvas_height < image.height() {
        return Err(TransformError::InvalidDimensions {
            width: canvas_width,
            height: canvas_height,
        });
    }

    let pad_x = canvas_width - image.width();
    let pad_y = canvas_height - image.height();

    let (left, right, top, bottom) = match alignment {
        Alignment::TopLeft => (0, pad_x, 0, pad_y),
        Alignment::TopCenter => (pad_x / 2, pad_x - pad_x / 2, 0, pad_y),
        Alignment::TopRight => (pad_x, 0, 0, pad_y),
        Alignment::MiddleLeft => (0, pad_x, pad_y / 2, pad_y - pad_y / 2),
        Alignment::Center => (pad_x / 2, pad_x - pad_x / 2, pad_y / 2, pad_y - pad_y / 2),
        Alignment::MiddleRight => (pad_x, 0, pad_y / 2, pad_y - pad_y / 2),
        Alignment::BottomLeft => (0, pad_x, pad_y, 0),
        Alignment::BottomCenter => (pad_x / 2, pad_x - pad_x / 2, pad_y, 0),
        Alignment::BottomRight => (pad_x, 0, pad_y, 0),
    };

    crate::transforms::canvas::pad(image, top, right, bottom, left, color)
}

/// Recenters visible content in a transparent image while preserving canvas size.
///
/// This detects non-transparent pixels (`alpha > 0`), computes their bounding box,
/// and places that content in the center of a new transparent canvas with the same
/// dimensions as the input.
///
/// For fully transparent images, this returns the original image unchanged.
pub fn recenter(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    recenter_with_alpha_threshold(image, 0)
}

/// Same as [`recenter`], but allows customizing the alpha threshold used to detect
/// visible content.
///
/// Pixels with `alpha <= alpha_threshold` are treated as transparent.
pub fn recenter_with_alpha_threshold(
    image: &DynamicImage,
    alpha_threshold: u8,
) -> Result<DynamicImage, TransformError> {
    if image.width() == 0 || image.height() == 0 {
        return Err(TransformError::InvalidDimensions {
            width: image.width(),
            height: image.height(),
        });
    }

    let rgba = image.to_rgba8();
    let Some((min_x, min_y, max_x, max_y)) = alpha_bounds(&rgba, alpha_threshold) else {
        return Ok(image.clone());
    };

    let object_width = max_x - min_x + 1;
    let object_height = max_y - min_y + 1;

    let content = image::imageops::crop_imm(&rgba, min_x, min_y, object_width, object_height).to_image();
    let mut output = RgbaImage::new(image.width(), image.height());

    let offset_x = ((image.width() - object_width) / 2) as i64;
    let offset_y = ((image.height() - object_height) / 2) as i64;
    image::imageops::overlay(&mut output, &content, offset_x, offset_y);

    Ok(DynamicImage::ImageRgba8(output))
}

fn alpha_bounds(
    image: &RgbaImage,
    alpha_threshold: u8,
) -> Option<(u32, u32, u32, u32)> {
    let (width, height) = image.dimensions();
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;

    for y in 0..height {
        for x in 0..width {
            if image.get_pixel(x, y)[3] > alpha_threshold {
                found = true;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }

    if found {
        Some((min_x, min_y, max_x, max_y))
    } else {
        None
    }
}
