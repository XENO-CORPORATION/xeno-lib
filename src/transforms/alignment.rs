use crate::error::TransformError;
use image::DynamicImage;

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
