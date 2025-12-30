use crate::error::TransformError;
use crate::transforms::utils::{
    allocate_pixel_storage, buffer_from_vec, channel_count, dispatch_on_dynamic_image,
};
use image::{DynamicImage, ImageBuffer, Pixel};
use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;

/// Creates a new image containing the requested rectangular region of the input.
pub fn crop(
    image: &DynamicImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<DynamicImage, TransformError> {
    if width == 0 || height == 0 {
        return Err(TransformError::CropOutOfBounds {
            x,
            y,
            width,
            height,
            image_width: image.width(),
            image_height: image.height(),
        });
    }

    let img_width = image.width();
    let img_height = image.height();
    if x >= img_width || y >= img_height {
        return Err(TransformError::CropOutOfBounds {
            x,
            y,
            width,
            height,
            image_width: img_width,
            image_height: img_height,
        });
    }

    let end_x = x
        .checked_add(width)
        .ok_or(TransformError::CropOutOfBounds {
            x,
            y,
            width,
            height,
            image_width: img_width,
            image_height: img_height,
        })?;
    let end_y = y
        .checked_add(height)
        .ok_or(TransformError::CropOutOfBounds {
            x,
            y,
            width,
            height,
            image_width: img_width,
            image_height: img_height,
        })?;

    if end_x > img_width || end_y > img_height {
        return Err(TransformError::CropOutOfBounds {
            x,
            y,
            width,
            height,
            image_width: img_width,
            image_height: img_height,
        });
    }

    dispatch_on_dynamic_image!(image, crop_impl, x, y, width, height)
}

fn crop_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel + 'static,
    P::Subpixel: image::Primitive + Default + Send + Sync,
{
    let channels = channel_count::<P>();
    let row_stride = width as usize * channels;

    let mut output_data = allocate_pixel_storage::<P>(width, height)?;
    let input_slice = input.as_raw();
    let src_width = input.width() as usize;
    let start_x = x as usize;
    let start_y = y as usize;

    output_data
        .par_chunks_mut(row_stride)
        .enumerate()
        .for_each(|(dest_row, dest_slice)| {
            let src_y = start_y + dest_row;
            let src_offset = (src_y * src_width + start_x) * channels;
            let src_slice = &input_slice[src_offset..src_offset + row_stride];
            dest_slice.copy_from_slice(src_slice);
        });

    buffer_from_vec(width, height, output_data)
}

/// Crops the image to the specified dimensions centered on the original image.
pub fn crop_center(
    image: &DynamicImage,
    width: u32,
    height: u32,
) -> Result<DynamicImage, TransformError> {
    if width == 0 || height == 0 {
        return Err(TransformError::InvalidDimensions { width, height });
    }

    let img_width = image.width();
    let img_height = image.height();

    if width > img_width || height > img_height {
        return Err(TransformError::CropOutOfBounds {
            x: 0,
            y: 0,
            width,
            height,
            image_width: img_width,
            image_height: img_height,
        });
    }

    let x = (img_width - width) / 2;
    let y = (img_height - height) / 2;

    crop(image, x, y, width, height)
}

/// Anchor position for cropping to aspect ratio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CropAnchor {
    /// Anchor to top-left corner
    TopLeft,
    /// Anchor to top-center
    TopCenter,
    /// Anchor to top-right corner
    TopRight,
    /// Anchor to middle-left
    MiddleLeft,
    /// Anchor to center (default)
    Center,
    /// Anchor to middle-right
    MiddleRight,
    /// Anchor to bottom-left corner
    BottomLeft,
    /// Anchor to bottom-center
    BottomCenter,
    /// Anchor to bottom-right corner
    BottomRight,
}

/// Crops the image to the specified aspect ratio.
pub fn crop_to_aspect(
    image: &DynamicImage,
    aspect_ratio: f32,
    anchor: CropAnchor,
) -> Result<DynamicImage, TransformError> {
    if !aspect_ratio.is_finite() || aspect_ratio <= 0.0 {
        return Err(TransformError::InvalidParameter {
            name: "aspect_ratio",
            value: aspect_ratio,
        });
    }

    let img_width = image.width() as f32;
    let img_height = image.height() as f32;
    let current_ratio = img_width / img_height;

    let (crop_width, crop_height) = if current_ratio > aspect_ratio {
        // Image is too wide, crop horizontally
        let new_width = (img_height * aspect_ratio).round() as u32;
        (new_width, image.height())
    } else {
        // Image is too tall, crop vertically
        let new_height = (img_width / aspect_ratio).round() as u32;
        (image.width(), new_height)
    };

    // Calculate x, y based on anchor
    let (x, y) = match anchor {
        CropAnchor::TopLeft => (0, 0),
        CropAnchor::TopCenter => ((image.width() - crop_width) / 2, 0),
        CropAnchor::TopRight => (image.width() - crop_width, 0),
        CropAnchor::MiddleLeft => (0, (image.height() - crop_height) / 2),
        CropAnchor::Center => (
            (image.width() - crop_width) / 2,
            (image.height() - crop_height) / 2,
        ),
        CropAnchor::MiddleRight => (image.width() - crop_width, (image.height() - crop_height) / 2),
        CropAnchor::BottomLeft => (0, image.height() - crop_height),
        CropAnchor::BottomCenter => ((image.width() - crop_width) / 2, image.height() - crop_height),
        CropAnchor::BottomRight => (image.width() - crop_width, image.height() - crop_height),
    };

    crop(image, x, y, crop_width, crop_height)
}

/// Crops the image by the specified percentage from each edge.
/// Percentages are in range 0.0 - 100.0.
pub fn crop_percentage(
    image: &DynamicImage,
    top_percent: f32,
    right_percent: f32,
    bottom_percent: f32,
    left_percent: f32,
) -> Result<DynamicImage, TransformError> {
    let validate_percent = |name: &'static str, value: f32| {
        if !value.is_finite() || value < 0.0 || value > 100.0 {
            return Err(TransformError::InvalidParameter { name, value });
        }
        Ok(())
    };

    validate_percent("top_percent", top_percent)?;
    validate_percent("right_percent", right_percent)?;
    validate_percent("bottom_percent", bottom_percent)?;
    validate_percent("left_percent", left_percent)?;

    let img_width = image.width() as f32;
    let img_height = image.height() as f32;

    let x = (img_width * left_percent / 100.0).round() as u32;
    let y = (img_height * top_percent / 100.0).round() as u32;
    let right_crop = (img_width * right_percent / 100.0).round() as u32;
    let bottom_crop = (img_height * bottom_percent / 100.0).round() as u32;

    let width = image
        .width()
        .saturating_sub(x)
        .saturating_sub(right_crop)
        .max(1);
    let height = image
        .height()
        .saturating_sub(y)
        .saturating_sub(bottom_crop)
        .max(1);

    crop(image, x, y, width, height)
}

/// Automatically crops uniform borders from the image.
/// Tolerance specifies how much variation is allowed (0-255).
pub fn autocrop(image: &DynamicImage, tolerance: u8) -> Result<DynamicImage, TransformError> {
    dispatch_on_dynamic_image!(image, autocrop_impl, tolerance)
}

fn autocrop_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    tolerance: u8,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + 'static,
    P::Subpixel: image::Primitive + Default + Send + Sync,
{
    let width = input.width() as usize;
    let height = input.height() as usize;

    if width == 0 || height == 0 {
        return Ok(input.clone());
    }

    let channels = channel_count::<P>();
    let input_slice = input.as_raw();

    // Get the reference color from top-left corner
    let ref_pixel = &input_slice[0..channels];

    // Helper to check if a pixel matches the reference within tolerance
    let matches_ref = |pixel: &[u8]| -> bool {
        pixel
            .iter()
            .zip(ref_pixel.iter())
            .all(|(a, b)| a.abs_diff(*b) <= tolerance)
    };

    // Find top border
    let mut top = 0usize;
    'top_loop: for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * channels;
            if !matches_ref(&input_slice[idx..idx + channels]) {
                break 'top_loop;
            }
        }
        top = y + 1;
    }

    // Find bottom border
    let mut bottom = height;
    'bottom_loop: for y in (0..height).rev() {
        for x in 0..width {
            let idx = (y * width + x) * channels;
            if !matches_ref(&input_slice[idx..idx + channels]) {
                break 'bottom_loop;
            }
        }
        bottom = y;
    }

    // Find left border
    let mut left = 0usize;
    'left_loop: for x in 0..width {
        for y in top..bottom {
            let idx = (y * width + x) * channels;
            if !matches_ref(&input_slice[idx..idx + channels]) {
                break 'left_loop;
            }
        }
        left = x + 1;
    }

    // Find right border
    let mut right = width;
    'right_loop: for x in (0..width).rev() {
        for y in top..bottom {
            let idx = (y * width + x) * channels;
            if !matches_ref(&input_slice[idx..idx + channels]) {
                break 'right_loop;
            }
        }
        right = x;
    }

    // Ensure we have at least 1x1 image
    if top >= bottom || left >= right {
        return Ok(input.clone());
    }

    let crop_width = (right - left) as u32;
    let crop_height = (bottom - top) as u32;

    crop_impl(input, left as u32, top as u32, crop_width, crop_height)
}

/// Crops the image to non-transparent content (RGBA images only).
/// For non-RGBA images, returns the original image.
pub fn crop_to_content(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    match image {
        DynamicImage::ImageRgba8(_) | DynamicImage::ImageLumaA8(_) => {
            // Autocrop with 0 tolerance will remove fully transparent pixels
            dispatch_on_dynamic_image!(image, crop_to_content_impl)
        }
        _ => Ok(image.clone()),
    }
}

fn crop_to_content_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + 'static,
    P::Subpixel: image::Primitive + Default + Send + Sync,
{
    let width = input.width() as usize;
    let height = input.height() as usize;

    if width == 0 || height == 0 {
        return Ok(input.clone());
    }

    let channels = channel_count::<P>();
    let input_slice = input.as_raw();

    // For images with alpha channel, find non-transparent bounds
    let alpha_channel_idx = channels - 1;

    // Helper to check if pixel is transparent
    let is_transparent = |pixel: &[u8]| -> bool { pixel[alpha_channel_idx] == 0 };

    // Find top border
    let mut top = 0usize;
    'top_loop: for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * channels;
            if !is_transparent(&input_slice[idx..idx + channels]) {
                break 'top_loop;
            }
        }
        top = y + 1;
    }

    // Find bottom border
    let mut bottom = height;
    'bottom_loop: for y in (0..height).rev() {
        for x in 0..width {
            let idx = (y * width + x) * channels;
            if !is_transparent(&input_slice[idx..idx + channels]) {
                break 'bottom_loop;
            }
        }
        bottom = y;
    }

    // Find left border
    let mut left = 0usize;
    'left_loop: for x in 0..width {
        for y in top..bottom {
            let idx = (y * width + x) * channels;
            if !is_transparent(&input_slice[idx..idx + channels]) {
                break 'left_loop;
            }
        }
        left = x + 1;
    }

    // Find right border
    let mut right = width;
    'right_loop: for x in (0..width).rev() {
        for y in top..bottom {
            let idx = (y * width + x) * channels;
            if !is_transparent(&input_slice[idx..idx + channels]) {
                break 'right_loop;
            }
        }
        right = x;
    }

    // Ensure we have at least 1x1 image
    if top >= bottom || left >= right {
        return Ok(input.clone());
    }

    let crop_width = (right - left) as u32;
    let crop_height = (bottom - top) as u32;

    crop_impl(input, left as u32, top as u32, crop_width, crop_height)
}
