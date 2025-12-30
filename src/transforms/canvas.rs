use crate::error::TransformError;
use crate::transforms::utils::{
    allocate_pixel_storage, buffer_from_vec, channel_count, dispatch_on_dynamic_image,
};
use image::{DynamicImage, ImageBuffer, Pixel};
use rayon::prelude::*;

/// Adds padding to all sides of the image.
/// Padding is filled with the specified color (RGBA format: [r, g, b, a]).
pub fn pad(
    image: &DynamicImage,
    top: u32,
    right: u32,
    bottom: u32,
    left: u32,
    color: [u8; 4],
) -> Result<DynamicImage, TransformError> {
    dispatch_on_dynamic_image!(image, pad_impl, top, right, bottom, left, color)
}

/// Pads the image to the specified total dimensions (centered).
pub fn pad_to_size(
    image: &DynamicImage,
    width: u32,
    height: u32,
    color: [u8; 4],
) -> Result<DynamicImage, TransformError> {
    if width < image.width() || height < image.height() {
        return Err(TransformError::InvalidDimensions { width, height });
    }

    let pad_x = width - image.width();
    let pad_y = height - image.height();

    let left = pad_x / 2;
    let right = pad_x - left;
    let top = pad_y / 2;
    let bottom = pad_y - top;

    pad(image, top, right, bottom, left, color)
}

/// Pads the image to achieve the specified aspect ratio.
pub fn pad_to_aspect(
    image: &DynamicImage,
    aspect_ratio: f32,
    color: [u8; 4],
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

    if (current_ratio - aspect_ratio).abs() < 0.001 {
        return Ok(image.clone());
    }

    let (target_width, target_height) = if current_ratio < aspect_ratio {
        // Image is too tall, pad horizontally
        let new_width = (img_height * aspect_ratio).round() as u32;
        (new_width, image.height())
    } else {
        // Image is too wide, pad vertically
        let new_height = (img_width / aspect_ratio).round() as u32;
        (image.width(), new_height)
    };

    pad_to_size(image, target_width, target_height, color)
}

/// Expands the canvas by the specified amount in all directions.
/// This is an alias for `pad`.
pub fn expand_canvas(
    image: &DynamicImage,
    amount: u32,
    color: [u8; 4],
) -> Result<DynamicImage, TransformError> {
    pad(image, amount, amount, amount, amount, color)
}

/// Trims uniform borders from the image edges.
/// Tolerance specifies how much variation is allowed (0-255).
pub fn trim(image: &DynamicImage, tolerance: u8) -> Result<DynamicImage, TransformError> {
    // Trim is essentially autocrop
    crate::transforms::crop::autocrop(image, tolerance)
}

fn pad_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    top: u32,
    right: u32,
    bottom: u32,
    left: u32,
    color: [u8; 4],
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let in_width = input.width() as usize;
    let in_height = input.height() as usize;

    let out_width = (in_width as u32)
        .checked_add(left)
        .and_then(|w| w.checked_add(right))
        .ok_or(TransformError::AllocationFailed {
            width: u32::MAX,
            height: u32::MAX,
        })?;

    let out_height = (in_height as u32)
        .checked_add(top)
        .and_then(|h| h.checked_add(bottom))
        .ok_or(TransformError::AllocationFailed {
            width: u32::MAX,
            height: u32::MAX,
        })?;

    let channels = channel_count::<P>();
    let in_row_stride = in_width * channels;
    let out_row_stride = out_width as usize * channels;

    let mut output_data = allocate_pixel_storage::<P>(out_width, out_height)?;
    let input_slice = input.as_raw();

    // Convert color to pixel format
    let fill_pixel: Vec<u8> = match channels {
        1 => vec![color[0]],                         // Grayscale
        2 => vec![color[0], color[3]],               // Grayscale + Alpha
        3 => vec![color[0], color[1], color[2]],     // RGB
        4 => vec![color[0], color[1], color[2], color[3]], // RGBA
        _ => vec![0; channels],
    };

    output_data
        .par_chunks_mut(out_row_stride)
        .enumerate()
        .for_each(|(dest_y, row)| {
            if dest_y < top as usize || dest_y >= (top as usize + in_height) {
                // Top or bottom padding row
                for chunk in row.chunks_exact_mut(channels) {
                    chunk.copy_from_slice(&fill_pixel);
                }
            } else {
                // Row with image data + left/right padding
                let src_y = dest_y - top as usize;
                let src_offset = src_y * in_row_stride;
                let src_row = &input_slice[src_offset..src_offset + in_row_stride];

                // Left padding
                for chunk in row[0..left as usize * channels].chunks_exact_mut(channels) {
                    chunk.copy_from_slice(&fill_pixel);
                }

                // Image data
                let img_start = left as usize * channels;
                let img_end = img_start + in_row_stride;
                row[img_start..img_end].copy_from_slice(src_row);

                // Right padding
                for chunk in row[img_end..].chunks_exact_mut(channels) {
                    chunk.copy_from_slice(&fill_pixel);
                }
            }
        });

    buffer_from_vec(out_width, out_height, output_data)
}
