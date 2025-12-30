use crate::error::TransformError;
use crate::transforms::utils::{
    allocate_pixel_storage, buffer_from_vec, channel_count, dispatch_on_dynamic_image,
};
use image::{DynamicImage, ImageBuffer, Pixel};
use rayon::prelude::*;

/// Transposes the image (swaps rows and columns).
/// Equivalent to rotating 90° clockwise and flipping horizontally.
pub fn transpose(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    dispatch_on_dynamic_image!(image, transpose_impl)
}

/// Transverses the image (flips along anti-diagonal).
/// Equivalent to rotating 90° counter-clockwise and flipping horizontally.
pub fn transverse(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    dispatch_on_dynamic_image!(image, transverse_impl)
}

fn transpose_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel + Send + Sync + 'static,
    P::Subpixel: Default + Send + Sync + image::Primitive,
{
    let width = input.width();
    let height = input.height();
    let out_width = height;
    let out_height = width;

    if out_width == 0 || out_height == 0 {
        let data = allocate_pixel_storage::<P>(out_width, out_height)?;
        return buffer_from_vec(out_width, out_height, data);
    }

    let channels = channel_count::<P>();
    let in_row_stride = width as usize * channels;
    let out_row_stride = out_width as usize * channels;

    let mut output_data = allocate_pixel_storage::<P>(out_width, out_height)?;
    let input_slice = input.as_raw();

    // Transpose: output[y][x] = input[x][y]
    output_data
        .par_chunks_mut(out_row_stride)
        .enumerate()
        .for_each(|(dest_y, row)| {
            for dest_x in 0..(out_width as usize) {
                let src_x = dest_y;
                let src_y = dest_x;
                let src_idx = src_y * in_row_stride + src_x * channels;
                let dst_idx = dest_x * channels;
                row[dst_idx..dst_idx + channels]
                    .copy_from_slice(&input_slice[src_idx..src_idx + channels]);
            }
        });

    buffer_from_vec(out_width, out_height, output_data)
}

fn transverse_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel + Send + Sync + 'static,
    P::Subpixel: Default + Send + Sync + image::Primitive,
{
    let width = input.width();
    let height = input.height();
    let out_width = height;
    let out_height = width;

    if out_width == 0 || out_height == 0 {
        let data = allocate_pixel_storage::<P>(out_width, out_height)?;
        return buffer_from_vec(out_width, out_height, data);
    }

    let channels = channel_count::<P>();
    let in_row_stride = width as usize * channels;
    let out_row_stride = out_width as usize * channels;

    let mut output_data = allocate_pixel_storage::<P>(out_width, out_height)?;
    let input_slice = input.as_raw();

    // Transverse: output[y][x] = input[width-1-x][height-1-y]
    output_data
        .par_chunks_mut(out_row_stride)
        .enumerate()
        .for_each(|(dest_y, row)| {
            for dest_x in 0..(out_width as usize) {
                let src_x = width as usize - 1 - dest_y;
                let src_y = height as usize - 1 - dest_x;
                let src_idx = src_y * in_row_stride + src_x * channels;
                let dst_idx = dest_x * channels;
                row[dst_idx..dst_idx + channels]
                    .copy_from_slice(&input_slice[src_idx..src_idx + channels]);
            }
        });

    buffer_from_vec(out_width, out_height, output_data)
}
