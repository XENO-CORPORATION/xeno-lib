use crate::error::TransformError;
use crate::transforms::interpolation::{BilinearInterpolation, InterpolationKernel};
use crate::transforms::utils::{
    allocate_pixel_storage, buffer_from_vec, channel_count, dispatch_on_dynamic_image,
};
use image::{DynamicImage, ImageBuffer, Pixel};
use rayon::prelude::*;

/// Applies horizontal shear (skew) to the image.
/// Factor determines the amount of shear (positive = right, negative = left).
pub fn shear_horizontal(
    image: &DynamicImage,
    factor: f32,
) -> Result<DynamicImage, TransformError> {
    if !factor.is_finite() {
        return Err(TransformError::InvalidParameter {
            name: "shear_factor",
            value: factor,
        });
    }
    dispatch_on_dynamic_image!(image, shear_horizontal_impl, factor)
}

/// Applies vertical shear (skew) to the image.
/// Factor determines the amount of shear (positive = down, negative = up).
pub fn shear_vertical(image: &DynamicImage, factor: f32) -> Result<DynamicImage, TransformError> {
    if !factor.is_finite() {
        return Err(TransformError::InvalidParameter {
            name: "shear_factor",
            value: factor,
        });
    }
    dispatch_on_dynamic_image!(image, shear_vertical_impl, factor)
}

/// Applies a 2x3 affine transformation matrix to the image.
/// Matrix format: [[a, b, c], [d, e, f]] where:
/// x' = a*x + b*y + c
/// y' = d*x + e*y + f
pub fn affine_transform(
    image: &DynamicImage,
    matrix: [[f32; 3]; 2],
) -> Result<DynamicImage, TransformError> {
    // Validate matrix values
    for row in &matrix {
        for &val in row {
            if !val.is_finite() {
                return Err(TransformError::InvalidParameter {
                    name: "affine_matrix",
                    value: val,
                });
            }
        }
    }
    dispatch_on_dynamic_image!(image, affine_transform_impl, matrix)
}

/// Translates (moves) the image by the specified offset.
/// Positive x moves right, positive y moves down.
/// The canvas size remains the same; parts that move outside are clipped.
pub fn translate(image: &DynamicImage, x: i32, y: i32) -> Result<DynamicImage, TransformError> {
    dispatch_on_dynamic_image!(image, translate_impl, x, y)
}

fn shear_horizontal_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    factor: f32,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let width = input.width();
    let height = input.height();

    if width == 0 || height == 0 {
        return Ok(input.clone());
    }

    let max_offset = (height as f32 * factor.abs()).ceil() as u32;
    let out_width = width + max_offset;

    let channels = channel_count::<P>();
    let in_row_stride = width as usize * channels;
    let out_row_stride = out_width as usize * channels;

    let mut output_data = allocate_pixel_storage::<P>(out_width, height)?;
    let input_slice = input.as_raw();

    output_data
        .par_chunks_mut(out_row_stride)
        .enumerate()
        .for_each(|(dest_y, row)| {
            let offset = (dest_y as f32 * factor).round() as i32;
            let offset_x = if factor >= 0.0 {
                offset
            } else {
                offset + max_offset as i32
            };

            for dest_x in 0..(out_width as usize) {
                let src_x = dest_x as i32 - offset_x;
                let dst_idx = dest_x * channels;

                if src_x >= 0 && src_x < width as i32 {
                    let src_idx = dest_y * in_row_stride + src_x as usize * channels;
                    row[dst_idx..dst_idx + channels]
                        .copy_from_slice(&input_slice[src_idx..src_idx + channels]);
                } else {
                    // Fill with transparent/black
                    row[dst_idx..dst_idx + channels].fill(P::Subpixel::default());
                }
            }
        });

    buffer_from_vec(out_width, height, output_data)
}

fn shear_vertical_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    factor: f32,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let width = input.width();
    let height = input.height();

    if width == 0 || height == 0 {
        return Ok(input.clone());
    }

    let max_offset = (width as f32 * factor.abs()).ceil() as u32;
    let out_height = height + max_offset;

    let channels = channel_count::<P>();
    let in_row_stride = width as usize * channels;
    let out_row_stride = width as usize * channels;

    let mut output_data = allocate_pixel_storage::<P>(width, out_height)?;
    let input_slice = input.as_raw();

    output_data
        .par_chunks_mut(out_row_stride)
        .enumerate()
        .for_each(|(dest_y, row)| {
            for dest_x in 0..(width as usize) {
                let offset = (dest_x as f32 * factor).round() as i32;
                let offset_y = if factor >= 0.0 {
                    offset
                } else {
                    offset + max_offset as i32
                };

                let src_y = dest_y as i32 - offset_y;
                let dst_idx = dest_x * channels;

                if src_y >= 0 && src_y < height as i32 {
                    let src_idx = src_y as usize * in_row_stride + dest_x * channels;
                    row[dst_idx..dst_idx + channels]
                        .copy_from_slice(&input_slice[src_idx..src_idx + channels]);
                } else {
                    // Fill with transparent/black
                    row[dst_idx..dst_idx + channels].fill(P::Subpixel::default());
                }
            }
        });

    buffer_from_vec(width, out_height, output_data)
}

fn affine_transform_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    matrix: [[f32; 3]; 2],
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let width = input.width();
    let height = input.height();

    if width == 0 || height == 0 {
        return Ok(input.clone());
    }

    // Calculate output bounds by transforming corners
    let corners = [
        (0.0, 0.0),
        (width as f32 - 1.0, 0.0),
        (0.0, height as f32 - 1.0),
        (width as f32 - 1.0, height as f32 - 1.0),
    ];

    let transformed_corners: Vec<(f32, f32)> = corners
        .iter()
        .map(|(x, y)| {
            let new_x = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2];
            let new_y = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2];
            (new_x, new_y)
        })
        .collect();

    let (min_x, max_x, min_y, max_y) = transformed_corners.iter().fold(
        (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
        |(min_x, max_x, min_y, max_y), (x, y)| {
            (min_x.min(*x), max_x.max(*x), min_y.min(*y), max_y.max(*y))
        },
    );

    let out_width = (max_x - min_x).ceil() as u32 + 1;
    let out_height = (max_y - min_y).ceil() as u32 + 1;

    // Compute inverse matrix for reverse mapping
    let det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if det.abs() < 1e-10 {
        return Err(TransformError::InvalidParameter {
            name: "affine_matrix_determinant",
            value: det,
        });
    }

    let inv_det = 1.0 / det;
    let inv_matrix = [
        [
            matrix[1][1] * inv_det,
            -matrix[0][1] * inv_det,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det,
        ],
        [
            -matrix[1][0] * inv_det,
            matrix[0][0] * inv_det,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv_det,
        ],
    ];

    let channels = channel_count::<P>();
    let in_row_stride = width as usize * channels;
    let out_row_stride = out_width as usize * channels;

    let mut output_data = allocate_pixel_storage::<P>(out_width, out_height)?;
    let input_slice = input.as_raw();

    output_data
        .par_chunks_mut(out_row_stride)
        .enumerate()
        .for_each(|(dest_y, row)| {
            let dest_y_f = dest_y as f32 + min_y;
            for dest_x in 0..(out_width as usize) {
                let dest_x_f = dest_x as f32 + min_x;

                // Apply inverse transform to find source coordinates
                let src_x = inv_matrix[0][0] * dest_x_f + inv_matrix[0][1] * dest_y_f
                    + inv_matrix[0][2];
                let src_y = inv_matrix[1][0] * dest_x_f + inv_matrix[1][1] * dest_y_f
                    + inv_matrix[1][2];

                let dst_idx = dest_x * channels;
                let out_pixel = &mut row[dst_idx..dst_idx + channels];

                if !BilinearInterpolation::sample_into(
                    input_slice,
                    in_row_stride,
                    channels,
                    src_x,
                    src_y,
                    out_pixel,
                ) {
                    out_pixel.fill(P::Subpixel::default());
                }
            }
        });

    buffer_from_vec(out_width, out_height, output_data)
}

fn translate_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    x_offset: i32,
    y_offset: i32,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let width = input.width();
    let height = input.height();

    if width == 0 || height == 0 {
        return Ok(input.clone());
    }

    let channels = channel_count::<P>();
    let row_stride = width as usize * channels;

    let mut output_data = allocate_pixel_storage::<P>(width, height)?;
    let input_slice = input.as_raw();

    output_data
        .par_chunks_mut(row_stride)
        .enumerate()
        .for_each(|(dest_y, row)| {
            let src_y = dest_y as i32 - y_offset;

            for dest_x in 0..(width as usize) {
                let src_x = dest_x as i32 - x_offset;
                let dst_idx = dest_x * channels;

                if src_x >= 0
                    && src_x < width as i32
                    && src_y >= 0
                    && src_y < height as i32
                {
                    let src_idx = src_y as usize * row_stride + src_x as usize * channels;
                    row[dst_idx..dst_idx + channels]
                        .copy_from_slice(&input_slice[src_idx..src_idx + channels]);
                } else {
                    // Fill with transparent/black
                    row[dst_idx..dst_idx + channels].fill(P::Subpixel::default());
                }
            }
        });

    buffer_from_vec(width, height, output_data)
}
