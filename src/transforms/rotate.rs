use crate::error::TransformError;
use crate::transforms::interpolation::{
    BilinearInterpolation, Interpolation, InterpolationKernel, NearestInterpolation,
};
use crate::transforms::utils::{
    allocate_pixel_storage, buffer_from_vec, channel_count, dispatch_on_dynamic_image,
};
use image::{DynamicImage, ImageBuffer, Pixel};
use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;

const RIGHT_ANGLE_EPSILON: f32 = 1e-4;

/// Rotates an image 90 degrees clockwise.
pub fn rotate_90(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    dispatch_on_dynamic_image!(image, rotate_90_impl)
}

/// Rotates an image 90 degrees clockwise (alias for rotate_90).
pub fn rotate_90_cw(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    rotate_90(image)
}

/// Rotates an image 90 degrees counter-clockwise (alias for rotate_270).
pub fn rotate_90_ccw(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    rotate_270(image)
}

/// Rotates an image 180 degrees.
pub fn rotate_180(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    dispatch_on_dynamic_image!(image, rotate_180_impl)
}

/// Rotates an image 270 degrees clockwise (or 90 degrees counter-clockwise).
pub fn rotate_270(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    dispatch_on_dynamic_image!(image, rotate_270_impl)
}

/// Rotates an image 270 degrees clockwise (alias for rotate_270).
pub fn rotate_270_cw(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    rotate_270(image)
}

/// Rotates an image by an arbitrary angle in degrees using the requested interpolation kernel.
/// The output canvas is expanded to fit the entire rotated image (no cropping).
/// This is an alias for `rotate_bounded`.
pub fn rotate(
    image: &DynamicImage,
    angle_degrees: f32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    rotate_bounded(image, angle_degrees, interpolation)
}

/// Rotates an image by an arbitrary angle and expands the canvas to fit the entire rotated image.
pub fn rotate_bounded(
    image: &DynamicImage,
    angle_degrees: f32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    if !angle_degrees.is_finite() {
        return Err(TransformError::InvalidAngle {
            angle: angle_degrees,
        });
    }

    let normalized = angle_degrees.rem_euclid(360.0);

    if (normalized - 0.0).abs() < RIGHT_ANGLE_EPSILON
        || (normalized - 360.0).abs() < RIGHT_ANGLE_EPSILON
    {
        return Ok(image.clone());
    }
    if (normalized - 90.0).abs() < RIGHT_ANGLE_EPSILON {
        return rotate_90(image);
    }
    if (normalized - 180.0).abs() < RIGHT_ANGLE_EPSILON {
        return rotate_180(image);
    }
    if (normalized - 270.0).abs() < RIGHT_ANGLE_EPSILON {
        return rotate_270(image);
    }

    match interpolation {
        Interpolation::Nearest => {
            dispatch_on_dynamic_image!(image, rotate_arbitrary_nearest_impl, normalized, false)
        }
        Interpolation::Bilinear => {
            dispatch_on_dynamic_image!(image, rotate_arbitrary_bilinear_impl, normalized, false)
        }
    }
}

/// Rotates an image by an arbitrary angle and maintains the original canvas dimensions.
/// Parts of the rotated image that don't fit will be cropped.
pub fn rotate_cropped(
    image: &DynamicImage,
    angle_degrees: f32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    if !angle_degrees.is_finite() {
        return Err(TransformError::InvalidAngle {
            angle: angle_degrees,
        });
    }

    let normalized = angle_degrees.rem_euclid(360.0);

    if (normalized - 0.0).abs() < RIGHT_ANGLE_EPSILON
        || (normalized - 360.0).abs() < RIGHT_ANGLE_EPSILON
    {
        return Ok(image.clone());
    }
    if (normalized - 90.0).abs() < RIGHT_ANGLE_EPSILON {
        return rotate_90(image);
    }
    if (normalized - 180.0).abs() < RIGHT_ANGLE_EPSILON {
        return rotate_180(image);
    }
    if (normalized - 270.0).abs() < RIGHT_ANGLE_EPSILON {
        return rotate_270(image);
    }

    match interpolation {
        Interpolation::Nearest => {
            dispatch_on_dynamic_image!(image, rotate_arbitrary_nearest_impl, normalized, true)
        }
        Interpolation::Bilinear => {
            dispatch_on_dynamic_image!(image, rotate_arbitrary_bilinear_impl, normalized, true)
        }
    }
}

fn rotate_90_impl<P>(
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

    output_data
        .par_chunks_mut(out_row_stride)
        .enumerate()
        .for_each(|(dest_y, row)| {
            for dest_x in 0..(out_width as usize) {
                let src_x = dest_y;
                let src_y = out_width as usize - 1 - dest_x;
                let src_idx = src_y * in_row_stride + src_x * channels;
                let dst_idx = dest_x * channels;
                row[dst_idx..dst_idx + channels]
                    .copy_from_slice(&input_slice[src_idx..src_idx + channels]);
            }
        });

    buffer_from_vec(out_width, out_height, output_data)
}

fn rotate_180_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel + Send + Sync + 'static,
    P::Subpixel: Default + Send + Sync + image::Primitive,
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
        .for_each(|(row_idx, row)| {
            let src_row = &input_slice[(height as usize - 1 - row_idx) * row_stride
                ..(height as usize - row_idx) * row_stride];
            for x in 0..(width as usize) {
                let src_idx = (width as usize - 1 - x) * channels;
                let dst_idx = x * channels;
                row[dst_idx..dst_idx + channels]
                    .copy_from_slice(&src_row[src_idx..src_idx + channels]);
            }
        });

    buffer_from_vec(width, height, output_data)
}

fn rotate_270_impl<P>(
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

    output_data
        .par_chunks_mut(out_row_stride)
        .enumerate()
        .for_each(|(dest_y, row)| {
            for dest_x in 0..(out_width as usize) {
                let src_x = width as usize - 1 - dest_y;
                let src_y = dest_x;
                let src_idx = src_y * in_row_stride + src_x * channels;
                let dst_idx = dest_x * channels;
                row[dst_idx..dst_idx + channels]
                    .copy_from_slice(&input_slice[src_idx..src_idx + channels]);
            }
        });

    buffer_from_vec(out_width, out_height, output_data)
}

fn rotate_arbitrary_impl<P, K>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    angle_degrees: f32,
    crop_to_original: bool,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
    K: InterpolationKernel + Sync,
{
    let width = input.width();
    let height = input.height();

    if width == 0 || height == 0 {
        let data = allocate_pixel_storage::<P>(width, height)?;
        return buffer_from_vec(width, height, data);
    }

    let width_f = width as f32;
    let height_f = height as f32;

    let angle_rad = angle_degrees.to_radians();
    let (sin_a, cos_a) = angle_rad.sin_cos();

    let cx = (width_f - 1.0) / 2.0;
    let cy = (height_f - 1.0) / 2.0;

    let (out_width, out_height, min_x, min_y) = if crop_to_original {
        // Keep original dimensions
        (width, height, -cx, -cy)
    } else {
        // Expand to fit rotated image
        let corners = [
            rotate_point(-cx, -cy, sin_a, cos_a),
            rotate_point(width_f - 1.0 - cx, -cy, sin_a, cos_a),
            rotate_point(-cx, height_f - 1.0 - cy, sin_a, cos_a),
            rotate_point(width_f - 1.0 - cx, height_f - 1.0 - cy, sin_a, cos_a),
        ];

        let (min_x, max_x, min_y, max_y) = corners.iter().fold(
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
            |(min_x, max_x, min_y, max_y), (x, y)| {
                (min_x.min(*x), max_x.max(*x), min_y.min(*y), max_y.max(*y))
            },
        );

        let out_width = (max_x - min_x).ceil() as u32 + 1;
        let out_height = (max_y - min_y).ceil() as u32 + 1;
        (out_width, out_height, min_x, min_y)
    };

    let mut output_data = allocate_pixel_storage::<P>(out_width, out_height)?;

    let channels = channel_count::<P>();
    let input_slice = input.as_raw();
    let row_stride = width as usize * channels;
    let output_row_stride = out_width as usize * channels;

    output_data
        .par_chunks_mut(output_row_stride)
        .enumerate()
        .for_each(|(dest_y, row)| {
            let dest_y = dest_y as f32;
            for dest_x in 0..(out_width as usize) {
                let dest_x_f = dest_x as f32;

                let x_prime = dest_x_f + min_x;
                let y_prime = dest_y + min_y;

                let src_x = cos_a * x_prime + sin_a * y_prime + cx;
                let src_y = -sin_a * x_prime + cos_a * y_prime + cy;

                let dst_idx = dest_x * channels;
                let out_pixel = &mut row[dst_idx..dst_idx + channels];
                if !K::sample_into(input_slice, row_stride, channels, src_x, src_y, out_pixel) {
                    out_pixel.fill(P::Subpixel::default());
                }
            }
        });

    buffer_from_vec(out_width, out_height, output_data)
}

#[inline]
fn rotate_point(x: f32, y: f32, sin_a: f32, cos_a: f32) -> (f32, f32) {
    (cos_a * x - sin_a * y, sin_a * x + cos_a * y)
}

fn rotate_arbitrary_nearest_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    angle_degrees: f32,
    crop_to_original: bool,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    rotate_arbitrary_impl::<P, NearestInterpolation>(input, angle_degrees, crop_to_original)
}

fn rotate_arbitrary_bilinear_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    angle_degrees: f32,
    crop_to_original: bool,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    rotate_arbitrary_impl::<P, BilinearInterpolation>(input, angle_degrees, crop_to_original)
}
