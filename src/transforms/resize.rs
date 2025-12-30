use crate::error::TransformError;
use crate::transforms::interpolation::{
    BilinearInterpolation, Interpolation, InterpolationKernel, NearestInterpolation,
};
use crate::transforms::utils::{
    allocate_pixel_storage, buffer_from_vec, channel_count, dispatch_on_dynamic_image,
};
use image::{DynamicImage, ImageBuffer, Pixel};
use rayon::prelude::*;

/// Resize the image to exact dimensions with aspect control (alias for resize_exact).
pub fn resize(
    image: &DynamicImage,
    new_width: u32,
    new_height: u32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    resize_exact(image, new_width, new_height, interpolation)
}

/// Resize the image to an exact width and height using the requested interpolation algorithm.
pub fn resize_exact(
    image: &DynamicImage,
    new_width: u32,
    new_height: u32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    if new_width == 0 || new_height == 0 {
        return Err(TransformError::InvalidDimensions {
            width: new_width,
            height: new_height,
        });
    }

    dispatch_on_dynamic_image!(
        image,
        resize_exact_impl,
        new_width,
        new_height,
        interpolation
    )
}

/// Resize the image by a uniform percentage (e.g. `50.0` halves both dimensions).
pub fn resize_by_percent(
    image: &DynamicImage,
    percent: f32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    if !percent.is_finite() || percent <= 0.0 {
        return Err(TransformError::InvalidScaleFactor { factor: percent });
    }

    let factor = percent / 100.0;
    let (width, height) = scale_dimensions_by_factor(image.width(), image.height(), factor)?;
    resize_exact(image, width, height, interpolation)
}

/// Scale to width, maintain aspect (alias for resize_to_width).
pub fn scale_width(
    image: &DynamicImage,
    target_width: u32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    resize_to_width(image, target_width, interpolation)
}

/// Resize the image to the requested width preserving aspect ratio.
pub fn resize_to_width(
    image: &DynamicImage,
    target_width: u32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    if target_width == 0 {
        return Err(TransformError::InvalidDimensions {
            width: target_width,
            height: image.height(),
        });
    }
    if image.width() == 0 || image.height() == 0 {
        return Ok(image.clone());
    }
    if image.width() == target_width {
        return Ok(image.clone());
    }

    let factor = target_width as f64 / image.width() as f64;
    let height = (image.height() as f64 * factor)
        .round()
        .clamp(1.0, u32::MAX as f64) as u32;
    resize_exact(image, target_width, height, interpolation)
}

/// Scale to height, maintain aspect (alias for resize_to_height).
pub fn scale_height(
    image: &DynamicImage,
    target_height: u32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    resize_to_height(image, target_height, interpolation)
}

/// Resize the image to the requested height preserving aspect ratio.
pub fn resize_to_height(
    image: &DynamicImage,
    target_height: u32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    if target_height == 0 {
        return Err(TransformError::InvalidDimensions {
            width: image.width(),
            height: target_height,
        });
    }
    if image.width() == 0 || image.height() == 0 {
        return Ok(image.clone());
    }
    if image.height() == target_height {
        return Ok(image.clone());
    }

    let factor = target_height as f64 / image.height() as f64;
    let width = (image.width() as f64 * factor)
        .round()
        .clamp(1.0, u32::MAX as f64) as u32;
    resize_exact(image, width, target_height, interpolation)
}

/// Resize the image so that it fits within the requested bounding box while maintaining aspect ratio.
/// This is an alias for `resize_to_fit`.
pub fn resize_fit(
    image: &DynamicImage,
    max_width: u32,
    max_height: u32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    resize_to_fit(image, max_width, max_height, interpolation)
}

/// Resize the image so that it fits within the requested bounding box while maintaining aspect ratio.
pub fn resize_to_fit(
    image: &DynamicImage,
    max_width: u32,
    max_height: u32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    if max_width == 0 || max_height == 0 {
        return Err(TransformError::InvalidDimensions {
            width: max_width,
            height: max_height,
        });
    }
    if image.width() == 0 || image.height() == 0 {
        return Ok(image.clone());
    }

    if image.width() <= max_width && image.height() <= max_height {
        return Ok(image.clone());
    }

    let width_ratio = max_width as f64 / image.width() as f64;
    let height_ratio = max_height as f64 / image.height() as f64;
    let factor = width_ratio.min(height_ratio);

    let new_width = (image.width() as f64 * factor)
        .round()
        .clamp(1.0, max_width as f64) as u32;
    let new_height = (image.height() as f64 * factor)
        .round()
        .clamp(1.0, max_height as f64) as u32;

    resize_exact(image, new_width, new_height, interpolation)
}

/// Create a thumbnail that fits inside a square of `max_dimension`.
pub fn thumbnail(
    image: &DynamicImage,
    max_dimension: u32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    resize_to_fit(image, max_dimension, max_dimension, interpolation)
}

fn resize_exact_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    new_width: u32,
    new_height: u32,
    interpolation: Interpolation,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let src_width = input.width();
    let src_height = input.height();

    if src_width == new_width && src_height == new_height {
        return Ok(input.clone());
    }

    let channels = channel_count::<P>();
    let src_width_usize = src_width as usize;
    let src_height_usize = src_height as usize;
    let dst_width_usize = new_width as usize;
    let dst_height_usize = new_height as usize;
    let src_row_stride = src_width_usize * channels;
    let dst_row_stride = dst_width_usize * channels;

    let input_slice = input.as_raw();
    let mut output_data = allocate_pixel_storage::<P>(new_width, new_height)?;

    match interpolation {
        Interpolation::Nearest => resize_with_kernel::<NearestInterpolation>(
            input_slice,
            &mut output_data,
            src_row_stride,
            dst_row_stride,
            channels,
            dst_width_usize,
            dst_height_usize,
            src_width_usize,
            src_height_usize,
        ),
        Interpolation::Bilinear => resize_with_kernel::<BilinearInterpolation>(
            input_slice,
            &mut output_data,
            src_row_stride,
            dst_row_stride,
            channels,
            dst_width_usize,
            dst_height_usize,
            src_width_usize,
            src_height_usize,
        ),
    }

    buffer_from_vec(new_width, new_height, output_data)
}

fn resize_with_kernel<K>(
    input: &[u8],
    output: &mut [u8],
    src_row_stride: usize,
    dst_row_stride: usize,
    channels: usize,
    dst_width: usize,
    dst_height: usize,
    src_width: usize,
    src_height: usize,
) where
    K: InterpolationKernel + Sync,
{
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        output.fill(0);
        return;
    }

    let x_ratio = if dst_width > 1 && src_width > 1 {
        (src_width - 1) as f32 / (dst_width - 1) as f32
    } else {
        0.0
    };
    let y_ratio = if dst_height > 1 && src_height > 1 {
        (src_height - 1) as f32 / (dst_height - 1) as f32
    } else {
        0.0
    };

    output
        .par_chunks_mut(dst_row_stride)
        .enumerate()
        .for_each(|(dst_y, row)| {
            let src_y = y_ratio * dst_y as f32;
            for dst_x in 0..dst_width {
                let src_x = x_ratio * dst_x as f32;
                let dst_idx = dst_x * channels;
                let pixel = &mut row[dst_idx..dst_idx + channels];
                if !K::sample_into(input, src_row_stride, channels, src_x, src_y, pixel) {
                    pixel.fill(0);
                }
            }
        });
}

fn scale_dimensions_by_factor(
    width: u32,
    height: u32,
    factor: f32,
) -> Result<(u32, u32), TransformError> {
    if !factor.is_finite() || factor <= 0.0 {
        return Err(TransformError::InvalidScaleFactor { factor });
    }

    let new_width = (width as f64 * factor as f64)
        .round()
        .clamp(1.0, u32::MAX as f64) as u32;
    let new_height = (height as f64 * factor as f64)
        .round()
        .clamp(1.0, u32::MAX as f64) as u32;

    Ok((new_width, new_height))
}

/// Resize the image to fill the requested bounding box while maintaining aspect ratio.
/// The image is scaled to cover the entire area, and overflow is cropped.
pub fn resize_fill(
    image: &DynamicImage,
    width: u32,
    height: u32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    if width == 0 || height == 0 {
        return Err(TransformError::InvalidDimensions { width, height });
    }
    if image.width() == 0 || image.height() == 0 {
        return Ok(image.clone());
    }

    let width_ratio = width as f64 / image.width() as f64;
    let height_ratio = height as f64 / image.height() as f64;
    let factor = width_ratio.max(height_ratio);

    let new_width = (image.width() as f64 * factor).round() as u32;
    let new_height = (image.height() as f64 * factor).round() as u32;

    let resized = resize_exact(image, new_width, new_height, interpolation)?;

    // Crop to exact dimensions if we overshot
    if new_width > width || new_height > height {
        let x = (new_width - width) / 2;
        let y = (new_height - height) / 2;
        crate::transforms::crop::crop(&resized, x, y, width, height)
    } else {
        Ok(resized)
    }
}

/// Resize the image to cover the requested bounding box while maintaining aspect ratio.
/// This is an alias for `resize_fill`.
pub fn resize_cover(
    image: &DynamicImage,
    width: u32,
    height: u32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    resize_fill(image, width, height, interpolation)
}

/// Scale the image by a uniform factor.
pub fn scale(
    image: &DynamicImage,
    factor: f32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    if !factor.is_finite() || factor <= 0.0 {
        return Err(TransformError::InvalidScaleFactor { factor });
    }

    let new_width = (image.width() as f64 * factor as f64)
        .round()
        .clamp(1.0, u32::MAX as f64) as u32;
    let new_height = (image.height() as f64 * factor as f64)
        .round()
        .clamp(1.0, u32::MAX as f64) as u32;

    resize_exact(image, new_width, new_height, interpolation)
}

/// Downscale the image to fit within the requested dimensions (only if larger).
/// If the image is already smaller, returns the original.
pub fn downscale(
    image: &DynamicImage,
    max_width: u32,
    max_height: u32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    if max_width == 0 || max_height == 0 {
        return Err(TransformError::InvalidDimensions {
            width: max_width,
            height: max_height,
        });
    }

    if image.width() <= max_width && image.height() <= max_height {
        return Ok(image.clone());
    }

    resize_to_fit(image, max_width, max_height, interpolation)
}

/// Upscale the image to at least the requested dimensions (only if smaller).
/// If the image is already larger, returns the original.
pub fn upscale(
    image: &DynamicImage,
    min_width: u32,
    min_height: u32,
    interpolation: Interpolation,
) -> Result<DynamicImage, TransformError> {
    if min_width == 0 || min_height == 0 {
        return Err(TransformError::InvalidDimensions {
            width: min_width,
            height: min_height,
        });
    }

    if image.width() >= min_width && image.height() >= min_height {
        return Ok(image.clone());
    }

    // Scale to meet the minimum dimension requirement
    let width_ratio = min_width as f64 / image.width() as f64;
    let height_ratio = min_height as f64 / image.height() as f64;
    let factor = width_ratio.max(height_ratio);

    let new_width = (image.width() as f64 * factor)
        .round()
        .clamp(min_width as f64, u32::MAX as f64) as u32;
    let new_height = (image.height() as f64 * factor)
        .round()
        .clamp(min_height as f64, u32::MAX as f64) as u32;

    resize_exact(image, new_width, new_height, interpolation)
}
