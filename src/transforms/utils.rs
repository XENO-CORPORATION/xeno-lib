use crate::error::TransformError;
use image::Primitive;
use image::{ImageBuffer, Pixel};

/// Helper macro to dispatch a transformation implemented for concrete pixel buffers over `DynamicImage`.
macro_rules! dispatch_on_dynamic_image {
    ($image:expr, $func:path $(, $args:expr )* ) => {{
        match $image {
            &image::DynamicImage::ImageLuma8(ref buffer) => $func(buffer $(, $args)*).map(image::DynamicImage::ImageLuma8),
            &image::DynamicImage::ImageLumaA8(ref buffer) => $func(buffer $(, $args)*).map(image::DynamicImage::ImageLumaA8),
            &image::DynamicImage::ImageRgb8(ref buffer) => $func(buffer $(, $args)*).map(image::DynamicImage::ImageRgb8),
            &image::DynamicImage::ImageRgba8(ref buffer) => $func(buffer $(, $args)*).map(image::DynamicImage::ImageRgba8),
            other => Err(TransformError::UnsupportedColorType(other.color())),
        }
    }};
}

pub(crate) use dispatch_on_dynamic_image;

/// Allocate a new image buffer with the requested dimensions, returning an error if the size exceeds addressable memory.
pub(crate) fn allocate_pixel_storage<P>(
    width: u32,
    height: u32,
) -> Result<Vec<P::Subpixel>, TransformError>
where
    P: Pixel + 'static,
    P::Subpixel: Primitive + Default + Send + Sync,
{
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(TransformError::AllocationFailed { width, height })?;
    let channels = P::CHANNEL_COUNT as usize;
    let len = pixels
        .checked_mul(channels)
        .ok_or(TransformError::AllocationFailed { width, height })?;

    let mut data = Vec::<P::Subpixel>::with_capacity(len);
    unsafe {
        // SAFETY: All callers fully initialize every element of `data` before it is read.
        data.set_len(len);
    }
    Ok(data)
}

/// Number of channels for supported pixel formats.
#[inline]
pub(crate) fn channel_count<P>() -> usize
where
    P: Pixel,
{
    P::CHANNEL_COUNT as usize
}

pub(crate) fn buffer_from_vec<P>(
    width: u32,
    height: u32,
    data: Vec<P::Subpixel>,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel + 'static,
    P::Subpixel: Primitive,
{
    ImageBuffer::from_vec(width, height, data)
        .ok_or(TransformError::AllocationFailed { width, height })
}
