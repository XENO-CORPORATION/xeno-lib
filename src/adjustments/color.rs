use crate::error::TransformError;
use crate::transforms::utils::{
    allocate_pixel_storage, buffer_from_vec, channel_count, dispatch_on_dynamic_image,
};
use image::{DynamicImage, ImageBuffer, Pixel};
use rayon::prelude::*;

const EPSILON: f32 = 1e-6;

/// Convert an image to grayscale while preserving alpha channels.
pub fn grayscale(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    dispatch_on_dynamic_image!(image, grayscale_impl)
}

/// Invert the color channels of an image, preserving alpha.
pub fn invert(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    dispatch_on_dynamic_image!(image, invert_impl)
}

/// Adjust image brightness by a percentage in the range [-100, 100].
pub fn adjust_brightness(
    image: &DynamicImage,
    amount: f32,
) -> Result<DynamicImage, TransformError> {
    if !amount.is_finite() {
        return Err(TransformError::InvalidParameter {
            name: "brightness",
            value: amount,
        });
    }
    dispatch_on_dynamic_image!(image, adjust_brightness_impl, amount.clamp(-100.0, 100.0))
}

/// Adjust contrast by a percentage in the range [-100, 100].
pub fn adjust_contrast(image: &DynamicImage, amount: f32) -> Result<DynamicImage, TransformError> {
    if !amount.is_finite() {
        return Err(TransformError::InvalidParameter {
            name: "contrast",
            value: amount,
        });
    }
    dispatch_on_dynamic_image!(image, adjust_contrast_impl, amount.clamp(-100.0, 100.0))
}

/// Adjust saturation by a percentage in the range [-100, 100].
pub fn adjust_saturation(
    image: &DynamicImage,
    amount: f32,
) -> Result<DynamicImage, TransformError> {
    if !amount.is_finite() {
        return Err(TransformError::InvalidParameter {
            name: "saturation",
            value: amount,
        });
    }
    dispatch_on_dynamic_image!(image, adjust_saturation_impl, amount.clamp(-100.0, 100.0))
}

/// Rotate hue by the given degrees.
pub fn adjust_hue(image: &DynamicImage, degrees: f32) -> Result<DynamicImage, TransformError> {
    if !degrees.is_finite() {
        return Err(TransformError::InvalidParameter {
            name: "hue",
            value: degrees,
        });
    }
    dispatch_on_dynamic_image!(image, adjust_hue_impl, degrees)
}

/// Apply exposure adjustment using photographic stops.
pub fn adjust_exposure(image: &DynamicImage, stops: f32) -> Result<DynamicImage, TransformError> {
    if !stops.is_finite() {
        return Err(TransformError::InvalidParameter {
            name: "exposure",
            value: stops,
        });
    }
    dispatch_on_dynamic_image!(image, adjust_exposure_impl, stops)
}

/// Apply gamma correction with a positive gamma value.
pub fn adjust_gamma(image: &DynamicImage, gamma: f32) -> Result<DynamicImage, TransformError> {
    if !gamma.is_finite() || gamma <= 0.0 {
        return Err(TransformError::InvalidParameter {
            name: "gamma",
            value: gamma,
        });
    }
    dispatch_on_dynamic_image!(image, adjust_gamma_impl, gamma)
}

fn grayscale_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let channels = channel_count::<P>();
    let mut output_data = allocate_pixel_storage::<P>(input.width(), input.height())?;
    let input_slice = input.as_raw();

    output_data
        .par_chunks_mut(channels)
        .enumerate()
        .for_each(|(idx, out_pixel)| {
            let src_start = idx * channels;
            let src = &input_slice[src_start..src_start + channels];

            match channels {
                1 => out_pixel.copy_from_slice(src),
                2 => {
                    out_pixel[0] = src[0];
                    out_pixel[1] = src[1];
                }
                3 => {
                    let lum = luminance(src[0], src[1], src[2]);
                    out_pixel[0] = lum;
                    out_pixel[1] = lum;
                    out_pixel[2] = lum;
                }
                4 => {
                    let lum = luminance(src[0], src[1], src[2]);
                    out_pixel[0] = lum;
                    out_pixel[1] = lum;
                    out_pixel[2] = lum;
                    out_pixel[3] = src[3];
                }
                _ => unreachable!("unsupported channel count: {}", channels),
            }
        });

    buffer_from_vec(input.width(), input.height(), output_data)
}

fn invert_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let channels = channel_count::<P>();
    let mut output_data = allocate_pixel_storage::<P>(input.width(), input.height())?;
    let input_slice = input.as_raw();
    let has_alpha = has_alpha(channels);

    output_data
        .par_chunks_mut(channels)
        .enumerate()
        .for_each(|(idx, out_pixel)| {
            let src_start = idx * channels;
            let src = &input_slice[src_start..src_start + channels];

            for (channel_idx, out_value) in out_pixel.iter_mut().enumerate() {
                if has_alpha && channel_idx == channels - 1 {
                    *out_value = src[channel_idx];
                } else {
                    *out_value = 255u8.saturating_sub(src[channel_idx]);
                }
            }
        });

    buffer_from_vec(input.width(), input.height(), output_data)
}

fn adjust_brightness_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    amount: f32,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let channels = channel_count::<P>();
    let mut output_data = allocate_pixel_storage::<P>(input.width(), input.height())?;
    let input_slice = input.as_raw();
    let has_alpha = has_alpha(channels);
    let delta = 255.0 * (amount / 100.0);

    output_data
        .par_chunks_mut(channels)
        .enumerate()
        .for_each(|(idx, out_pixel)| {
            let src_start = idx * channels;
            let src = &input_slice[src_start..src_start + channels];

            for channel_idx in 0..channels {
                if has_alpha && channel_idx == channels - 1 {
                    out_pixel[channel_idx] = src[channel_idx];
                } else {
                    let value = src[channel_idx] as f32 + delta;
                    out_pixel[channel_idx] = clamp_to_u8(value);
                }
            }
        });

    buffer_from_vec(input.width(), input.height(), output_data)
}

fn adjust_contrast_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    amount: f32,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let channels = channel_count::<P>();
    let mut output_data = allocate_pixel_storage::<P>(input.width(), input.height())?;
    let input_slice = input.as_raw();
    let has_alpha = has_alpha(channels);
    let c = amount * 255.0 / 100.0;
    let factor = (259.0 * (c + 255.0)) / (255.0 * (259.0 - c));

    output_data
        .par_chunks_mut(channels)
        .enumerate()
        .for_each(|(idx, out_pixel)| {
            let src_start = idx * channels;
            let src = &input_slice[src_start..src_start + channels];

            for channel_idx in 0..channels {
                if has_alpha && channel_idx == channels - 1 {
                    out_pixel[channel_idx] = src[channel_idx];
                } else {
                    let centered = src[channel_idx] as f32 - 128.0;
                    let value = centered.mul_add(factor, 128.0);
                    out_pixel[channel_idx] = clamp_to_u8(value);
                }
            }
        });

    buffer_from_vec(input.width(), input.height(), output_data)
}

fn adjust_saturation_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    amount: f32,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let channels = channel_count::<P>();
    let mut output_data = allocate_pixel_storage::<P>(input.width(), input.height())?;
    let input_slice = input.as_raw();
    let factor = 1.0 + amount / 100.0;

    output_data
        .par_chunks_mut(channels)
        .enumerate()
        .for_each(|(idx, out_pixel)| {
            let src_start = idx * channels;
            let src = &input_slice[src_start..src_start + channels];

            match channels {
                1 => out_pixel[0] = src[0],
                2 => {
                    out_pixel[0] = src[0];
                    out_pixel[1] = src[1];
                }
                3 | 4 => {
                    let r = src[0] as f32;
                    let g = src[1] as f32;
                    let b = src[2] as f32;
                    let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    let adjust = |channel: f32| clamp_to_u8(lum + (channel - lum) * factor);
                    out_pixel[0] = adjust(r);
                    out_pixel[1] = adjust(g);
                    out_pixel[2] = adjust(b);
                    if channels == 4 {
                        out_pixel[3] = src[3];
                    }
                }
                _ => unreachable!("unsupported channel count: {}", channels),
            }
        });

    buffer_from_vec(input.width(), input.height(), output_data)
}

fn adjust_hue_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    degrees: f32,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let channels = channel_count::<P>();
    if channels < 3 {
        return Ok(input.clone());
    }

    let mut output_data = allocate_pixel_storage::<P>(input.width(), input.height())?;
    let input_slice = input.as_raw();
    let angle = degrees.to_radians();
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    output_data
        .par_chunks_mut(channels)
        .enumerate()
        .for_each(|(idx, out_pixel)| {
            let src_start = idx * channels;
            let src = &input_slice[src_start..src_start + channels];

            let (r, g, b) = (
                src[0] as f32 / 255.0,
                src[1] as f32 / 255.0,
                src[2] as f32 / 255.0,
            );

            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            let i = 0.596 * r - 0.274 * g - 0.322 * b;
            let q = 0.211 * r - 0.523 * g + 0.312 * b;

            let i_rot = i * cos_a - q * sin_a;
            let q_rot = i * sin_a + q * cos_a;

            let r_new = y + 0.956 * i_rot + 0.621 * q_rot;
            let g_new = y - 0.272 * i_rot - 0.647 * q_rot;
            let b_new = y - 1.105 * i_rot + 1.702 * q_rot;

            out_pixel[0] = clamp_to_u8(r_new * 255.0);
            out_pixel[1] = clamp_to_u8(g_new * 255.0);
            out_pixel[2] = clamp_to_u8(b_new * 255.0);

            if channels == 4 {
                out_pixel[3] = src[3];
            }
        });

    buffer_from_vec(input.width(), input.height(), output_data)
}

fn adjust_exposure_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    stops: f32,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let channels = channel_count::<P>();
    let mut output_data = allocate_pixel_storage::<P>(input.width(), input.height())?;
    let input_slice = input.as_raw();
    let has_alpha = has_alpha(channels);
    let factor = 2f32.powf(stops);

    output_data
        .par_chunks_mut(channels)
        .enumerate()
        .for_each(|(idx, out_pixel)| {
            let src_start = idx * channels;
            let src = &input_slice[src_start..src_start + channels];

            for channel_idx in 0..channels {
                if has_alpha && channel_idx == channels - 1 {
                    out_pixel[channel_idx] = src[channel_idx];
                } else {
                    let value = src[channel_idx] as f32 * factor;
                    out_pixel[channel_idx] = clamp_to_u8(value);
                }
            }
        });

    buffer_from_vec(input.width(), input.height(), output_data)
}

fn adjust_gamma_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
    gamma: f32,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let channels = channel_count::<P>();
    let mut output_data = allocate_pixel_storage::<P>(input.width(), input.height())?;
    let input_slice = input.as_raw();
    let has_alpha = has_alpha(channels);
    let gamma = gamma.max(EPSILON);

    output_data
        .par_chunks_mut(channels)
        .enumerate()
        .for_each(|(idx, out_pixel)| {
            let src_start = idx * channels;
            let src = &input_slice[src_start..src_start + channels];

            for channel_idx in 0..channels {
                if has_alpha && channel_idx == channels - 1 {
                    out_pixel[channel_idx] = src[channel_idx];
                } else {
                    let normalized = src[channel_idx] as f32 / 255.0;
                    let corrected = normalized.powf(gamma);
                    out_pixel[channel_idx] = clamp_to_u8(corrected * 255.0);
                }
            }
        });

    buffer_from_vec(input.width(), input.height(), output_data)
}

#[inline]
fn luminance(r: u8, g: u8, b: u8) -> u8 {
    let r = r as f32;
    let g = g as f32;
    let b = b as f32;
    clamp_to_u8(0.2126 * r + 0.7152 * g + 0.0722 * b)
}

#[inline]
fn has_alpha(channels: usize) -> bool {
    matches!(channels, 2 | 4)
}

#[inline]
fn clamp_to_u8(value: f32) -> u8 {
    value.max(0.0).min(255.0).round() as u8
}
