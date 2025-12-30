use super::flip_simd;
use crate::error::TransformError;
use crate::transforms::utils::{
    allocate_pixel_storage, buffer_from_vec, channel_count, dispatch_on_dynamic_image,
};
use image::{DynamicImage, ImageBuffer, Pixel};
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, _mm_sfence, _mm256_loadu_si256, _mm256_permutevar8x32_epi32, _mm256_setr_epi32,
    _mm256_storeu_si256, _mm256_stream_si256,
};

/// Returns a horizontally flipped copy of the provided image.
pub fn flip_horizontal(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    dispatch_on_dynamic_image!(image, flip_horizontal_impl)
}

/// Returns a vertically flipped copy of the provided image.
pub fn flip_vertical(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    dispatch_on_dynamic_image!(image, flip_vertical_impl)
}

/// Returns a copy of the image flipped both horizontally and vertically (equivalent to 180-degree rotation).
pub fn flip_both(image: &DynamicImage) -> Result<DynamicImage, TransformError> {
    dispatch_on_dynamic_image!(image, flip_both_impl)
}

pub(crate) fn flip_horizontal_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let width = input.width() as usize;
    let height = input.height();
    if width == 0 || height == 0 {
        return Ok(input.clone());
    }
    let channels = channel_count::<P>();
    let row_stride = width * channels;

    let mut output_data = allocate_pixel_storage::<P>(input.width(), height)?; // safe: we fill all bytes below
    let input_slice = input.as_raw();

    output_data
        .par_chunks_mut(row_stride)
        .with_min_len(32)
        .enumerate()
        .for_each(|(row_idx, dst_row)| {
            let src_row_start = row_idx * row_stride;
            let src_row = &input_slice[src_row_start..src_row_start + row_stride];

            if channels == 4 {
                let src_u32 =
                    unsafe { std::slice::from_raw_parts(src_row.as_ptr() as *const u32, width) };
                let dst_u32 = unsafe {
                    std::slice::from_raw_parts_mut(dst_row.as_mut_ptr() as *mut u32, width)
                };

                unsafe { flip_row_rgba_u32(src_u32.as_ptr(), dst_u32.as_mut_ptr(), width) };
            } else {
                let processed_pixels =
                    flip_simd::horizontal_rows(src_row, dst_row, width, channels, row_stride, 1);
                flip_horizontal_scalar_tail(src_row, dst_row, width, channels, processed_pixels);
            }
        });

    buffer_from_vec(input.width(), height, output_data)
}

pub(crate) fn flip_vertical_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let width = input.width() as usize;
    let height = input.height();
    if width == 0 || height == 0 {
        return Ok(input.clone());
    }
    let channels = channel_count::<P>();
    let row_stride = width * channels;

    let mut output_data = allocate_pixel_storage::<P>(input.width(), height)?; // safe: every row is written below
    let input_slice = input.as_raw();
    let height_usize = height as usize;

    output_data
        .par_chunks_mut(row_stride)
        .with_min_len(32)
        .enumerate()
        .for_each(|(dst_row_idx, dst_row)| {
            let src_row_idx = height_usize - 1 - dst_row_idx;
            let src_offset = src_row_idx * row_stride;
            let src_row = &input_slice[src_offset..src_offset + row_stride];

            if !flip_simd::copy_row(src_row, dst_row) {
                dst_row.copy_from_slice(src_row);
            }
        });

    buffer_from_vec(input.width(), height, output_data)
}

pub(crate) fn flip_both_impl<P>(
    input: &ImageBuffer<P, Vec<P::Subpixel>>,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, TransformError>
where
    P: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let width = input.width() as usize;
    let height = input.height();
    if width == 0 || height == 0 {
        return Ok(input.clone());
    }
    let channels = channel_count::<P>();
    let row_stride = width * channels;

    let mut output_data = allocate_pixel_storage::<P>(input.width(), height)?;
    let input_slice = input.as_raw();
    let height_usize = height as usize;

    // Flip both axes: reverse row order AND reverse pixel order within each row
    output_data
        .par_chunks_mut(row_stride)
        .with_min_len(32)
        .enumerate()
        .for_each(|(dst_row_idx, dst_row)| {
            let src_row_idx = height_usize - 1 - dst_row_idx;
            let src_offset = src_row_idx * row_stride;
            let src_row = &input_slice[src_offset..src_offset + row_stride];

            if channels == 4 {
                let src_u32 =
                    unsafe { std::slice::from_raw_parts(src_row.as_ptr() as *const u32, width) };
                let dst_u32 = unsafe {
                    std::slice::from_raw_parts_mut(dst_row.as_mut_ptr() as *mut u32, width)
                };

                unsafe { flip_row_rgba_u32(src_u32.as_ptr(), dst_u32.as_mut_ptr(), width) };
            } else {
                let processed_pixels =
                    flip_simd::horizontal_rows(src_row, dst_row, width, channels, row_stride, 1);
                flip_horizontal_scalar_tail(src_row, dst_row, width, channels, processed_pixels);
            }
        });

    buffer_from_vec(input.width(), height, output_data)
}

fn flip_horizontal_scalar_tail(
    src_row: &[u8],
    dst_row: &mut [u8],
    width_pixels: usize,
    channels: usize,
    start_pixel: usize,
) {
    if start_pixel >= width_pixels {
        return;
    }

    for dst_pixel in start_pixel..width_pixels {
        let src_pixel = width_pixels - 1 - dst_pixel;
        let dst_offset = dst_pixel * channels;
        let src_offset = src_pixel * channels;
        dst_row[dst_offset..dst_offset + channels]
            .copy_from_slice(&src_row[src_offset..src_offset + channels]);
    }
}

#[inline]
unsafe fn flip_row_rgba_u32(src: *const u32, dst: *mut u32, width: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 is detected, pointers are valid per caller contract
            unsafe { flip_row_avx2_u32(src, dst, width) };
            return;
        }
    }

    // SAFETY: Pointers are valid per caller contract
    unsafe { flip_row_scalar_u32(src, dst, width) };
}

#[inline]
unsafe fn flip_row_scalar_u32(src: *const u32, dst: *mut u32, width: usize) {
    for i in 0..width {
        // SAFETY: Caller guarantees src and dst point to valid memory of at least width elements
        unsafe {
            let value = src.add(i).read();
            dst.add(width - 1 - i).write(value);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn flip_row_avx2_u32(src: *const u32, dst: *mut u32, width: usize) {
    #[target_feature(enable = "avx2")]
    unsafe fn inner(src: *const u32, dst: *mut u32, width: usize) {
        // SAFETY: This function is only called when AVX2 is available and pointers are valid
        unsafe {
            let mask = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            let chunks = width / 8;
            let mut streamed = false;

            for chunk in 0..chunks {
                let src_idx = chunk * 8;
                let dst_idx = width - (chunk + 1) * 8;

                let pixels = _mm256_loadu_si256(src.add(src_idx) as *const __m256i);
                let reversed = _mm256_permutevar8x32_epi32(pixels, mask);
                let dst_ptr = dst.add(dst_idx) as *mut __m256i;

                if (dst_ptr as usize & 31) == 0 {
                    _mm256_stream_si256(dst_ptr, reversed);
                    streamed = true;
                } else {
                    _mm256_storeu_si256(dst_ptr, reversed);
                }
            }

            let remainder_start = chunks * 8;
            for i in remainder_start..width {
                let value = src.add(i).read();
                dst.add(width - 1 - i).write(value);
            }

            if streamed {
                _mm_sfence();
            }
        }
    }

    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected, pointers are valid per caller contract
        unsafe { inner(src, dst, width) };
    } else {
        // SAFETY: Pointers are valid per caller contract
        unsafe { flip_row_scalar_u32(src, dst, width) };
    }
}
