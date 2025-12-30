#![allow(unsafe_op_in_unsafe_fn)]
//! SIMD-accelerated helpers for image flipping operations.
//!
//! The horizontal helpers return the number of pixels handled via SIMD so the
//! caller can finish any remainder using the scalar fallback. Vertical helpers
//! expose fast row copies used when reversing row order.

use std::ptr;

/// Attempts to horizontally flip `rows` consecutive rows in `dst` using SIMD.
/// Returns the number of pixels processed per row (0 if SIMD was not applied).
#[inline]
pub(crate) fn horizontal_rows(
    src: &[u8],
    dst: &mut [u8],
    width_pixels: usize,
    channels: usize,
    row_stride: usize,
    rows: usize,
) -> usize {
    if width_pixels == 0 || channels == 0 || rows == 0 {
        return 0;
    }

    if channels == 3 {
        return flip_rows_rgb_scalar(src, dst, width_pixels, row_stride, rows);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            let processed =
                unsafe { horizontal_rows_avx2(src, dst, width_pixels, channels, row_stride, rows) };
            if processed > 0 {
                return processed;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let processed =
            unsafe { horizontal_rows_neon(src, dst, width_pixels, channels, row_stride, rows) };
        if processed > 0 {
            return processed;
        }
    }

    0
}

/// SIMD-assisted memcpy for a single row. Returns `true` if SIMD was used.
#[inline]
pub(crate) fn copy_row(src: &[u8], dst: &mut [u8]) -> bool {
    if src.len() != dst.len() || src.is_empty() {
        return false;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            if unsafe { copy_row_avx2(src, dst) } {
                return true;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if unsafe { copy_row_neon(src, dst) } {
            return true;
        }
    }

    false
}

#[inline]
fn flip_rows_rgb_scalar(
    src: &[u8],
    dst: &mut [u8],
    width_pixels: usize,
    row_stride: usize,
    rows: usize,
) -> usize {
    for row in 0..rows {
        let offset = row * row_stride;
        let src_row = &src[offset..offset + row_stride];
        let dst_row = &mut dst[offset..offset + row_stride];

        let mut dst_pixel = 0usize;
        while dst_pixel < width_pixels {
            let src_pixel = width_pixels - 1 - dst_pixel;
            let src_offset = src_pixel * 3;
            let dst_offset = dst_pixel * 3;
            unsafe {
                ptr::copy_nonoverlapping(
                    src_row.as_ptr().add(src_offset),
                    dst_row.as_mut_ptr().add(dst_offset),
                    3,
                );
            }
            dst_pixel += 1;
        }
    }

    width_pixels
}

// ---------------------------------------------------------------------------
// AVX2 implementations

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, _mm_sfence, _mm256_loadu_si256, _mm256_permute2x128_si256,
    _mm256_permutevar8x32_epi32, _mm256_setr_epi8, _mm256_setr_epi32, _mm256_shuffle_epi8,
    _mm256_storeu_si256, _mm256_stream_si256,
};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_rows_avx2(
    src: &[u8],
    dst: &mut [u8],
    width_pixels: usize,
    channels: usize,
    row_stride: usize,
    rows: usize,
) -> usize {
    let mut processed = 0usize;
    for row in 0..rows {
        let offset = row * row_stride;
        let src_row = &src[offset..offset + row_stride];
        let dst_row = &mut dst[offset..offset + row_stride];
        let stream_hint = (dst_row.as_ptr() as usize & 31) == 0;

        processed = match channels {
            1 => flip_row_avx2_bytes(src_row, dst_row, width_pixels, stream_hint),
            2 => flip_row_avx2_pairs(src_row, dst_row, width_pixels, stream_hint),
            4 => flip_row_avx2_rgba(src_row, dst_row, width_pixels, stream_hint),
            _ => 0,
        };

        if processed == 0 {
            break;
        }
    }

    _mm_sfence();
    processed
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn reverse_bytes_32(v: __m256i) -> __m256i {
    let shuffle = _mm256_setr_epi8(
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
        5, 4, 3, 2, 1, 0,
    );
    let lane_reversed = _mm256_shuffle_epi8(v, shuffle);
    _mm256_permute2x128_si256(lane_reversed, lane_reversed, 0x01)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn reverse_pairs_32(v: __m256i) -> __m256i {
    let shuffle = _mm256_setr_epi8(
        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7,
        4, 5, 2, 3, 0, 1,
    );
    let lane_reversed = _mm256_shuffle_epi8(v, shuffle);
    _mm256_permute2x128_si256(lane_reversed, lane_reversed, 0x01)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn store_m256(ptr: *mut __m256i, data: __m256i, stream_hint: bool) {
    if stream_hint && (ptr as usize & 31) == 0 {
        _mm256_stream_si256(ptr, data);
    } else {
        _mm256_storeu_si256(ptr, data);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn flip_row_avx2_bytes(
    src: &[u8],
    dst: &mut [u8],
    width_pixels: usize,
    stream_hint: bool,
) -> usize {
    const PIXELS_PER_CHUNK: usize = 32;
    let chunks = width_pixels / PIXELS_PER_CHUNK;
    if chunks == 0 {
        return 0;
    }

    let mut chunk_idx = 0usize;
    while chunk_idx + 1 < chunks {
        let src_pixel0 = width_pixels - (chunk_idx + 1) * PIXELS_PER_CHUNK;
        let src_pixel1 = width_pixels - (chunk_idx + 2) * PIXELS_PER_CHUNK;
        let dst_pixel0 = chunk_idx * PIXELS_PER_CHUNK;
        let dst_pixel1 = (chunk_idx + 1) * PIXELS_PER_CHUNK;

        let src_ptr0 = src.as_ptr().add(src_pixel0) as *const __m256i;
        let src_ptr1 = src.as_ptr().add(src_pixel1) as *const __m256i;
        let dst_ptr0 = dst.as_mut_ptr().add(dst_pixel0) as *mut __m256i;
        let dst_ptr1 = dst.as_mut_ptr().add(dst_pixel1) as *mut __m256i;

        let data0 = _mm256_loadu_si256(src_ptr0);
        let data1 = _mm256_loadu_si256(src_ptr1);
        let reversed0 = reverse_bytes_32(data0);
        let reversed1 = reverse_bytes_32(data1);
        store_m256(dst_ptr0, reversed0, stream_hint);
        store_m256(dst_ptr1, reversed1, stream_hint);

        chunk_idx += 2;
    }

    if chunk_idx < chunks {
        let src_pixel = width_pixels - (chunk_idx + 1) * PIXELS_PER_CHUNK;
        let dst_pixel = chunk_idx * PIXELS_PER_CHUNK;
        let src_ptr = src.as_ptr().add(src_pixel) as *const __m256i;
        let dst_ptr = dst.as_mut_ptr().add(dst_pixel) as *mut __m256i;

        let data = _mm256_loadu_si256(src_ptr);
        let reversed = reverse_bytes_32(data);
        store_m256(dst_ptr, reversed, stream_hint);
    }

    chunks * PIXELS_PER_CHUNK
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn flip_row_avx2_pairs(
    src: &[u8],
    dst: &mut [u8],
    width_pixels: usize,
    stream_hint: bool,
) -> usize {
    const PIXELS_PER_CHUNK: usize = 16;
    let chunks = width_pixels / PIXELS_PER_CHUNK;
    if chunks == 0 {
        return 0;
    }

    let mut chunk_idx = 0usize;
    while chunk_idx + 1 < chunks {
        let src_pixel0 = width_pixels - (chunk_idx + 1) * PIXELS_PER_CHUNK;
        let src_pixel1 = width_pixels - (chunk_idx + 2) * PIXELS_PER_CHUNK;
        let dst_pixel0 = chunk_idx * PIXELS_PER_CHUNK;
        let dst_pixel1 = (chunk_idx + 1) * PIXELS_PER_CHUNK;

        let src_ptr0 = src.as_ptr().add(src_pixel0 * 2) as *const __m256i;
        let src_ptr1 = src.as_ptr().add(src_pixel1 * 2) as *const __m256i;
        let dst_ptr0 = dst.as_mut_ptr().add(dst_pixel0 * 2) as *mut __m256i;
        let dst_ptr1 = dst.as_mut_ptr().add(dst_pixel1 * 2) as *mut __m256i;

        let data0 = _mm256_loadu_si256(src_ptr0);
        let data1 = _mm256_loadu_si256(src_ptr1);
        let reversed0 = reverse_pairs_32(data0);
        let reversed1 = reverse_pairs_32(data1);
        store_m256(dst_ptr0, reversed0, stream_hint);
        store_m256(dst_ptr1, reversed1, stream_hint);

        chunk_idx += 2;
    }

    if chunk_idx < chunks {
        let src_pixel = width_pixels - (chunk_idx + 1) * PIXELS_PER_CHUNK;
        let dst_pixel = chunk_idx * PIXELS_PER_CHUNK;
        let src_ptr = src.as_ptr().add(src_pixel * 2) as *const __m256i;
        let dst_ptr = dst.as_mut_ptr().add(dst_pixel * 2) as *mut __m256i;

        let data = _mm256_loadu_si256(src_ptr);
        let reversed = reverse_pairs_32(data);
        store_m256(dst_ptr, reversed, stream_hint);
    }

    chunks * PIXELS_PER_CHUNK
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn flip_row_avx2_rgba(
    src: &[u8],
    dst: &mut [u8],
    width_pixels: usize,
    stream_hint: bool,
) -> usize {
    const PIXELS_PER_CHUNK: usize = 8;
    let chunks = width_pixels / PIXELS_PER_CHUNK;
    if chunks == 0 {
        return 0;
    }

    let permute_indices = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    let mut chunk_idx = 0usize;
    while chunk_idx + 1 < chunks {
        let src_pixel0 = width_pixels - (chunk_idx + 1) * PIXELS_PER_CHUNK;
        let src_pixel1 = width_pixels - (chunk_idx + 2) * PIXELS_PER_CHUNK;
        let dst_pixel0 = chunk_idx * PIXELS_PER_CHUNK;
        let dst_pixel1 = (chunk_idx + 1) * PIXELS_PER_CHUNK;

        let src_ptr0 = src.as_ptr().add(src_pixel0 * 4) as *const __m256i;
        let src_ptr1 = src.as_ptr().add(src_pixel1 * 4) as *const __m256i;
        let dst_ptr0 = dst.as_mut_ptr().add(dst_pixel0 * 4) as *mut __m256i;
        let dst_ptr1 = dst.as_mut_ptr().add(dst_pixel1 * 4) as *mut __m256i;

        let data0 = _mm256_loadu_si256(src_ptr0);
        let data1 = _mm256_loadu_si256(src_ptr1);
        let reversed0 = _mm256_permutevar8x32_epi32(data0, permute_indices);
        let reversed1 = _mm256_permutevar8x32_epi32(data1, permute_indices);
        store_m256(dst_ptr0, reversed0, stream_hint);
        store_m256(dst_ptr1, reversed1, stream_hint);

        chunk_idx += 2;
    }

    if chunk_idx < chunks {
        let src_pixel = width_pixels - (chunk_idx + 1) * PIXELS_PER_CHUNK;
        let dst_pixel = chunk_idx * PIXELS_PER_CHUNK;
        let src_ptr = src.as_ptr().add(src_pixel * 4) as *const __m256i;
        let dst_ptr = dst.as_mut_ptr().add(dst_pixel * 4) as *mut __m256i;

        let data = _mm256_loadu_si256(src_ptr);
        let reversed = _mm256_permutevar8x32_epi32(data, permute_indices);
        store_m256(dst_ptr, reversed, stream_hint);
    }

    chunks * PIXELS_PER_CHUNK
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_row_avx2(src: &[u8], dst: &mut [u8]) -> bool {
    let len = src.len();
    let mut offset = 0usize;
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let stream_hint = (dst_ptr as usize & 31) == 0;

    while offset + 64 <= len {
        let src0 = src_ptr.add(offset) as *const __m256i;
        let src1 = src_ptr.add(offset + 32) as *const __m256i;
        let dst0 = dst_ptr.add(offset) as *mut __m256i;
        let dst1 = dst_ptr.add(offset + 32) as *mut __m256i;

        let data0 = _mm256_loadu_si256(src0);
        let data1 = _mm256_loadu_si256(src1);
        store_m256(dst0, data0, stream_hint);
        store_m256(dst1, data1, stream_hint);

        offset += 64;
    }

    if offset + 32 <= len {
        let src0 = src_ptr.add(offset) as *const __m256i;
        let dst0 = dst_ptr.add(offset) as *mut __m256i;
        let data0 = _mm256_loadu_si256(src0);
        store_m256(dst0, data0, stream_hint);
        offset += 32;
    }

    let remaining = len - offset;
    if remaining > 0 {
        ptr::copy_nonoverlapping(src_ptr.add(offset), dst_ptr.add(offset), remaining);
    }

    _mm_sfence();
    true
}

// ---------------------------------------------------------------------------
// NEON implementations

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    uint8x16_t, uint16x8_t, uint32x4_t, vcombine_u8, vcombine_u16, vcombine_u32, vget_high_u8,
    vget_high_u16, vget_high_u32, vget_low_u8, vget_low_u16, vget_low_u32, vld1q_u8, vld1q_u16,
    vld1q_u32, vrev64q_u8, vrev64q_u16, vrev64q_u32, vst1q_u8, vst1q_u16, vst1q_u32,
};

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn horizontal_rows_neon(
    src: &[u8],
    dst: &mut [u8],
    width_pixels: usize,
    channels: usize,
    row_stride: usize,
    rows: usize,
) -> usize {
    let mut processed = 0usize;
    for row in 0..rows {
        let offset = row * row_stride;
        let src_row = &src[offset..offset + row_stride];
        let dst_row = &mut dst[offset..offset + row_stride];

        processed = match channels {
            1 => flip_row_neon_bytes(src_row, dst_row, width_pixels),
            2 => flip_row_neon_pairs(src_row, dst_row, width_pixels),
            4 => flip_row_neon_rgba(src_row, dst_row, width_pixels),
            _ => 0,
        };

        if processed == 0 {
            break;
        }
    }

    processed
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn flip_row_neon_bytes(src: &[u8], dst: &mut [u8], width_pixels: usize) -> usize {
    const PIXELS_PER_CHUNK: usize = 16;
    let chunks = width_pixels / PIXELS_PER_CHUNK;
    if chunks == 0 {
        return 0;
    }

    for chunk_idx in 0..chunks {
        let src_pixel = width_pixels - (chunk_idx + 1) * PIXELS_PER_CHUNK;
        let dst_pixel = chunk_idx * PIXELS_PER_CHUNK;
        let src_ptr = src.as_ptr().add(src_pixel);
        let dst_ptr = dst.as_mut_ptr().add(dst_pixel);

        let data: uint8x16_t = vld1q_u8(src_ptr);
        let reversed64 = vrev64q_u8(data);
        let reversed = vcombine_u8(vget_high_u8(reversed64), vget_low_u8(reversed64));
        vst1q_u8(dst_ptr, reversed);
    }

    chunks * PIXELS_PER_CHUNK
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn flip_row_neon_pairs(src: &[u8], dst: &mut [u8], width_pixels: usize) -> usize {
    const PIXELS_PER_CHUNK: usize = 8;
    let chunks = width_pixels / PIXELS_PER_CHUNK;
    if chunks == 0 {
        return 0;
    }

    for chunk_idx in 0..chunks {
        let src_pixel = width_pixels - (chunk_idx + 1) * PIXELS_PER_CHUNK;
        let dst_pixel = chunk_idx * PIXELS_PER_CHUNK;
        let src_ptr = src.as_ptr().add(src_pixel * 2) as *const u16;
        let dst_ptr = dst.as_mut_ptr().add(dst_pixel * 2) as *mut u16;

        let data: uint16x8_t = vld1q_u16(src_ptr);
        let reversed64 = vrev64q_u16(data);
        let reversed = vcombine_u16(vget_high_u16(reversed64), vget_low_u16(reversed64));
        vst1q_u16(dst_ptr, reversed);
    }

    chunks * PIXELS_PER_CHUNK
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn flip_row_neon_rgba(src: &[u8], dst: &mut [u8], width_pixels: usize) -> usize {
    const PIXELS_PER_CHUNK: usize = 4;
    let chunks = width_pixels / PIXELS_PER_CHUNK;
    if chunks == 0 {
        return 0;
    }

    for chunk_idx in 0..chunks {
        let src_pixel = width_pixels - (chunk_idx + 1) * PIXELS_PER_CHUNK;
        let dst_pixel = chunk_idx * PIXELS_PER_CHUNK;
        let src_ptr = src.as_ptr().add(src_pixel * 4) as *const u32;
        let dst_ptr = dst.as_mut_ptr().add(dst_pixel * 4) as *mut u32;

        let data: uint32x4_t = vld1q_u32(src_ptr);
        let reversed64 = vrev64q_u32(data);
        let reversed = vcombine_u32(vget_high_u32(reversed64), vget_low_u32(reversed64));
        vst1q_u32(dst_ptr, reversed);
    }

    chunks * PIXELS_PER_CHUNK
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn copy_row_neon(src: &[u8], dst: &mut [u8]) -> bool {
    let len = src.len();
    let mut offset = 0usize;
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    while offset + 32 <= len {
        let data0: uint8x16_t = vld1q_u8(src_ptr.add(offset));
        let data1: uint8x16_t = vld1q_u8(src_ptr.add(offset + 16));
        vst1q_u8(dst_ptr.add(offset), data0);
        vst1q_u8(dst_ptr.add(offset + 16), data1);
        offset += 32;
    }

    if offset + 16 <= len {
        let data0: uint8x16_t = vld1q_u8(src_ptr.add(offset));
        vst1q_u8(dst_ptr.add(offset), data0);
        offset += 16;
    }

    let remaining = len - offset;
    if remaining > 0 {
        ptr::copy_nonoverlapping(src_ptr.add(offset), dst_ptr.add(offset), remaining);
    }

    true
}
