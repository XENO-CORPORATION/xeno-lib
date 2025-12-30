//! Shared interpolation kernels for sampling pixel data.

/// Interpolation strategy for resampling operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Nearest,
    Bilinear,
}

pub(crate) trait InterpolationKernel {
    fn sample_into(
        input: &[u8],
        row_stride: usize,
        channels: usize,
        src_x: f32,
        src_y: f32,
        out: &mut [u8],
    ) -> bool;
}

pub(crate) struct NearestInterpolation;
pub(crate) struct BilinearInterpolation;

impl InterpolationKernel for NearestInterpolation {
    #[inline]
    fn sample_into(
        input: &[u8],
        row_stride: usize,
        channels: usize,
        src_x: f32,
        src_y: f32,
        out: &mut [u8],
    ) -> bool {
        let width = row_stride / channels;
        let height = input.len() / row_stride;

        if src_x < 0.0 || src_y < 0.0 || src_x > (width - 1) as f32 || src_y > (height - 1) as f32 {
            return false;
        }

        let src_x = src_x.round().clamp(0.0, (width - 1) as f32) as usize;
        let src_y = src_y.round().clamp(0.0, (height - 1) as f32) as usize;
        let src_idx = src_y * row_stride + src_x * channels;

        out.copy_from_slice(&input[src_idx..src_idx + channels]);
        true
    }
}

impl InterpolationKernel for BilinearInterpolation {
    #[inline]
    fn sample_into(
        input: &[u8],
        row_stride: usize,
        channels: usize,
        src_x: f32,
        src_y: f32,
        out: &mut [u8],
    ) -> bool {
        let width = row_stride / channels;
        let height = input.len() / row_stride;

        if src_x < 0.0 || src_y < 0.0 || src_x > (width - 1) as f32 || src_y > (height - 1) as f32 {
            return false;
        }

        let x0 = src_x.floor();
        let y0 = src_y.floor();
        let x1 = (x0 + 1.0).min((width - 1) as f32);
        let y1 = (y0 + 1.0).min((height - 1) as f32);

        let dx = src_x - x0;
        let dy = src_y - y0;

        let x0 = x0 as usize;
        let y0 = y0 as usize;
        let x1 = x1 as usize;
        let y1 = y1 as usize;

        for channel in 0..channels {
            let c00_idx = y0 * row_stride + x0 * channels + channel;
            let c10_idx = y0 * row_stride + x1 * channels + channel;
            let c01_idx = y1 * row_stride + x0 * channels + channel;
            let c11_idx = y1 * row_stride + x1 * channels + channel;

            let c00 = input[c00_idx] as f32;
            let c10 = input[c10_idx] as f32;
            let c01 = input[c01_idx] as f32;
            let c11 = input[c11_idx] as f32;

            let top = c00.mul_add(1.0 - dx, c10 * dx);
            let bottom = c01.mul_add(1.0 - dx, c11 * dx);
            let value = top.mul_add(1.0 - dy, bottom * dy);
            out[channel] = value.round().clamp(0.0, 255.0) as u8;
        }

        true
    }
}
