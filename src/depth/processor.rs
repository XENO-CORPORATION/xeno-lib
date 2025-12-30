//! Depth estimation processing logic.

use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage, imageops::FilterType};
use ndarray::Array4;

use crate::error::TransformError;
use super::model::DepthSession;

/// Result of depth estimation.
#[derive(Debug, Clone)]
pub struct DepthMap {
    /// Depth values as 2D array (height x width).
    pub values: ndarray::Array2<f32>,
    /// Minimum depth value.
    pub min_depth: f32,
    /// Maximum depth value.
    pub max_depth: f32,
}

impl DepthMap {
    /// Converts depth map to grayscale image.
    pub fn to_grayscale(&self) -> GrayImage {
        let (height, width) = self.values.dim();
        let range = self.max_depth - self.min_depth;

        let mut image = GrayImage::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let normalized = if range > 0.0 {
                    (self.values[[y, x]] - self.min_depth) / range
                } else {
                    0.5
                };
                let gray = (normalized * 255.0).clamp(0.0, 255.0) as u8;
                image.put_pixel(x as u32, y as u32, Luma([gray]));
            }
        }

        image
    }

    /// Converts depth map to colored visualization.
    pub fn to_colored(&self) -> RgbImage {
        let (height, width) = self.values.dim();
        let range = self.max_depth - self.min_depth;

        let mut image = RgbImage::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let normalized = if range > 0.0 {
                    (self.values[[y, x]] - self.min_depth) / range
                } else {
                    0.5
                };

                // Apply colormap (Viridis-like)
                let rgb = depth_to_color(normalized);
                image.put_pixel(x as u32, y as u32, Rgb(rgb));
            }
        }

        image
    }

    /// Gets depth value at a specific point.
    pub fn depth_at(&self, x: usize, y: usize) -> Option<f32> {
        if y < self.values.nrows() && x < self.values.ncols() {
            Some(self.values[[y, x]])
        } else {
            None
        }
    }
}

/// Estimates depth from an image.
pub fn estimate_depth_impl(
    image: &DynamicImage,
    session: &mut DepthSession,
) -> Result<DepthMap, TransformError> {
    let (input_w, input_h) = session.input_size();
    let original_width = image.width();
    let original_height = image.height();

    // Resize to model input size
    let resized = image.resize_exact(input_w, input_h, FilterType::Lanczos3);

    // Convert to tensor
    let input_tensor = image_to_tensor(&resized)?;

    // Run depth estimation
    let mut depth_output = session.run(&input_tensor)?;

    // Normalize if configured
    let config = session.config();
    let mut min_depth = depth_output.iter().copied().fold(f32::INFINITY, f32::min);
    let mut max_depth = depth_output.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if config.normalize_output {
        let range = max_depth - min_depth;
        if range > 0.0 {
            for val in depth_output.iter_mut() {
                *val = (*val - min_depth) / range;
            }
            min_depth = 0.0;
            max_depth = 1.0;
        }
    }

    // Invert if configured
    if config.invert_depth {
        for val in depth_output.iter_mut() {
            *val = max_depth - *val + min_depth;
        }
        std::mem::swap(&mut min_depth, &mut max_depth);
    }

    // Resize depth map to original dimensions
    let depth_map = resize_depth_map(&depth_output, original_width as usize, original_height as usize);

    Ok(DepthMap {
        values: depth_map,
        min_depth,
        max_depth,
    })
}

/// Resizes a depth map using bilinear interpolation.
fn resize_depth_map(
    input: &ndarray::Array2<f32>,
    target_width: usize,
    target_height: usize,
) -> ndarray::Array2<f32> {
    let (src_height, src_width) = input.dim();
    let mut output = ndarray::Array2::<f32>::zeros((target_height, target_width));

    let scale_x = src_width as f32 / target_width as f32;
    let scale_y = src_height as f32 / target_height as f32;

    for y in 0..target_height {
        for x in 0..target_width {
            let src_x = x as f32 * scale_x;
            let src_y = y as f32 * scale_y;

            let x0 = (src_x.floor() as usize).min(src_width - 1);
            let x1 = (x0 + 1).min(src_width - 1);
            let y0 = (src_y.floor() as usize).min(src_height - 1);
            let y1 = (y0 + 1).min(src_height - 1);

            let dx = src_x - x0 as f32;
            let dy = src_y - y0 as f32;

            // Bilinear interpolation
            let v00 = input[[y0, x0]];
            let v01 = input[[y0, x1]];
            let v10 = input[[y1, x0]];
            let v11 = input[[y1, x1]];

            let v0 = v00 * (1.0 - dx) + v01 * dx;
            let v1 = v10 * (1.0 - dx) + v11 * dx;

            output[[y, x]] = v0 * (1.0 - dy) + v1 * dy;
        }
    }

    output
}

/// Converts image to tensor normalized to [0, 1].
fn image_to_tensor(image: &DynamicImage) -> Result<Array4<f32>, TransformError> {
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                tensor[[0, c, y, x]] = pixel[c] as f32 / 255.0;
            }
        }
    }

    Ok(tensor)
}

/// Converts depth value to RGB color (Viridis-like colormap).
fn depth_to_color(depth: f32) -> [u8; 3] {
    let d = depth.clamp(0.0, 1.0);

    // Simple Viridis approximation
    let r = (0.267 + d * (1.0 - 0.993 * d + 0.906 * d * d)) * 255.0;
    let g = (0.004 + d * (1.327 - 0.228 * d - 0.318 * d * d)) * 255.0;
    let b = (0.329 + d * (1.932 - 3.177 * d + 1.362 * d * d)) * 255.0;

    [
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth_map_grayscale() {
        let values = ndarray::Array2::from_shape_vec(
            (2, 2),
            vec![0.0, 0.5, 0.5, 1.0],
        ).unwrap();

        let depth_map = DepthMap {
            values,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let gray = depth_map.to_grayscale();
        assert_eq!(gray.width(), 2);
        assert_eq!(gray.height(), 2);
        assert_eq!(gray.get_pixel(0, 0)[0], 0);
        assert_eq!(gray.get_pixel(1, 1)[0], 255);
    }

    #[test]
    fn test_depth_at() {
        let values = ndarray::Array2::from_shape_vec(
            (3, 3),
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ).unwrap();

        let depth_map = DepthMap {
            values,
            min_depth: 0.1,
            max_depth: 0.9,
        };

        assert_eq!(depth_map.depth_at(1, 1), Some(0.5));
        assert_eq!(depth_map.depth_at(10, 10), None);
    }

    #[test]
    fn test_resize_depth_map() {
        let input = ndarray::Array2::from_shape_vec(
            (2, 2),
            vec![0.0, 1.0, 1.0, 0.0],
        ).unwrap();

        let resized = resize_depth_map(&input, 4, 4);
        assert_eq!(resized.dim(), (4, 4));
    }

    #[test]
    fn test_depth_to_color() {
        let black = depth_to_color(0.0);
        let white = depth_to_color(1.0);

        // Should produce different colors
        assert_ne!(black, white);
    }
}
