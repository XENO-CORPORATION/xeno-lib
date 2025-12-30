//! Image compositing helpers: overlay, watermark, borders, and frames.

use crate::error::TransformError;
use image::{DynamicImage, Rgba, RgbaImage};

/// Overlay `layer` onto `base` at the provided top-left offset.
pub fn overlay(
    base: &DynamicImage,
    layer: &DynamicImage,
    offset_x: u32,
    offset_y: u32,
) -> Result<DynamicImage, TransformError> {
    composite(base, layer, offset_x, offset_y, 1.0)
}

/// Apply `layer` as a watermark with configurable opacity (0.0 - 1.0).
pub fn watermark(
    base: &DynamicImage,
    layer: &DynamicImage,
    offset_x: u32,
    offset_y: u32,
    opacity: f32,
) -> Result<DynamicImage, TransformError> {
    let opacity = opacity.clamp(0.0, 1.0);
    composite(base, layer, offset_x, offset_y, opacity)
}

/// Add a solid border around the image.
pub fn border(
    base: &DynamicImage,
    thickness: u32,
    color: Rgba<u8>,
) -> Result<DynamicImage, TransformError> {
    if thickness == 0 {
        return Ok(base.clone());
    }

    let mut extended = RgbaImage::from_pixel(
        base.width() + thickness * 2,
        base.height() + thickness * 2,
        color,
    );

    let base_rgba = base.to_rgba8();
    image::imageops::overlay(
        &mut extended,
        &base_rgba,
        thickness as i64,
        thickness as i64,
    );
    Ok(DynamicImage::ImageRgba8(extended))
}

/// Create a two-tone frame around the image (mat + outer border).
pub fn frame(
    base: &DynamicImage,
    mat_thickness: u32,
    mat_color: Rgba<u8>,
    outer_thickness: u32,
    outer_color: Rgba<u8>,
) -> Result<DynamicImage, TransformError> {
    let with_mat = border(base, mat_thickness, mat_color)?;
    border(&with_mat, outer_thickness, outer_color)
}

fn composite(
    base: &DynamicImage,
    layer: &DynamicImage,
    offset_x: u32,
    offset_y: u32,
    opacity: f32,
) -> Result<DynamicImage, TransformError> {
    if layer.width() == 0 || layer.height() == 0 {
        return Ok(base.clone());
    }

    if offset_x
        .checked_add(layer.width())
        .map(|w| w > base.width())
        .unwrap_or(true)
        || offset_y
            .checked_add(layer.height())
            .map(|h| h > base.height())
            .unwrap_or(true)
    {
        return Err(TransformError::OverlayOutOfBounds {
            x: offset_x,
            y: offset_y,
            width: layer.width(),
            height: layer.height(),
            image_width: base.width(),
            image_height: base.height(),
        });
    }

    let mut base_rgba = base.to_rgba8();
    let layer_rgba = layer.to_rgba8();

    for y in 0..layer.height() {
        for x in 0..layer.width() {
            let src = layer_rgba.get_pixel(x, y).0;
            if src[3] == 0 {
                continue;
            }

            let dst = base_rgba.get_pixel_mut(offset_x + x, offset_y + y);
            let src_alpha = (src[3] as f32 / 255.0) * opacity;
            if src_alpha <= 0.0 {
                continue;
            }
            let dst_alpha = dst[3] as f32 / 255.0;
            let out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha);
            let out_alpha_u8 = (out_alpha * 255.0).round().clamp(0.0, 255.0) as u8;

            for channel in 0..3 {
                let src_c = src[channel] as f32 / 255.0;
                let dst_c = dst[channel] as f32 / 255.0;
                let out_c = (src_c * src_alpha + dst_c * dst_alpha * (1.0 - src_alpha))
                    / out_alpha.max(1e-6);
                dst[channel] = (out_c * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            dst[3] = out_alpha_u8;
        }
    }

    Ok(DynamicImage::ImageRgba8(base_rgba))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, color: Rgba<u8>) -> DynamicImage {
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(width, height, color))
    }

    #[test]
    fn overlay_in_bounds() {
        let base = solid(4, 4, Rgba([0, 0, 0, 255]));
        let layer = solid(2, 2, Rgba([255, 0, 0, 255]));
        let composed = overlay(&base, &layer, 1, 1).expect("overlay");
        assert_eq!(composed.width(), 4);
        assert_eq!(composed.height(), 4);
        assert_eq!(composed.to_rgba8().get_pixel(1, 1), &Rgba([255, 0, 0, 255]));
    }

    #[test]
    fn watermark_applies_opacity() {
        let base = solid(2, 2, Rgba([0, 0, 0, 255]));
        let layer = solid(2, 2, Rgba([255, 255, 255, 255]));
        let composed = watermark(&base, &layer, 0, 0, 0.5).expect("watermark");
        let composed_rgba = composed.to_rgba8();
        let px = composed_rgba.get_pixel(0, 0);
        assert!(px[0] > 120 && px[0] < 200);
        assert_eq!(px[3], 255);
    }

    #[test]
    fn border_expands_dimensions() {
        let base = solid(2, 3, Rgba([10, 10, 10, 255]));
        let bordered = border(&base, 2, Rgba([255, 0, 0, 255])).expect("border");
        assert_eq!(bordered.width(), 6);
        assert_eq!(bordered.height(), 7);
    }

    #[test]
    fn frame_applies_nested_borders() {
        let base = solid(2, 2, Rgba([0, 0, 0, 255]));
        let framed = frame(
            &base,
            1,
            Rgba([10, 10, 10, 255]),
            2,
            Rgba([200, 200, 200, 255]),
        )
        .expect("frame");
        assert_eq!(framed.width(), 8);
        assert_eq!(framed.height(), 8);
    }

    #[test]
    fn overlay_oob_errors() {
        let base = solid(2, 2, Rgba([0, 0, 0, 255]));
        let layer = solid(2, 2, Rgba([255, 0, 0, 255]));
        assert!(matches!(
            overlay(&base, &layer, 2, 0),
            Err(TransformError::OverlayOutOfBounds { .. })
        ));
    }
}
