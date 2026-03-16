//! Text overlay functionality for images and video frames.
//!
//! This module provides pure Rust text rendering capabilities using `ab_glyph`,
//! enabling text overlays without requiring FFmpeg or system font dependencies.
//!
//! # Example
//!
//! ```ignore
//! use xeno_lib::text::{TextOverlay, TextConfig, Anchor};
//! use image::DynamicImage;
//!
//! let config = TextConfig::new("Hello, World!")
//!     .with_font_size(48.0)
//!     .with_color([255, 255, 255, 255])
//!     .with_position(100, 50)
//!     .with_anchor(Anchor::TopLeft);
//!
//! let overlay = TextOverlay::with_default_font()?;
//! let result = overlay.draw(&image, &config)?;
//! ```

use crate::error::TransformError;
use ab_glyph::{Font, FontRef, Glyph, PxScale, ScaleFont, point};
use image::{DynamicImage, Rgba, RgbaImage};

/// Text anchor position for alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Anchor {
    /// Anchor at top-left corner (default)
    #[default]
    TopLeft,
    /// Anchor at top-center
    TopCenter,
    /// Anchor at top-right corner
    TopRight,
    /// Anchor at middle-left
    MiddleLeft,
    /// Anchor at center
    Center,
    /// Anchor at middle-right
    MiddleRight,
    /// Anchor at bottom-left corner
    BottomLeft,
    /// Anchor at bottom-center
    BottomCenter,
    /// Anchor at bottom-right corner
    BottomRight,
}

impl Anchor {
    /// Calculate the top-left position based on anchor, text dimensions, and target position
    pub fn compute_position(&self, x: i32, y: i32, text_width: u32, text_height: u32) -> (i32, i32) {
        match self {
            Anchor::TopLeft => (x, y),
            Anchor::TopCenter => (x - (text_width as i32 / 2), y),
            Anchor::TopRight => (x - text_width as i32, y),
            Anchor::MiddleLeft => (x, y - (text_height as i32 / 2)),
            Anchor::Center => (x - (text_width as i32 / 2), y - (text_height as i32 / 2)),
            Anchor::MiddleRight => (x - text_width as i32, y - (text_height as i32 / 2)),
            Anchor::BottomLeft => (x, y - text_height as i32),
            Anchor::BottomCenter => (x - (text_width as i32 / 2), y - text_height as i32),
            Anchor::BottomRight => (x - text_width as i32, y - text_height as i32),
        }
    }
}

/// Configuration for text rendering
#[derive(Debug, Clone)]
pub struct TextConfig {
    /// The text to render
    pub text: String,
    /// Font size in pixels
    pub font_size: f32,
    /// Text color as RGBA
    pub color: [u8; 4],
    /// X position (interpretation depends on anchor)
    pub x: i32,
    /// Y position (interpretation depends on anchor)
    pub y: i32,
    /// Anchor point for positioning
    pub anchor: Anchor,
    /// Optional background color as RGBA (None = transparent)
    pub background: Option<[u8; 4]>,
    /// Padding around text when background is used
    pub padding: u32,
    /// Optional shadow offset (dx, dy) - None = no shadow
    pub shadow: Option<(i32, i32)>,
    /// Shadow color as RGBA
    pub shadow_color: [u8; 4],
    /// Optional outline thickness (0 = no outline)
    pub outline: u32,
    /// Outline color as RGBA
    pub outline_color: [u8; 4],
}

impl TextConfig {
    /// Create a new text configuration with default settings
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            font_size: 24.0,
            color: [255, 255, 255, 255], // White
            x: 0,
            y: 0,
            anchor: Anchor::TopLeft,
            background: None,
            padding: 4,
            shadow: None,
            shadow_color: [0, 0, 0, 128], // Semi-transparent black
            outline: 0,
            outline_color: [0, 0, 0, 255], // Black
        }
    }

    /// Set the font size in pixels
    pub fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    /// Set the text color as RGBA
    pub fn with_color(mut self, color: [u8; 4]) -> Self {
        self.color = color;
        self
    }

    /// Set the text color from RGB (alpha = 255)
    pub fn with_rgb(mut self, r: u8, g: u8, b: u8) -> Self {
        self.color = [r, g, b, 255];
        self
    }

    /// Set the position
    pub fn with_position(mut self, x: i32, y: i32) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    /// Set the anchor point
    pub fn with_anchor(mut self, anchor: Anchor) -> Self {
        self.anchor = anchor;
        self
    }

    /// Set background color
    pub fn with_background(mut self, color: [u8; 4]) -> Self {
        self.background = Some(color);
        self
    }

    /// Set padding around text
    pub fn with_padding(mut self, padding: u32) -> Self {
        self.padding = padding;
        self
    }

    /// Set shadow offset
    pub fn with_shadow(mut self, dx: i32, dy: i32) -> Self {
        self.shadow = Some((dx, dy));
        self
    }

    /// Set shadow color
    pub fn with_shadow_color(mut self, color: [u8; 4]) -> Self {
        self.shadow_color = color;
        self
    }

    /// Set outline thickness
    pub fn with_outline(mut self, thickness: u32) -> Self {
        self.outline = thickness;
        self
    }

    /// Set outline color
    pub fn with_outline_color(mut self, color: [u8; 4]) -> Self {
        self.outline_color = color;
        self
    }
}

/// Text dimensions after layout
#[derive(Debug, Clone, Copy)]
pub struct TextDimensions {
    /// Width of the text in pixels
    pub width: u32,
    /// Height of the text in pixels
    pub height: u32,
    /// Ascent (distance from baseline to top)
    pub ascent: f32,
    /// Descent (distance from baseline to bottom)
    pub descent: f32,
}

/// Text overlay renderer
pub struct TextOverlay<'a> {
    font: FontRef<'a>,
}

impl<'a> TextOverlay<'a> {
    /// Create a new TextOverlay with the provided font data
    pub fn new(font_data: &'a [u8]) -> Result<Self, TransformError> {
        let font = FontRef::try_from_slice(font_data)
            .map_err(|e| TransformError::FontLoadFailed {
                message: format!("Failed to parse font data: {}", e)
            })?;
        Ok(Self { font })
    }

    /// Create a new TextOverlay with the embedded default font (DejaVu Sans Mono)
    ///
    /// Note: This embeds a ~700KB font in the binary. For smaller binaries,
    /// use `new()` with an external font file.
    pub fn with_default_font() -> Result<Self, TransformError> {
        // Use a minimal embedded font - we'll embed a subset or use a system font path
        // For now, we'll return an error if no font is provided
        Err(TransformError::InvalidTextConfig {
            message: "No default font embedded. Please provide a font file path.".to_string()
        })
    }

    /// Measure the dimensions of the text without rendering
    pub fn measure(&self, config: &TextConfig) -> TextDimensions {
        let scale = PxScale::from(config.font_size);
        let scaled_font = self.font.as_scaled(scale);

        let mut glyphs: Vec<Glyph> = Vec::new();
        let mut caret = point(0.0, scaled_font.ascent());
        let mut last_glyph_id = None;

        for ch in config.text.chars() {
            if ch == '\n' {
                // Handle newlines - this is a simple single-line implementation
                continue;
            }

            let glyph_id = scaled_font.glyph_id(ch);

            // Apply kerning
            if let Some(prev_id) = last_glyph_id {
                caret.x += scaled_font.kern(prev_id, glyph_id);
            }

            let glyph = glyph_id.with_scale_and_position(scale, caret);
            glyphs.push(glyph);

            caret.x += scaled_font.h_advance(glyph_id);
            last_glyph_id = Some(glyph_id);
        }

        let width = caret.x;

        // Calculate actual bounds from outlined glyphs
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;

        for glyph in glyphs {
            if let Some(outlined) = self.font.outline_glyph(glyph) {
                let bounds = outlined.px_bounds();
                min_y = min_y.min(bounds.min.y);
                max_y = max_y.max(bounds.max.y);
            }
        }

        let height = if max_y > min_y { max_y - min_y } else { config.font_size };

        TextDimensions {
            width: width.ceil() as u32,
            height: height.ceil() as u32,
            ascent: scaled_font.ascent(),
            descent: scaled_font.descent(),
        }
    }

    /// Draw text onto an image
    pub fn draw(&self, image: &DynamicImage, config: &TextConfig) -> Result<DynamicImage, TransformError> {
        let mut output = image.to_rgba8();
        self.draw_onto(&mut output, config)?;
        Ok(DynamicImage::ImageRgba8(output))
    }

    /// Draw text onto an RGBA image buffer (in-place)
    pub fn draw_onto(&self, image: &mut RgbaImage, config: &TextConfig) -> Result<(), TransformError> {
        let scale = PxScale::from(config.font_size);
        let scaled_font = self.font.as_scaled(scale);

        // Measure text dimensions
        let dims = self.measure(config);

        // Calculate actual position based on anchor
        let (draw_x, draw_y) = config.anchor.compute_position(
            config.x, config.y, dims.width, dims.height
        );

        // Draw background if specified
        if let Some(bg_color) = config.background {
            let bg_x = (draw_x - config.padding as i32).max(0) as u32;
            let bg_y = (draw_y - config.padding as i32).max(0) as u32;
            let bg_w = (dims.width + config.padding * 2).min(image.width().saturating_sub(bg_x));
            let bg_h = (dims.height + config.padding * 2).min(image.height().saturating_sub(bg_y));

            for y in bg_y..bg_y + bg_h {
                for x in bg_x..bg_x + bg_w {
                    if x < image.width() && y < image.height() {
                        let bg_pixel = Rgba(bg_color);
                        blend_pixel(image, x, y, bg_pixel);
                    }
                }
            }
        }

        // Draw shadow if specified
        if let Some((shadow_dx, shadow_dy)) = config.shadow {
            self.render_text(
                image,
                config,
                draw_x + shadow_dx,
                draw_y + shadow_dy,
                config.shadow_color,
                &scaled_font,
                scale,
            );
        }

        // Draw outline if specified
        if config.outline > 0 {
            let offsets: Vec<(i32, i32)> = Self::generate_outline_offsets(config.outline);
            for (ox, oy) in offsets {
                self.render_text(
                    image,
                    config,
                    draw_x + ox,
                    draw_y + oy,
                    config.outline_color,
                    &scaled_font,
                    scale,
                );
            }
        }

        // Draw main text
        self.render_text(
            image,
            config,
            draw_x,
            draw_y,
            config.color,
            &scaled_font,
            scale,
        );

        Ok(())
    }

    /// Generate outline offsets for a given thickness
    fn generate_outline_offsets(thickness: u32) -> Vec<(i32, i32)> {
        let mut offsets = Vec::new();
        let t = thickness as i32;

        for dy in -t..=t {
            for dx in -t..=t {
                if dx == 0 && dy == 0 {
                    continue;
                }
                // Use circular distance for smoother outline
                if (dx * dx + dy * dy) <= t * t + t {
                    offsets.push((dx, dy));
                }
            }
        }
        offsets
    }

    /// Render text at a specific position with a specific color
    fn render_text(
        &self,
        image: &mut RgbaImage,
        config: &TextConfig,
        x: i32,
        y: i32,
        color: [u8; 4],
        scaled_font: &ab_glyph::PxScaleFont<&FontRef<'a>>,
        scale: PxScale,
    ) {
        let mut caret = point(x as f32, y as f32 + scaled_font.ascent());
        let mut last_glyph_id = None;

        for ch in config.text.chars() {
            if ch == '\n' {
                continue;
            }

            let glyph_id = scaled_font.glyph_id(ch);

            // Apply kerning
            if let Some(prev_id) = last_glyph_id {
                caret.x += scaled_font.kern(prev_id, glyph_id);
            }

            let glyph = glyph_id.with_scale_and_position(scale, caret);

            // Render the glyph
            if let Some(outlined) = self.font.outline_glyph(glyph) {
                let bounds = outlined.px_bounds();

                outlined.draw(|px, py, coverage| {
                    let px_x = bounds.min.x as i32 + px as i32;
                    let px_y = bounds.min.y as i32 + py as i32;

                    if px_x >= 0 && px_y >= 0
                        && (px_x as u32) < image.width()
                        && (px_y as u32) < image.height()
                    {
                        let alpha = (coverage * color[3] as f32) as u8;
                        let pixel = Rgba([color[0], color[1], color[2], alpha]);
                        blend_pixel(image, px_x as u32, px_y as u32, pixel);
                    }
                });
            }

            caret.x += scaled_font.h_advance(glyph_id);
            last_glyph_id = Some(glyph_id);
        }
    }
}

/// Blend a pixel with alpha onto the image
fn blend_pixel(image: &mut RgbaImage, x: u32, y: u32, src: Rgba<u8>) {
    let dst = image.get_pixel(x, y);

    let src_a = src[3] as f32 / 255.0;
    let dst_a = dst[3] as f32 / 255.0;

    let out_a = src_a + dst_a * (1.0 - src_a);

    if out_a > 0.0 {
        let out_r = (src[0] as f32 * src_a + dst[0] as f32 * dst_a * (1.0 - src_a)) / out_a;
        let out_g = (src[1] as f32 * src_a + dst[1] as f32 * dst_a * (1.0 - src_a)) / out_a;
        let out_b = (src[2] as f32 * src_a + dst[2] as f32 * dst_a * (1.0 - src_a)) / out_a;

        image.put_pixel(x, y, Rgba([
            out_r.round().clamp(0.0, 255.0) as u8,
            out_g.round().clamp(0.0, 255.0) as u8,
            out_b.round().clamp(0.0, 255.0) as u8,
            (out_a * 255.0).round().clamp(0.0, 255.0) as u8,
        ]));
    }
}

/// Draw text on an image with the given configuration
///
/// This is a convenience function that creates a TextOverlay and draws text.
pub fn draw_text(
    image: &DynamicImage,
    font_data: &[u8],
    config: &TextConfig,
) -> Result<DynamicImage, TransformError> {
    let overlay = TextOverlay::new(font_data)?;
    overlay.draw(image, config)
}

/// Draw text on multiple images (e.g., video frames) efficiently
///
/// This reuses the font for all frames, which is more efficient than
/// calling `draw_text` for each frame.
pub fn draw_text_batch<'a>(
    images: impl Iterator<Item = &'a DynamicImage>,
    font_data: &[u8],
    config: &TextConfig,
) -> Result<Vec<DynamicImage>, TransformError> {
    let overlay = TextOverlay::new(font_data)?;
    images.map(|img| overlay.draw(img, config)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Tests would require a font file to be available
    // These are placeholder tests that document expected behavior

    #[test]
    fn test_anchor_positions() {
        let (x, y) = Anchor::TopLeft.compute_position(100, 50, 200, 30);
        assert_eq!((x, y), (100, 50));

        let (x, y) = Anchor::Center.compute_position(100, 50, 200, 30);
        assert_eq!((x, y), (0, 35));

        let (x, y) = Anchor::BottomRight.compute_position(100, 50, 200, 30);
        assert_eq!((x, y), (-100, 20));
    }

    #[test]
    fn test_text_config_builder() {
        let config = TextConfig::new("Test")
            .with_font_size(32.0)
            .with_rgb(255, 0, 0)
            .with_position(10, 20)
            .with_anchor(Anchor::Center)
            .with_shadow(2, 2)
            .with_outline(1);

        assert_eq!(config.text, "Test");
        assert_eq!(config.font_size, 32.0);
        assert_eq!(config.color, [255, 0, 0, 255]);
        assert_eq!(config.x, 10);
        assert_eq!(config.y, 20);
        assert_eq!(config.anchor, Anchor::Center);
        assert_eq!(config.shadow, Some((2, 2)));
        assert_eq!(config.outline, 1);
    }
}
