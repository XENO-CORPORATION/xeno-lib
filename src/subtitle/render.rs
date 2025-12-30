//! Subtitle rendering for burning subtitles into video frames.

use image::{DynamicImage, Rgba, RgbaImage};

use super::{SubtitleAlign, SubtitleCue};

/// Subtitle rendering style.
#[derive(Debug, Clone)]
pub struct SubtitleStyle {
    /// Font size as percentage of frame height.
    pub font_size_percent: f32,
    /// Font color.
    pub color: Rgba<u8>,
    /// Outline/stroke color.
    pub outline_color: Rgba<u8>,
    /// Outline width in pixels.
    pub outline_width: u32,
    /// Background box color (if enabled).
    pub background_color: Option<Rgba<u8>>,
    /// Vertical position (0.0 = top, 1.0 = bottom).
    pub vertical_position: f32,
    /// Text alignment.
    pub align: SubtitleAlign,
    /// Margin from edges as percentage.
    pub margin_percent: f32,
    /// Bold text.
    pub bold: bool,
    /// Italic text.
    pub italic: bool,
    /// Shadow offset (pixels).
    pub shadow_offset: Option<(i32, i32)>,
    /// Shadow color.
    pub shadow_color: Rgba<u8>,
}

impl Default for SubtitleStyle {
    fn default() -> Self {
        Self {
            font_size_percent: 5.0, // 5% of frame height
            color: Rgba([255, 255, 255, 255]), // White
            outline_color: Rgba([0, 0, 0, 255]), // Black
            outline_width: 2,
            background_color: None,
            vertical_position: 0.9, // Near bottom
            align: SubtitleAlign::Center,
            margin_percent: 5.0,
            bold: true,
            italic: false,
            shadow_offset: Some((2, 2)),
            shadow_color: Rgba([0, 0, 0, 180]),
        }
    }
}

impl SubtitleStyle {
    /// Create a Netflix-style subtitle appearance.
    pub fn netflix() -> Self {
        Self {
            font_size_percent: 4.5,
            color: Rgba([255, 255, 255, 255]),
            outline_color: Rgba([0, 0, 0, 255]),
            outline_width: 2,
            background_color: None,
            vertical_position: 0.92,
            align: SubtitleAlign::Center,
            margin_percent: 10.0,
            bold: false,
            italic: false,
            shadow_offset: Some((1, 1)),
            shadow_color: Rgba([0, 0, 0, 200]),
        }
    }

    /// Create a YouTube-style subtitle appearance.
    pub fn youtube() -> Self {
        Self {
            font_size_percent: 4.0,
            color: Rgba([255, 255, 255, 255]),
            outline_color: Rgba([0, 0, 0, 0]),
            outline_width: 0,
            background_color: Some(Rgba([0, 0, 0, 180])),
            vertical_position: 0.9,
            align: SubtitleAlign::Center,
            margin_percent: 5.0,
            bold: false,
            italic: false,
            shadow_offset: None,
            shadow_color: Rgba([0, 0, 0, 0]),
        }
    }

    /// Create a karaoke-style appearance.
    pub fn karaoke() -> Self {
        Self {
            font_size_percent: 6.0,
            color: Rgba([255, 255, 0, 255]), // Yellow
            outline_color: Rgba([0, 0, 0, 255]),
            outline_width: 3,
            background_color: None,
            vertical_position: 0.85,
            align: SubtitleAlign::Center,
            margin_percent: 5.0,
            bold: true,
            italic: false,
            shadow_offset: Some((3, 3)),
            shadow_color: Rgba([0, 0, 128, 200]),
        }
    }

    /// Create subtitles at the top of the frame.
    pub fn top() -> Self {
        Self {
            vertical_position: 0.1,
            ..Self::default()
        }
    }
}

/// Render subtitle text onto a frame.
///
/// Note: This is a simplified renderer. For production use with proper
/// font rendering, enable the `text-overlay` feature and use ab_glyph.
pub fn render_subtitle(
    frame: &DynamicImage,
    text: &str,
    style: &SubtitleStyle,
) -> DynamicImage {
    let mut rgba = frame.to_rgba8();
    let (width, height) = (rgba.width(), rgba.height());

    if text.is_empty() {
        return DynamicImage::ImageRgba8(rgba);
    }

    // Calculate font size in pixels
    let font_size = (height as f32 * style.font_size_percent / 100.0) as u32;

    // Split text into lines
    let lines: Vec<&str> = text.lines().collect();
    let line_height = font_size + font_size / 4;
    let total_text_height = lines.len() as u32 * line_height;

    // Calculate vertical position
    let y_base = (height as f32 * style.vertical_position) as u32;
    let y_start = y_base.saturating_sub(total_text_height / 2);

    // Calculate margins
    let margin = (width as f32 * style.margin_percent / 100.0) as u32;
    let available_width = width.saturating_sub(2 * margin);

    // Render each line
    for (line_idx, line) in lines.iter().enumerate() {
        let y_pos = y_start + line_idx as u32 * line_height;

        // Estimate text width (simplified - assumes ~0.6 char width ratio)
        let char_width = (font_size as f32 * 0.55) as u32;
        let text_width = line.chars().count() as u32 * char_width;

        // Calculate x position based on alignment
        let x_pos = match style.align {
            SubtitleAlign::Left => margin,
            SubtitleAlign::Center => {
                margin + (available_width.saturating_sub(text_width)) / 2
            }
            SubtitleAlign::Right => {
                width.saturating_sub(margin).saturating_sub(text_width)
            }
        };

        // Draw background box if enabled
        if let Some(bg_color) = style.background_color {
            let padding = font_size / 4;
            let box_x = x_pos.saturating_sub(padding);
            let box_y = y_pos.saturating_sub(padding);
            let box_w = text_width + 2 * padding;
            let box_h = line_height;

            draw_rect(&mut rgba, box_x, box_y, box_w, box_h, bg_color);
        }

        // Draw shadow if enabled
        if let Some((sx, sy)) = style.shadow_offset {
            draw_text_simple(
                &mut rgba,
                line,
                (x_pos as i32 + sx) as u32,
                (y_pos as i32 + sy) as u32,
                font_size,
                style.shadow_color,
            );
        }

        // Draw outline
        if style.outline_width > 0 {
            let ow = style.outline_width as i32;
            for dy in -ow..=ow {
                for dx in -ow..=ow {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    draw_text_simple(
                        &mut rgba,
                        line,
                        (x_pos as i32 + dx) as u32,
                        (y_pos as i32 + dy) as u32,
                        font_size,
                        style.outline_color,
                    );
                }
            }
        }

        // Draw main text
        draw_text_simple(&mut rgba, line, x_pos, y_pos, font_size, style.color);
    }

    DynamicImage::ImageRgba8(rgba)
}

/// Render subtitle cue onto a frame.
pub fn render_cue(
    frame: &DynamicImage,
    cue: &SubtitleCue,
    style: &SubtitleStyle,
) -> DynamicImage {
    render_subtitle(frame, &cue.text, style)
}

/// Draw a simple filled rectangle.
fn draw_rect(img: &mut RgbaImage, x: u32, y: u32, w: u32, h: u32, color: Rgba<u8>) {
    let (img_w, img_h) = (img.width(), img.height());

    for dy in 0..h {
        for dx in 0..w {
            let px = x + dx;
            let py = y + dy;
            if px < img_w && py < img_h {
                blend_pixel(img, px, py, color);
            }
        }
    }
}

/// Simple text drawing using basic bitmap characters.
/// This is a placeholder - for real font rendering, use ab_glyph.
fn draw_text_simple(
    img: &mut RgbaImage,
    text: &str,
    x: u32,
    y: u32,
    font_size: u32,
    color: Rgba<u8>,
) {
    let char_width = (font_size as f32 * 0.55) as u32;
    let char_height = font_size;

    for (i, _ch) in text.chars().enumerate() {
        let cx = x + i as u32 * char_width;

        // Draw a simple rectangle for each character (placeholder)
        // Real implementation would render actual glyphs
        for dy in 0..char_height {
            for dx in 0..char_width.saturating_sub(2) {
                let px = cx + dx;
                let py = y + dy;

                // Simple pattern to make text somewhat readable
                // This creates a basic block character effect
                if px < img.width() && py < img.height() {
                    // Only draw interior pixels to create character-like shapes
                    let in_bounds = dx > 0 && dx < char_width - 2 && dy > font_size / 6 && dy < char_height - font_size / 6;
                    if in_bounds {
                        blend_pixel(img, px, py, color);
                    }
                }
            }
        }
    }
}

/// Blend a pixel with alpha compositing.
fn blend_pixel(img: &mut RgbaImage, x: u32, y: u32, color: Rgba<u8>) {
    if x >= img.width() || y >= img.height() {
        return;
    }

    let dst = img.get_pixel(x, y);
    let src_a = color[3] as f32 / 255.0;
    let dst_a = dst[3] as f32 / 255.0;

    let out_a = src_a + dst_a * (1.0 - src_a);

    if out_a > 0.0 {
        let blend = |s: u8, d: u8| -> u8 {
            let s = s as f32 / 255.0;
            let d = d as f32 / 255.0;
            let out = (s * src_a + d * dst_a * (1.0 - src_a)) / out_a;
            (out * 255.0) as u8
        };

        img.put_pixel(
            x,
            y,
            Rgba([
                blend(color[0], dst[0]),
                blend(color[1], dst[1]),
                blend(color[2], dst[2]),
                (out_a * 255.0) as u8,
            ]),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subtitle_style_presets() {
        let default = SubtitleStyle::default();
        assert!(default.font_size_percent > 0.0);

        let netflix = SubtitleStyle::netflix();
        assert!(netflix.vertical_position > 0.5);

        let youtube = SubtitleStyle::youtube();
        assert!(youtube.background_color.is_some());

        let top = SubtitleStyle::top();
        assert!(top.vertical_position < 0.5);
    }

    #[test]
    fn test_render_subtitle() {
        let frame = DynamicImage::new_rgba8(640, 480);
        let result = render_subtitle(&frame, "Hello, World!", &SubtitleStyle::default());
        assert_eq!(result.width(), 640);
        assert_eq!(result.height(), 480);
    }

    #[test]
    fn test_render_empty_text() {
        let frame = DynamicImage::new_rgba8(640, 480);
        let result = render_subtitle(&frame, "", &SubtitleStyle::default());
        assert_eq!(result.width(), 640);
    }

    #[test]
    fn test_render_multiline() {
        let frame = DynamicImage::new_rgba8(640, 480);
        let result = render_subtitle(&frame, "Line 1\nLine 2", &SubtitleStyle::default());
        assert_eq!(result.height(), 480);
    }
}
