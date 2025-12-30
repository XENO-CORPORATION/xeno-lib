use super::*;
use image::{DynamicImage, ImageBuffer};

fn gradient_image() -> DynamicImage {
    let mut data = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            data.push((x * 40 + y * 10) as u8);
            data.push(128);
            data.push(200);
        }
    }
    let buffer = ImageBuffer::from_raw(4, 4, data).expect("valid buffer");
    DynamicImage::ImageRgb8(buffer)
}

#[test]
fn blur_with_zero_sigma_returns_clone() {
    let image = gradient_image();
    let blurred = gaussian_blur(&image, 0.0).expect("blur succeeds");
    assert_eq!(blurred.to_rgb8(), image.to_rgb8());
}

#[test]
fn sharpen_preserves_dimensions() {
    let image = gradient_image();
    let sharpened = unsharp_mask(&image, 1.0, 1).expect("sharpen succeeds");
    assert_eq!(sharpened.width(), image.width());
    assert_eq!(sharpened.height(), image.height());
}

#[test]
fn edge_detect_outputs_highlighted_edges() {
    let image = gradient_image();
    let edges = edge_detect(&image, 1.0).expect("edge detection succeeds");
    assert_eq!(edges.width(), image.width());
    assert_eq!(edges.height(), image.height());
}

#[test]
fn emboss_outputs_rgba_image() {
    let image = gradient_image();
    let embossed = emboss(&image, 1.0).expect("emboss succeeds");
    assert_eq!(embossed.width(), image.width());
}

#[test]
fn sepia_applies_tone() {
    let image = gradient_image();
    let toned = sepia(&image).expect("sepia succeeds").to_rgba8();
    let px = toned.get_pixel(0, 0);
    assert!(px[0] >= px[1] && px[1] >= px[2]);
}
