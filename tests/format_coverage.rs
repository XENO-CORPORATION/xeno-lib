use image::{DynamicImage, ImageBuffer, ImageFormat, Rgba};
use tempfile::tempdir;
use xeno_lib::{
    Interpolation, crop, flip_horizontal, flip_vertical, rotate, rotate_90, rotate_180, rotate_270,
};

fn sample_image() -> DynamicImage {
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 24;

    let buffer = ImageBuffer::from_fn(WIDTH, HEIGHT, |x, y| {
        let r = (x * 3 % 256) as u8;
        let g = (y * 5 % 256) as u8;
        let b = ((x + y) * 11 % 256) as u8;
        Rgba([r, g, b, 255])
    });

    DynamicImage::ImageRgba8(buffer)
}

#[test]
fn transforms_operate_on_common_file_formats() {
    let image = sample_image();
    let dir = tempdir().expect("tempdir");

    let formats = [
        ("png", ImageFormat::Png),
        ("jpg", ImageFormat::Jpeg),
        ("webp", ImageFormat::WebP),
    ];

    for (extension, format) in formats {
        let path = dir.path().join(format!("sample.{extension}"));
        image
            .save_with_format(&path, format)
            .unwrap_or_else(|e| panic!("failed to save {extension}: {e}"));

        let loaded =
            image::open(&path).unwrap_or_else(|e| panic!("failed to reopen {extension}: {e}"));

        let flipped_h = flip_horizontal(&loaded).expect("flip horizontal");
        assert_eq!(flipped_h.width(), loaded.width());
        assert_eq!(flipped_h.height(), loaded.height());

        let flipped_v = flip_vertical(&loaded).expect("flip vertical");
        assert_eq!(flipped_v.width(), loaded.width());
        assert_eq!(flipped_v.height(), loaded.height());

        let rot90 = rotate_90(&loaded).expect("rotate 90");
        assert_eq!(rot90.width(), loaded.height());
        assert_eq!(rot90.height(), loaded.width());

        let rot180 = rotate_180(&loaded).expect("rotate 180");
        assert_eq!(rot180.width(), loaded.width());
        assert_eq!(rot180.height(), loaded.height());

        let rot270 = rotate_270(&loaded).expect("rotate 270");
        assert_eq!(rot270.width(), loaded.height());
        assert_eq!(rot270.height(), loaded.width());

        let arbitrary =
            rotate(&loaded, 37.0, Interpolation::Bilinear).expect("rotate arbitrary bilinear");
        assert!(arbitrary.width() >= loaded.width());
        assert!(arbitrary.height() >= loaded.height());

        let cropped = crop(&loaded, 4, 3, 8, 6).expect("crop");
        assert_eq!(cropped.width(), 8);
        assert_eq!(cropped.height(), 6);
    }
}
