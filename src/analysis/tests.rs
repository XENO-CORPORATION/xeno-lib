use super::{histogram, image_info, read_exif_from_reader, sniff_format};
use image::{ColorType, DynamicImage, ImageBuffer, ImageEncoder, codecs::png::PngEncoder};

fn sample_rgba_image() -> DynamicImage {
    let data = vec![
        255, 0, 0, 255, //
        0, 255, 0, 128, //
        0, 0, 255, 0, //
        255, 255, 255, 255,
    ];
    let buffer = ImageBuffer::from_raw(2, 2, data).expect("buffer");
    DynamicImage::ImageRgba8(buffer)
}

#[test]
fn info_reports_dimensions_and_channels() {
    let image = sample_rgba_image();
    let info = image_info(&image);
    assert_eq!(info.width, 2);
    assert_eq!(info.height, 2);
    assert_eq!(info.channels, 4);
    assert!(info.has_alpha);
    assert_eq!(info.color, ColorType::Rgba8);
}

#[test]
fn histogram_counts_pixels_per_channel() {
    let image = sample_rgba_image();
    let hist = histogram(&image);
    assert_eq!(hist.channels.len(), 4);
    for channel in &hist.channels {
        assert_eq!(channel.total(), 4);
    }
    assert!(hist.channels[0].bins[255] > 0);
}

#[test]
fn sniff_detects_png_bytes() {
    let image = sample_rgba_image();
    let mut buf = Vec::new();
    let encoder = PngEncoder::new(&mut buf);
    encoder
        .write_image(
            image.as_rgba8().expect("rgba view").as_raw(),
            image.width(),
            image.height(),
            image::ExtendedColorType::Rgba8,
        )
        .expect("encode");
    let format = sniff_format(&buf).expect("format detected");
    assert_eq!(format, image::ImageFormat::Png);
}

#[test]
fn exif_reader_reports_error_for_invalid_data() {
    use crate::error::TransformError;
    use std::io::Cursor;

    let mut tiff = vec![
        0x49, 0x49, 0x2A, 0x00, // little endian TIFF header
        0x08, 0x00, 0x00, 0x00, // offset to first IFD
        0x01, 0x00, // one entry
        0x32, 0x01, // tag = 0x0132 (DateTime)
        0x02, 0x00, // type = ASCII
        0x14, 0x00, 0x00, 0x00, // count = 20 bytes
        0x1A, 0x00, 0x00, 0x00, // offset to string (26 bytes from TIFF start)
        0x00, 0x00, 0x00, 0x00, // next IFD offset = 0
    ];

    tiff.extend_from_slice(b"2025:01:02 03:04:05\0");
    tiff.resize(80, 0);

    let cursor = Cursor::new(tiff);
    let err = read_exif_from_reader(cursor).expect_err("should fail");
    assert!(matches!(err, TransformError::ExifRead { .. }));
}
