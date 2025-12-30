use std::error::Error;
use std::fs;
use std::path::Path;

use image::{DynamicImage, ImageBuffer, Rgba};
use xeno_lib::{
    Interpolation, crop, flip_horizontal, flip_vertical, rotate, rotate_90, rotate_180, rotate_270,
};

fn base_image() -> DynamicImage {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 48;

    let buffer = ImageBuffer::from_fn(WIDTH, HEIGHT, |x, y| {
        let r = ((x * 5 + y * 3) % 256) as u8;
        let g = ((x * 7) % 256) as u8;
        let b = ((y * 11) % 256) as u8;
        let a = if (x + y) % 13 == 0 { 128 } else { 255 };
        Rgba([r, g, b, a])
    });

    DynamicImage::ImageRgba8(buffer)
}

fn save_image(path: &Path, image: &DynamicImage) -> Result<(), Box<dyn Error>> {
    image.save(path)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = Path::new("tests/goldens");
    fs::create_dir_all(out_dir)?;

    let base = base_image();
    save_image(&out_dir.join("base.png"), &base)?;

    let hflip = flip_horizontal(&base)?;
    save_image(&out_dir.join("flip_horizontal.png"), &hflip)?;

    let vflip = flip_vertical(&base)?;
    save_image(&out_dir.join("flip_vertical.png"), &vflip)?;

    let rot90 = rotate_90(&base)?;
    save_image(&out_dir.join("rotate_90.png"), &rot90)?;

    let rot180 = rotate_180(&base)?;
    save_image(&out_dir.join("rotate_180.png"), &rot180)?;

    let rot270 = rotate_270(&base)?;
    save_image(&out_dir.join("rotate_270.png"), &rot270)?;

    let rot33_bilinear = rotate(&base, 33.0, Interpolation::Bilinear)?;
    save_image(&out_dir.join("rotate_33_bilinear.png"), &rot33_bilinear)?;

    let rot120_nearest = rotate(&base, 120.0, Interpolation::Nearest)?;
    save_image(&out_dir.join("rotate_120_nearest.png"), &rot120_nearest)?;

    let cropped = crop(&base, 8, 6, 24, 20)?;
    save_image(&out_dir.join("crop_24x20_at_8x6.png"), &cropped)?;

    println!("Golden images written to {}", out_dir.display());
    Ok(())
}
