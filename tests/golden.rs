use std::path::Path;

use image::DynamicImage;
use xeno_lib::{
    Interpolation, crop, flip_horizontal, flip_vertical, rotate, rotate_90, rotate_180, rotate_270,
};

fn load_image(name: &str) -> DynamicImage {
    let path = Path::new("tests").join("goldens").join(name);
    image::open(&path).unwrap_or_else(|e| {
        panic!(
            "missing golden image {} ({}) – run `cargo run --example generate_goldens`",
            name, e
        )
    })
}

fn assert_image_eq(expected: &DynamicImage, actual: &DynamicImage, context: &str) {
    let expected = expected.to_rgba8();
    let actual = actual.to_rgba8();
    assert_eq!(
        expected.dimensions(),
        actual.dimensions(),
        "{} dimensions differ",
        context
    );
    assert_eq!(
        expected.as_raw(),
        actual.as_raw(),
        "{} pixel data differs",
        context
    );
}

#[test]
fn transforms_match_golden_outputs() {
    let base = load_image("base.png");

    let expected_h = load_image("flip_horizontal.png");
    let actual_h = flip_horizontal(&base).expect("flip_horizontal");
    assert_image_eq(&expected_h, &actual_h, "flip_horizontal");

    let expected_v = load_image("flip_vertical.png");
    let actual_v = flip_vertical(&base).expect("flip_vertical");
    assert_image_eq(&expected_v, &actual_v, "flip_vertical");

    let expected_90 = load_image("rotate_90.png");
    let actual_90 = rotate_90(&base).expect("rotate_90");
    assert_image_eq(&expected_90, &actual_90, "rotate_90");

    let expected_180 = load_image("rotate_180.png");
    let actual_180 = rotate_180(&base).expect("rotate_180");
    assert_image_eq(&expected_180, &actual_180, "rotate_180");

    let expected_270 = load_image("rotate_270.png");
    let actual_270 = rotate_270(&base).expect("rotate_270");
    assert_image_eq(&expected_270, &actual_270, "rotate_270");

    let expected_rot33 = load_image("rotate_33_bilinear.png");
    let actual_rot33 = rotate(&base, 33.0, Interpolation::Bilinear).expect("rotate bilinear");
    assert_image_eq(&expected_rot33, &actual_rot33, "rotate 33 bilinear");

    let expected_rot120 = load_image("rotate_120_nearest.png");
    let actual_rot120 = rotate(&base, 120.0, Interpolation::Nearest).expect("rotate nearest");
    assert_image_eq(&expected_rot120, &actual_rot120, "rotate 120 nearest");

    let expected_crop = load_image("crop_24x20_at_8x6.png");
    let actual_crop = crop(&base, 8, 6, 24, 20).expect("crop");
    assert_image_eq(&expected_crop, &actual_crop, "crop 24x20");
}
