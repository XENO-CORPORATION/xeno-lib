fn main() {
    let img = image::DynamicImage::new_rgb8(800, 600);
    img.save("../xeno-edit/test_input.png").unwrap();
    println!("Created test_input.png (800x600)");
}
