use xeno_lib::*;

fn main() -> Result<(), TransformError> {
    println!("🖼️  xeno-lib Practical Usage Examples");
    println!("======================================\n");

    // Example 1: Load, transform, and save a single image
    println!("📸 Example 1: Basic Image Transformation");
    println!("   1. Load an image");
    println!("   2. Resize to 800x600");
    println!("   3. Rotate 45 degrees");
    println!("   4. Save result");
    println!();
    println!("   Code:");
    println!("   let img = image::open(\"input.jpg\")?;");
    println!("   let resized = resize(&img, 800, 600, Interpolation::Bilinear)?;");
    println!("   let rotated = rotate(&resized, 45.0, Interpolation::Bilinear)?;");
    println!("   rotated.save(\"output.jpg\")?;\n");

    // Example 2: Create a thumbnail with aspect ratio
    println!("🔍 Example 2: Create Thumbnail");
    println!("   let img = image::open(\"photo.jpg\")?;");
    println!("   let thumb = resize_fit(&img, 200, 200, Interpolation::Bilinear)?;");
    println!("   thumb.save(\"thumbnail.jpg\")?;\n");

    // Example 3: Crop to specific aspect ratio
    println!("✂️  Example 3: Crop to 16:9 Aspect Ratio");
    println!("   let img = image::open(\"video_frame.png\")?;");
    println!("   let cropped = crop_to_aspect(&img, 16.0/9.0, CropAnchor::Center)?;");
    println!("   cropped.save(\"16_9.png\")?;\n");

    // Example 4: Pipeline multiple transformations
    println!("🔄 Example 4: Transform Pipeline");
    println!("   let pipeline = TransformPipeline::new()");
    println!("       .add(|i| flip_horizontal(&i))");
    println!("       .add(|i| crop_center(&i, 500, 500))");
    println!("       .add(|i| rotate_90_cw(&i));");
    println!();
    println!("   let img = image::open(\"input.jpg\")?;");
    println!("   let result = pipeline.execute(img)?;");
    println!("   result.save(\"output.jpg\")?;\n");

    // Example 5: Batch process multiple images
    println!("📦 Example 5: Batch Processing");
    println!("   let images = vec![");
    println!("       image::open(\"img1.jpg\")?,");
    println!("       image::open(\"img2.jpg\")?,");
    println!("       image::open(\"img3.jpg\")?,");
    println!("   ];");
    println!();
    println!("   let thumbnails = batch_transform(&images, |img| {{");
    println!("       resize_fit(img, 150, 150, Interpolation::Bilinear)");
    println!("   }})?;");
    println!();
    println!("   for (i, thumb) in thumbnails.iter().enumerate() {{");
    println!("       thumb.save(format!(\"thumb_{{}}.jpg\", i))?;");
    println!("   }}\n");

    // Example 6: Process video frames
    println!("🎬 Example 6: Video Frame Processing");
    println!("   // Process frames 0-99 from frame_0000.png to frame_0099.png");
    println!("   sequence_transform(");
    println!("       \"input/frame_%04d.png\",");
    println!("       0,");
    println!("       99,");
    println!("       |img| resize(&img, 1920, 1080, Interpolation::Bilinear),");
    println!("       \"output/frame_%04d.png\"");
    println!("   )?;\n");

    // Example 7: Advanced transformations
    println!("🎨 Example 7: Advanced Transformations");
    println!("   let img = image::open(\"input.jpg\")?;");
    println!();
    println!("   // Perspective transform");
    println!("   let src = [(0.0, 0.0), (800.0, 0.0), (800.0, 600.0), (0.0, 600.0)];");
    println!("   let dst = [(50.0, 50.0), (750.0, 30.0), (750.0, 570.0), (50.0, 590.0)];");
    println!("   let warped = perspective_transform(&img, src, dst, 800, 600)?;");
    println!();
    println!("   // Shear transformation");
    println!("   let sheared = shear_horizontal(&img, 0.3)?;");
    println!();
    println!("   // Custom affine transformation");
    println!("   let matrix = [[1.2, 0.1, 0.0], [0.1, 1.2, 0.0]];");
    println!("   let transformed = affine_transform(&img, matrix)?;\n");

    // Example 8: Configuration
    println!("⚙️  Example 8: Global Configuration");
    println!("   // Set default interpolation method");
    println!("   set_interpolation(Interpolation::Nearest);");
    println!();
    println!("   // Set default background color (RGBA)");
    println!("   set_background([255, 255, 255, 255]); // White");
    println!();
    println!("   // Preserve alpha channel");
    println!("   preserve_alpha(true);");
    println!();
    println!("   // Optimize memory usage");
    println!("   optimize_memory(true);\n");

    // Demonstrate actual working example
    println!("🚀 Running Live Example:");
    println!("   Creating a test image and applying transformations...\n");

    let img = image::DynamicImage::new_rgb8(400, 300);

    // Create a simple pipeline
    let pipeline = TransformPipeline::new()
        .add(|i| resize(&i, 200, 150, Interpolation::Bilinear))
        .add(|i| flip_horizontal(&i))
        .add(|i| crop_center(&i, 100, 100));

    let result = pipeline.execute(img)?;

    println!("   ✅ Pipeline executed successfully!");
    println!("   ✅ Result dimensions: {}x{}", result.width(), result.height());
    println!("   ✅ Ready to save with: result.save(\"output.png\")?;\n");

    println!("📚 All 52 operations are ready to use!");
    println!("   See examples/test_all_operations.rs for complete operation list.");
    println!("   See README.md for detailed API documentation.\n");

    Ok(())
}
