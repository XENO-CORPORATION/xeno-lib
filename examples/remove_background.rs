//! Example: Remove background from an image using ONNX AI model.
//!
//! This example demonstrates how to use the background removal feature
//! to create images with transparent backgrounds.
//!
//! # Prerequisites
//!
//! 1. Download the RMBG-1.4 model:
//!    ```bash
//!    mkdir -p ~/.xeno-lib/models
//!    curl -L "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx" \
//!         -o ~/.xeno-lib/models/rmbg-1.4.onnx
//!    ```
//!
//! 2. Run with the background-removal feature:
//!    ```bash
//!    cargo run --example remove_background --features background-removal -- input.jpg output.png
//!    ```
//!
//! # Usage
//!
//! ```bash
//! # Basic usage
//! cargo run --example remove_background --features background-removal -- photo.jpg result.png
//!
//! # With CUDA acceleration
//! cargo run --example remove_background --features background-removal-cuda -- photo.jpg result.png
//!
//! # Custom model path
//! cargo run --example remove_background --features background-removal -- \
//!     --model /path/to/model.onnx photo.jpg result.png
//! ```

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use xeno_lib::background::{load_model, remove_background, BackgroundRemovalConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let (input_path, output_path, model_path) = parse_args(&args)?;

    println!("Background Removal Example");
    println!("==========================");
    println!();

    // Configure model
    let config = if let Some(model) = model_path {
        BackgroundRemovalConfig {
            model_path: PathBuf::from(model),
            ..Default::default()
        }
    } else {
        BackgroundRemovalConfig::default()
    };

    println!("Model path: {}", config.model_path.display());
    println!("Use GPU: {}", config.use_gpu);
    println!("Confidence threshold: {}", config.confidence_threshold);
    println!();

    // Load model
    println!("Loading model...");
    let load_start = Instant::now();
    let mut session = load_model(&config)?;
    println!("Model loaded in {:?}", load_start.elapsed());
    println!();

    // Load input image
    println!("Loading image: {}", input_path);
    let load_image_start = Instant::now();
    let input_image = image::open(&input_path)?;
    println!(
        "Image loaded in {:?} ({}x{})",
        load_image_start.elapsed(),
        input_image.width(),
        input_image.height()
    );
    println!();

    // Remove background
    println!("Removing background...");
    let inference_start = Instant::now();
    let output_image = remove_background(&input_image, &mut session)?;
    let inference_time = inference_start.elapsed();
    println!("Background removed in {:?}", inference_time);
    println!();

    // Save output
    println!("Saving result to: {}", output_path);
    let save_start = Instant::now();
    output_image.save(&output_path)?;
    println!("Saved in {:?}", save_start.elapsed());
    println!();

    // Summary
    println!("Summary");
    println!("-------");
    println!("Input:  {} ({}x{})", input_path, input_image.width(), input_image.height());
    println!("Output: {} ({}x{})", output_path, output_image.width(), output_image.height());
    println!("Inference time: {:?}", inference_time);
    println!(
        "Throughput: {:.2} megapixels/second",
        (input_image.width() as f64 * input_image.height() as f64) / 1_000_000.0
            / inference_time.as_secs_f64()
    );

    Ok(())
}

fn parse_args(args: &[String]) -> Result<(String, String, Option<String>), &'static str> {
    if args.len() < 3 {
        print_usage(&args[0]);
        return Err("Not enough arguments");
    }

    let mut input = None;
    let mut output = None;
    let mut model = None;
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                if i + 1 >= args.len() {
                    return Err("--model requires a path argument");
                }
                model = Some(args[i + 1].clone());
                i += 2;
            }
            "--help" | "-h" => {
                print_usage(&args[0]);
                std::process::exit(0);
            }
            arg if arg.starts_with('-') => {
                return Err("Unknown option");
            }
            _ => {
                if input.is_none() {
                    input = Some(args[i].clone());
                } else if output.is_none() {
                    output = Some(args[i].clone());
                }
                i += 1;
            }
        }
    }

    match (input, output) {
        (Some(i), Some(o)) => Ok((i, o, model)),
        _ => {
            print_usage(&args[0]);
            Err("Missing required arguments")
        }
    }
}

fn print_usage(program: &str) {
    eprintln!("Usage: {} [OPTIONS] <INPUT> <OUTPUT>", program);
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  <INPUT>   Input image path (jpg, png, webp, etc.)");
    eprintln!("  <OUTPUT>  Output image path (should be .png for transparency)");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -m, --model <PATH>  Path to ONNX model file");
    eprintln!("  -h, --help          Show this help message");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {} photo.jpg result.png", program);
    eprintln!("  {} --model custom.onnx photo.jpg result.png", program);
}
