use crate::error::TransformError;
use image::DynamicImage;
use rayon::prelude::*;

/// Applies a transformation function to multiple images in parallel.
pub fn batch_transform<F>(
    images: &[DynamicImage],
    transform_fn: F,
) -> Result<Vec<DynamicImage>, TransformError>
where
    F: Fn(&DynamicImage) -> Result<DynamicImage, TransformError> + Sync,
{
    images
        .par_iter()
        .map(|img| transform_fn(img))
        .collect::<Result<Vec<_>, _>>()
}

/// Applies a transformation to a numbered sequence of images.
/// Pattern example: "frame_%04d.jpg" will match frame_0001.jpg, frame_0002.jpg, etc.
pub fn sequence_transform<F>(
    pattern: &str,
    start: usize,
    end: usize,
    transform_fn: F,
    output_pattern: &str,
) -> Result<(), TransformError>
where
    F: Fn(&DynamicImage) -> Result<DynamicImage, TransformError> + Sync + Send,
{
    let pattern = pattern.to_string();
    let output_pattern = output_pattern.to_string();

    (start..=end)
        .into_par_iter()
        .try_for_each(|i| {
            let input_path = format_path_pattern_str(&pattern, i);
            let output_path = format_path_pattern_str(&output_pattern, i);

            let img = image::open(&input_path)
                .map_err(|_| TransformError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Failed to open image: {}", input_path),
                )))?;

            let transformed = transform_fn(&img)?;

            transformed.save(&output_path)
                .map_err(|_| TransformError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to save image: {}", output_path),
                )))?;

            Ok(())
        })
}

/// Applies transformations in parallel with explicit thread control.
pub fn parallel_batch<F>(
    images: &[DynamicImage],
    transform_fn: F,
    num_threads: usize,
) -> Result<Vec<DynamicImage>, TransformError>
where
    F: Fn(&DynamicImage) -> Result<DynamicImage, TransformError> + Sync + Send,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|_| TransformError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to create thread pool",
        )))?;

    pool.install(|| batch_transform(images, transform_fn))
}

/// Applies a transformation to images read from stdin and written to stdout.
/// Useful for zero-disk I/O pipelines.
/// Note: This function uses a temporary buffer since stdin/stdout don't support seeking.
pub fn stream_transform<F>(
    transform_fn: F,
) -> Result<(), TransformError>
where
    F: Fn(DynamicImage) -> Result<DynamicImage, TransformError>,
{
    use std::io::{stdin, stdout, Read, Write, BufReader};

    // Read stdin into a buffer
    let stdin = stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut buffer = Vec::new();
    reader
        .read_to_end(&mut buffer)
        .map_err(|_| TransformError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Failed to read from stdin",
        )))?;

    // Load image from buffer
    let img = image::load_from_memory(&buffer)
        .map_err(|_| TransformError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Failed to decode image from stdin",
        )))?;

    let transformed = transform_fn(img)?;

    // Write to stdout through a buffer
    let mut output_buffer = Vec::new();
    transformed
        .write_to(&mut std::io::Cursor::new(&mut output_buffer), image::ImageFormat::Png)
        .map_err(|_| TransformError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to encode image",
        )))?;

    let stdout = stdout();
    let mut writer = stdout.lock();
    writer
        .write_all(&output_buffer)
        .map_err(|_| TransformError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to write to stdout",
        )))?;

    Ok(())
}

/// A pipeline that chains multiple transformation operations.
pub struct TransformPipeline {
    operations: Vec<Box<dyn Fn(DynamicImage) -> Result<DynamicImage, TransformError> + Send + Sync>>,
}

impl TransformPipeline {
    /// Creates a new empty pipeline.
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Adds a transformation operation to the pipeline.
    pub fn add<F>(mut self, operation: F) -> Self
    where
        F: Fn(DynamicImage) -> Result<DynamicImage, TransformError> + Send + Sync + 'static,
    {
        self.operations.push(Box::new(operation));
        self
    }

    /// Executes the pipeline on an image.
    pub fn execute(&self, image: DynamicImage) -> Result<DynamicImage, TransformError> {
        self.operations.iter().try_fold(image, |img, op| op(img))
    }

    /// Executes the pipeline on multiple images in parallel.
    pub fn execute_batch(&self, images: &[DynamicImage]) -> Result<Vec<DynamicImage>, TransformError> {
        images
            .par_iter()
            .map(|img| self.execute(img.clone()))
            .collect()
    }
}

impl Default for TransformPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Chain multiple geometric operations into a pipeline.
/// This is a convenience wrapper around TransformPipeline.
pub fn pipeline_transform(
    image: DynamicImage,
    operations: Vec<Box<dyn Fn(DynamicImage) -> Result<DynamicImage, TransformError> + Send + Sync>>,
) -> Result<DynamicImage, TransformError> {
    let mut pipeline = TransformPipeline::new();
    for op in operations {
        pipeline.operations.push(op);
    }
    pipeline.execute(image)
}

fn format_path_pattern_str(pattern: &str, index: usize) -> String {
    // Simple pattern replacement for %04d, %05d, etc.
    if let Some(pos) = pattern.find('%') {
        if let Some(end) = pattern[pos..].find('d') {
            let format_spec = &pattern[pos..pos + end + 1];
            let width = format_spec
                .chars()
                .filter(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<usize>()
                .unwrap_or(4);

            let formatted = format!("{:0width$}", index, width = width);
            return pattern.replace(format_spec, &formatted);
        }
    }

    pattern.to_string()
}
