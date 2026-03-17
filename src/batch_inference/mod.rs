//! Batch inference utilities for processing multiple inputs in parallel.
//!
//! Provides optimized batch processing for AI models, allowing multiple images
//! to be processed in a single forward pass when supported by the model, or
//! efficiently parallelized across CPU cores.
//!
//! # Architecture (2025-2026 Research)
//!
//! ## Batched ONNX Inference
//! Most ONNX models support dynamic batch dimensions. Instead of running N
//! individual inferences, batch them into a single [N, C, H, W] tensor for
//! significantly better GPU utilization.
//!
//! ## Parallel CPU Processing
//! For CPU-only models or when GPU batching isn't supported, use rayon to
//! process items across all CPU cores.
//!
//! ## Pipeline Parallelism
//! For multi-stage pipelines (preprocess -> inference -> postprocess),
//! overlap stages using async channels for maximum throughput.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::batch_inference::{BatchProcessor, BatchConfig};
//!
//! let config = BatchConfig::default();
//! let processor = BatchProcessor::new(config);
//! // Process images in batches
//! let batch_size = processor.optimal_batch_size(1024, 1024);
//! println!("Optimal batch size: {}", batch_size);
//! ```

use std::time::Instant;
use rayon::prelude::*;
use image::DynamicImage;

use crate::error::TransformError;

/// Configuration for batch inference.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size for GPU inference. Default: 4.
    pub max_gpu_batch: usize,
    /// Maximum batch size for CPU inference. Default: number of CPU cores.
    pub max_cpu_batch: usize,
    /// Maximum GPU memory to use per batch in MB. Default: 2048.
    pub max_batch_memory_mb: u64,
    /// Whether to use GPU batching when available. Default: true.
    pub use_gpu: bool,
    /// Whether to use rayon for CPU parallelism. Default: true.
    pub parallel_cpu: bool,
    /// Number of CPU threads. 0 = auto. Default: 0.
    pub num_threads: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_gpu_batch: 4,
            max_cpu_batch: num_cpus(),
            max_batch_memory_mb: 2048,
            use_gpu: true,
            parallel_cpu: true,
            num_threads: 0,
        }
    }
}

impl BatchConfig {
    /// Set maximum GPU batch size.
    pub fn with_gpu_batch(mut self, size: usize) -> Self {
        self.max_gpu_batch = size.max(1);
        self
    }

    /// Set maximum CPU batch size.
    pub fn with_cpu_batch(mut self, size: usize) -> Self {
        self.max_cpu_batch = size.max(1);
        self
    }

    /// Set memory budget.
    pub fn with_memory_budget(mut self, mb: u64) -> Self {
        self.max_batch_memory_mb = mb;
        self
    }

    /// Disable GPU batching.
    pub fn cpu_only(mut self) -> Self {
        self.use_gpu = false;
        self
    }
}

/// Batch processor for AI model inference.
pub struct BatchProcessor {
    config: BatchConfig,
}

impl BatchProcessor {
    /// Creates a new batch processor.
    pub fn new(config: BatchConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Calculates optimal batch size based on image dimensions and available memory.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width.
    /// * `height` - Image height.
    ///
    /// # Returns
    ///
    /// Optimal batch size that fits within the memory budget.
    pub fn optimal_batch_size(&self, width: u32, height: u32) -> usize {
        // Estimate memory per image: width * height * 4 channels * 4 bytes (f32) * 2 (input+output)
        let mem_per_image_mb =
            (width as u64 * height as u64 * 4 * 4 * 2) / (1024 * 1024);
        let mem_per_image_mb = mem_per_image_mb.max(1);

        let max_by_memory = (self.config.max_batch_memory_mb / mem_per_image_mb) as usize;
        let max_by_config = if self.config.use_gpu {
            self.config.max_gpu_batch
        } else {
            self.config.max_cpu_batch
        };

        max_by_memory.min(max_by_config).max(1)
    }

    /// Processes images in parallel using rayon.
    ///
    /// Applies a function to each image in the batch using available CPU cores.
    ///
    /// # Arguments
    ///
    /// * `images` - Slice of input images.
    /// * `process_fn` - Function to apply to each image.
    ///
    /// # Returns
    ///
    /// Vector of results, one per input image.
    pub fn process_parallel<F, T>(
        &self,
        images: &[DynamicImage],
        process_fn: F,
    ) -> Vec<Result<T, TransformError>>
    where
        F: Fn(&DynamicImage) -> Result<T, TransformError> + Sync,
        T: Send,
    {
        if self.config.parallel_cpu && images.len() > 1 {
            images
                .par_iter()
                .map(|img| process_fn(img))
                .collect()
        } else {
            images
                .iter()
                .map(|img| process_fn(img))
                .collect()
        }
    }

    /// Processes images in batches with progress tracking.
    ///
    /// Splits the input into optimal batch sizes, processes each batch,
    /// and calls the progress callback between batches.
    ///
    /// # Arguments
    ///
    /// * `images` - Slice of input images.
    /// * `batch_fn` - Function to process a batch of images.
    /// * `progress` - Optional progress callback.
    pub fn process_batched<F, T>(
        &self,
        images: &[DynamicImage],
        batch_fn: F,
        progress: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,
    ) -> Vec<Result<T, TransformError>>
    where
        F: Fn(&[DynamicImage]) -> Vec<Result<T, TransformError>>,
    {
        if images.is_empty() {
            return Vec::new();
        }

        // Determine batch size from first image
        let batch_size = if !images.is_empty() {
            self.optimal_batch_size(images[0].width(), images[0].height())
        } else {
            1
        };

        let total = images.len();
        let mut all_results = Vec::with_capacity(total);
        let mut processed = 0;

        for chunk in images.chunks(batch_size) {
            let batch_results = batch_fn(chunk);
            all_results.extend(batch_results);

            processed += chunk.len();
            if let Some(ref cb) = progress {
                cb(processed, total);
            }
        }

        all_results
    }
}

/// Batch processing result with timing information.
#[derive(Debug, Clone)]
pub struct BatchResult<T> {
    /// Individual results.
    pub results: Vec<Result<T, String>>,
    /// Total processing time in milliseconds.
    pub total_ms: u64,
    /// Average time per item in milliseconds.
    pub avg_per_item_ms: f64,
    /// Throughput in items per second.
    pub throughput: f64,
    /// Batch size used.
    pub batch_size: usize,
    /// Number of successful items.
    pub succeeded: usize,
    /// Number of failed items.
    pub failed: usize,
}

/// Times a batch operation and returns results with metrics.
pub fn timed_batch<F, T, I>(
    items: &[I],
    process_fn: F,
) -> BatchResult<T>
where
    F: Fn(&I) -> Result<T, String>,
{
    let start = Instant::now();
    let mut succeeded = 0usize;
    let mut failed = 0usize;

    let results: Vec<_> = items
        .iter()
        .map(|item| {
            let result = process_fn(item);
            match &result {
                Ok(_) => succeeded += 1,
                Err(_) => failed += 1,
            }
            result
        })
        .collect();

    let elapsed = start.elapsed();
    let total_ms = elapsed.as_millis() as u64;
    let count = items.len().max(1);

    BatchResult {
        results,
        total_ms,
        avg_per_item_ms: total_ms as f64 / count as f64,
        throughput: count as f64 / elapsed.as_secs_f64().max(0.001),
        batch_size: count,
        succeeded,
        failed,
    }
}

/// Returns the number of available CPU cores.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_default_config() {
        let config = BatchConfig::default();
        assert_eq!(config.max_gpu_batch, 4);
        assert!(config.parallel_cpu);
    }

    #[test]
    fn test_optimal_batch_size() {
        let processor = BatchProcessor::new(BatchConfig::default());

        // Small images = larger batches
        let small = processor.optimal_batch_size(256, 256);
        assert!(small >= 1);

        // Large images = smaller batches
        let large = processor.optimal_batch_size(4096, 4096);
        assert!(large >= 1);
        assert!(large <= small || small == 1);
    }

    #[test]
    fn test_process_parallel() {
        let processor = BatchProcessor::new(BatchConfig::default());
        let images: Vec<DynamicImage> = (0..4)
            .map(|_| DynamicImage::ImageRgb8(RgbImage::new(64, 64)))
            .collect();

        let results = processor.process_parallel(&images, |img| {
            Ok(img.width() * img.height())
        });

        assert_eq!(results.len(), 4);
        for r in &results {
            assert_eq!(*r.as_ref().unwrap(), 64 * 64);
        }
    }

    #[test]
    fn test_process_batched() {
        let processor = BatchProcessor::new(BatchConfig::default());
        let images: Vec<DynamicImage> = (0..6)
            .map(|_| DynamicImage::ImageRgb8(RgbImage::new(64, 64)))
            .collect();

        let mut progress_calls = 0u32;
        let results = processor.process_batched(
            &images,
            |batch| {
                batch.iter().map(|img| Ok(img.width())).collect()
            },
            Some(Box::new(move |_current, _total| {
                // Progress callback
            })),
        );

        assert_eq!(results.len(), 6);
    }

    #[test]
    fn test_timed_batch() {
        let items = vec![1, 2, 3, 4, 5];
        let result = timed_batch(&items, |&x| Ok(x * 2));

        assert_eq!(result.succeeded, 5);
        assert_eq!(result.failed, 0);
        assert_eq!(result.batch_size, 5);
        assert!(result.throughput > 0.0);
    }

    #[test]
    fn test_timed_batch_with_failures() {
        let items = vec![1, 2, 0, 4, 5];
        let result = timed_batch(&items, |&x| {
            if x == 0 {
                Err("division by zero".to_string())
            } else {
                Ok(10 / x)
            }
        });

        assert_eq!(result.succeeded, 4);
        assert_eq!(result.failed, 1);
    }
}
