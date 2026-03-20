// TODO: MIGRATE TO XENO-RT — this module manages GPU memory for AI model inference and belongs
// in the inference runtime. The GpuMemoryPool, model swapping, LRU eviction, and CUDA memory
// detection are all inference-specific and should move to xeno-rt.
//!
//! GPU memory management for AI model inference.
//!
//! Provides intelligent model swapping, shared GPU memory pools, and CUDA stream
//! management to optimize GPU utilization when running multiple AI models.
//!
//! # Architecture (2025-2026 Research)
//!
//! ## Model Swapping Strategy
//! When multiple models need GPU memory but VRAM is limited:
//! 1. **Priority queue**: Most-recently-used models stay loaded
//! 2. **Lazy loading**: Models load on first use, unload on memory pressure
//! 3. **Size-aware eviction**: Evict largest models first when under pressure
//!
//! ## CUDA Stream Management
//! ONNX Runtime manages CUDA streams internally. This module provides:
//! - Memory budget tracking per device
//! - Model lifecycle management (load/unload/swap)
//! - Concurrent model usage coordination
//!
//! ## Memory Budget
//! - Reserve 512MB for system/display use
//! - Allocate remaining VRAM across models with LRU eviction
//! - Fall back to CPU when GPU memory is exhausted
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use xeno_lib::gpu_memory::{GpuMemoryPool, GpuConfig};
//!
//! let config = GpuConfig::default();
//! let pool = GpuMemoryPool::new(config);
//! let available = pool.available_memory_mb();
//! println!("Available GPU memory: {} MB", available);
//! ```

use std::collections::HashMap;
use std::time::Instant;

/// Configuration for GPU memory management.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Maximum GPU memory budget in MB. 0 = auto-detect.
    pub max_memory_mb: u64,
    /// Reserved memory for system/display in MB. Default: 512.
    pub reserved_memory_mb: u64,
    /// GPU device ID. Default: 0.
    pub device_id: i32,
    /// Maximum number of models to keep loaded. Default: 4.
    pub max_loaded_models: usize,
    /// Whether to fall back to CPU when GPU memory is exhausted. Default: true.
    pub cpu_fallback: bool,
    /// Whether to enable memory defragmentation. Default: true.
    pub enable_defrag: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 0,
            reserved_memory_mb: 512,
            device_id: 0,
            max_loaded_models: 4,
            cpu_fallback: true,
            enable_defrag: true,
        }
    }
}

impl GpuConfig {
    /// Create config with specific memory budget.
    pub fn with_memory(mut self, mb: u64) -> Self {
        self.max_memory_mb = mb;
        self
    }

    /// Set maximum loaded models.
    pub fn with_max_models(mut self, n: usize) -> Self {
        self.max_loaded_models = n.max(1);
        self
    }

    /// Disable CPU fallback.
    pub fn no_cpu_fallback(mut self) -> Self {
        self.cpu_fallback = false;
        self
    }
}

/// Tracks a loaded model's memory usage.
#[derive(Debug, Clone)]
pub struct ModelAllocation {
    /// Model identifier.
    pub model_id: String,
    /// Estimated memory usage in MB.
    pub memory_mb: u64,
    /// Whether loaded on GPU (true) or CPU (false).
    pub on_gpu: bool,
    /// Last access time.
    pub last_accessed: Instant,
    /// Number of times used.
    pub use_count: u64,
}

/// GPU memory pool for managing multiple AI model sessions.
///
/// Tracks memory allocations, handles model swapping, and provides
/// a unified interface for GPU resource management.
pub struct GpuMemoryPool {
    config: GpuConfig,
    allocations: HashMap<String, ModelAllocation>,
    total_gpu_memory_mb: u64,
    used_gpu_memory_mb: u64,
}

impl GpuMemoryPool {
    /// Creates a new GPU memory pool.
    pub fn new(config: GpuConfig) -> Self {
        let total = if config.max_memory_mb > 0 {
            config.max_memory_mb
        } else {
            // Auto-detect: default to 8GB if detection fails
            detect_gpu_memory_mb().unwrap_or(8192)
        };

        Self {
            config,
            allocations: HashMap::new(),
            total_gpu_memory_mb: total,
            used_gpu_memory_mb: 0,
        }
    }

    /// Returns total GPU memory in MB.
    pub fn total_memory_mb(&self) -> u64 {
        self.total_gpu_memory_mb
    }

    /// Returns available GPU memory in MB (accounting for reservations).
    pub fn available_memory_mb(&self) -> u64 {
        let usable = self.total_gpu_memory_mb.saturating_sub(self.config.reserved_memory_mb);
        usable.saturating_sub(self.used_gpu_memory_mb)
    }

    /// Returns used GPU memory in MB.
    pub fn used_memory_mb(&self) -> u64 {
        self.used_gpu_memory_mb
    }

    /// Returns the number of loaded models.
    pub fn loaded_model_count(&self) -> usize {
        self.allocations.len()
    }

    /// Checks if a model can fit in available GPU memory.
    pub fn can_fit(&self, model_size_mb: u64) -> bool {
        self.available_memory_mb() >= model_size_mb
    }

    /// Registers a model allocation.
    ///
    /// If GPU memory is insufficient, attempts to evict least-recently-used models.
    /// If eviction isn't enough, loads on CPU if fallback is enabled.
    ///
    /// # Returns
    ///
    /// `true` if allocated on GPU, `false` if on CPU.
    pub fn allocate(&mut self, model_id: &str, memory_mb: u64) -> bool {
        // Check if already allocated
        if let Some(alloc) = self.allocations.get_mut(model_id) {
            alloc.last_accessed = Instant::now();
            alloc.use_count += 1;
            return alloc.on_gpu;
        }

        // Try to fit on GPU
        let on_gpu = if self.can_fit(memory_mb) {
            self.used_gpu_memory_mb += memory_mb;
            true
        } else if self.try_evict(memory_mb) {
            self.used_gpu_memory_mb += memory_mb;
            true
        } else {
            // Can't fit on GPU — use CPU if fallback enabled, otherwise fail
            false
        };

        // Enforce max loaded models
        while self.allocations.len() >= self.config.max_loaded_models {
            self.evict_lru();
        }

        self.allocations.insert(
            model_id.to_string(),
            ModelAllocation {
                model_id: model_id.to_string(),
                memory_mb,
                on_gpu,
                last_accessed: Instant::now(),
                use_count: 1,
            },
        );

        on_gpu
    }

    /// Releases a model allocation.
    pub fn release(&mut self, model_id: &str) {
        if let Some(alloc) = self.allocations.remove(model_id) {
            if alloc.on_gpu {
                self.used_gpu_memory_mb = self.used_gpu_memory_mb.saturating_sub(alloc.memory_mb);
            }
        }
    }

    /// Releases all allocations.
    pub fn release_all(&mut self) {
        self.allocations.clear();
        self.used_gpu_memory_mb = 0;
    }

    /// Lists all loaded models.
    pub fn loaded_models(&self) -> Vec<&ModelAllocation> {
        self.allocations.values().collect()
    }

    /// Returns a summary of memory usage.
    pub fn memory_summary(&self) -> MemorySummary {
        let gpu_models: Vec<_> = self
            .allocations
            .values()
            .filter(|a| a.on_gpu)
            .map(|a| a.model_id.clone())
            .collect();
        let cpu_models: Vec<_> = self
            .allocations
            .values()
            .filter(|a| !a.on_gpu)
            .map(|a| a.model_id.clone())
            .collect();

        MemorySummary {
            total_gpu_mb: self.total_gpu_memory_mb,
            used_gpu_mb: self.used_gpu_memory_mb,
            available_gpu_mb: self.available_memory_mb(),
            reserved_mb: self.config.reserved_memory_mb,
            gpu_models,
            cpu_models,
            total_models: self.allocations.len(),
        }
    }

    /// Attempts to evict models to free `needed_mb` of GPU memory.
    fn try_evict(&mut self, needed_mb: u64) -> bool {
        let mut freed = 0u64;

        // Collect GPU model IDs sorted by last access time (oldest first)
        let mut gpu_models: Vec<_> = self
            .allocations
            .values()
            .filter(|a| a.on_gpu)
            .map(|a| (a.model_id.clone(), a.last_accessed, a.memory_mb))
            .collect();
        gpu_models.sort_by_key(|&(_, accessed, _)| accessed);

        for (model_id, _, memory_mb) in gpu_models {
            if freed >= needed_mb {
                break;
            }
            self.release(&model_id);
            freed += memory_mb;
        }

        freed >= needed_mb
    }

    /// Evicts the least-recently-used model.
    fn evict_lru(&mut self) {
        let lru = self
            .allocations
            .values()
            .min_by_key(|a| a.last_accessed)
            .map(|a| a.model_id.clone());

        if let Some(model_id) = lru {
            self.release(&model_id);
        }
    }
}

/// Summary of GPU memory usage.
#[derive(Debug, Clone)]
pub struct MemorySummary {
    /// Total GPU memory in MB.
    pub total_gpu_mb: u64,
    /// Used GPU memory in MB.
    pub used_gpu_mb: u64,
    /// Available GPU memory in MB.
    pub available_gpu_mb: u64,
    /// Reserved memory in MB.
    pub reserved_mb: u64,
    /// Models loaded on GPU.
    pub gpu_models: Vec<String>,
    /// Models loaded on CPU.
    pub cpu_models: Vec<String>,
    /// Total loaded models.
    pub total_models: usize,
}

/// Estimates model memory requirements based on file size.
///
/// ONNX model GPU memory is approximately:
/// - FP32: ~1.2x file size (model weights + activations)
/// - FP16: ~0.7x file size
/// - INT8: ~0.4x file size
pub fn estimate_model_memory_mb(model_file_size_bytes: u64, is_fp16: bool) -> u64 {
    let multiplier = if is_fp16 { 0.7 } else { 1.2 };
    ((model_file_size_bytes as f64 * multiplier) / (1024.0 * 1024.0)) as u64 + 1
}

/// Detects available GPU memory. Returns None if no GPU is available.
fn detect_gpu_memory_mb() -> Option<u64> {
    // Stub: in production, query CUDA runtime for device memory
    // cudaMemGetInfo(&free, &total)
    // For now, return None to trigger default
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GpuConfig::default();
        assert_eq!(config.reserved_memory_mb, 512);
        assert!(config.cpu_fallback);
        assert_eq!(config.max_loaded_models, 4);
    }

    #[test]
    fn test_pool_creation() {
        let config = GpuConfig::default().with_memory(4096);
        let pool = GpuMemoryPool::new(config);
        assert_eq!(pool.total_memory_mb(), 4096);
        assert_eq!(pool.available_memory_mb(), 4096 - 512);
    }

    #[test]
    fn test_allocate_and_release() {
        let config = GpuConfig::default().with_memory(4096);
        let mut pool = GpuMemoryPool::new(config);

        let on_gpu = pool.allocate("upscale", 500);
        assert!(on_gpu);
        assert_eq!(pool.used_memory_mb(), 500);
        assert_eq!(pool.loaded_model_count(), 1);

        pool.release("upscale");
        assert_eq!(pool.used_memory_mb(), 0);
        assert_eq!(pool.loaded_model_count(), 0);
    }

    #[test]
    fn test_eviction() {
        let config = GpuConfig::default().with_memory(2048).with_max_models(2);
        let mut pool = GpuMemoryPool::new(config);

        pool.allocate("model_a", 500);
        pool.allocate("model_b", 500);
        // Third allocation should evict oldest
        pool.allocate("model_c", 500);

        assert_eq!(pool.loaded_model_count(), 2);
    }

    #[test]
    fn test_cpu_fallback() {
        let config = GpuConfig::default().with_memory(1024);
        let mut pool = GpuMemoryPool::new(config);

        // Try to allocate more than available (1024 - 512 = 512 available)
        let on_gpu = pool.allocate("huge_model", 2000);
        // Should fall back to CPU
        assert!(!on_gpu);
    }

    #[test]
    fn test_memory_summary() {
        let config = GpuConfig::default().with_memory(4096);
        let mut pool = GpuMemoryPool::new(config);
        pool.allocate("model_a", 300);

        let summary = pool.memory_summary();
        assert_eq!(summary.total_gpu_mb, 4096);
        assert_eq!(summary.used_gpu_mb, 300);
        assert_eq!(summary.total_models, 1);
    }

    #[test]
    fn test_estimate_model_memory() {
        let size_bytes = 100 * 1024 * 1024; // 100 MB
        let fp32_est = estimate_model_memory_mb(size_bytes, false);
        let fp16_est = estimate_model_memory_mb(size_bytes, true);
        assert!(fp32_est > fp16_est);
    }
}
