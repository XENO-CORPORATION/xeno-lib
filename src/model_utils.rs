//! Shared utilities for AI model path resolution and home directory detection.
//!
//! This module provides common functions used by all AI model modules to locate
//! ONNX model files on disk. It avoids duplicating the home directory detection
//! and default model path logic across every model config module.

use std::path::PathBuf;

/// Cross-platform home directory detection.
///
/// Returns the user's home directory using environment variables.
/// On Windows, uses `USERPROFILE`; on other platforms, uses `HOME`.
pub fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE").ok().map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}

/// Returns the default model path for a given model filename.
///
/// The default model directory is `~/.xeno-lib/models/`.
/// If the home directory cannot be determined, falls back to the
/// current directory.
///
/// # Arguments
///
/// * `filename` - The ONNX model filename (e.g., "birefnet-general.onnx")
///
/// # Returns
///
/// The full path to the model file.
pub fn default_model_path(filename: &str) -> PathBuf {
    let home = home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".xeno-lib").join("models").join(filename)
}

/// Returns the default models directory.
///
/// The default model directory is `~/.xeno-lib/models/`.
pub fn models_dir() -> PathBuf {
    let home = home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".xeno-lib").join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model_path_contains_filename() {
        let path = default_model_path("test_model.onnx");
        assert!(path.to_string_lossy().contains("test_model.onnx"));
        assert!(path.to_string_lossy().contains(".xeno-lib"));
        assert!(path.to_string_lossy().contains("models"));
    }

    #[test]
    fn test_models_dir_exists_in_path() {
        let dir = models_dir();
        assert!(dir.to_string_lossy().contains("models"));
    }

    #[test]
    fn test_home_dir_returns_something() {
        // On CI or testing environments, home dir should be available
        let home = home_dir();
        // Don't assert Some because some CI environments may not have HOME set
        if let Some(h) = home {
            assert!(!h.as_os_str().is_empty());
        }
    }
}
