use crate::transforms::Interpolation;
use std::sync::Mutex;

/// Global configuration for transform operations.
pub struct TransformConfig {
    /// Default interpolation method for transforms
    pub default_interpolation: Interpolation,
    /// Background color for transforms (RGBA)
    pub background_color: [u8; 4],
    /// Preserve alpha channel in transformations
    pub preserve_alpha: bool,
    /// Use in-place operations where possible for memory optimization
    pub optimize_memory: bool,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            default_interpolation: Interpolation::Bilinear,
            background_color: [0, 0, 0, 0], // Transparent black
            preserve_alpha: true,
            optimize_memory: false,
        }
    }
}

static CONFIG: Mutex<TransformConfig> = Mutex::new(TransformConfig {
    default_interpolation: Interpolation::Bilinear,
    background_color: [0, 0, 0, 0],
    preserve_alpha: true,
    optimize_memory: false,
});

/// Sets the default interpolation method for all transform operations.
pub fn set_interpolation(interpolation: Interpolation) {
    if let Ok(mut config) = CONFIG.lock() {
        config.default_interpolation = interpolation;
    }
}

/// Gets the current default interpolation method.
pub fn get_interpolation() -> Interpolation {
    CONFIG
        .lock()
        .map(|config| config.default_interpolation)
        .unwrap_or(Interpolation::Bilinear)
}

/// Sets the default background color for transform operations (RGBA format).
pub fn set_background(color: [u8; 4]) {
    if let Ok(mut config) = CONFIG.lock() {
        config.background_color = color;
    }
}

/// Gets the current default background color.
pub fn get_background() -> [u8; 4] {
    CONFIG
        .lock()
        .map(|config| config.background_color)
        .unwrap_or([0, 0, 0, 0])
}

/// Sets whether to preserve alpha channel in transformations.
pub fn preserve_alpha(preserve: bool) {
    if let Ok(mut config) = CONFIG.lock() {
        config.preserve_alpha = preserve;
    }
}

/// Gets whether alpha channel preservation is enabled.
pub fn get_preserve_alpha() -> bool {
    CONFIG
        .lock()
        .map(|config| config.preserve_alpha)
        .unwrap_or(true)
}

/// Sets whether to optimize memory usage with in-place operations where possible.
pub fn optimize_memory(optimize: bool) {
    if let Ok(mut config) = CONFIG.lock() {
        config.optimize_memory = optimize;
    }
}

/// Gets whether memory optimization is enabled.
pub fn get_optimize_memory() -> bool {
    CONFIG
        .lock()
        .map(|config| config.optimize_memory)
        .unwrap_or(false)
}
