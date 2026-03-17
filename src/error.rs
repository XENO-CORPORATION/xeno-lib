use image::ColorType;
use thiserror::Error;

/// Errors that can occur while performing geometric image transformations.
#[derive(Debug, Error)]
pub enum TransformError {
    /// The requested color type or pixel format is not yet supported.
    #[error("unsupported color type {0:?}")]
    UnsupportedColorType(ColorType),

    /// The provided byte stream does not match any supported image format.
    #[error("unable to detect image format from provided data")]
    UnsupportedFormat,

    /// The requested crop rectangle exceeds the source image bounds.
    #[error(
        "crop rectangle (x: {x}, y: {y}, width: {width}, height: {height}) exceeds image bounds ({image_width}x{image_height})"
    )]
    CropOutOfBounds {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        image_width: u32,
        image_height: u32,
    },

    /// The provided rotation angle is invalid (e.g. NaN or infinite).
    #[error("rotation angle must be finite, got {angle}")]
    InvalidAngle { angle: f32 },

    /// The requested output dimensions are invalid (width or height is zero).
    #[error("image dimensions must be greater than zero, got {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    /// The provided scale factor is invalid (zero, negative, or non-finite).
    #[error("scale factor must be positive and finite, got {factor}")]
    InvalidScaleFactor { factor: f32 },

    /// A named parameter is outside the accepted range.
    #[error("parameter `{name}` is out of range with value {value}")]
    InvalidParameter { name: &'static str, value: f32 },

    /// EXIF metadata could not be parsed from the provided input.
    #[error("failed to parse EXIF data: {message}")]
    ExifRead { message: String },

    /// Requested overlay/write exceeds image bounds.
    #[error(
        "overlay starting at ({x},{y}) with dimensions {width}x{height} exceeds image bounds {image_width}x{image_height}"
    )]
    OverlayOutOfBounds {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        image_width: u32,
        image_height: u32,
    },

    /// Inputs have mismatched dimensions and cannot be processed together.
    #[error(
        "image dimension mismatch: left={left_width}x{left_height}, right={right_width}x{right_height}"
    )]
    DimensionMismatch {
        left_width: u32,
        left_height: u32,
        right_width: u32,
        right_height: u32,
    },

    /// Allocation for an output image buffer failed due to size overflow or capacity exhaustion.
    #[error("failed to allocate buffer for {width}x{height} image")]
    AllocationFailed { width: u32, height: u32 },

    /// I/O operation failed while accessing external data.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// ONNX model file was not found at the specified path.
    #[cfg(any(
        feature = "background-removal",
        feature = "upscale",
        feature = "face-restore",
        feature = "frame-interpolate",
        feature = "colorize",
        feature = "inpaint",
        feature = "face-detect",
        feature = "depth",
        feature = "transcribe",
        feature = "audio-separate",
        feature = "style-transfer",
        feature = "ocr",
        feature = "pose",
        feature = "face-analysis",
        feature = "text-to-3d",
        feature = "voice-clone",
        feature = "music-gen",
        feature = "video-gen"
    ))]
    #[error("model file not found at {path}")]
    ModelNotFound { path: String },

    /// Failed to load or initialize an ONNX model.
    #[cfg(any(
        feature = "background-removal",
        feature = "upscale",
        feature = "face-restore",
        feature = "frame-interpolate",
        feature = "colorize",
        feature = "inpaint",
        feature = "face-detect",
        feature = "depth",
        feature = "transcribe",
        feature = "audio-separate",
        feature = "style-transfer",
        feature = "ocr",
        feature = "pose",
        feature = "face-analysis",
        feature = "text-to-3d",
        feature = "voice-clone",
        feature = "music-gen",
        feature = "video-gen"
    ))]
    #[error("failed to load ONNX model: {message}")]
    ModelLoadFailed { message: String },

    /// ONNX inference failed during execution.
    #[cfg(any(
        feature = "background-removal",
        feature = "upscale",
        feature = "face-restore",
        feature = "frame-interpolate",
        feature = "colorize",
        feature = "inpaint",
        feature = "face-detect",
        feature = "depth",
        feature = "transcribe",
        feature = "audio-separate",
        feature = "style-transfer",
        feature = "ocr",
        feature = "pose",
        feature = "face-analysis",
        feature = "text-to-3d",
        feature = "voice-clone",
        feature = "music-gen",
        feature = "video-gen"
    ))]
    #[error("ONNX inference failed: {message}")]
    InferenceFailed { message: String },

    /// Failed to load a font file for text rendering.
    #[cfg(feature = "text-overlay")]
    #[error("failed to load font: {message}")]
    FontLoadFailed { message: String },

    /// Text overlay configuration is invalid.
    #[cfg(feature = "text-overlay")]
    #[error("invalid text configuration: {message}")]
    InvalidTextConfig { message: String },

    /// Parsing failed for a text-based format.
    #[cfg(feature = "subtitle")]
    #[error("failed to parse: {message}")]
    ParseFailed { message: String },

    /// Failed to read a file.
    #[cfg(feature = "subtitle")]
    #[error("failed to read file: {message}")]
    FileReadFailed { message: String },

    /// Failed to write a file.
    #[cfg(feature = "subtitle")]
    #[error("failed to write file: {message}")]
    FileWriteFailed { message: String },
}
