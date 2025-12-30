//! Video frame type.

use image::DynamicImage;

/// A single decoded video frame.
///
/// Contains the pixel data as a `DynamicImage` along with timing metadata.
/// The image can be processed using any xeno-lib function.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Pixel data as RGBA image.
    pub image: DynamicImage,

    /// Presentation timestamp in milliseconds from start of video.
    pub pts_ms: i64,

    /// Duration of this frame in milliseconds.
    pub duration_ms: i64,

    /// Frame number (0-indexed).
    pub frame_number: u64,

    /// Whether this is a keyframe (I-frame).
    pub is_keyframe: bool,
}

impl VideoFrame {
    /// Create a new video frame.
    pub fn new(image: DynamicImage, pts_ms: i64, frame_number: u64) -> Self {
        Self {
            image,
            pts_ms,
            duration_ms: 0,
            frame_number,
            is_keyframe: false,
        }
    }

    /// Create a new video frame with all metadata.
    pub fn with_metadata(
        image: DynamicImage,
        pts_ms: i64,
        duration_ms: i64,
        frame_number: u64,
        is_keyframe: bool,
    ) -> Self {
        Self {
            image,
            pts_ms,
            duration_ms,
            frame_number,
            is_keyframe,
        }
    }

    /// Get the width of the frame in pixels.
    pub fn width(&self) -> u32 {
        self.image.width()
    }

    /// Get the height of the frame in pixels.
    pub fn height(&self) -> u32 {
        self.image.height()
    }

    /// Get the timestamp as seconds.
    pub fn pts_secs(&self) -> f64 {
        self.pts_ms as f64 / 1000.0
    }
}
