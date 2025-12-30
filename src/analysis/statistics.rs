use image::{DynamicImage, GenericImageView};

/// High-level metadata describing an image buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub color: image::ColorType,
    pub channels: u8,
    pub has_alpha: bool,
}

impl ImageInfo {
    pub(crate) fn from_dynamic_image(image: &DynamicImage) -> Self {
        let (width, height) = image.dimensions();
        let color = image.color();
        let channels = color.channel_count();
        let has_alpha = color.has_alpha();

        Self {
            width,
            height,
            color,
            channels,
            has_alpha,
        }
    }
}

/// Histogram values for each channel in the image (256 bins).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Histogram {
    pub channels: Vec<HistogramChannel>,
}

impl Histogram {
    pub(crate) fn from_dynamic_image(image: &DynamicImage) -> Self {
        match image {
            DynamicImage::ImageLuma8(buffer) => {
                let mut channel = HistogramChannel::new("Luma");
                fill_channel(&mut channel.bins, buffer.as_raw());
                Self {
                    channels: vec![channel],
                }
            }
            DynamicImage::ImageLumaA8(buffer) => {
                let mut luma = HistogramChannel::new("Luma");
                let mut alpha = HistogramChannel::new("Alpha");
                for chunk in buffer.as_raw().chunks_exact(2) {
                    luma.bins[chunk[0] as usize] += 1;
                    alpha.bins[chunk[1] as usize] += 1;
                }
                Self {
                    channels: vec![luma, alpha],
                }
            }
            DynamicImage::ImageRgb8(buffer) => {
                let mut r = HistogramChannel::new("Red");
                let mut g = HistogramChannel::new("Green");
                let mut b = HistogramChannel::new("Blue");
                for chunk in buffer.as_raw().chunks_exact(3) {
                    r.bins[chunk[0] as usize] += 1;
                    g.bins[chunk[1] as usize] += 1;
                    b.bins[chunk[2] as usize] += 1;
                }
                Self {
                    channels: vec![r, g, b],
                }
            }
            DynamicImage::ImageRgba8(buffer) => {
                let mut r = HistogramChannel::new("Red");
                let mut g = HistogramChannel::new("Green");
                let mut b = HistogramChannel::new("Blue");
                let mut a = HistogramChannel::new("Alpha");
                for chunk in buffer.as_raw().chunks_exact(4) {
                    r.bins[chunk[0] as usize] += 1;
                    g.bins[chunk[1] as usize] += 1;
                    b.bins[chunk[2] as usize] += 1;
                    a.bins[chunk[3] as usize] += 1;
                }
                Self {
                    channels: vec![r, g, b, a],
                }
            }
            other => {
                // Fallback to RGBA conversion for unsupported formats.
                let rgba = other.to_rgba8();
                let mut r = HistogramChannel::new("Red");
                let mut g = HistogramChannel::new("Green");
                let mut b = HistogramChannel::new("Blue");
                let mut a = HistogramChannel::new("Alpha");
                for chunk in rgba.as_raw().chunks_exact(4) {
                    r.bins[chunk[0] as usize] += 1;
                    g.bins[chunk[1] as usize] += 1;
                    b.bins[chunk[2] as usize] += 1;
                    a.bins[chunk[3] as usize] += 1;
                }
                Self {
                    channels: vec![r, g, b, a],
                }
            }
        }
    }
}

/// Histogram details for a single channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistogramChannel {
    pub name: &'static str,
    pub bins: [u32; 256],
}

impl HistogramChannel {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            bins: [0; 256],
        }
    }

    /// Total sample count represented by this channel.
    pub fn total(&self) -> u32 {
        self.bins.iter().sum()
    }
}

#[inline]
fn fill_channel(bins: &mut [u32; 256], data: &[u8]) {
    for &value in data {
        bins[value as usize] += 1;
    }
}
