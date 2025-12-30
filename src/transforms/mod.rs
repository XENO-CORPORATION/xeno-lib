//! Core geometric transformation entry points and helpers.

mod affine;
mod alignment;
mod batch;
mod canvas;
mod config;
mod crop;
mod flip;
mod flip_simd;
mod interpolation;
mod matrix;
mod perspective;
mod resize;
mod rotate;
mod sequence;
pub(crate) mod utils;

pub use affine::{affine_transform, shear_horizontal, shear_vertical, translate};
pub use alignment::{align, center_on_canvas, Alignment};
pub use batch::{
    batch_transform, parallel_batch, pipeline_transform, sequence_transform, stream_transform,
    TransformPipeline,
};
pub use canvas::{expand_canvas, pad, pad_to_aspect, pad_to_size, trim};
pub use config::{
    get_background, get_interpolation, get_optimize_memory, get_preserve_alpha, optimize_memory,
    preserve_alpha, set_background, set_interpolation,
};
pub use crop::{
    autocrop, crop, crop_center, crop_percentage, crop_to_aspect, crop_to_content, CropAnchor,
};
pub use flip::{flip_both, flip_horizontal, flip_vertical};
pub use interpolation::Interpolation;
pub use matrix::{transpose, transverse};
pub use perspective::{homography, perspective_correct, perspective_transform};
pub use resize::{
    downscale, resize, resize_by_percent, resize_cover, resize_exact, resize_fill, resize_fit,
    resize_to_fit, resize_to_height, resize_to_width, scale, scale_height, scale_width, thumbnail,
    upscale,
};
pub use rotate::{
    rotate, rotate_90, rotate_90_ccw, rotate_90_cw, rotate_180, rotate_270, rotate_270_cw,
    rotate_bounded, rotate_cropped,
};
pub use sequence::{load_sequence, save_sequence, sequence_info, validate_sequence, SequenceInfo};

#[cfg(test)]
mod tests;
