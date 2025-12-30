#![doc = r#"
High-performance geometric image transformations for the XENO tooling stack.

`xeno-lib` provides safe, allocation-conscious implementations for the most
common geometric transforms and color adjustments required by higher-level
imaging workflows. The library builds on top of the `image` crate and embraces
data-parallel patterns via `rayon` to ensure excellent performance on modern
multi-core hardware.

## Provided operations

- `flip_horizontal` and `flip_vertical` mirror an image across the X or Y axis without modifying metadata.
- `rotate_90`, `rotate_180`, and `rotate_270` provide optimized, allocation-minimal quarter-turn rotations.
- `rotate` supports arbitrary angles with selectable interpolation (`Nearest` or `Bilinear`) and sensible bounds handling.
- `crop` extracts bounded sub-regions while preserving the original pixel format.
- `resize_exact`, `resize_by_percent`, and helpers resize images with bilinear or nearest-neighbour sampling.
- `grayscale`, `invert`, `adjust_brightness`, `adjust_contrast`, `adjust_saturation`, `adjust_hue`, `adjust_exposure`, and `adjust_gamma`
  provide tone and color control while respecting alpha channels.
- `gaussian_blur`, `unsharp_mask`, `edge_detect`, `emboss`, and `sepia` deliver creative filter effects ready for pipelines.
- `overlay`, `watermark`, `border`, and `frame` simplify compositing and presentation tasks.
- `image_info`, `histogram`, and `sniff_format` expose analysis utilities for metadata inspection and diagnostics.

Every operation accepts an immutable `DynamicImage` reference and returns a freshly
allocated `DynamicImage`, ensuring pure functional semantics suitable for pipeline
composition.

## Error handling

All APIs return `Result<T, TransformError>`. Errors are reported for unsupported pixel
formats, invalid rotation angles, and out-of-bounds crop requests.

## Testing & benchmarking

Unit tests and Criterion benchmarks can be executed locally:

- `cargo test` validates behaviour across RGB, RGBA, and grayscale samples.
- `cargo bench --bench transforms` measures performance on 10‑megapixel RGBA inputs and
  highlights regressions early in development.

## Known limitations

- Only 8-bit grayscale and RGB(A) pixel formats are currently supported; higher bit
  depths will be added in a follow-up phase.
- Arbitrary-angle rotation fills uncovered regions with zeroed pixels (transparent for RGBA,
  black otherwise). Future work may expose configurable background fills.
- SIMD-accelerated resizing via `fast_image_resize` is feature-gated but not yet wired into
  the core API.
"#]

pub mod adjustments;
pub mod agent;
pub mod analysis;
pub mod composite;
pub mod error;
pub mod filters;
pub mod transforms;

#[cfg(feature = "background-removal")]
pub mod background;

#[cfg(feature = "upscale")]
pub mod upscale;

#[cfg(feature = "face-restore")]
pub mod face_restore;

#[cfg(feature = "colorize")]
pub mod colorize;

#[cfg(feature = "frame-interpolate")]
pub mod frame_interpolate;

#[cfg(feature = "inpaint")]
pub mod inpaint;

#[cfg(feature = "face-detect")]
pub mod face_detect;

#[cfg(feature = "depth")]
pub mod depth;

#[cfg(feature = "transcribe")]
pub mod transcribe;

#[cfg(feature = "audio-separate")]
pub mod audio_separate;

#[cfg(any(feature = "video", feature = "video-encode"))]
pub mod video;

#[cfg(feature = "audio")]
pub mod audio;

#[cfg(feature = "text-overlay")]
pub mod text;

pub use crate::adjustments::{
    adjust_brightness, adjust_contrast, adjust_exposure, adjust_gamma, adjust_hue,
    adjust_saturation, grayscale, invert,
};
pub use crate::analysis::{
    Histogram, HistogramChannel, ImageInfo, histogram, image_info, sniff_format,
};
pub use crate::composite::{border, frame, overlay, watermark};
pub use crate::error::TransformError;
pub use crate::filters::{
    edge_detect, emboss, gaussian_blur, sepia, unsharp_mask,
    vignette, denoise, chromakey, remove_green_screen, remove_blue_screen,
    deinterlace, posterize, solarize, color_temperature, tint, vibrance,
};
pub use crate::transforms::{
    affine_transform, align, autocrop, batch_transform, center_on_canvas, crop, crop_center,
    crop_percentage, crop_to_aspect, crop_to_content, downscale, expand_canvas, flip_both,
    flip_horizontal, flip_vertical, get_background, get_interpolation, get_optimize_memory,
    get_preserve_alpha, homography, load_sequence, optimize_memory, pad, pad_to_aspect,
    pad_to_size, parallel_batch, perspective_correct, perspective_transform, pipeline_transform,
    preserve_alpha, resize, resize_by_percent, resize_cover, resize_exact, resize_fill,
    resize_fit, resize_to_fit, resize_to_height, resize_to_width, rotate, rotate_90,
    rotate_90_ccw, rotate_90_cw, rotate_180, rotate_270, rotate_270_cw, rotate_bounded,
    rotate_cropped, save_sequence, scale, scale_height, scale_width, sequence_info,
    sequence_transform, set_background, set_interpolation, shear_horizontal, shear_vertical,
    stream_transform, thumbnail, translate, transpose, transverse, trim, upscale,
    validate_sequence, Alignment, CropAnchor, Interpolation, SequenceInfo, TransformPipeline,
};

#[cfg(feature = "background-removal")]
pub use crate::background::{
    load_model, remove_background, remove_background_batch, BackgroundRemovalConfig, ModelSession,
};

#[cfg(feature = "upscale")]
pub use crate::upscale::{
    load_upscaler, upscale as ai_upscale, upscale_batch, upscale_quick,
    UpscaleConfig, UpscaleModel, UpscalerSession,
};

#[cfg(feature = "video")]
pub use crate::video::{
    detect_format_from_extension, is_supported_extension, AudioCodec, ContainerFormat, VideoCodec,
    VideoError, VideoFrame, VideoMetadata, VideoResult, SUPPORTED_EXTENSIONS,
};

#[cfg(feature = "video-encode")]
pub use crate::video::{
    encode_to_ivf, encode_to_mp4, Av1Encoder, Av1EncoderConfig, EncodingSpeed,
};

#[cfg(feature = "video-decode")]
pub use crate::video::decode::{
    best_decoder_for, decode_ivf, extract_frames, DecodeCodec, DecodedFrame, DecoderBackend,
    DecoderCapabilities, DecoderConfig, OutputFormat,
};

#[cfg(feature = "audio")]
pub use crate::audio::{
    decode_file as decode_audio_file, extract_audio_from_video, AudioCodec as AudioCodecType,
    AudioDecoder, AudioError, AudioFormat, AudioInfo, AudioResult, AudioSamples, DecodedAudio,
};

#[cfg(feature = "text-overlay")]
pub use crate::text::{
    draw_text, draw_text_batch, Anchor, TextConfig, TextDimensions, TextOverlay,
};

// AI Feature Exports

#[cfg(feature = "face-restore")]
pub use crate::face_restore::{
    load_restorer, restore_faces, restore_faces_batch,
    FaceRestoreConfig, FaceRestoreModel, FaceRestorerSession,
};

#[cfg(feature = "colorize")]
pub use crate::colorize::{
    colorize, colorize_batch, colorize_quick, load_colorizer,
    ColorizeConfig, ColorizeModel, ColorizerSession,
};

#[cfg(feature = "frame-interpolate")]
pub use crate::frame_interpolate::{
    interpolate_frame, interpolate_frames, interpolate_quick, is_scene_change, load_interpolator,
    InterpolationConfig, InterpolationModel, InterpolatorSession,
};

#[cfg(feature = "inpaint")]
pub use crate::inpaint::{
    create_mask, inpaint, inpaint_quick, load_inpainter,
    InpaintConfig, InpaintModel, InpainterSession, MaskRegion,
};

#[cfg(feature = "face-detect")]
pub use crate::face_detect::{
    crop_faces, detect_faces, detect_faces_quick, load_detector, visualize_detections,
    DetectedFace, FaceDetectConfig, FaceDetectModel, FaceDetectorSession, FaceLandmarks,
};

#[cfg(feature = "depth")]
pub use crate::depth::{
    apply_depth_blur, estimate_depth, estimate_depth_quick, load_depth_estimator,
    DepthConfig, DepthMap, DepthModel, DepthSession,
};

#[cfg(feature = "transcribe")]
pub use crate::transcribe::{
    to_srt, to_vtt, transcribe, transcribe_quick, load_transcriber,
    Language, Transcript, TranscribeConfig, TranscribeModel, TranscriberSession, TranscriptSegment,
};

#[cfg(feature = "audio-separate")]
pub use crate::audio_separate::{
    isolate_vocals, remove_vocals, separate, separate_quick, load_separator,
    AudioStem, SeparatedAudio, SeparationConfig, SeparationModel, SeparatorSession, StereoAudio,
};

// New Feature Modules

#[cfg(feature = "subtitle")]
pub mod subtitle;

#[cfg(feature = "style-transfer")]
pub mod style_transfer;

#[cfg(feature = "ocr")]
pub mod ocr;

#[cfg(feature = "pose")]
pub mod pose;

#[cfg(feature = "face-analysis")]
pub mod face_analysis;

#[cfg(feature = "qrcode")]
pub mod qrcode;

#[cfg(feature = "quality")]
pub mod quality;

#[cfg(feature = "document")]
pub mod document;

// New Feature Exports

#[cfg(feature = "subtitle")]
pub use crate::subtitle::{
    Subtitles, SubtitleCue, SubtitleFormat, SubtitleStyle, SubtitlePosition,
    parse_srt, parse_vtt, parse_ass, render_subtitle, render_cue,
};

#[cfg(feature = "style-transfer")]
pub use crate::style_transfer::{
    stylize, stylize_blended, load_style_model,
    PretrainedStyle, StyleConfig, StyleSession,
};

#[cfg(feature = "ocr")]
pub use crate::ocr::{
    extract_text, extract_text_quick, load_ocr_model, visualize_ocr,
    OcrConfig, OcrModel, OcrResult, OcrSession, TextBlock, TextBox,
};

#[cfg(feature = "pose")]
pub use crate::pose::{
    detect_pose, detect_poses, load_pose_model, visualize_pose,
    BodyKeypoint, DetectedPose, PoseConfig, PoseModel, PoseSession,
};

#[cfg(feature = "face-analysis")]
pub use crate::face_analysis::{
    analyze_face, analyze_faces, load_analyzer, visualize_analysis,
    Emotion, FaceAnalysisConfig, FaceAnalysisResult, FaceAnalyzerSession, Gender,
};

#[cfg(feature = "qrcode")]
pub use crate::qrcode::{
    decode_qr, generate_qr, generate_barcode,
    BarcodeFormat, DecodeResult, ErrorCorrection, QrConfig, QrError,
};

#[cfg(feature = "quality")]
pub use crate::quality::{
    assess_quality, find_best_image, is_acceptable_quality, rank_images,
    QualityConfig, QualityGrade, QualityIssue, QualityMetrics, QualityReport,
};

#[cfg(feature = "document")]
pub use crate::document::{
    deskew, detect_skew, process_document, quick_deskew, scan_enhance,
    DocumentConfig, DocumentResult, ProcessingStats,
};

#[cfg(feature = "audio")]
pub use crate::audio::{
    // Audio effects
    effects::{
        reverb, equalizer, pitch_shift, delay, distortion, chorus, flanger, noise_gate,
        ReverbConfig, EqConfig, EqBand, FilterType, PitchShiftConfig, DelayConfig,
        DistortionConfig, ChorusConfig, FlangerConfig, GateConfig, ReverbPreset,
    },
    // Audio visualization
    visualization::{
        render_waveform, render_spectrum, render_spectrogram,
        WaveformConfig, SpectrumConfig, ColorMap,
    },
};

#[cfg(feature = "video")]
pub use crate::video::edit::{
    trim_video, cut_video, concat_videos, change_speed,
    TrimConfig, ConcatConfig, SpeedConfig, VideoSegment,
};
