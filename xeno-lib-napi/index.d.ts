/* Auto-generated TypeScript declarations for @xeno/lib N-API bindings.
 *
 * These types mirror the #[napi] exports in the Rust source.
 * When building with `napi build`, napi-rs will regenerate this file
 * automatically from the Rust source — this hand-written version serves
 * as documentation and a fallback.
 */

// ---------------------------------------------------------------------------
// Image Processing (Priority 1 — xeno-pixel)
// ---------------------------------------------------------------------------

/**
 * Apply a Gaussian blur to an RGBA image.
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @param radius - Blur sigma (must be > 0, finite)
 * @returns RGBA u8 buffer of the blurred image (same dimensions)
 * @throws If buffer size !== width * height * 4, or dimensions are zero, or radius is invalid
 */
export function applyGaussianBlur(buffer: Buffer, width: number, height: number, radius: number): Buffer;

/**
 * Sharpen an RGBA image using unsharp mask.
 * @param buffer   - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width    - Image width in pixels (must be > 0)
 * @param height   - Image height in pixels (must be > 0)
 * @param strength - Sharpening sigma (must be > 0, finite)
 * @returns RGBA u8 buffer of the sharpened image (same dimensions)
 * @throws If buffer size !== width * height * 4, or dimensions are zero, or strength is invalid
 */
export function applySharpen(buffer: Buffer, width: number, height: number, strength: number): Buffer;

/**
 * Adjust brightness of an RGBA image.
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @param amount - Brightness adjustment (typical: -100 to 100, must be finite)
 * @returns RGBA u8 buffer with adjusted brightness (same dimensions)
 * @throws If buffer size !== width * height * 4, or dimensions are zero, or amount is NaN/Infinity
 */
export function adjustBrightness(buffer: Buffer, width: number, height: number, amount: number): Buffer;

/**
 * Adjust contrast of an RGBA image.
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @param amount - Contrast adjustment (typical: -100 to 100, must be finite)
 * @returns RGBA u8 buffer with adjusted contrast (same dimensions)
 * @throws If buffer size !== width * height * 4, or dimensions are zero, or amount is NaN/Infinity
 */
export function adjustContrast(buffer: Buffer, width: number, height: number, amount: number): Buffer;

/**
 * Rotate the hue of an RGBA image.
 * @param buffer  - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width   - Image width in pixels (must be > 0)
 * @param height  - Image height in pixels (must be > 0)
 * @param degrees - Hue rotation angle in degrees (must be finite)
 * @returns RGBA u8 buffer with rotated hue (same dimensions)
 * @throws If buffer size !== width * height * 4, or dimensions are zero, or degrees is NaN/Infinity
 */
export function adjustHue(buffer: Buffer, width: number, height: number, degrees: number): Buffer;

/**
 * Adjust saturation of an RGBA image.
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @param amount - Saturation adjustment (typical: -100 to 100, must be finite)
 * @returns RGBA u8 buffer with adjusted saturation (same dimensions)
 * @throws If buffer size !== width * height * 4, or dimensions are zero, or amount is NaN/Infinity
 */
export function adjustSaturation(buffer: Buffer, width: number, height: number, amount: number): Buffer;

/**
 * Resize an RGBA image to new dimensions.
 * @param buffer    - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width     - Current image width in pixels (must be > 0)
 * @param height    - Current image height in pixels (must be > 0)
 * @param newWidth  - Target width in pixels (must be > 0)
 * @param newHeight - Target height in pixels (must be > 0)
 * @param method    - Interpolation method: "nearest" or "bilinear" (default)
 * @returns RGBA u8 buffer of the resized image (newWidth * newHeight * 4 bytes)
 * @throws If any dimension is zero, or buffer size !== width * height * 4
 */
export function resizeImage(
  buffer: Buffer,
  width: number,
  height: number,
  newWidth: number,
  newHeight: number,
  method: string,
): Buffer;

// ---------------------------------------------------------------------------
// Image Encoding (Priority 1 — xeno-pixel export)
// ---------------------------------------------------------------------------

/**
 * Encode an RGBA buffer to PNG format.
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @returns Buffer containing compressed PNG file bytes
 * @throws If buffer size !== width * height * 4, or dimensions are zero
 */
export function encodePng(buffer: Buffer, width: number, height: number): Buffer;

/**
 * Encode an RGBA buffer to JPEG format. Alpha channel is discarded.
 * @param buffer  - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width   - Image width in pixels (must be > 0)
 * @param height  - Image height in pixels (must be > 0)
 * @param quality - JPEG quality 1-100 (clamped if out of range)
 * @returns Buffer containing compressed JPEG file bytes
 * @throws If buffer size !== width * height * 4, or dimensions are zero
 */
export function encodeJpeg(buffer: Buffer, width: number, height: number, quality: number): Buffer;

/**
 * Encode an RGBA buffer to WebP format.
 * @param buffer  - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width   - Image width in pixels (must be > 0)
 * @param height  - Image height in pixels (must be > 0)
 * @param quality - WebP quality 1-100 (clamped if out of range)
 * @returns Buffer containing compressed WebP file bytes
 * @throws If buffer size !== width * height * 4, or dimensions are zero
 */
export function encodeWebp(buffer: Buffer, width: number, height: number, quality: number): Buffer;

/**
 * Encode an RGBA buffer to AVIF format.
 * @param buffer  - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width   - Image width in pixels (must be > 0)
 * @param height  - Image height in pixels (must be > 0)
 * @param quality - AVIF quality 1-100 (clamped if out of range)
 * @returns Buffer containing compressed AVIF file bytes
 * @throws If buffer size !== width * height * 4, or dimensions are zero
 */
export function encodeAvif(buffer: Buffer, width: number, height: number, quality: number): Buffer;

/**
 * Decode AVIF data to an RGBA buffer.
 * @param data - Raw AVIF file bytes
 * @returns RGBA u8 buffer of the decoded image
 * @throws If data is empty or invalid AVIF
 */
export function decodeAvif(data: Buffer): Buffer;

// ---------------------------------------------------------------------------
// Audio Processing (Priority 2 — xeno-sound)
// ---------------------------------------------------------------------------

/** Decoded audio data. */
export interface AudioData {
  /** Interleaved f32 PCM samples (-1.0 to 1.0), encoded as f64 for JS. */
  samples: number[];
  /** Sample rate in Hz (e.g., 44100, 48000). */
  sampleRate: number;
  /** Number of channels (1 = mono, 2 = stereo). */
  channels: number;
  /** Total duration in milliseconds. */
  durationMs: number;
}

/**
 * Decode an audio file to raw PCM samples.
 * Supports MP3, AAC, FLAC, Vorbis, ALAC, WAV, AIFF, Ogg, etc.
 * @param filePath - Path to audio file (must exist and be non-empty)
 * @returns AudioData with interleaved PCM samples
 * @throws If file path is empty, file doesn't exist, or format is unsupported
 */
export function decodeAudio(filePath: string): Promise<AudioData>;

/**
 * Encode PCM samples to WAV format.
 * @param samples    - f64 array of interleaved PCM samples (must not be empty, no NaN/Infinity)
 * @param sampleRate - Sample rate in Hz (1-384000)
 * @param channels   - Number of channels (1-32)
 * @param bitDepth   - Bits per sample (8, 16, 24, or 32)
 * @returns Buffer containing complete WAV file bytes
 * @throws If samples empty, sample rate/channels out of range, invalid bit depth, or NaN/Infinity samples
 */
export function encodeWav(samples: Float64Array, sampleRate: number, channels: number, bitDepth: number): Buffer;

/**
 * Encode PCM samples to AAC format (lossy compression for video export).
 * Currently a stub - returns an error until fdk-aac C bindings are integrated.
 * Use Opus encoding as an alternative.
 * @param samples    - f64 array of interleaved PCM samples (must not be empty, no NaN/Infinity)
 * @param sampleRate - Sample rate in Hz (1-96000)
 * @param channels   - Number of channels (1-8)
 * @param bitrate    - Target bitrate in bps (e.g., 128000, 192000, 256000)
 * @returns Buffer containing AAC encoded bytes
 * @throws If samples empty, parameters out of range, or encoder not available
 */
export function encodeAac(samples: Float64Array, sampleRate: number, channels: number, bitrate: number): Buffer;

/**
 * Encode PCM samples to FLAC format (lossless compression).
 * @param samples    - f64 array of interleaved PCM samples (must not be empty, no NaN/Infinity)
 * @param sampleRate - Sample rate in Hz (1-384000)
 * @param channels   - Number of channels (1-32)
 * @returns Buffer containing complete FLAC file bytes
 * @throws If samples empty, sample rate/channels out of range, or NaN/Infinity samples
 */
export function encodeFlac(samples: Float64Array, sampleRate: number, channels: number): Buffer;

// ---------------------------------------------------------------------------
// AI Models (Priority 3 — all apps)
// ---------------------------------------------------------------------------

/**
 * Remove the background from an RGBA image using AI (BiRefNet).
 * Model is lazily loaded on first call and cached.
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @returns RGBA buffer with background made transparent
 * @throws If buffer size mismatch, zero dimensions, model missing from ~/.xeno-lib/models/, or inference failure
 */
export function removeBackground(buffer: Buffer, width: number, height: number): Promise<Buffer>;

/**
 * Upscale an RGBA image using AI (Real-ESRGAN).
 * Model is lazily loaded on first call and cached.
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @param scale  - Upscale factor (must be 2 or 4)
 * @returns Upscaled RGBA buffer (width*scale x height*scale)
 * @throws If buffer size mismatch, zero dimensions, invalid scale, model missing, or inference failure
 */
export function upscaleImage(buffer: Buffer, width: number, height: number, scale: number): Promise<Buffer>;

/**
 * Denoise an RGBA image using a spatial denoise filter.
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @returns Denoised RGBA buffer (same dimensions)
 * @throws If buffer size mismatch, or zero dimensions
 */
export function denoiseImage(buffer: Buffer, width: number, height: number): Promise<Buffer>;

/**
 * Denoise audio using limiter + peak normalization pipeline.
 * @param samples    - f64 array of interleaved PCM samples (must not be empty, no NaN/Infinity)
 * @param sampleRate - Sample rate in Hz (1-384000)
 * @returns Denoised f32 PCM samples as Float64Array
 * @throws If samples empty, sample rate out of range, or NaN/Infinity samples
 */
export function denoiseAudio(samples: Float64Array, sampleRate: number): Promise<Float64Array>;

/** A single transcription segment with timestamps. */
export interface TranscriptionSegment {
  /** Start time in milliseconds. */
  startMs: number;
  /** End time in milliseconds. */
  endMs: number;
  /** Transcribed text for this segment. */
  text: string;
}

/** Full transcription result. */
export interface TranscriptionResult {
  /** Individual segments with timestamps. */
  segments: TranscriptionSegment[];
  /** Detected language (ISO 639-1 code, e.g., "en"). Empty string if not detected. */
  language: string;
}

/** Stem separation result. */
export interface StemSeparationResult {
  /** Interleaved stereo vocal track samples (f64). */
  vocals: number[];
  /** Interleaved stereo drum track samples (f64). */
  drums: number[];
  /** Interleaved stereo bass track samples (f64). */
  bass: number[];
  /** Interleaved stereo other instruments samples (f64). */
  other: number[];
  /** Sample rate in Hz. */
  sampleRate: number;
}

/**
 * Separate audio into stems (vocals, drums, bass, other) using AI (HTDemucs).
 * Model is lazily loaded on first call and cached.
 * @param filePath - Path to audio file (must exist and be non-empty)
 * @returns StemSeparationResult with four stereo stems
 * @throws If file path empty, file missing, decode failure, model missing, or inference failure
 */
export function separateStems(filePath: string): Promise<StemSeparationResult>;

/**
 * Transcribe audio from a file using AI (Whisper).
 * Model is lazily loaded on first call and cached.
 * @param filePath - Path to audio file (must exist and be non-empty)
 * @returns TranscriptionResult with timestamped segments and detected language
 * @throws If file path empty, file missing, decode failure, model missing, or inference failure
 */
export function transcribeAudio(filePath: string): Promise<TranscriptionResult>;

/**
 * Interpolate a frame between two RGBA frames using AI (RIFE).
 * Model is lazily loaded on first call and cached.
 * @param frame1  - First RGBA u8 frame (4 bytes per pixel, row-major)
 * @param frame2  - Second RGBA u8 frame (same dimensions as frame1)
 * @param width   - Frame width in pixels (must be > 0)
 * @param height  - Frame height in pixels (must be > 0)
 * @param factor  - Interpolation position (0.0 = frame1, 1.0 = frame2, must be in [0.0, 1.0])
 * @returns Interpolated RGBA buffer (same dimensions as input)
 * @throws If buffer size mismatch, zero dimensions, factor out of range, model missing, or inference failure
 */
export function interpolateFrames(
  frame1: Buffer,
  frame2: Buffer,
  width: number,
  height: number,
  factor: number,
): Promise<Buffer>;

// ---------------------------------------------------------------------------
// AI Models — Extended (all 17 models exposed via N-API)
// ---------------------------------------------------------------------------

/**
 * Restore faces in an RGBA image using AI (GFPGAN).
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @returns RGBA u8 buffer with restored faces (same dimensions)
 * @throws If buffer size mismatch, zero dimensions, model missing, or inference failure
 */
export function restoreFaces(buffer: Buffer, width: number, height: number): Promise<Buffer>;

/**
 * Colorize a grayscale RGBA image using AI (DDColor).
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @returns Colorized RGBA u8 buffer (same dimensions)
 * @throws If buffer size mismatch, zero dimensions, model missing, or inference failure
 */
export function colorize(buffer: Buffer, width: number, height: number): Promise<Buffer>;

/**
 * Inpaint (fill) masked regions of an RGBA image using AI (LaMa).
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param mask   - Single-channel u8 mask (width*height bytes; 255=inpaint, 0=keep)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @returns Inpainted RGBA u8 buffer (same dimensions)
 * @throws If buffer/mask size mismatch, zero dimensions, model missing, or inference failure
 */
export function inpaint(buffer: Buffer, mask: Buffer, width: number, height: number): Promise<Buffer>;

/** A detected face bounding box. */
export interface DetectedFace {
  /** Bounding box x coordinate (pixels). */
  x: number;
  /** Bounding box y coordinate (pixels). */
  y: number;
  /** Bounding box width (pixels). */
  width: number;
  /** Bounding box height (pixels). */
  height: number;
  /** Detection confidence (0.0 to 1.0). */
  confidence: number;
}

/**
 * Detect faces in an RGBA image using AI (SCRFD).
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @returns Array of detected face bounding boxes
 * @throws If buffer size mismatch, zero dimensions, model missing, or inference failure
 */
export function detectFaces(buffer: Buffer, width: number, height: number): Promise<DetectedFace[]>;

/** Depth estimation result. */
export interface DepthMapResult {
  /** Depth values as number array (0.0=near, 1.0=far), row-major. */
  data: number[];
  /** Width of the depth map. */
  width: number;
  /** Height of the depth map. */
  height: number;
}

/**
 * Estimate depth from an RGBA image using AI (Depth Anything).
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @returns DepthMapResult with depth values
 * @throws If buffer size mismatch, zero dimensions, model missing, or inference failure
 */
export function estimateDepth(buffer: Buffer, width: number, height: number): Promise<DepthMapResult>;

/**
 * Apply neural style transfer to an RGBA image.
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @param style  - Style name: "mosaic", "candy", "rain_princess", "udnie", "pointilism"
 * @returns Stylized RGBA u8 buffer (same dimensions)
 * @throws If buffer size mismatch, zero dimensions, model missing, or inference failure
 */
export function styleTransfer(buffer: Buffer, width: number, height: number, style: string): Promise<Buffer>;

/** OCR text block. */
export interface OcrTextBlock {
  /** Recognized text. */
  text: string;
  /** Bounding box x (pixels). */
  x: number;
  /** Bounding box y (pixels). */
  y: number;
  /** Bounding box width (pixels). */
  width: number;
  /** Bounding box height (pixels). */
  height: number;
  /** Recognition confidence (0.0 to 1.0). */
  confidence: number;
}

/** OCR result. */
export interface OcrResult {
  /** All recognized text blocks. */
  blocks: OcrTextBlock[];
  /** Full concatenated text. */
  fullText: string;
}

/**
 * Extract text from an RGBA image using AI (PaddleOCR).
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @returns OcrResult with text blocks and full text
 * @throws If buffer size mismatch, zero dimensions, model missing, or inference failure
 */
export function extractText(buffer: Buffer, width: number, height: number): Promise<OcrResult>;

/** Body pose keypoint. */
export interface PoseKeypoint {
  /** Keypoint name (e.g., "nose", "left_shoulder"). */
  name: string;
  /** X coordinate (pixels). */
  x: number;
  /** Y coordinate (pixels). */
  y: number;
  /** Detection confidence (0.0 to 1.0). */
  confidence: number;
}

/**
 * Detect body poses in an RGBA image using AI (MoveNet).
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @returns Array of pose keypoint arrays (one per detected person)
 * @throws If buffer size mismatch, zero dimensions, model missing, or inference failure
 */
export function detectPoses(buffer: Buffer, width: number, height: number): Promise<PoseKeypoint[][]>;

/** Face analysis result. */
export interface FaceAnalysisResult {
  /** Estimated age. */
  age: number;
  /** Detected gender ("male" or "female"). */
  gender: string;
  /** Gender confidence (0.0 to 1.0). */
  genderConfidence: number;
  /** Dominant emotion. */
  emotion: string;
  /** Emotion confidence (0.0 to 1.0). */
  emotionConfidence: number;
}

/**
 * Analyze faces in an RGBA image (age, gender, emotion).
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel, row-major)
 * @param width  - Image width in pixels (must be > 0)
 * @param height - Image height in pixels (must be > 0)
 * @returns Array of face analysis results
 * @throws If buffer size mismatch, zero dimensions, model missing, or inference failure
 */
export function analyzeFaces(buffer: Buffer, width: number, height: number): Promise<FaceAnalysisResult[]>;

// ---------------------------------------------------------------------------
// Model Management (native Rust bindings)
// ---------------------------------------------------------------------------

/**
 * Get the path to the model directory (~/.xeno-lib/models/).
 * Creates the directory if it doesn't exist.
 * @returns Absolute path to model directory
 */
export function getModelDir(): string;

/**
 * Check if a specific model file exists in the model directory.
 * @param modelName - Model file name (e.g., "birefnet-general.onnx")
 * @returns true if the model file exists
 */
export function isModelAvailable(modelName: string): boolean;

/**
 * List all .onnx model files in the model directory.
 * @returns Array of model file names
 */
export function listAvailableModels(): string[];

/**
 * Get the list of model files required for a specific AI function.
 * @param functionName - Name of the AI function (e.g., "removeBackground")
 * @returns Array of model file names needed
 */
export function getRequiredModels(functionName: string): string[];

/**
 * Get the download URL for a model file.
 * @param modelName - Model file name (e.g., "birefnet-general.onnx")
 * @returns Full download URL on updates.xenostudio.ai
 */
export function getModelDownloadUrl(modelName: string): string;

/**
 * Check which models are missing for a given AI function.
 * @param functionName - Name of the AI function
 * @returns Array of missing model file names (empty if all present)
 */
export function getMissingModels(functionName: string): string[];

// ---------------------------------------------------------------------------
// Model Download Manager (JavaScript)
// ---------------------------------------------------------------------------

export { ModelManager } from './model-manager';
export type {
  ModelInfo,
  ModelManifest,
  DownloadProgress,
  ModelStatus,
  BatchDownloadProgress,
  DiskUsage,
  ModelManagerOptions,
} from './model-manager';

// ---------------------------------------------------------------------------
// Hardware Detection (Phase 2 — hardware encoder/decoder capability)
// ---------------------------------------------------------------------------

/** NVIDIA GPU information. */
export interface NvidiaInfoJs {
  /** GPU device name (e.g., "NVIDIA GeForce RTX 4090"). */
  gpuName: string;
  /** Driver version string (e.g., "560.35"). */
  driverVersion: string;
  /** Whether NVENC (hardware video encoding) is available. */
  nvencAvailable: boolean;
  /** Whether NVDEC (hardware video decoding) is available. */
  nvdecAvailable: boolean;
  /** Video RAM in megabytes. */
  vramMb: number;
}

/** Intel GPU information. */
export interface IntelInfoJs {
  /** GPU device name. */
  gpuName: string;
  /** Whether Intel Quick Sync Video is available. */
  qsvAvailable: boolean;
}

/** AMD GPU information. */
export interface AmdInfoJs {
  /** GPU device name. */
  gpuName: string;
  /** Whether AMD Advanced Media Framework is available. */
  amfAvailable: boolean;
}

/** Hardware acceleration capabilities detected on the current system. */
export interface HardwareCapabilitiesJs {
  /** NVIDIA GPU info, or null/undefined if no NVIDIA GPU is detected. */
  nvidia?: NvidiaInfoJs;
  /** Intel GPU info, or null/undefined if no Intel GPU is detected. */
  intel?: IntelInfoJs;
  /** AMD GPU info, or null/undefined if no AMD GPU is detected. */
  amd?: AmdInfoJs;
}

/** Codec capability for a single codec. */
export interface CodecCapabilityJs {
  /** Whether software encoding is available. */
  encode: boolean;
  /** Whether software decoding is available. */
  decode: boolean;
  /** Whether hardware-accelerated encoding is available. */
  hardwareEncode: boolean;
  /** Whether hardware-accelerated decoding is available. */
  hardwareDecode: boolean;
}

/** Codec support for all known codecs. */
export interface CodecSupportJs {
  /** H.264/AVC support. */
  h264: CodecCapabilityJs;
  /** H.265/HEVC support. */
  h265: CodecCapabilityJs;
  /** AV1 support. */
  av1: CodecCapabilityJs;
  /** VP9 support. */
  vp9: CodecCapabilityJs;
}

// ---------------------------------------------------------------------------
// Video Decode/Encode (Phase 2 — HEVC + NVENC)
// ---------------------------------------------------------------------------

/**
 * Decode a single HEVC/H.265 frame from raw NAL unit data to RGBA pixels.
 *
 * Takes Annex B format HEVC data and returns an RGBA buffer.
 * Currently returns an error until libde265 is linked. The NAL unit parsing
 * and YUV→RGBA conversion pipeline are fully implemented and ready.
 *
 * @param data - Raw HEVC NAL unit data (Annex B format)
 * @returns RGBA u8 buffer of the decoded frame
 * @throws If data is empty, or decoder is not yet available
 */
export function decodeHevcFrame(data: Buffer): Promise<Buffer>;

/**
 * Check if NVENC hardware encoding is available on the current system.
 * Returns true if the NVIDIA NVENC library can be loaded.
 * This is a lightweight check that does not create an encoder session.
 * @returns true if NVENC is available
 */
export function isNvencAvailable(): boolean;

// ---------------------------------------------------------------------------
// Hardware Detection (Phase 2 — hardware encoder/decoder capability)
// ---------------------------------------------------------------------------

/**
 * Detect available hardware acceleration on the current system.
 * Dynamically loads vendor-specific libraries to detect GPUs.
 * Never throws — missing hardware is reported as undefined fields.
 * @returns Hardware capabilities with NVIDIA/Intel/AMD info
 */
export function detectHardware(): HardwareCapabilitiesJs;

/**
 * Get supported codecs based on compiled features and detected hardware.
 * @returns Codec support information for H.264, H.265, AV1, VP9
 */
export function getSupportedCodecs(): CodecSupportJs;

// ---------------------------------------------------------------------------
// LaTeX Compilation (Tectonic engine — no external TeX distribution needed)
// ---------------------------------------------------------------------------

/**
 * Compile LaTeX source code to PDF using the embedded Tectonic engine.
 * No external TeX distribution (MiKTeX, TeX Live, etc.) is required.
 *
 * Note: The first invocation may be slower as Tectonic downloads
 * required LaTeX packages (~200MB cached locally).
 *
 * @param texSource - LaTeX source code string
 * @returns The compiled PDF as a Buffer
 * @throws If the LaTeX source has errors or compilation fails
 */
export function compileLatex(texSource: string): Promise<Buffer>;
