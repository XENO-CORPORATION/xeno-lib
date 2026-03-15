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
