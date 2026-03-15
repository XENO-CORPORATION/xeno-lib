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
 * @param buffer - RGBA u8 pixel data (4 bytes per pixel)
 * @param width  - Image width in pixels
 * @param height - Image height in pixels
 * @param radius - Blur sigma (> 0)
 * @returns RGBA u8 buffer of the blurred image
 */
export function applyGaussianBlur(buffer: Buffer, width: number, height: number, radius: number): Buffer;

/**
 * Sharpen an RGBA image using unsharp mask.
 * @param strength - Sharpening sigma (> 0)
 */
export function applySharpen(buffer: Buffer, width: number, height: number, strength: number): Buffer;

/**
 * Adjust brightness of an RGBA image.
 * @param amount - Brightness adjustment [-100, 100]
 */
export function adjustBrightness(buffer: Buffer, width: number, height: number, amount: number): Buffer;

/**
 * Adjust contrast of an RGBA image.
 * @param amount - Contrast adjustment [-100, 100]
 */
export function adjustContrast(buffer: Buffer, width: number, height: number, amount: number): Buffer;

/**
 * Rotate the hue of an RGBA image.
 * @param degrees - Hue rotation angle in degrees
 */
export function adjustHue(buffer: Buffer, width: number, height: number, degrees: number): Buffer;

/**
 * Adjust saturation of an RGBA image.
 * @param amount - Saturation adjustment [-100, 100]
 */
export function adjustSaturation(buffer: Buffer, width: number, height: number, amount: number): Buffer;

/**
 * Resize an RGBA image.
 * @param method - Interpolation method: "nearest" or "bilinear"
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

/** Encode an RGBA buffer to PNG. */
export function encodePng(buffer: Buffer, width: number, height: number): Buffer;

/** Encode an RGBA buffer to JPEG. quality: 1-100. */
export function encodeJpeg(buffer: Buffer, width: number, height: number, quality: number): Buffer;

/** Encode an RGBA buffer to WebP. quality: 1-100. */
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
 */
export function decodeAudio(filePath: string): Promise<AudioData>;

/**
 * Encode PCM samples to WAV format.
 * @param samples   - f64 array of interleaved PCM samples
 * @param sampleRate - Sample rate in Hz
 * @param channels   - Number of channels
 * @param bitDepth   - Bits per sample (8, 16, 24, or 32)
 * @returns Buffer containing WAV file bytes
 */
export function encodeWav(samples: Float64Array, sampleRate: number, channels: number, bitDepth: number): Buffer;

/**
 * Encode PCM samples to FLAC format.
 * @returns Buffer containing FLAC file bytes
 */
export function encodeFlac(samples: Float64Array, sampleRate: number, channels: number): Buffer;

// ---------------------------------------------------------------------------
// AI Models (Priority 3 — all apps)
// ---------------------------------------------------------------------------

/**
 * Remove the background from an RGBA image using AI (BiRefNet).
 * Model is lazily loaded on first call and cached.
 * @returns RGBA buffer with background made transparent
 */
export function removeBackground(buffer: Buffer, width: number, height: number): Promise<Buffer>;

/**
 * Upscale an RGBA image using AI (Real-ESRGAN).
 * @param scale - Upscale factor (2 or 4)
 * @returns Upscaled RGBA buffer
 */
export function upscaleImage(buffer: Buffer, width: number, height: number, scale: number): Promise<Buffer>;

/**
 * Denoise an RGBA image using a spatial denoise filter.
 * @returns Denoised RGBA buffer
 */
export function denoiseImage(buffer: Buffer, width: number, height: number): Promise<Buffer>;

/**
 * Denoise audio using audio filters.
 * @returns Denoised f32 PCM samples (as Float64Array)
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
  /** Detected language (if auto-detect was used). */
  language: string;
}

/** Stem separation result. */
export interface StemSeparationResult {
  /** Interleaved vocal track samples (f64). */
  vocals: number[];
  /** Interleaved drum track samples (f64). */
  drums: number[];
  /** Interleaved bass track samples (f64). */
  bass: number[];
  /** Interleaved other instruments samples (f64). */
  other: number[];
  /** Sample rate in Hz. */
  sampleRate: number;
}

/**
 * Separate audio into stems (vocals, drums, bass, other) using AI (HTDemucs).
 * Model is lazily loaded on first call and cached.
 */
export function separateStems(filePath: string): Promise<StemSeparationResult>;

/**
 * Transcribe audio from a file using AI (Whisper).
 * Model is lazily loaded on first call and cached.
 */
export function transcribeAudio(filePath: string): Promise<TranscriptionResult>;

/**
 * Interpolate a frame between two RGBA frames using AI (RIFE).
 * @param factor - Interpolation position (0.0 = frame1, 1.0 = frame2)
 * @returns Interpolated RGBA buffer
 */
export function interpolateFrames(
  frame1: Buffer,
  frame2: Buffer,
  width: number,
  height: number,
  factor: number,
): Promise<Buffer>;
