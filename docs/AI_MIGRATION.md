# AI Migration Manifest: xeno-lib -> xeno-rt

## Overview

This document tracks the migration of all AI inference code from `xeno-lib` (pure processing
library) to `xeno-rt` (unified inference runtime). After migration, `xeno-lib` will contain
ONLY pure processing: codecs, transforms, effects, capture, format I/O. All ONNX Runtime
inference will live in `xeno-rt`.

**Status**: Planning (code annotated with `TODO: MIGRATE TO XENO-RT` comments)

**Date**: 2026-03-20

---

## Architecture After Migration

```
xeno-rt (inference runtime)
  - LLM inference (GGUF, already there)
  - Task-specific ONNX models (migrating from xeno-lib)
  - GPU memory management
  - Model quantization tooling
  - Batch inference utilities
  - Model path resolution

xeno-lib (processing library)
  - Video encode/decode (rav1e, OpenH264, dav1d, NVDEC)
  - Audio decode/encode (symphonia, hound, flacenc, audiopus)
  - Image transforms (resize, crop, flip, rotate, affine, perspective)
  - Color adjustments (brightness, contrast, saturation, hue, exposure, gamma)
  - Image filters (blur, sharpen, edge detect, emboss, denoise, chromakey)
  - Image compositing (overlay, watermark, border, frame)
  - Audio effects (reverb, EQ, pitch shift, delay, distortion, chorus, flanger, gate)
  - Audio visualization (waveform, spectrum, spectrogram)
  - Format I/O (PNG, JPEG, WebP, AVIF, HEIF, MP4, MKV, WAV, FLAC)
  - Text overlay, subtitle parsing/rendering
  - QR code generation/decoding
  - Image quality assessment (non-AI, pure metrics)
  - Document processing (deskew, binarization)
  - Raster-to-vector (SVG)
  - LaTeX compilation
  - Hardware detection
  - Pure utility data types shared with xeno-rt
```

---

## AI Modules to Migrate (21 total)

### Image AI Models (7 modules)

| # | Module | Directory | Model | Dependencies | What STAYS in xeno-lib |
|---|--------|-----------|-------|-------------|----------------------|
| 1 | Background Removal | `src/background/` | BiRefNet (ONNX) | `ort`, `ndarray` | `postprocess::apply_mask()` could stay as a general mask-application utility |
| 2 | AI Upscale | `src/upscale/` | Real-ESRGAN (ONNX) | `ort`, `ndarray` | Nothing (traditional resize stays in `src/transforms/resize.rs`) |
| 3 | Face Restore | `src/face_restore/` | GFPGAN/CodeFormer (ONNX) | `ort`, `ndarray` | `FaceRegion`, `FaceLandmarks` structs (shared types) |
| 4 | Colorize | `src/colorize/` | DDColor (ONNX) | `ort`, `ndarray` | Nothing |
| 5 | Inpaint | `src/inpaint/` | LaMa (ONNX) | `ort`, `ndarray` | `create_mask()`, `MaskRegion` enum (pure geometry) |
| 6 | Face Detect | `src/face_detect/` | SCRFD (ONNX) | `ort`, `ndarray` | `crop_faces()`, `visualize_detections()` (pure image ops), `DetectedFace`/`FaceLandmarks` structs |
| 7 | Depth Estimation | `src/depth/` | MiDaS/Depth Anything (ONNX) | `ort`, `ndarray` | `apply_depth_blur()` (pure pixel processing), `DepthMap` struct |

### Audio AI Models (3 modules)

| # | Module | Directory | Model | Dependencies | What STAYS in xeno-lib |
|---|--------|-----------|-------|-------------|----------------------|
| 8 | Transcription | `src/transcribe/` | Whisper (ONNX) | `ort`, `ndarray` | `to_srt()`, `to_vtt()` (pure text formatting), `Transcript`/`TranscriptSegment` structs |
| 9 | Audio Separation | `src/audio_separate/` | HTDemucs (ONNX) | `ort`, `ndarray` | `StereoAudio` struct, audio interleaving utils |
| 10 | Noise Reduction | `src/noise_reduce/` | RNNoise/DTLN/DeepFilterNet (ONNX) | `ort` (planned) | Audio resampling, frame-based DSP pipeline (xeno-rt provides per-frame gain mask) |

### Video AI Models (1 module)

| # | Module | Directory | Model | Dependencies | What STAYS in xeno-lib |
|---|--------|-----------|-------|-------------|----------------------|
| 11 | Frame Interpolation | `src/frame_interpolate/` | RIFE (ONNX) | `ort`, `ndarray` | `is_scene_change()` (pure pixel comparison) |

### Vision AI Models (4 modules)

| # | Module | Directory | Model | Dependencies | What STAYS in xeno-lib |
|---|--------|-----------|-------|-------------|----------------------|
| 12 | Style Transfer | `src/style_transfer/` | Fast NST (ONNX) | `ort`, `ndarray` | Nothing |
| 13 | OCR | `src/ocr/` | PaddleOCR (ONNX) | `ort`, `ndarray` | `visualize_ocr()` drawing, `OcrResult`/`TextBlock` structs |
| 14 | Pose Estimation | `src/pose/` | MoveNet (ONNX) | `ort`, `ndarray` | `BodyKeypoint` enum, `SKELETON_CONNECTIONS`, `visualize_pose()` drawing |
| 15 | Face Analysis | `src/face_analysis/` | Multi-task CNN (ONNX) | `ort`, `ndarray` | `Gender`, `Emotion` enums, `FaceAnalysisResult` struct, `visualize_analysis()` |

### Generative AI Models (4 modules)

| # | Module | Directory | Model | Dependencies | What STAYS in xeno-lib |
|---|--------|-----------|-------|-------------|----------------------|
| 16 | Text-to-3D | `src/text_to_3d/` | TripoSR/InstantMesh (ONNX) | `ort`, `ndarray` | `Vertex`, `Triangle`, `GeneratedMesh` structs, OBJ/STL export, marching cubes |
| 17 | Voice Clone | `src/voice_clone/` | XTTS/Bark/Tortoise (ONNX) | `ort`, `ndarray` | `VoiceEmbedding`, `SynthesizedAudio` structs (shared types) |
| 18 | Music Generation | `src/music_gen/` | MusicGen/Riffusion (ONNX) | `ort`, `ndarray` | `GeneratedMusic` struct (shared type) |
| 19 | Video Generation | `src/video_gen/` | SVD/AnimateDiff (ONNX) | `ort`, `ndarray` | `GeneratedVideo` struct (shared type) |

### AI Infrastructure (3 modules)

| # | Module | Directory | Purpose | What STAYS in xeno-lib |
|---|--------|-----------|---------|----------------------|
| 20 | Model Utils | `src/model_utils.rs` | Model path resolution, home dir detection | `home_dir()` utility (generic) |
| 21 | Batch Inference | `src/batch_inference/` | Batched ONNX processing, parallel CPU fallback | Generic `process_parallel()` rayon wrapper |
| 22 | GPU Memory | `src/gpu_memory/` | VRAM management, model swapping, LRU eviction | Nothing |
| 23 | Model Quantize | `src/quantize/` | ONNX model FP16/INT8 quantization | Nothing |

---

## Files with ONNX Runtime (`ort`) / `ndarray` Usage (34 files)

These files contain `use ort` or `use ndarray` imports and need to migrate:

### Model loading (ONNX session creation):
- `src/background/model.rs`
- `src/upscale/model.rs`
- `src/face_restore/model.rs`
- `src/colorize/model.rs`
- `src/inpaint/model.rs`
- `src/face_detect/model.rs`
- `src/depth/model.rs`
- `src/transcribe/model.rs`
- `src/audio_separate/model.rs`
- `src/frame_interpolate/model.rs`
- `src/style_transfer/model.rs`
- `src/ocr/model.rs`
- `src/pose/model.rs`
- `src/face_analysis/model.rs`
- `src/text_to_3d/model.rs`
- `src/voice_clone/model.rs`
- `src/music_gen/model.rs`
- `src/video_gen/model.rs`

### Preprocessing (image/audio -> tensor):
- `src/background/preprocess.rs`
- `src/upscale/processor.rs` (partial: tile logic)
- `src/colorize/processor.rs`
- `src/inpaint/processor.rs`
- `src/depth/processor.rs`
- `src/face_detect/processor.rs`
- `src/face_restore/processor.rs`
- `src/frame_interpolate/processor.rs`
- `src/transcribe/processor.rs`
- `src/audio_separate/processor.rs`
- `src/style_transfer/processor.rs`
- `src/ocr/processor.rs`
- `src/pose/processor.rs`
- `src/face_analysis/processor.rs`
- `src/text_to_3d/processor.rs`
- `src/voice_clone/processor.rs`
- `src/music_gen/processor.rs`

### Postprocessing (tensor -> output):
- `src/background/postprocess.rs`

---

## Cargo.toml Dependencies to Remove After Migration

When all AI modules are removed from xeno-lib, these dependencies become unnecessary:

```toml
# REMOVE after migration:
ort = { version = "=2.0.0-rc.10", optional = true }
ndarray = { version = "0.16", optional = true }
```

### Feature Flags to Remove

All AI-related feature flags will be removed from xeno-lib's Cargo.toml:

```
background-removal, background-removal-cuda
upscale, upscale-cuda
face-restore, face-restore-cuda
colorize, colorize-cuda
inpaint, inpaint-cuda
face-detect, face-detect-cuda
depth, depth-cuda
transcribe, transcribe-cuda
audio-separate, audio-separate-cuda
frame-interpolate, frame-interpolate-cuda
style-transfer, style-transfer-cuda
ocr, ocr-cuda
pose, pose-cuda
face-analysis, face-analysis-cuda
text-to-3d, text-to-3d-cuda
voice-clone, voice-clone-cuda
music-gen, music-gen-cuda
video-gen, video-gen-cuda
noise-reduce
model-quantize
gpu-memory
batch-inference
ai, ai-cuda
ai-video, ai-video-cuda
ai-vision, ai-vision-cuda
ai-generative, ai-generative-cuda
ai-audio
ai-full, ai-full-cuda
perf-utils
```

The `full` and `full-cuda` bundles will be updated to exclude AI features.

---

## N-API Bindings to Deprecate

**File**: `xeno-lib-napi/src/ai_models.rs`

All AI functions are marked `DEPRECATED: will be served by xeno-rt inference API instead`:

| Function | Model | Status |
|----------|-------|--------|
| `remove_background()` | BiRefNet | DEPRECATED |
| `upscale_image()` | Real-ESRGAN | DEPRECATED |
| `separate_stems()` | HTDemucs | DEPRECATED |
| `transcribe_audio()` | Whisper | DEPRECATED |
| `interpolate_frames()` | RIFE | DEPRECATED |
| `restore_faces()` | GFPGAN | DEPRECATED |
| `colorize()` | DDColor | DEPRECATED |
| `inpaint()` | LaMa | DEPRECATED |
| `detect_faces()` | SCRFD | DEPRECATED |
| `estimate_depth()` | Depth Anything | DEPRECATED |
| `style_transfer()` | Fast NST | DEPRECATED |
| `extract_text()` | PaddleOCR | DEPRECATED |
| `detect_poses()` | MoveNet | DEPRECATED |
| `analyze_faces()` | Multi-task CNN | DEPRECATED |
| `get_model_dir()` | N/A (utility) | DEPRECATED |
| `is_model_available()` | N/A (utility) | DEPRECATED |
| `list_available_models()` | N/A (utility) | DEPRECATED |

**NOT deprecated** (stay in xeno-lib NAPI):
- `denoise_image()` -- uses spatial filter, NOT AI (should move to `image_processing.rs`)
- `denoise_audio()` -- uses limiter/normalization, NOT AI (should move to `audio_processing.rs`)

---

## Pure Processing Code That STAYS in xeno-lib

These modules have NO AI dependencies and are confirmed clean:

| Module | Directory | Purpose |
|--------|-----------|---------|
| Transforms | `src/transforms/` | 52 geometric transforms (flip, rotate, crop, resize, affine, perspective) |
| Adjustments | `src/adjustments/` | Color adjustments (brightness, contrast, saturation, hue, exposure, gamma) |
| Filters | `src/filters/` | Image filters (blur, sharpen, edge detect, emboss, denoise, vignette, chromakey) |
| Composite | `src/composite/` | Image compositing (overlay, watermark, border, frame) |
| Analysis | `src/analysis/` | Image analysis (histogram, EXIF, statistics, comparison) |
| Audio | `src/audio/` | Audio decode, encode, effects, filters, visualization |
| Video | `src/video/` | Video encode/decode/mux (rav1e, OpenH264, dav1d, NVDEC, MP4, MKV) |
| Text | `src/text.rs` | Text overlay and font rendering |
| Subtitle | `src/subtitle/` | Subtitle parsing (SRT, VTT, ASS) and rendering |
| QR Code | `src/qrcode/` | QR/barcode generation and decoding (pure Rust) |
| Quality | `src/quality/` | Image quality assessment (pure metrics, no AI) |
| Document | `src/document/` | Document processing (deskew, binarization) |
| Vectorize | `src/vectorize/` | Raster-to-vector SVG conversion |
| Formats | `src/formats/` | Modern image formats (AVIF, HEIF, WebP) |
| Hardware | `src/hardware/` | GPU/encoder hardware detection |
| LaTeX | `src/latex/` | LaTeX compilation via Tectonic |
| Agent | `src/agent.rs` | Agent API (JSON-serializable results) |
| Error | `src/error.rs` | Error types (shared) |

---

## Migration Steps (Recommended Order)

### Phase 1: Absorb into xeno-rt
1. Copy all 19 AI model modules into xeno-rt's codebase
2. Copy `model_utils.rs`, `batch_inference/`, `gpu_memory/`, `quantize/` to xeno-rt
3. Add `ort` and `ndarray` dependencies to xeno-rt's Cargo.toml
4. Verify all models build and tests pass in xeno-rt

### Phase 2: Create shared types crate (optional)
- Consider a `xeno-types` crate for shared structs (DetectedFace, DepthMap, Transcript, etc.)
- Or just re-export from xeno-rt with the same type signatures

### Phase 3: Update consuming apps
1. Update xeno-pixel, xeno-motion, xeno-sound, xeno-hub to call xeno-rt for AI inference
2. Keep calling xeno-lib for pure processing (resize, filters, codecs, etc.)

### Phase 4: Remove AI from xeno-lib
1. Remove all AI modules from `src/`
2. Remove `ort` and `ndarray` from Cargo.toml
3. Remove all AI feature flags
4. Remove AI exports from `lib.rs`
5. Remove `xeno-lib-napi/src/ai_models.rs` (or keep only the non-AI functions)
6. Update `CLAUDE.md` to remove references to 17 AI models
7. Bump major version

### Phase 5: Cleanup
1. Update documentation on xenostudio.ai
2. Update the Ecosystem Report
3. Verify all consuming apps still build and work

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Output format changes during migration | High -- 5+ apps break | Keep identical output contracts in xeno-rt |
| Model file path changes | Medium -- users must re-download | Keep `~/.xeno-lib/models/` path or add migration |
| API surface change | High -- all consumers affected | Maintain identical public API in xeno-rt |
| Performance regression | Medium -- different code path | Benchmark before/after migration |
| Build time regression for consuming apps | Low | xeno-rt is a separate binary, not linked into apps |

---

## Output Contracts (MUST be preserved in xeno-rt)

| Output Type | Format | Consumers |
|-------------|--------|-----------|
| Image outputs | RGBA u8 (4 bytes/pixel) | xeno-pixel, xeno-motion, xeno-hub |
| Audio outputs | f32 PCM (-1.0 to 1.0) at input sample rate | xeno-sound, xeno-motion |
| Mask outputs | Single-channel u8 (0=bg, 255=fg) | xeno-pixel |
| Depth maps | Single-channel f32 (0.0 to 1.0) | xeno-pixel |
| Structured data | JSON via serde | All apps, xeno-agent-sdk |
