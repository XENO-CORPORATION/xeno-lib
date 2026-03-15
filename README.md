# xeno-lib

**Pure Rust multimedia processing library with 17 AI models.** The compute backbone of the entire XENO platform — replacing all dependency on FFmpeg and proprietary codecs with custom, pure Rust implementations.

> xeno-lib is not a wrapper. It is a ground-up Rust media processing engine with SIMD-accelerated transforms, state-of-the-art neural networks via ONNX Runtime + CUDA, pure Rust video/audio codecs, and professional utilities. Every XENO app depends on this library.

---

## Vision

**Replace FFmpeg entirely.** No C dependency chains, no GPL licensing headaches, no opaque binary blobs. xeno-lib delivers:

- Pure Rust codecs for every format we ship (AV1 via rav1e, H.264 via OpenH264, audio via symphonia)
- 17 AI models that FFmpeg cannot match (upscaling, background removal, transcription, pose estimation, etc.)
- SIMD-accelerated image transforms (AVX2 on x86_64, NEON on ARM)
- GPU acceleration via ONNX Runtime + CUDA for all AI inference
- Memory safety guaranteed by the Rust compiler — no CVEs from buffer overflows
- N-API bindings (planned) for seamless Electron integration across xeno-hub, xeno-pixel, xeno-motion, xeno-sound
- WASM compilation target (planned) for browser-based processing

This is not incremental improvement over FFmpeg. This is a replacement built for AI-native creative applications.

---

## Architecture: Where xeno-lib Fits

```
                    LAYER 5 — CREATIVE APPS
         ┌──────────────┬──────────────┬──────────────┐
         │  xeno-pixel  │  xeno-motion │  xeno-sound  │
         │  (images)    │  (video)     │  (audio/DAW) │
         └──────┬───────┴──────┬───────┴──────┬───────┘
                │              │              │
         ┌──────┴──────────────┴──────────────┴───────┐
         │              @xeno/core                     │
         │   Rendering, compositing, color, UI, plugins│
         └─────────────────────┬───────────────────────┘
                               │
                    LAYER 4 — SHARED ENGINE
                               │
    ┌──────────────────────────┴───────────────────────┐
    │                    xeno-lib                       │
    │                                                   │
    │  ┌─────────────┐  ┌──────────┐  ┌─────────────┐ │
    │  │ 17 AI Models│  │ Codecs   │  │ Transforms  │ │
    │  │ ONNX + CUDA │  │ Pure Rust│  │ SIMD AVX2   │ │
    │  └─────────────┘  └──────────┘  └─────────────┘ │
    │  ┌─────────────┐  ┌──────────┐  ┌─────────────┐ │
    │  │ Audio FX    │  │ Video    │  │ Pro Utils   │ │
    │  │ Effects/Viz │  │ Edit/Mux │  │ QR/Doc/IQA  │ │
    │  └─────────────┘  └──────────┘  └─────────────┘ │
    └──────────────────────────────────────────────────┘
                               │
                    LAYER 2 — COMPUTE & AI
                               │
              ┌────────────────┴────────────────┐
              │ xeno-rt (LLM inference runtime) │
              │ xeno-mind (resonance research)  │
              └─────────────────────────────────┘
```

**Data format contracts (never change without ecosystem coordination):**

| Data Type | Format | Contract |
|-----------|--------|----------|
| Image output | RGBA u8 | All image model outputs are RGBA, 8 bits per channel |
| Audio output | f32 PCM | All audio at input sample rate, interleaved f32 |
| Mask output | Single-channel u8 | 0 = background, 255 = foreground |
| Depth maps | Single-channel f32 | Normalized 0.0 to 1.0 |
| Structured data | JSON | Bounding boxes, keypoints, transcripts, metadata |

---

## AI Models (17)

All models run via ONNX Runtime with optional CUDA acceleration. Models are stored in `~/.xeno-lib/models/` as ONNX files.

### Image AI — Used by xeno-pixel

| Model | Capability | Input | Output | Size |
|-------|-----------|-------|--------|------|
| **Real-ESRGAN** | 2x/4x/8x neural super-resolution | RGBA image | RGBA image (upscaled) | ~67MB |
| **BiRefNet** | Deep learning background removal | RGBA image | Alpha mask (u8) | ~176MB |
| **GFPGAN** | Photo-quality face restoration | RGBA face crop | RGBA restored face | ~348MB |
| **DDColor** | Automatic B&W photo colorization | Grayscale image | RGB colorized image | ~93MB |
| **LaMa** | Content-aware object removal (inpainting) | RGBA image + mask | RGBA filled image | ~52MB |
| **SCRFD** | Fast face detection with landmarks | RGBA image | Bounding boxes + landmarks (JSON) | ~27MB |
| **MiDaS** | Monocular depth estimation | RGBA image | Depth map (f32) | ~400MB |
| **Fast NST** | Neural artistic style transfer | Content + style images | Stylized image (u8) | ~7MB |
| **PaddleOCR** | Multi-language text recognition | RGBA image | Text + positions (JSON) | ~12MB |
| **MoveNet** | Real-time body pose estimation (17 keypoints) | RGBA image | Keypoints (JSON) | ~9MB |
| **Multi-task CNN** | Age, gender, and emotion detection | RGBA face crop | Structured analysis (JSON) | ~25MB |

### Video/Audio AI — Used by xeno-motion, xeno-sound

| Model | Capability | Input | Output | Size |
|-------|-----------|-------|--------|------|
| **RIFE v4.6** | Optical flow frame interpolation (slow-mo) | Two RGBA frames | Interpolated RGBA frame | ~15MB |
| **Whisper Base** | Speech-to-text with word timestamps | f32 PCM audio | Text + timestamps (JSON) | ~145MB |
| **HTDemucs** | Source separation (vocals/drums/bass/other) | f32 PCM audio | 4 stem tracks (f32 PCM each) | ~81MB |

### Planned Models

| Model | Capability | Target App |
|-------|-----------|------------|
| YOLOv8 | Object detection with bounding boxes | xeno-motion |
| SAM2 | Interactive segmentation (click-to-select) | xeno-pixel |
| RNNoise | Real-time noise suppression | xeno-sound |
| NAFNet | Image denoising | xeno-pixel |
| Depth Anything v2 | Improved depth estimation | xeno-pixel |
| Color Transfer | Reference-based color grading | xeno-pixel, xeno-motion |

---

## Pure Rust Codec Stack

xeno-lib builds its own media processing pipeline. No FFmpeg. No libav. No GPL dependencies.

### Current Implementations

| Codec/Format | Library | Direction | Status |
|-------------|---------|-----------|--------|
| **AV1** | rav1e (pure Rust) | Encode | Shipping |
| **AV1** | dav1d (C bindings) | Decode | Shipping |
| **H.264** | OpenH264 (Cisco, BSD) | Encode + Decode | Shipping |
| **MP4 container** | mp4 crate | Mux/Demux | Shipping |
| **MKV/WebM** | matroska crate | Demux | Shipping |
| **MP3** | symphonia (pure Rust) | Decode | Shipping |
| **AAC** | symphonia (pure Rust) | Decode | Shipping |
| **FLAC** | symphonia (decode) + flacenc (encode) | Both | Shipping |
| **Vorbis** | symphonia (pure Rust) | Decode | Shipping |
| **ALAC** | symphonia (pure Rust) | Decode | Shipping |
| **WAV** | hound (pure Rust) | Encode | Shipping |
| **Opus** | audiopus (libopus bindings) | Encode | Shipping |
| **PNG/JPEG/WebP/GIF/BMP/TIFF** | image crate (pure Rust) | Both | Shipping |
| **SVG vectorization** | vtracer (vendored) | Raster-to-SVG | Shipping |

### Planned Codec Work

| Codec | Priority | Notes |
|-------|----------|-------|
| **H.265/HEVC** | High | Pure Rust or minimal C bindings, royalty considerations |
| **VP9** | Medium | WebM compatibility, pure Rust decoder |
| **NVENC/NVDEC** | High | NVIDIA hardware encode/decode via libloading (no static linking) |
| **QSV** | Medium | Intel Quick Sync Video for Intel GPUs |
| **AMF** | Medium | AMD hardware encoding |
| **AAC encode** | High | Pure Rust AAC encoder for MP4 output |
| **Opus (pure Rust)** | Medium | Replace libopus binding with pure Rust |
| **AV1 hardware decode** | Low | Via NVDEC when available |

### Video Decoding Architecture

GPU-accelerated decoding uses dynamic loading (`libloading`) to avoid compile-time CUDA version constraints:

```
Video File → Container Demux (mp4/matroska) → Codec Detection
    ├── NVDEC available? → GPU decode (libloading → nvcuvid)
    ├── Software fallback → dav1d (AV1) / OpenH264 (H.264)
    └── Frame output → RGBA u8 buffer
```

---

## Audio Processing Pipeline

### Decode (symphonia — pure Rust)
Formats: MP3, AAC, FLAC, Vorbis, ALAC, WAV, AIFF, MKV audio, MP4 audio, OGG, WebM audio.

### Effects (pure Rust DSP)
| Effect | Parameters | Use Case |
|--------|-----------|----------|
| **Reverb** | Room size, damping, wet/dry, presets (hall, room, plate) | Spatial audio, music production |
| **Parametric EQ** | Frequency bands, gain, Q factor, filter types | Mixing, mastering, correction |
| **Pitch Shift** | Semitones, fine-tune cents | Key changes, vocal tuning |
| **Delay** | Time, feedback, wet/dry | Echo effects, creative |
| **Distortion** | Drive, tone, mix | Guitar amp simulation, creative |
| **Chorus** | Rate, depth, voices | Thickening, detuning effects |
| **Flanger** | Rate, depth, feedback | Sweeping modulation effects |

### Visualization (pure Rust)
| Type | Output | Use Case |
|------|--------|----------|
| **Waveform** | RGBA image | Timeline display in xeno-sound |
| **Spectrum Analyzer** | RGBA image (FFT bins) | Frequency analysis |
| **Spectrogram** | RGBA image (time x frequency) | Audio forensics, mastering |

### Encode
| Format | Library | Notes |
|--------|---------|-------|
| WAV | hound (pure Rust) | Lossless, any sample rate/bit depth |
| FLAC | flacenc (pure Rust) | Lossless compressed |
| Opus | audiopus (libopus) | High-quality lossy at low bitrates |

---

## Video Processing Pipeline

### Encoding
```rust
use xeno_lib::video::edit::{trim_video, concat_videos, TrimConfig, ConcatConfig};

// Trim video segment
let config = TrimConfig::new(10.0, 30.0);
trim_video("input.mp4", "trimmed.mp4", config)?;

// Concatenate clips
concat_videos(&["clip1.mp4", "clip2.mp4"], "combined.mp4", ConcatConfig::default())?;
```

### Supported Operations
- **Encode**: AV1 (rav1e), H.264 (OpenH264) to MP4 container
- **Decode**: AV1 (dav1d), H.264 (OpenH264), GPU-accelerated (NVDEC)
- **Edit**: Trim, cut, concatenate, speed change, subtitle burn-in
- **Mux**: Audio + video sync to MP4 with proper timestamps
- **Transcode**: Format conversion between supported codecs
- **Frame extraction**: Video to frame sequence (PNG/JPEG)
- **GIF/WebP**: Animated GIF and animated WebP creation

---

## Image Transforms (52 Geometric Operations)

All geometric transforms are SIMD-accelerated (AVX2 on x86_64, scalar fallback for other targets).

### Categories
- **Geometric**: flip, rotate (90/180/270/arbitrary), crop, resize, perspective warp, affine transform
- **Subject layout**: recenter transparent subjects on-canvas with padding control
- **Vectorization**: raster (PNG/JPG/etc.) to SVG vector paths via vtracer
- **Color adjustments**: brightness, contrast, saturation, hue rotation, gamma, exposure, levels
- **Filters**: Gaussian blur, sharpen, edge detect, emboss, sepia, denoise, median
- **Compositing**: overlay, watermark, border/frame, text overlay with font rendering

---

## Performance Benchmarks

**Test hardware:** NVIDIA RTX 4090 + AMD Ryzen 9950X

### AI Model Inference (GPU)

| Operation | Resolution | Time | Notes |
|-----------|-----------|------|-------|
| Background removal | 2048x2048 | ~35ms | BiRefNet, CUDA |
| AI Upscale 4x | 512 to 2048 | ~120ms | Real-ESRGAN, CUDA |
| Face restoration | 512x512 | ~45ms | GFPGAN, CUDA |
| Style transfer | 512x512 | ~60ms | Fast NST, CUDA |
| Pose estimation | 512x512 | ~25ms | MoveNet, CUDA |
| OCR | 1024x768 | ~80ms | PaddleOCR, CUDA |

### Image Transforms (CPU, SIMD)

| Operation | Resolution | Time | Notes |
|-----------|-----------|------|-------|
| Flip horizontal | 4000x2500 | ~12ms | AVX2 accelerated |
| Rotate 90 degrees | 4000x2500 | ~16ms | Cache-optimized |
| Quality assessment | 4000x2500 | ~5ms | Pure Rust |
| Document deskew | 2000x3000 | ~15ms | Pure Rust |

---

## Integration with XENO Apps

### xeno-pixel (Image Editor)

| Feature | Model | Use Case |
|---------|-------|----------|
| AI Upscale | Real-ESRGAN | Enlarge images without quality loss |
| Background Removal | BiRefNet | One-click subject isolation |
| Inpainting | LaMa | Content-aware fill / object removal |
| Face Restore | GFPGAN | Fix blurry or damaged faces |
| Colorize | DDColor | Automatic B&W photo colorization |
| Style Transfer | Fast NST | Apply artistic styles to photos |
| Face Detection | SCRFD | Detect faces for auto-crop, retouch |
| Depth Estimation | MiDaS | Generate depth maps for 3D effects |
| OCR | PaddleOCR | Extract text from images |
| Pose Estimation | MoveNet | Body keypoint overlay |
| Segmentation | SAM2 (planned) | Click-to-select subjects |
| Denoise | NAFNet (planned) | Remove image noise |

### xeno-motion (Video Editor)

| Feature | Model | Use Case |
|---------|-------|----------|
| Frame Interpolation | RIFE | Smooth slow-motion, frame rate conversion |
| Transcription | Whisper | Auto-generate subtitles from speech |
| Audio Separation | HTDemucs | Isolate dialogue, music, sound effects |
| Object Detection | YOLOv8 (planned) | Track objects across frames |
| Scene Detection | (planned) | Auto-detect scene changes for editing |

### xeno-sound (Audio/DAW)

| Feature | Model/Engine | Use Case |
|---------|-------------|----------|
| Noise Reduction | RNNoise (planned) | Clean up recordings |
| Stem Separation | HTDemucs | Isolate vocals/drums/bass/other |
| Transcription | Whisper | Speech-to-text for podcasts |
| Pitch Detection | (planned) | Auto-tune, pitch correction |
| Audio Effects | Pure Rust DSP | Reverb, EQ, delay, chorus, etc. |
| Visualization | Pure Rust | Waveform, spectrum, spectrogram rendering |

---

## N-API Binding Layer (Planned)

All XENO desktop apps run on Electron. xeno-lib needs N-API bindings to be callable from Node.js/Electron.

### Architecture
```
Electron Main Process
    └── N-API (napi-rs)
        └── xeno-lib (Rust)
            ├── AI models (ONNX Runtime + CUDA)
            ├── Codecs (rav1e, symphonia, OpenH264)
            ├── Transforms (SIMD AVX2)
            └── DSP (audio effects, visualization)
```

### Implementation Plan
- **napi-rs** for Rust-to-Node bindings (not node-bindgen, not neon)
- Async operations for all model inference (non-blocking main thread)
- Streaming APIs for video/audio processing
- Typed TypeScript declarations auto-generated from Rust types
- Platform-specific prebuilt binaries (Windows x64, macOS ARM64/x64, Linux x64)

### WASM Compilation Target (Planned)

For browser-based previews and the web version of XENO apps:
- All pure-compute functions compile to `wasm32-unknown-unknown`
- CPU-only fallback (no CUDA in browser)
- SIMD via `wasm-simd` where supported
- Target: lightweight preview operations, not full model inference

---

## Professional Utilities (Pure Rust)

### Subtitle Processing
- Parse: SRT, VTT, ASS/SSA formats
- Render: Burn subtitles onto video frames with configurable styles
- Time: Shift, stretch, sync subtitle timing

### QR Code and Barcode
- Generate: QR, Code128, EAN-13, UPC-A
- Decode: QR code reading from images
- Customize: Size, error correction level, colors

### Image Quality Assessment
- Metrics: Sharpness, noise level, exposure, contrast, color balance
- Scoring: Overall quality score with letter grade
- Issues: Automatic detection of blur, noise, over/underexposure

### Document Processing
- Deskew: Automatic rotation correction for scanned documents
- Binarization: Adaptive thresholding for OCR preprocessing
- Perspective: Correct perspective distortion from camera captures

---

## Quick Start

### Basic Image Processing

```rust
use xeno_lib::{flip_horizontal, adjust_brightness, gaussian_blur};

let img = image::open("photo.jpg")?;
let flipped = flip_horizontal(&img)?;
let brightened = adjust_brightness(&flipped, 1.2)?;
let blurred = gaussian_blur(&brightened, 2.0)?;
blurred.save("output.png")?;
```

### AI Background Removal

```rust
use xeno_lib::{load_model, remove_background, BackgroundRemovalConfig};

let config = BackgroundRemovalConfig::default();
let mut session = load_model(&config)?;

let input = image::open("portrait.jpg")?;
let output = remove_background(&input, &mut session)?;
output.save("portrait_nobg.png")?;
```

### AI Upscaling (4x)

```rust
use xeno_lib::{load_upscaler, ai_upscale, UpscaleConfig, UpscaleModel};

let config = UpscaleConfig::new(UpscaleModel::RealEsrganX4);
let mut upscaler = load_upscaler(&config)?;

let input = image::open("low_res.jpg")?;
let upscaled = ai_upscale(&input, &mut upscaler)?;
upscaled.save("high_res.png")?;
```

### Audio Effects

```rust
use xeno_lib::audio::effects::{reverb, equalizer, ReverbConfig, EqConfig};

let samples: Vec<f32> = load_audio("input.wav")?;

// Apply hall reverb
let reverbed = reverb(&samples, 44100, ReverbConfig::hall());

// Apply EQ with bass boost
let eq_config = EqConfig::default().with_bass_boost(6.0);
let processed = equalizer(&reverbed, 44100, &eq_config);
```

### Audio Visualization

```rust
use xeno_lib::audio::visualization::{render_waveform, render_spectrogram, WaveformConfig};

let samples: Vec<f32> = load_audio("music.wav")?;

let config = WaveformConfig::default().with_color([0, 255, 128, 255]);
let waveform = render_waveform(&samples, &config);
waveform.save("waveform.png")?;

let spectrogram = render_spectrogram(&samples, 44100, 800, 400, ColorMap::Viridis);
spectrogram.save("spectrogram.png")?;
```

### Video Encoding

```rust
use xeno_lib::video::edit::{trim_video, concat_videos, TrimConfig};

// Trim 10s to 30s
trim_video("input.mp4", "trimmed.mp4", TrimConfig::new(10.0, 30.0))?;

// Concatenate clips
concat_videos(&["clip1.mp4", "clip2.mp4"], "combined.mp4", Default::default())?;
```

---

## Feature Flags

```toml
[dependencies]
xeno-lib = { version = "0.1", features = ["full"] }
```

### AI Features

| Feature | Description | GPU Variant |
|---------|-------------|-------------|
| `background-removal` | BiRefNet background removal | `background-removal-cuda` |
| `upscale` | Real-ESRGAN 2x/4x/8x upscaling | `upscale-cuda` |
| `face-restore` | GFPGAN/CodeFormer face restoration | `face-restore-cuda` |
| `colorize` | DDColor/DeOldify colorization | `colorize-cuda` |
| `frame-interpolate` | RIFE frame interpolation | `frame-interpolate-cuda` |
| `inpaint` | LaMa object removal | `inpaint-cuda` |
| `face-detect` | SCRFD face detection | `face-detect-cuda` |
| `depth` | MiDaS depth estimation | `depth-cuda` |
| `transcribe` | Whisper speech-to-text | `transcribe-cuda` |
| `audio-separate` | Demucs voice isolation | `audio-separate-cuda` |
| `style-transfer` | Neural style transfer | `style-transfer-cuda` |
| `ocr` | PaddleOCR text recognition | `ocr-cuda` |
| `pose` | MoveNet pose estimation | `pose-cuda` |
| `face-analysis` | Age/gender/emotion detection | `face-analysis-cuda` |

### Media Features

| Feature | Description |
|---------|-------------|
| `video` | Video types, container detection, editing |
| `video-encode` | AV1 encoding via rav1e |
| `video-encode-h264` | H.264 encoding via OpenH264 |
| `video-decode` | Video decoding via NVDEC (NVIDIA GPU) |
| `video-decode-sw` | Software AV1 + H.264 decoding (dav1d + OpenH264) |
| `audio` | Audio decoding via symphonia (MP3/AAC/FLAC/Vorbis/ALAC/WAV) |
| `audio-encode` | WAV encoding (hound) |
| `audio-encode-flac` | FLAC encoding (pure Rust) |
| `audio-encode-opus` | Opus encoding (libopus bindings) |
| `text-overlay` | Text rendering on images (ab_glyph) |
| `subtitle` | SRT/VTT/ASS subtitle parsing and burn-in |
| `containers` | MKV/WebM + MP4 demuxing |

### Professional Utilities (Pure Rust)

| Feature | Description |
|---------|-------------|
| `qrcode` | QR code and barcode generation/decoding |
| `quality` | Image quality assessment |
| `document` | Document deskew/binarization |
| `vectorize` | Raster-to-SVG vectorization (vtracer) |

### Feature Bundles

| Bundle | Includes |
|--------|----------|
| `ai` | All core AI image features (bg removal, upscale, face restore, colorize, inpaint, face detect, depth) |
| `ai-cuda` | All core AI image features with CUDA GPU acceleration |
| `ai-video` | Frame interpolation, transcription, audio separation |
| `ai-vision` | Style transfer, OCR, pose estimation, face analysis |
| `ai-full` | All AI features combined |
| `ai-full-cuda` | All AI features with GPU acceleration |
| `multimedia` | Video (encode + decode) + audio (decode + encode) + subtitles + text overlay |
| `pro-utils` | QR code, quality assessment, document processing, vectorization |
| `full` | Everything |
| `full-cuda` | Everything with GPU acceleration |

### Platform-Specific Notes

**`video-decode-sw` (dav1d + OpenH264):**
- Linux: install `libdav1d-dev` and `pkg-config` (or set `SYSTEM_DEPS_DAV1D_BUILD_INTERNAL=always`)
- Windows: install `pkg-config` and `dav1d:x64-windows-static` via `vcpkg`, set `PKG_CONFIG_PATH` and `PKG_CONFIG_ALL_STATIC=1`

---

## Model Downloads

Download ONNX models to `~/.xeno-lib/models/`:

| Model | Size | Use |
|-------|------|-----|
| BiRefNet General | ~176MB | Background removal |
| Real-ESRGAN x4 | ~67MB | Upscaling |
| GFPGAN | ~348MB | Face restoration |
| DDColor | ~93MB | Colorization |
| RIFE v4.6 | ~15MB | Frame interpolation |
| LaMa | ~52MB | Inpainting |
| SCRFD | ~27MB | Face detection |
| MiDaS v3.1 | ~400MB | Depth estimation |
| Whisper Base | ~145MB | Speech-to-text |
| Demucs | ~81MB | Voice isolation |
| Fast NST | ~7MB | Style transfer |
| PaddleOCR | ~12MB | Text recognition |
| MoveNet | ~9MB | Pose estimation |
| Age/Gender/Emotion | ~25MB | Face analysis |

---

## FFmpeg Replacement Roadmap

### The Goal

Replace all dependency on FFmpeg with XENO-owned infrastructure. NOT "zero C code" — that's unrealistic for hardware encoders. The goal is: **zero FFmpeg dependency, full pipeline control, AI capabilities FFmpeg can never have.**

Some things will never be pure Rust and that's OK:
- NVENC/QSV/AMF — proprietary GPU hardware encoders require C/C++ vendor SDKs
- ONNX Runtime — C++ library, `ort` crate is bindings
- Some codecs (H.264/H.265 are patent-encumbered)

C bindings to specific vendor SDKs (NVIDIA, Intel) are fine. Depending on a monolithic 3M-line C project (FFmpeg) is not.

FFmpeg parity is tracked by a generated, spec-driven matrix:
- Spec: `benchmarks/ffmpeg/parity_spec.json`
- Strategic roadmap: `benchmarks/ffmpeg/OVERCOME_TRACKER.md`
- Current generated report: `benchmarks/ffmpeg/results/latest.md` (CI artifact)
- Regression baseline: `benchmarks/ffmpeg/baseline.json`

### Phase 1 — Foundation (COMPLETE)

- Pure Rust image processing (52 transforms, SIMD AVX2)
- Pure Rust audio decode (symphonia — MP3, AAC, FLAC, Vorbis, ALAC, WAV, AIFF)
- Pure Rust audio encode (WAV via hound, FLAC via flacenc, Opus via audiopus)
- Pure Rust AV1 encode (rav1e)
- H.264 encode (OpenH264 — Cisco C library, BSD licensed)
- 17 AI models via ONNX Runtime + CUDA
- Agent-friendly JSON API
- MP4 container muxing

### Phase 2 — Electron Integration & Decode Expansion

- N-API bindings via napi-rs (so Electron apps call xeno-lib directly, not subprocess)
- Async N-API with streaming results for large operations
- Platform-specific prebuilt binaries (Windows x64, macOS ARM64, Linux x64)
- H.265/HEVC decode (via minimal C binding or pure Rust when available)
- VP9 decode
- Hardware decode: NVDEC (NVIDIA GPU) — already partially implemented via libloading
- Software AV1 decode improvement (dav1d integration hardening)

### Phase 3 — Hardware Encoding & Codec Expansion

- NVENC (NVIDIA hardware H.264/H.265/AV1 encode) — C SDK bindings, 10-50x faster than CPU
- QSV (Intel Quick Sync) — C SDK bindings
- AMF (AMD Advanced Media Framework) — C SDK bindings
- VideoToolbox (macOS hardware encode) — ObjC/C bindings
- ProRes encode/decode (reverse-engineered or FFmpeg-independent implementation)
- DNxHR/DNxHD for professional video workflows
- AAC encode (via fdk-aac bindings or pure Rust implementation when mature)
- MP3 encode (via lame bindings)
- MKV/WebM container support (matroska crate expansion)
- MOV container support

### Phase 4 — Professional Feature Parity

- 100+ filters/effects that creative apps actually need (not all 400+ FFmpeg filters — most are niche)
- Professional color grading pipeline (LUT application, color space conversions)
- Video stabilization
- Advanced audio effects (multiband compression, limiter, de-esser, noise gate)
- Subtitle burning/rendering
- Multi-stream muxing (multiple audio tracks, subtitle tracks)
- Chapter/metadata support
- Thumbnail/poster frame extraction at scale

### What We Will NEVER Build (and why)

- Streaming protocols (RTMP/RTSP/HLS/DASH) — XENO is a creative suite, not a streaming platform
- 400+ niche FFmpeg filters — we build what our apps need, not a kitchen sink
- Legacy format support (FLV, WMV, RealMedia) — nobody needs these in 2026

### What We Have That FFmpeg NEVER Will

- 17 AI models integrated (upscale, bg removal, inpainting, face restore, depth, OCR, pose, transcription, stem separation, noise reduction, style transfer, segmentation, frame interpolation, face detection, face analysis, colorization, color transfer)
- Memory-safe Rust codebase
- Native Rust API with proper types (not CLI string parsing)
- WASM compilation target (runs in browser)
- Agent-friendly JSON API for AI automation
- Designed for real-time creative app embedding, not batch processing

### Parity Achieved (Phase 1)

- Image transforms (flip, rotate, crop, resize, perspective, affine)
- Color adjustments (brightness, contrast, saturation, hue, gamma, exposure)
- Filters (blur, sharpen, edge detect, denoise, median)
- Format conversion (PNG, JPEG, WebP, GIF, BMP, TIFF, etc.)
- Video encoding (AV1, H.264) to MP4
- Video editing (trim, cut, concat, speed change)
- Audio decoding (MP3, AAC, FLAC, Vorbis, ALAC, WAV)
- Audio encoding (Opus, FLAC, WAV)
- Audio effects (reverb, EQ, pitch shift, delay, chorus, flanger, distortion)
- Subtitle processing (SRT, VTT, ASS)
- Animated GIF/WebP creation

---

## Repository Layout

```
xeno-lib/
├── src/
│   ├── lib.rs                 # Public API exports
│   ├── adjustments/           # Color adjustments (brightness, contrast, hue, etc.)
│   ├── analysis/              # Image analysis (statistics, comparison, EXIF)
│   ├── audio/                 # Audio processing
│   │   ├── effects.rs         # Reverb, EQ, pitch shift, delay, chorus, flanger
│   │   ├── visualization.rs   # Waveform, spectrum analyzer, spectrogram
│   │   └── filters.rs         # Audio filters
│   ├── audio_separate/        # Demucs voice/stem isolation
│   ├── background/            # BiRefNet background removal
│   ├── colorize/              # DDColor colorization
│   ├── composite/             # Overlay, watermark, borders, frames
│   ├── depth/                 # MiDaS depth estimation
│   ├── document/              # Document deskew, binarization, perspective
│   ├── face_analysis/         # Age/gender/emotion detection
│   ├── face_detect/           # SCRFD face detection
│   ├── face_restore/          # GFPGAN face restoration
│   ├── filters/               # Image filters (blur, sharpen, edge detect, etc.)
│   ├── frame_interpolate/     # RIFE frame interpolation
│   ├── inpaint/               # LaMa object removal
│   ├── ocr/                   # PaddleOCR text recognition
│   ├── pose/                  # MoveNet pose estimation
│   ├── qrcode/                # QR code and barcode generation/decoding
│   ├── quality/               # Image quality assessment
│   ├── style_transfer/        # Neural style transfer
│   ├── subtitle/              # SRT/VTT/ASS parsing and rendering
│   ├── text.rs                # Text overlay with font rendering
│   ├── transcribe/            # Whisper speech-to-text
│   ├── transforms/            # 52 geometric transforms (SIMD AVX2)
│   ├── upscale/               # Real-ESRGAN upscaling
│   └── video/                 # Video encoding, decoding, editing, muxing
├── xeno-edit/                 # CLI tool (xeno-edit binary)
├── vendor/                    # Vendored dependencies (vtracer, flacenc, rav1e)
├── examples/                  # Usage examples
├── tests/                     # Integration tests
├── benches/                   # Criterion performance benchmarks
├── benchmarks/                # FFmpeg parity and competitive benchmark tracking
│   ├── ffmpeg/                # FFmpeg parity spec, baseline, results
│   └── competitors/           # Competitive benchmark suite
├── tools/                     # Build and benchmark tooling
└── docs/                      # Additional documentation
```

---

## Development

### Build and Test

```bash
# Run tests (default features)
cargo test

# Run with all features
cargo test --features full

# Run with all features + locked dependencies
cargo test --all-features --locked

# Run benchmarks
cargo bench --bench transforms

# Build release CLI
cd xeno-edit && cargo build --release
```

### Competitive Benchmarking

Benchmark xeno against FFmpeg, ImageMagick, and libvips:

```bash
# Build xeno-edit first
cargo build --manifest-path xeno-edit/Cargo.toml --release

# Run benchmark suite
cargo run --manifest-path tools/competitive-bench/Cargo.toml -- run \
  --xeno-bin xeno-edit/target/release/xeno-edit \
  --output benchmarks/competitors/results/latest.json

# Gate regressions
cargo run --manifest-path tools/competitive-bench/Cargo.toml -- gate \
  --current benchmarks/competitors/results/latest.json \
  --baseline benchmarks/competitors/baseline.json
```

### FFmpeg Parity Matrix

```bash
python tools/ffmpeg-parity/generate_matrix.py \
  --xeno-bin xeno-edit/target/release/xeno-edit \
  --spec benchmarks/ffmpeg/parity_spec.json \
  --output-json benchmarks/ffmpeg/results/latest.json \
  --output-md benchmarks/ffmpeg/results/latest.md \
  --baseline benchmarks/ffmpeg/baseline.json \
  --baseline-candidate benchmarks/ffmpeg/results/baseline-candidate.json \
  --fail-on-regression
```

### Test Coverage

- 82+ unit tests covering all modules
- Integration tests for transforms and filters
- Criterion benchmarks for performance tracking
- CI runs on: Ubuntu, Windows, macOS

---

## xeno-edit CLI

Command-line tool for image/video/audio editing powered by xeno-lib.

```bash
# Background removal
xeno-edit remove-bg photo.jpg

# Format conversion
xeno-edit convert webp --quality 90 photo.png

# Raster to SVG vectorization
xeno-edit convert svg --svg-preset photo input.png

# Recenter transparent subject
xeno-edit recenter logo.png --resize 512x512

# Create animated GIF
xeno-edit gif output.gif -d 100 frame*.png

# Video encoding (AV1)
xeno-edit video-encode output.mp4 -f 30 frame*.png

# H.264 encoding
xeno-edit h264-encode output.mp4 -f 30 frame*.png
```

### Command Groups

- **Image**: `remove-bg`, `convert`, `recenter`, `image-filter`, `gif`, `awebp`, `text-overlay`
- **Video**: `video-info`, `video-encode`, `h264-encode`, `video-frames`, `video-to-gif`, `video-thumbnail`, `encode-sequence`, `video-transcode`, `video-trim`, `video-concat`
- **Audio**: `audio-info`, `extract-audio`, `audio-encode`
- **Agent/Automation**: `capabilities`, `gpu-info`, `exec`, `template`

Full CLI reference: `xeno-edit/CLI_COMMANDS.md`

---

## License

Licensed under the [Apache License 2.0](LICENSE).

Important: source code in this repository is Apache-2.0, but optional models,
runtime backends, codecs, and external toolchains may have their own licenses,
patent terms, or redistribution rules. Review upstream terms before shipping a
commercial product that enables those components. See `docs/LICENSE_POLICY.md`
for the current dependency license policy, accepted exceptions, and known
commercial-use cautions.

---

## Contributing

Contributions are welcome. Before your first PR can be merged, you must agree
to the terms in [CLA.md](CLA.md).

See the full project policies:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)
- [RELEASE.md](RELEASE.md)
- [CHANGELOG.md](CHANGELOG.md)

Baseline expectations:

- All relevant tests pass (`cargo test --all-features`)
- Code is formatted (`cargo fmt --check`)
- No new clippy warnings in touched areas
- Docs, CLI help, and parity tracking are updated when behavior changes

---

**Built with Rust by XENO Corporation**
