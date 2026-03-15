# CLAUDE.md — XENO Lib Engineering Standards

## You Are Working On

**xeno-lib** — pure Rust multimedia processing library with 17 AI models. The compute engine of the entire XENO platform.

## Critical Context

Part of a 16+ repo ecosystem. Read `../XENO CORPORATION - Full Ecosystem Report.md`.

```
YOUR REPO: xeno-lib (Layer 2 — Compute & AI)
    ↑ consumed by: xeno-pixel (bg removal, upscale, denoise, inpainting, segmentation)
    ↑ consumed by: xeno-motion (transcription, voice isolation, object detection)
    ↑ consumed by: xeno-sound (voice isolation, noise reduction, transcription)
    ↑ consumed by: xeno-hub (bg removal tool, format converter)
    ↑ invoked by: xeno-agent-sdk (agents call models programmatically)
```

## ABSOLUTE RULES

### 1. Every Model Output Must Be Stable
- 5+ apps consume your model outputs. Changing output format breaks everything.
- **NEVER change model output dimensions, data types, or value ranges** without coordinating with all consumers.
- Image outputs: always RGBA u8 unless documented otherwise.
- Audio outputs: always f32 PCM at the input sample rate.
- Mask outputs: always single-channel u8 (0=background, 255=foreground).

### 2. Never Remove a Model
- Apps depend on specific models. Removing one breaks features.
- Deprecate first, maintain for at least one major version.

### 3. WASM Compatibility
- All pure-compute functions should compile to `wasm32-unknown-unknown`.
- GPU-specific code (CUDA) must have CPU fallbacks.
- SIMD optimizations: always have scalar fallback path.

### 4. Performance is the Product
- This library exists because JavaScript is too slow. If a Rust function is slower than the JS equivalent, something is wrong.
- Benchmark against FFmpeg for media operations. Match or beat.
- SIMD (AVX2/NEON) for all hot paths.

## The 17 Models

Every model must maintain its API contract:

| # | Model | Input | Output | Format |
|---|-------|-------|--------|--------|
| 1 | Real-ESRGAN | RGBA image | RGBA image (2x/4x) | u8 |
| 2 | BiRefNet/RMBG-2.0 | RGBA image | Alpha mask | u8 single-channel |
| 3 | GFPGAN | RGBA image (face) | RGBA image (restored) | u8 |
| 4 | DDColor | Grayscale image | RGB image | u8 |
| 5 | Whisper | Audio (f32 PCM) | Text + timestamps | JSON |
| 6 | HTDemucs | Audio (f32 PCM) | 4 stems (f32 PCM each) | f32 |
| 7 | YOLOv8 | RGBA image | Bounding boxes + labels | JSON |
| 8 | SAM2 | RGBA image + points | Segmentation mask | u8 |
| 9 | Depth Anything | RGBA image | Depth map | f32 single-channel |
| 10 | PaddleOCR | RGBA image | Text + positions | JSON |
| 11 | OpenPose | RGBA image | Keypoints | JSON |
| 12 | RNNoise | Audio (f32 PCM) | Audio (f32 PCM, cleaned) | f32 |
| 13 | NAFNet | RGBA image | RGBA image (denoised) | u8 |
| 14 | LaMa | RGBA image + mask | RGBA image (filled) | u8 |
| 15 | Style Transfer | Content + style images | Stylized image | u8 |
| 16 | Face Detection | RGBA image | Face boxes + landmarks | JSON |
| 17 | Color Transfer | Source + reference | Recolored image | u8 |

## Code Quality

- Pure Rust. No unsafe unless absolutely necessary (and documented why).
- Every public function needs doc comments with examples.
- Benchmarks for all performance-critical paths (criterion).
- CI must run on all targets: x86_64 (Linux/Windows/macOS), aarch64 (macOS/Linux).
