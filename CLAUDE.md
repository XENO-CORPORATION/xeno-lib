# CLAUDE.md — XENO Lib Engineering Standards

## You Are Working On

**xeno-lib** — the COMPUTE layer of the entire XENO platform. Pure Rust multimedia processing library with 17 AI models. Performance is everything. This library exists because JavaScript is too slow for professional media processing, and FFmpeg is a GPL-licensed C dependency chain we refuse to adopt.

**Mandate: NO FFmpeg dependency. Build our own pure Rust codec and processing stack.**

## Critical Context

Part of a 16+ repo ecosystem. Read `../XENO CORPORATION - Full Ecosystem Report.md`.

```
YOUR REPO: xeno-lib (Layer 2 — Compute & AI)
    ↑ consumed by: xeno-pixel (upscale, bg removal, inpaint, denoise, style transfer, face detect/restore, depth, OCR, pose, segmentation)
    ↑ consumed by: xeno-motion (frame interpolation, transcription, audio separation, object detection, scene detection)
    ↑ consumed by: xeno-sound (noise reduction, stem separation, transcription, pitch detection, audio effects, visualization)
    ↑ consumed by: xeno-hub (bg removal tool, format converter)
    ↑ invoked by: xeno-agent-sdk (agents call models programmatically via agent.rs)
    ↑ changes here affect 5+ downstream consumers — NEVER change outputs without coordination
```

## ABSOLUTE RULES

### 1. Performance Is The Product

- This library exists because JavaScript and FFmpeg are not fast enough. If a Rust function is slower than the JS or FFmpeg equivalent, something is wrong.
- SIMD (AVX2 on x86_64, NEON on aarch64) for ALL hot paths. Always provide a scalar fallback.
- Benchmark every performance-critical path with criterion. Regressions are bugs.
- GPU acceleration (ONNX Runtime + CUDA) for all AI model inference.
- Target: match or beat FFmpeg for every media operation we implement.

### 2. Pure Rust Over C Bindings

- **Default to pure Rust.** Only use C bindings when no viable pure Rust alternative exists.
- Current C binding exceptions (documented and justified):
  - `ort` (ONNX Runtime) — no pure Rust ONNX inference engine exists at production quality
  - `openh264` — Cisco's BSD-licensed H.264, compiled from source via `cc`
  - `audiopus` — libopus bindings (planned: replace with pure Rust Opus encoder)
  - `dav1d` — AV1 decoder (C bindings, required for performance parity)
- Every C dependency must be justified in a comment in `Cargo.toml`.
- When a pure Rust alternative matures, migrate to it.

### 3. Every Model Output Must Be Stable

5+ apps consume your model outputs. Changing output format breaks the entire ecosystem.

**NEVER change model output dimensions, data types, or value ranges** without coordinating with ALL consumers (xeno-pixel, xeno-motion, xeno-sound, xeno-hub, xeno-agent-sdk).

| Output Type | Contract | Consumers |
|-------------|----------|-----------|
| Image outputs | RGBA u8 (always) | All apps |
| Audio outputs | f32 PCM at input sample rate | xeno-sound, xeno-motion |
| Mask outputs | Single-channel u8 (0=bg, 255=fg) | xeno-pixel |
| Depth maps | Single-channel f32 (0.0 to 1.0) | xeno-pixel |
| Structured data | JSON-serializable | All apps, xeno-agent-sdk |

### 4. Never Remove a Model

Apps depend on specific models by name and output format. Removing one breaks features across the platform. Deprecate first, maintain for at least one major version, coordinate removal with all consumers.

### 5. No FFmpeg — Build Our Own

The user has explicitly mandated: **no FFmpeg dependency**. "Zero FFmpeg" means no FFmpeg dependency, NOT no C code. C bindings to vendor SDKs (NVIDIA NVENC, Intel QSV, AMD AMF) are acceptable and necessary for hardware acceleration. Pure Rust is preferred where crates exist; C bindings are acceptable for hardware/proprietary interfaces. Depending on a monolithic 3M-line C project (FFmpeg) is not acceptable.

Current codec stack:
- Video encode: rav1e (AV1, pure Rust), OpenH264 (H.264, BSD C)
- Video decode: dav1d (AV1, C), OpenH264 (H.264), NVDEC (GPU, dynamic loading)
- Audio decode: symphonia (pure Rust — MP3, AAC, FLAC, Vorbis, ALAC, WAV)
- Audio encode: hound (WAV), flacenc (FLAC), audiopus (Opus)
- Container: mp4 crate (MP4 mux/demux), matroska crate (MKV/WebM demux)
- Image: image crate (pure Rust — PNG, JPEG, WebP, GIF, BMP, TIFF)

#### FFmpeg Replacement Roadmap

- **Phase 1 (COMPLETE):** Foundation — pure Rust image/audio processing, AV1+H.264 encode, 17 AI models, MP4 muxing, agent JSON API.
- **Phase 2 (Next):** Electron integration via N-API bindings + decode expansion (H.265, VP9, NVDEC hardening, platform prebuilds).
- **Phase 3:** Hardware encoding (NVENC, QSV, AMF, VideoToolbox) + codec expansion (ProRes, DNxHR, AAC/MP3 encode, MKV/MOV containers).
- **Phase 4:** Professional feature parity — 100+ filters, color grading pipeline, video stabilization, advanced audio effects, multi-stream muxing.

Priority order: Phase 1 (done) -> Phase 2 (N-API + decode) -> Phase 3 (hardware encode) -> Phase 4 (professional parity).

## The 17 Models

Every model must maintain its API contract. Changes require ecosystem-wide coordination.

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
| 9 | Depth Anything / MiDaS | RGBA image | Depth map | f32 single-channel |
| 10 | PaddleOCR | RGBA image | Text + positions | JSON |
| 11 | MoveNet | RGBA image | Keypoints | JSON |
| 12 | RNNoise | Audio (f32 PCM) | Audio (f32 PCM, cleaned) | f32 |
| 13 | NAFNet | RGBA image | RGBA image (denoised) | u8 |
| 14 | LaMa | RGBA image + mask | RGBA image (filled) | u8 |
| 15 | Fast NST | Content + style images | Stylized image | u8 |
| 16 | SCRFD | RGBA image | Face boxes + landmarks | JSON |
| 17 | Multi-task CNN | RGBA face crop | Age/gender/emotion | JSON |

## Feature Flag Organization

Feature flags control what gets compiled. They are organized in bundles for convenience.

### AI Bundles
- `ai` — Core image AI (bg removal, upscale, face restore, colorize, inpaint, face detect, depth)
- `ai-video` — Video/audio AI (frame interpolation, transcription, audio separation)
- `ai-vision` — Vision AI (style transfer, OCR, pose estimation, face analysis)
- `ai-full` — All AI models combined
- Each has a `-cuda` variant for GPU acceleration

### Media Bundles
- `multimedia` — Video (encode + decode) + audio (decode + encode) + subtitles + text overlay
- `pro-utils` — QR code, quality assessment, document processing, vectorization

### Top-Level Bundles
- `full` — Everything (CPU inference)
- `full-cuda` — Everything (GPU inference)

When adding a new feature:
1. Create a standalone feature flag in Cargo.toml
2. Add it to the appropriate bundle
3. Create a `-cuda` variant if it uses ONNX Runtime
4. Update the README feature flag table

## N-API Binding Development (Planned)

When implementing N-API bindings for Electron integration:

- Use **napi-rs** (not neon, not node-bindgen)
- Every binding function must be async (non-blocking Electron main thread)
- TypeScript type declarations auto-generated from `#[napi]` macros
- Streaming APIs for video/audio (don't buffer entire files in memory)
- Platform prebuilds: Windows x64, macOS ARM64 + x64, Linux x64
- Test bindings with the same inputs as the Rust unit tests
- N-API crate lives in a separate `xeno-lib-napi/` workspace member

## WASM Compilation Requirements

- All pure-compute functions (transforms, adjustments, filters, DSP) must compile to `wasm32-unknown-unknown`
- GPU-specific code (CUDA, NVDEC) must be behind feature flags that are excluded from WASM builds
- SIMD: use `wasm-simd` feature detection, always have scalar fallback
- No `std::fs`, `std::net`, or `std::process` in WASM-compatible code paths
- Test with `wasm-pack test --headless --chrome` for browser targets

## Cross-Repo Impact Analysis

Before ANY change to a public API, model output, or data format:

1. **Check all consumers**: xeno-pixel, xeno-motion, xeno-sound, xeno-hub, xeno-agent-sdk
2. **Verify the change doesn't break downstream**: model outputs, function signatures, feature flags
3. **If it breaks consumers**: plan migration across ALL affected repos, get user approval
4. **Communicate the change**: document in CHANGELOG.md, notify the orchestrator (root CLAUDE.md)

High-impact areas (changes here break multiple apps):
- Model output formats (RGBA u8, f32 PCM, JSON schemas)
- Public function signatures in `lib.rs`
- Feature flag names (apps reference these in their Cargo.toml or build scripts)
- Model file names and paths (`~/.xeno-lib/models/`)
- Agent API (`agent.rs` — xeno-agent-sdk calls this)

## Code Quality Standards

### Rust Standards
- Pure Rust. No `unsafe` unless absolutely necessary (and documented with `// SAFETY:` comment explaining why).
- Every public function needs doc comments with examples.
- All errors use `thiserror` derive macros with descriptive messages.
- No panics in library code — return `Result<T, E>` everywhere.
- No `unwrap()` in library code (use `expect()` with context or propagate with `?`).

### Testing Requirements
- **Unit tests**: Every module has `#[cfg(test)] mod tests` with coverage of edge cases
- **Benchmarks**: Every performance-critical path has a criterion benchmark in `benches/`
- **Integration tests**: End-to-end tests in `tests/` for multi-module workflows
- **CI gates**: `cargo test --all-features --locked`, `cargo build` on Ubuntu/Windows/macOS
- **Parity tests**: FFmpeg comparison matrix must not regress
- **Competitive benchmarks**: Must match or beat FFmpeg/ImageMagick/libvips on implemented features

### Performance Requirements
- All image transforms: SIMD (AVX2) with scalar fallback
- All AI inference: GPU path (CUDA) with CPU fallback
- All audio DSP: optimized for real-time processing at 44100/48000 Hz
- Benchmark before and after every performance-related change
- Document benchmark results in PR descriptions

## Build and Test

```bash
# Standard test run
cargo test

# Full feature test
cargo test --features full

# Locked dependency test (CI equivalent)
cargo test --all-features --locked

# Performance benchmarks
cargo bench --bench transforms

# Build CLI tool
cargo build --manifest-path xeno-edit/Cargo.toml --release

# Format check
cargo fmt --check

# Lint
cargo clippy --all-features -- -D warnings
```

## CI Expectations

The repository enforces these checks:

- `Check (ubuntu-latest)` — Compilation on Linux
- `Check (windows-latest)` — Compilation on Windows
- `Check (macos-latest)` — Compilation on macOS
- `Tests (Ubuntu)` — Full test suite
- `FFmpeg Parity Matrix (Ubuntu)` — Feature parity tracking
- `Competitive Benchmarks (Ubuntu)` — Performance regression gating
- `License Policy (Ubuntu)` — Dependency license compliance
- `cla` — Contributor License Agreement verification

Do not merge changes that break these signals.

## Release & Documentation Protocol

### When Releasing a New Version
1. **Version bump**: Update `version` in package.json (or Cargo.toml for Rust)
2. **Git tag**: `git tag vX.Y.Z && git push --tags`
3. **GitHub Release**: Create release with changelog via `gh release create`
4. **Build installers**: Run `npm run package` (for Electron apps) or `cargo build --release` (for Rust)
5. **Upload artifacts**: Push to Cloudflare R2 (`xeno-hub-releases` bucket) and GitHub Releases
6. **Update product pages**: Update release notes on xenostudio.ai

### Product Pages on xenostudio.ai
Every product has dedicated pages at `xenostudio.ai/products/{product-name}/`:
- `/products/{name}/` — Product overview (features, hero, download CTA)
- `/products/{name}/release-notes/` — Version history with changelogs
- `/products/{name}/download/` — Download page with OS detection
- `/products/{name}/docs/` — Documentation (getting started, features, API)

When a new version is released, the release notes page MUST be updated with:
- Version number and date
- New features (bullet list)
- Bug fixes (bullet list)
- Known issues
- Breaking changes (if any)

### Documentation Standards
- Every new feature MUST be documented before release
- Documentation lives on xenostudio.ai/products/{name}/docs/
- Use clear, concise language. Include screenshots where helpful.
- API documentation must include examples
- Keyboard shortcuts must be listed

### Versioning
- Follow semantic versioning: MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes
