# agents.md — XENO Lib (for Codex CLI and AI agents)

## Identity

You are working on **xeno-lib**, the pure Rust multimedia processing library with 17 AI models. This is the COMPUTE layer of the XENO platform — Layer 2 in the architecture. Performance is everything.

## Ecosystem

Read `../XENO CORPORATION - Full Ecosystem Report.md`. You are the compute backbone consumed by every creative app and the agent runtime. Other agents work on consumer repos simultaneously. Never assume you are alone.

```
YOUR REPO: xeno-lib (Layer 2 — Compute & AI)
    ↑ consumed by: xeno-pixel (image AI: upscale, bg removal, inpaint, face restore, depth, OCR, pose, style transfer)
    ↑ consumed by: xeno-motion (video AI: frame interpolation, transcription, audio separation)
    ↑ consumed by: xeno-sound (audio AI: stem separation, noise reduction, transcription, DSP effects)
    ↑ consumed by: xeno-hub (bg removal, format conversion via xeno-edit CLI)
    ↑ invoked by: xeno-agent-sdk (agents call models via agent.rs JSON API)
```

## Safety

1. **NEVER change model output formats.** Image outputs are RGBA u8. Audio outputs are f32 PCM. Masks are single-channel u8. Structured data is JSON. 5+ apps depend on these contracts.
2. **NEVER remove a model.** Deprecate first, maintain for one major version, coordinate with all consumers.
3. **NEVER add an FFmpeg dependency.** The mandate is pure Rust codecs. C bindings only when no viable Rust alternative exists (and must be justified).
4. **NEVER change public function signatures or feature flag names** without checking all downstream consumers first.
5. **NEVER merge code that regresses performance benchmarks or FFmpeg parity.**
6. **Ask before any destructive or cross-repo-impacting action.**

## Stack

- **Language**: Rust (2021 edition), pure Rust wherever possible
- **AI inference**: ONNX Runtime + CUDA (via `ort` crate)
- **SIMD**: AVX2 on x86_64, scalar fallbacks for all targets
- **Video encode**: rav1e (AV1, pure Rust), OpenH264 (H.264, BSD C source)
- **Video decode**: dav1d (AV1, C), OpenH264 (H.264), NVDEC (GPU, dynamic loading)
- **Audio decode**: symphonia (pure Rust — MP3, AAC, FLAC, Vorbis, ALAC, WAV)
- **Audio encode**: hound (WAV), flacenc (FLAC, pure Rust), audiopus (Opus)
- **Audio DSP**: Pure Rust effects (reverb, EQ, pitch shift, delay, distortion, chorus, flanger)
- **Image**: image crate (pure Rust — PNG, JPEG, WebP, GIF, BMP, TIFF)
- **Containers**: mp4 crate (MP4), matroska crate (MKV/WebM)
- **Vectorization**: vtracer (vendored, raster-to-SVG)

## Dependencies (Upstream)

- `ort` (ONNX Runtime) — AI model inference
- `ndarray` — tensor manipulation for model I/O
- `image` — image loading, saving, pixel manipulation
- `rayon` — parallel iteration
- `symphonia` — audio decoding
- `rav1e` — AV1 video encoding (vendored)
- `flacenc` — FLAC audio encoding (vendored)

## Consumers (Downstream)

| Consumer | What It Uses |
|----------|-------------|
| xeno-pixel | bg removal, upscale, inpaint, face restore, colorize, face detect, depth, OCR, pose, style transfer |
| xeno-motion | frame interpolation, transcription, audio separation |
| xeno-sound | audio separation, transcription, audio effects, visualization |
| xeno-hub | bg removal, format conversion (via xeno-edit CLI) |
| xeno-agent-sdk | agent.rs JSON API for programmatic model invocation |

## Critical Data Format Contracts

These are the integration points between xeno-lib and the rest of the platform. Breaking these breaks the ecosystem.

| Data | Format | Contract |
|------|--------|----------|
| Image model output | RGBA u8 | 4 channels, 8 bits per channel, premultiplied alpha |
| Audio model output | f32 PCM | Interleaved, at input sample rate |
| Mask output | Single-channel u8 | 0 = background, 255 = foreground |
| Depth map output | Single-channel f32 | Normalized 0.0 (near) to 1.0 (far) |
| Bounding boxes | JSON | `{ x, y, width, height, confidence, label }` |
| Keypoints | JSON | `{ points: [{ x, y, confidence }], connections }` |
| Transcription | JSON | `{ segments: [{ text, start, end, words }] }` |
| Model files | ONNX | Stored in `~/.xeno-lib/models/`, downloaded on first use |

## Quality Gates

- `cargo test --all-features --locked` must pass
- `cargo fmt --check` must pass
- `cargo clippy --all-features -- -D warnings` must pass
- FFmpeg parity matrix must not regress
- Competitive benchmarks must not regress
- No `unsafe` without `// SAFETY:` justification
- No `unwrap()` in library code
- No `console.log` equivalent — use `tracing` or `log` crate
- Every public function has doc comments
