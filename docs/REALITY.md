# xeno-lib Reality Check

> Status note (March 15, 2026): use `benchmarks/ffmpeg/OVERCOME_TRACKER.md` for the active roadmap and `benchmarks/ffmpeg/results/latest.md` for the CI-generated parity snapshot. This file is still useful context, but it is not the operational source of truth.

> **This document is an honest assessment of where xeno-lib stands compared to FFmpeg and other industry tools. No marketing, just facts.**

---

## The Hard Truth

**We're at maybe 15-20% of FFmpeg's capabilities for video/audio — but we have a clear 4-phase plan to close the gap on everything that matters for creative apps, and we already surpass FFmpeg in AI capabilities it will never have.**

The goal is NOT "zero C code" — that's unrealistic. The goal is **zero FFmpeg dependency, full pipeline control, AI capabilities FFmpeg can never have.** C bindings to specific vendor SDKs (NVIDIA NVENC, Intel QSV, AMD AMF) are acceptable and necessary. Depending on a monolithic 3M-line C project is not.

---

## What FFmpeg Actually Has (That We Don't)

| Category | FFmpeg | xeno-lib | Gap |
|----------|--------|----------|-----|
| **Video Codecs** | H.264, H.265, VP8, VP9, AV1, ProRes, DNxHD, MPEG-2/4, ~100+ more | AV1 (slow), H.264 (basic) | **Massive** |
| **Hardware Encoding** | NVENC, QSV, AMF, VideoToolbox, VAAPI | None | **Critical** |
| **Hardware Decoding** | NVDEC, CUVID, QSV, DXVA2, VideoToolbox | Partial NVDEC | **Major** |
| **Audio Codecs** | AAC, MP3, Opus, AC3, DTS, FLAC, Vorbis, 50+ more | Opus*, FLAC, WAV | **Large** |
| **Containers** | MP4, MKV, AVI, MOV, WebM, FLV, TS, MXF, 50+ more | MP4 (basic), IVF | **Massive** |
| **Filters** | 400+ video/audio filters, complex graphs | ~20 basic filters | **Massive** |
| **Streaming** | RTMP, RTSP, HLS, DASH, SRT, RTP | None | **Total** |
| **Subtitles** | Full ASS/SRT/VTT with styling, burn-in, extract | Basic parsing | **Large** |

*Opus uses libopus C bindings, not pure Rust

---

## Honest Feature-by-Feature Assessment

### Video Encoding - WEAK

```
FFmpeg:   x264 encodes 4K60 at 200+ fps with NVENC
xeno-lib: rav1e does maybe 2-5 fps for 1080p on CPU
```

| What We Have | What's Missing |
|--------------|----------------|
| AV1 via rav1e (pure Rust, slow) | H.265/HEVC encoding |
| H.264 via OpenH264 (baseline only) | Hardware encoding (NVENC/QSV/AMF) |
| | VP9 encoding |
| | ProRes/DNxHD (professional) |
| | x264/x265 quality encoders |

**Reality:** rav1e is quality-focused but painfully slow. OpenH264 is limited to baseline profile. No hardware acceleration means we can't compete on speed.

### Video Decoding - WEAK

| What We Have | What's Missing |
|--------------|----------------|
| AV1 via dav1d (good) | H.264 software decoder |
| Partial NVDEC | H.265 software decoder |
| | VP9 decoder |
| | Complete hardware decoding |

**Reality:** If someone hands us an H.264 file (90% of videos), we can't decode it in software.

### Audio - MEDIOCRE

| What We Have | What's Missing |
|--------------|----------------|
| Symphonia decoding (good coverage) | AAC encoding |
| Opus encoding (via libopus) | MP3 encoding |
| FLAC encoding (pure Rust) | AC3/DTS encoding |
| WAV encoding (pure Rust) | Advanced audio filters |
| Basic effects (reverb, EQ, etc.) | Production-tested effects |

**Reality:** Decoding is decent thanks to Symphonia. Encoding options are limited. New effects are untested.

### Container Support - WEAK

| What We Have | What's Missing |
|--------------|----------------|
| MP4 muxing (basic) | Full MP4 with all features |
| IVF (raw AV1) | MKV muxing |
| MKV demuxing (read only) | AVI support |
| | MOV support |
| | WebM support |
| | FLV/TS/MXF |

**Reality:** We can create basic MP4 files. That's about it.

### Filters - MINIMAL

| What We Have (~20) | What FFmpeg Has (400+) |
|--------------------|------------------------|
| Flip, rotate, crop, resize | Complex filter graphs |
| Brightness, contrast, saturation | Temporal filters |
| Blur, sharpen, edge detect | Motion compensation |
| Basic color adjustments | Professional color grading |
| | Scene detection |
| | Noise reduction (advanced) |
| | Deinterlacing (multiple methods) |
| | Stabilization |

### Video Editing - STUBS ONLY

The `video/edit.rs` module has code but:
- Requires multiple features enabled simultaneously
- Never been tested with real video files
- No crossfades or transitions
- No complex timeline editing
- No audio sync handling for edits

### Streaming - INTENTIONALLY OUT OF SCOPE

| Protocol | Status |
|----------|--------|
| RTMP | Not planned |
| RTSP | Not planned |
| HLS | Not planned |
| DASH | Not planned |
| SRT | Not planned |
| WebRTC | Not planned |

**Decision:** XENO is a creative suite (image editor, video editor, DAW), not a streaming platform. Streaming protocols are out of scope. This is not a gap — it's a deliberate boundary.

---

## The AI "Advantage" - Context

### What the AI features actually are:

```
AI Module = Config struct + ONNX model loading + Tensor conversion + Model inference
```

We're essentially **ONNX Runtime wrappers** with nice Rust APIs.

The actual intelligence comes from:
- Pre-trained models (someone else trained them)
- ONNX Runtime (Microsoft's C++ library)
- CUDA (NVIDIA's runtime)

### What a Python user can achieve:

```python
# Literally the same thing in 5 lines
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'])
output = session.run(None, {"input": preprocessed_image})
```

### Important notes:
- Models are NOT included - users download separately (~1.5GB total)
- Requires CUDA for reasonable performance
- We provide nice APIs, but the heavy lifting is external

### What IS genuinely valuable:
- Unified Rust API for multiple AI tasks
- Proper error handling and type safety
- Integration with image/video pipeline
- GPU memory management
- Batch processing support

---

## The "Pure Rust" Claim - Audit

| Dependency | Reality | Pure Rust? |
|------------|---------|------------|
| `image` | Pure Rust image processing | Yes |
| `rav1e` | Pure Rust AV1 encoder | Yes |
| `symphonia` | Pure Rust audio decoder | Yes |
| `hound` | Pure Rust WAV encoder | Yes |
| `flacenc` | Pure Rust FLAC encoder | Yes |
| `ort` | Links to ONNX Runtime C++ | **No** |
| `openh264` | Cisco's C library | **No** |
| `audiopus` | Links to libopus C | **No** |
| `dav1d` | C library bindings | **No** |
| `libloading` | For NVDEC (CUDA) | **No** |

**Reality:** Core image transforms are pure Rust. Most video/audio codecs use C libraries.

---

## Production Reality

| Metric | FFmpeg | xeno-lib |
|--------|--------|----------|
| Years in production | 20+ | 0 |
| Companies using | Netflix, YouTube, Facebook, Twitch, Disney+, etc. | None |
| Bug fixes applied | Thousands | None (too new) |
| CVE patches | Hundreds | None yet |
| Documentation | Comprehensive wiki, man pages, examples | README only |
| Community | Massive (IRC, mailing lists, forums) | None |
| Stack Overflow answers | 50,000+ | 0 |
| Books written about it | Multiple | 0 |

---

## Honest Comparison Summary

| Area | vs FFmpeg | Phase to Address | Notes |
|------|-----------|-----------------|-------|
| Video codec support | **5%** | Phase 2-3 | Only AV1/H.264, H.265/VP9/ProRes coming |
| Audio codec support | **20%** | Phase 3 | Decode OK, AAC/MP3 encode coming |
| Container support | **10%** | Phase 3 | MP4 basic; MKV/MOV/WebM planned |
| Filters | **5%** | Phase 4 | ~20 vs 400+; targeting 100+ that matter |
| Hardware acceleration | **5%** | Phase 2-3 | Partial NVDEC; NVENC/QSV/AMF planned |
| Streaming | **N/A** | Never | Out of scope by design |
| Performance (CPU) | **10-20%** | Phase 3-4 | SIMD for transforms; codecs need hardware accel |
| Stability | **?%** | Ongoing | Untested in production |
| AI features | **Unique** | Done | 17 models FFmpeg will never have |
| API ergonomics | **Better** | Done | Rust > CLI string parsing |
| Memory safety | **Better** | Done | Rust compiler guarantees |
| WASM/browser target | **Unique** | Done | FFmpeg can't run in browser natively |

---

## Roadmap to FFmpeg Independence

The goal is NOT to replicate FFmpeg's entire 3M-line codebase. It's to build everything our creative apps (xeno-pixel, xeno-motion, xeno-sound) actually need, with AI capabilities FFmpeg will never have.

### Phase 1 — Foundation (COMPLETE)
- [x] Pure Rust image processing (52 transforms, SIMD AVX2)
- [x] Pure Rust audio decode (symphonia — MP3, AAC, FLAC, Vorbis, ALAC, WAV, AIFF)
- [x] Pure Rust audio encode (WAV via hound, FLAC via flacenc, Opus via audiopus)
- [x] Pure Rust AV1 encode (rav1e)
- [x] H.264 encode (OpenH264 — Cisco C library, BSD licensed)
- [x] 17 AI models via ONNX Runtime + CUDA
- [x] Agent-friendly JSON API
- [x] MP4 container muxing

### Phase 2 — Electron Integration & Decode Expansion
- [ ] N-API bindings via napi-rs (Electron apps call xeno-lib directly, not subprocess)
- [ ] Async N-API with streaming results for large operations
- [ ] Platform-specific prebuilt binaries (Windows x64, macOS ARM64, Linux x64)
- [ ] H.265/HEVC decode (via minimal C binding or pure Rust when available)
- [ ] VP9 decode
- [ ] Hardware decode: NVDEC (NVIDIA GPU) — already partially implemented via libloading
- [ ] Software AV1 decode improvement (dav1d integration hardening)

### Phase 3 — Hardware Encoding & Codec Expansion
- [ ] NVENC (NVIDIA hardware H.264/H.265/AV1 encode) — C SDK bindings, 10-50x faster than CPU
- [ ] QSV (Intel Quick Sync) — C SDK bindings
- [ ] AMF (AMD Advanced Media Framework) — C SDK bindings
- [ ] VideoToolbox (macOS hardware encode) — ObjC/C bindings
- [ ] ProRes encode/decode (reverse-engineered or FFmpeg-independent implementation)
- [ ] DNxHR/DNxHD for professional video workflows
- [ ] AAC encode (via fdk-aac bindings or pure Rust implementation when mature)
- [ ] MP3 encode (via lame bindings)
- [ ] MKV/WebM container support (matroska crate expansion)
- [ ] MOV container support

### Phase 4 — Professional Feature Parity
- [ ] 100+ filters/effects that creative apps actually need (not all 400+ FFmpeg filters — most are niche)
- [ ] Professional color grading pipeline (LUT application, color space conversions)
- [ ] Video stabilization
- [ ] Advanced audio effects (multiband compression, limiter, de-esser, noise gate)
- [ ] Subtitle burning/rendering
- [ ] Multi-stream muxing (multiple audio tracks, subtitle tracks)
- [ ] Chapter/metadata support
- [ ] Thumbnail/poster frame extraction at scale

### What We Will NEVER Build (and why)
- Streaming protocols (RTMP/RTSP/HLS/DASH) — XENO is a creative suite, not a streaming platform
- 400+ niche FFmpeg filters — we build what our apps need, not a kitchen sink
- Legacy format support (FLV, WMV, RealMedia) — nobody needs these in 2026

---

## What xeno-lib Actually Is (Honest Positioning)

### What we ARE:
- A Rust multimedia library purpose-built for AI-native creative applications
- A growing FFmpeg-independent media pipeline with a clear 4-phase roadmap
- 17 AI models integrated that FFmpeg will never have
- A memory-safe alternative to the 3M-line C codebase that is FFmpeg
- The compute backbone for xeno-pixel, xeno-motion, and xeno-sound

### What we ARE NOT (yet):
- Feature-complete compared to FFmpeg for traditional codec/filter work (Phase 2-4 will close this gap)
- Production-hardened at FFmpeg's scale (20+ years vs months)
- A streaming solution (and never will be — out of scope by design)

### What we ARE NOT (ever):
- "Pure Rust, zero C code" — that was never realistic. Hardware encoders (NVENC, QSV, AMF) require vendor C SDKs. The actual goal is zero FFmpeg dependency, not zero C.
- A kitchen-sink filter library — we build the 100+ filters creative apps need, not 400+ niche ones

### Best use cases today:
1. AI-powered image processing (upscaling, background removal, inpainting, face restore, depth, etc.)
2. Image transforms and color adjustments (52 SIMD-accelerated operations)
3. Audio decode + effects pipeline for DAW integration
4. Basic video encoding (AV1, H.264) for creative app export
5. Agent-driven media processing via JSON API

---

## Tracking Progress

Use this section to track improvements:

### Completed
- [x] Basic image transforms
- [x] Color adjustments
- [x] AV1 encoding (rav1e)
- [x] H.264 encoding (OpenH264, basic)
- [x] Audio decoding (Symphonia)
- [x] MP4 muxing (basic)
- [x] 10 AI features (Phase 1-2)
- [x] 4 new AI features (Phase 3)
- [x] Subtitle parsing
- [x] QR code support
- [x] Audio effects
- [x] Quality assessment
- [x] Document processing

### In Progress
- [ ] Video editing (code exists, needs testing)
- [ ] Audio visualization (code exists, needs testing)

### Not Started — Phase 2 (Next Priority)
- [ ] N-API bindings for Electron integration
- [ ] H.265/HEVC decode
- [ ] VP9 decode
- [ ] NVDEC hardening

### Not Started — Phase 3
- [ ] NVENC hardware encoding
- [ ] QSV / AMF hardware encoding
- [ ] ProRes / DNxHR support
- [ ] AAC / MP3 encoding
- [ ] MKV/MOV container muxing

### Not Started — Phase 4
- [ ] 100+ creative filters
- [ ] Professional color grading pipeline
- [ ] Video stabilization
- [ ] Advanced audio effects (multiband compression, limiter, de-esser)

### Explicitly Out of Scope
- Streaming protocols (RTMP/RTSP/HLS/DASH) — not a streaming platform
- 400+ niche FFmpeg filters — only what creative apps need
- Legacy formats (FLV, WMV, RealMedia)

---

*Last updated: 2026-03-15*
*This document should be updated as capabilities improve.*
