# FFmpeg Overcome Tracker

This is the working roadmap for making `xeno-lib` and `xeno-edit` beat FFmpeg in the workflows we care about.

## Source Of Truth

- Objective parity spec: `benchmarks/ffmpeg/parity_spec.json`
- Regression baseline: `benchmarks/ffmpeg/baseline.json`
- Generated report artifact: `benchmarks/ffmpeg/results/latest.md`
- Competitive benchmark gate: `benchmarks/competitors/baseline.json`

## Current Snapshot

Current local baseline from `benchmarks/ffmpeg/results/latest.json`:

- Objective rows tracked: `46`
- `HAVE`: `33`
- `PARTIAL`: `5`
- `MISSING`: `8`
- Weighted parity score: `77.17%`

Interpretation:

- We are strong on image transforms, AI utilities, and basic CLI ergonomics.
- We are still materially behind on codec breadth, container breadth, encode backends, and cross-platform decode fallback.
- Parity alone is not enough. To actually beat FFmpeg, we also need benchmark wins, production hardening, and better AI-native workflows.

## Definition Of "Overcome FFmpeg"

We only claim this when all of the following are true:

- Objective parity tracker has no `MISSING` rows in target workflows.
- Current `PARTIAL` rows are either promoted to `HAVE` or explicitly accepted as non-goals.
- Competitive benchmark gate passes against FFmpeg, ImageMagick, and libvips for our target image workflows.
- The xeno-only advantages stay ahead: AI pipelines, agent-native JSON control, recenter/layout-aware transforms, vectorization, and workflow composition.
- New media features are production-hardened with tests, docs, and CI coverage.

## P0: Close Current Red Gaps

These are direct gaps already visible in the parity matrix and should be the first execution lane.

- [ ] H.265/HEVC encode
- [ ] VP9 encode
- [ ] Hardware video encode surface (`gpu_encode`)
- [x] Software AV1 decode fallback in the shipped CLI build
- [x] Software H.264 decode fallback in the shipped CLI build
- [ ] AV1-in-MP4 muxing
- [ ] WebM muxing
- [ ] MKV muxing
- [x] AVI demux packet iteration and seek
- [x] Opus encode in `xeno-edit audio-encode`
- [ ] AAC encode in `xeno-edit audio-encode`
- [ ] MP3 encode in `xeno-edit audio-encode`

## P0: Turn Current Yellow Gaps Green

These exist today but are incomplete or too backend-dependent.

- [x] AV1 decode without relying only on NVDEC
- [x] H.264 decode without relying only on NVDEC
- [ ] H.265 decode without relying only on NVDEC
- [ ] VP8 decode without relying only on NVDEC
- [ ] VP9 decode without relying only on NVDEC
- [ ] MKV demux frame/audio iteration
- [ ] MKV demux seeking
- [ ] WebM demux frame/audio iteration
- [ ] WebM demux seeking

## P1: Big Systems Still Missing From The Parity Matrix

These are major FFmpeg-class capabilities we still need to track and eventually add to the objective matrix once surfaced in code/CLI.

- [ ] Filter graph system for chaining non-trivial audio/video pipelines
- [ ] Streaming protocols: HLS, DASH, RTMP, RTSP, SRT
- [ ] Cross-platform hardware backend matrix: NVENC, QSV, AMF, VideoToolbox, VAAPI
- [ ] Timeline editing with audio sync guarantees, transitions, and overlays
- [ ] Subtitle extraction, muxing, burn-in, and style parity
- [ ] Broader container support: MOV, TS, FLV, MXF, OGG/OGM
- [ ] More video codec coverage where it matters commercially: ProRes, DNxHD/DNxHR, MPEG-2

## P1: Performance And Quality Requirements

Matching feature names is not enough. We need measurable wins or at least non-losses on the tasks we claim.

- [ ] Add codec-focused benchmarks, not just transform benchmarks
- [ ] Add quality gates for encode/decode output where applicable
- [ ] Add long-running soak tests for mux/decode/transcode paths
- [ ] Add Windows CLI smoke coverage for container H.264 transcode so NVDEC/OpenH264 selection regressions fail in CI
- [ ] Add memory ceilings for large batch/video workflows
- [ ] Track Windows, Linux, and macOS behavior separately where backends differ

## Interop Bugs Surfaced

- [x] Make xeno-generated H.264 MP4 output round-trip through our own MP4 demux path
  Current state: fixed by switching H.264 MP4 writing to `mp4::Mp4Writer`; `open_container()` now parses xeno-generated MP4 output.
- [x] Fix Windows `video-transcode` failure on H.264 MP4 when the NVDEC path is selected
  Current state: fixed by preferring the validated OpenH264 software path for container H.264 transcoding; `mp4 -> av1` and `mp4 -> h264` transcode both succeed locally on Windows.

## P1: Keep The Xeno Advantage

These are the reasons to choose xeno even before full FFmpeg parity. They need to keep getting stronger while parity work continues.

- [ ] Agent-first JSON execution remains first-class
- [ ] AI pipelines stay faster to use than hand-built FFmpeg + Python glue
- [ ] Recenter, remove-bg, vectorize, and subject-aware tooling remain easier than competitor stacks
- [ ] Cross-feature composition improves: transform + AI + export in one pipeline

## Workstream Owners In Code

- Video encode and mux: `src/video/encode/`, `src/video/mux.rs`, `xeno-edit/src/main.rs`
- Container work: `src/video/container/`
- Audio encode surface: `src/audio/encode/`, `xeno-edit/src/main.rs`
- Capability reporting: `src/agent.rs`
- FFmpeg parity tracking: `benchmarks/ffmpeg/`, `tools/ffmpeg-parity/`
- Competitor benchmark gate: `benchmarks/competitors/`, `tools/competitive-bench/`
- CI enforcement: `.github/workflows/ci.yml`

## Completion Rule For Any New Media Feature

A feature does not count as done until all of the following exist:

- Library API
- CLI surface in `xeno-edit`
- Capability reporting in `src/agent.rs` when relevant
- Tests for happy path and failure path
- Docs update
- Parity spec update if it affects FFmpeg comparison
- Benchmark or quality gate if performance/quality claims are involved

## Recommended Execution Order

1. Audio encode surface: finish AAC/MP3 path, keeping Opus/WAV/FLAC capability-gated in CLI and parity tracking.
2. Container completion: AV1-in-MP4, WebM mux, MKV mux.
3. Decode hardening: add software fallback paths so current decode rows stop being GPU-only partials.
4. Encode expansion: H.265 and VP9.
5. Backend expansion: hardware encode and broader platform coverage.
6. Only after that, move into streaming and full filter-graph work.
