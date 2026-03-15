# Contributing to xeno-lib

We welcome contributions to xeno-lib and xeno-edit. This document explains how to work on the project without creating regressions in media correctness, performance, or licensing posture.

## Contributor License Agreement (CLA)

Before your first pull request can be merged, you must agree to the terms in [CLA.md](CLA.md).

On your first pull request, add this exact sentence as a PR comment:

`I have read the CLA Document and I hereby sign the CLA`

That acknowledgement is required before a maintainer will merge your first contribution.
If the `cla` check ran before your comment was posted, ask a maintainer to rerun it.

**Why we require a CLA:** XENO Corporation intends to use xeno-lib in both open-source and commercial products. The CLA ensures the project can accept outside contributions while preserving clear IP rights for maintainers, users, and downstream commercial distribution.

## Development Setup

### Prerequisites

- Rust stable toolchain
- Git
- NASM for accelerated `rav1e` builds on supported platforms
- Platform-specific dependencies for optional features:
  - `dav1d` + `pkg-config` for `video-decode-sw`
  - NVIDIA drivers for NVDEC paths
  - ONNX Runtime prerequisites for AI features where applicable

### Build

```bash
cargo build --locked
cargo build --manifest-path xeno-edit/Cargo.toml --release --locked
```

### Test

```bash
cargo test --locked
cargo test --all-features
cargo test --all-features h264_mp4_output_round_trips_through_mp4_demuxer -- --nocapture
```

### Benchmarks and Gates

```bash
cargo bench --bench transforms

cargo run --manifest-path tools/competitive-bench/Cargo.toml -- run \
  --xeno-bin xeno-edit/target/release/xeno-edit \
  --output benchmarks/competitors/results/latest.json

python tools/ffmpeg-parity/generate_matrix.py \
  --xeno-bin xeno-edit/target/release/xeno-edit \
  --spec benchmarks/ffmpeg/parity_spec.json \
  --output-json benchmarks/ffmpeg/results/latest.json \
  --output-md benchmarks/ffmpeg/results/latest.md \
  --baseline benchmarks/ffmpeg/baseline.json \
  --baseline-candidate benchmarks/ffmpeg/results/baseline-candidate.json \
  --fail-on-regression
```

## Before You Open a PR

Ensure the relevant checks pass for your change:

```bash
cargo fmt --check
cargo clippy --all-features --all-targets
cargo test --all-features
cargo check --manifest-path xeno-edit/Cargo.toml --release --locked
```

If your change touches parity claims or benchmark-sensitive paths, also run the relevant parity or competitor benchmark command.

## PR Guidelines

- Keep PRs focused. One feature or one fix per PR.
- Add or update tests for new behavior and regressions.
- Update documentation when CLI behavior, feature flags, or supported formats change.
- Do not claim parity, quality, or performance improvements without evidence.
- If you add a new media feature, update the library API, CLI surface, capability reporting, tests, docs, and parity tracking together.
- Call out platform-specific assumptions, especially for Windows GPU or codec paths.

## Where To Put Changes

| Change Type | Location |
|---|---|
| Library API and transforms | `src/` |
| CLI commands and UX | `xeno-edit/src/main.rs` |
| FFmpeg parity tracking | `benchmarks/ffmpeg/`, `tools/ffmpeg-parity/` |
| Competitor benchmarks | `benchmarks/competitors/`, `tools/competitive-bench/` |
| CI enforcement | `.github/workflows/ci.yml` |
| Examples | `examples/` |
| Integration tests | `tests/` |

## Code Style

- Use `cargo fmt` and keep changes rustfmt-clean.
- Prefer explicit, testable behavior over clever abstractions.
- Keep `unsafe` minimized and document invariants if it is required.
- Prefer primary-source documentation when adding codec, container, or model support.
- Do not change parity or benchmark baselines casually; treat them as controlled evidence files.

## Reporting Bugs and Security Issues

- Normal bugs and feature requests: open a GitHub issue.
- Security-sensitive issues: do **not** open a public issue. Follow [SECURITY.md](SECURITY.md).

## License

By contributing to xeno-lib, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE), subject to the terms of [CLA.md](CLA.md).
