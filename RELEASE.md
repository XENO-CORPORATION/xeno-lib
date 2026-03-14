# Release Process

## Purpose

This document defines how xeno-lib and xeno-edit are versioned, stabilized, and shipped. The goal is predictable releases without weakening correctness, capability tracking, or platform-specific media behavior.

## Versioning Policy

xeno-lib follows Semantic Versioning 2.0.

- MAJOR releases are for breaking API, CLI, feature-flag, or behavior changes that require downstream migration.
- MINOR releases add backwards-compatible functionality, new supported formats, new AI capabilities, or major performance improvements.
- PATCH releases are limited to backwards-compatible fixes for correctness, packaging, security, and severe regressions.

Pre-release candidates may use the form `vX.Y.Z-rc.N`.

## Branching Strategy

- `master` is the main integration branch.
- Release branches may be cut as `release/x.y` when a stable line needs patch support.
- Any release-only fix must be merged back into `master` immediately after the release.

## Release Checklist

Before any stable release, confirm:

- `CHANGELOG.md` reflects user-visible changes.
- CI is green.
- `cargo fmt --check`, `cargo clippy --all-features --all-targets`, and `cargo test --all-features` pass on the release branch.
- `cargo build --manifest-path xeno-edit/Cargo.toml --release --locked` succeeds.
- Windows runtime smoke checks pass for container H.264 transcode and capability reporting.
- FFmpeg parity generation passes without regression.
- Competitive benchmark gates are reviewed for any area claiming speed or quality wins.
- Docs are updated for new CLI commands, feature flags, model requirements, or platform prerequisites.

## Patch Release Rules

A change qualifies for patch release only if it:

- fixes a security issue
- fixes incorrect output, corrupted files, or broken round-trips
- restores a broken build, packaging path, or supported platform
- resolves a documented regression in CLI or library behavior

The following do not qualify by themselves:

- new features
- broad refactors without urgent user impact
- parity wish-list work that does not fix an existing regression
- benchmark work without a correctness or stability reason

## Validation Expectations

Validation should match the area that changed:

- Image transforms: relevant unit tests and golden outputs
- Audio/video encode or decode: focused round-trip tests and CLI smoke runs
- Container work: demux, mux, and transcode verification across representative files
- Parity changes: regenerate `benchmarks/ffmpeg/results/latest.json` and gate against the current baseline
- Performance-sensitive changes: run or spot-check the affected benchmarks

## Ownership

The release manager for a cycle is responsible for driving the checklist, tagging releases, and coordinating any go or no-go decision.
