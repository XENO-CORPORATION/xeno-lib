# License Policy

`xeno-lib` source code is licensed under Apache-2.0. That is the repository
license for code authored in this project. Dependency and toolchain licenses
are enforced separately through `cargo-deny`.

## Policy

- Do not introduce GPL, AGPL, or LGPL-only dependencies into the shipped graph.
- Keep dependency licensing explicit and reviewable.
- Prefer permissive licenses by default.
- Scope non-permissive or unusual licenses to crate-level exceptions instead of
  allowing them globally.

## Current Scoped Exceptions

- `MPL-2.0` for the `symphonia` audio decoding stack.
  - This is a deliberate exception, not a global allowance.
  - If the project moves to a permissive-only policy, `symphonia` is the first
    dependency family that must be replaced.
- `NCSA` for `libfuzzer-sys`.
  - This enters through the AVIF and AV1 image stack.
  - It is permissive, but it is tracked explicitly rather than allowed
    repository-wide.
- `CDLA-Permissive-2.0` for `webpki-root-certs`.
  - This appears in the ONNX Runtime download and build path through `ort`.
  - It is tracked as a build-time supply-chain exception, not a blanket
    allowance.

## Clarified Crates

- `nom-exif` is treated as `MIT` based on its shipped `LICENSE` file.
  - The crate does not publish an SPDX `license` field in its manifest, so the
    policy records a file-hash-based clarification for reproducibility.

## Additional Commercial Caution

Licensing is not the whole commercial-risk picture for multimedia software.
Codec patents and redistribution terms can still matter even when the source
license is permissive.

Current areas that require separate commercial review:

- `openh264` and H.264 functionality
- ONNX Runtime binaries and model artifacts
- external pretrained model weights
- platform SDKs and GPU backends

## Enforcement

The repository enforces this policy with:

- `deny.toml`
- `cargo deny --all-features --locked check licenses`
- CI checks on pull requests and default-branch pushes

This document is an engineering policy summary, not legal advice.
