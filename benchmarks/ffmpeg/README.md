# FFmpeg Parity Tracking

This folder tracks **objective FFmpeg parity** for `xeno-lib`/`xeno-edit` using a spec-driven matrix.

## Files

- `parity_spec.json`: Capability rows and status rules (`have`, `partial`, `missing`).
- `OVERCOME_TRACKER.md`: Strategic roadmap for closing parity gaps and surpassing FFmpeg in target workflows.
- `baseline.json`: Regression baseline used by CI.
- `results/latest.json`: Generated machine-readable report (CI artifact).
- `results/latest.md`: Generated human-readable matrix (CI artifact).
- `results/baseline-candidate.json`: Generated status snapshot for baseline updates.

## Generate Matrix Locally

Build `xeno-edit` first:

```bash
cargo build --manifest-path xeno-edit/Cargo.toml --release
```

Then generate and gate:

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

On Windows, use `xeno-edit/target/release/xeno-edit.exe` for `--xeno-bin`.

## Status Semantics

- `HAVE`: feature exists in current CLI capability surface.
- `PARTIAL`: feature exists but with known limits (e.g., NVDEC dependency, demux limitations).
- `MISSING`: feature is not exposed in current CLI capability surface.

## Updating Baseline

Only update `baseline.json` intentionally when capability changes are accepted.
Use `results/baseline-candidate.json` as the source of truth for proposed updates.
