# Competitive Benchmarks

This suite benchmarks `xeno-edit` against:

- `ffmpeg`
- `ImageMagick` (`magick`)
- `libvips` (`vips`)

for shared image operations and computes:

- throughput (`mean_ms`, `p95_ms`)
- memory (`max_rss_kb` on Linux)
- quality (`PSNR`, `SSIM`, `MAE`) against deterministic references

## Scenarios

- `resize_4k_to_1080p`
- `gaussian_blur_sigma2`
- `rotate_90`

## Run

```bash
# Build xeno CLI first
cargo build --manifest-path xeno-edit/Cargo.toml --release

# Run benchmark
cargo run --manifest-path tools/competitive-bench/Cargo.toml -- run \
  --xeno-bin xeno-edit/target/release/xeno-edit \
  --output benchmarks/competitors/results/latest.json
```

On Windows use `xeno-edit/target/release/xeno-edit.exe`.
If `libvips` is installed via winget and not on `PATH`, the runner auto-detects it; pass `--vips-bin <full-path-to-vips.exe>` only if detection fails.

If tools are missing locally, add `--allow-missing-tools`.

## Update Baseline

```bash
cargo run --manifest-path tools/competitive-bench/Cargo.toml -- update-baseline \
  --from benchmarks/competitors/results/latest.json \
  --to benchmarks/competitors/baseline.json
```

## Gate

```bash
cargo run --manifest-path tools/competitive-bench/Cargo.toml -- gate \
  --current benchmarks/competitors/results/latest.json \
  --baseline benchmarks/competitors/baseline.json \
  --max-time-regression-pct 12 \
  --max-psnr-drop 0.15 \
  --max-ssim-drop 0.0015 \
  --max-slowdown-vs-best-pct 10 \
  --max-p95-slowdown-vs-best-pct 12 \
  --max-psnr-gap-vs-best 0.20 \
  --max-ssim-gap-vs-best 0.0015 \
  --min-psnr 35 \
  --min-ssim 0.97 \
  --require-competitors \
  --require-all-competitors
```

The gate fails if xeno quality drops, timing regresses against baseline (same OS), any competitor is unavailable (when required), xeno p95/mean latency is too slow vs the best competitor, or xeno quality trails the best competitor beyond the configured margin.

Note:
- Baseline timing regression is only applied when `baseline.host_os` matches the current run OS.
- Keep a Linux baseline for CI (`ubuntu-latest`) to enforce cross-PR timing regressions in GitHub Actions.
- CI benchmark job uploads both `latest.json` and `linux-baseline-candidate.json` artifacts so you can refresh `benchmarks/competitors/baseline.json` from measured Linux runs.
