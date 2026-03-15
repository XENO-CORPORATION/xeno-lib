# Xeno Lib Agent Guide

This repository is a public, production-grade open-source project. Treat every change as if it will be reviewed externally, preserved in long-term history, and used to judge the engineering quality of `xeno-lib`.

## Operating Standard

- Optimize for durable repository quality, not short-term convenience.
- Prefer explicit, reviewable changes over clever shortcuts.
- Keep the repository in a state that is safe to show publicly at any time.
- Maintain truthful docs, honest capability reporting, and reproducible builds.

## Git And PR Workflow

- Do not push feature or maintenance work directly to `master`.
- Create a branch, push it, and land changes through a pull request.
- Keep commit history intentional:
  - use focused commits
  - use clear conventional-style commit subjects
  - avoid noisy fixup chains unless they will be squashed before merge
- Preserve branch protection. Do not weaken repository rules unless the user explicitly asks, and restore them immediately after any temporary exception.
- Prefer `rebase` merges to keep history linear unless the user requests otherwise.

## Commit Identity

- Commits intended for PR review must use a GitHub-linked author identity.
- If you create commits on behalf of the repo owner, use the GitHub-linked noreply email for that account.
- If commits are authored with an email that GitHub cannot map to a user, the CLA workflow will flag the PR as unknown/unlinked.

## Required Quality Gates

Changes should be validated against the real repository standards, not a reduced local bar.

- For code or dependency changes, prefer running:
  - `cargo test --all-features --locked`
  - `cargo build --manifest-path xeno-edit/Cargo.toml --release --locked`
- If dependency graph changes are involved, check the effective graph, not just manifests.
- If workflows are changed, validate YAML locally before pushing.
- If a change affects public capability claims, parity, or benchmarks, update the relevant docs and baselines.

## CI Expectations

The repository intentionally enforces visible, objective checks.

- `Check (ubuntu-latest)`
- `Check (windows-latest)`
- `Check (macos-latest)`
- `Tests (Ubuntu)`
- `FFmpeg Parity Matrix (Ubuntu)`
- `Competitive Benchmarks (Ubuntu)`
- `cla`

Do not merge changes that knowingly break these signals.

## Security And Dependency Policy

- Keep the default branch free of open Dependabot alerts whenever practical.
- Remove vulnerable dependencies from the actual resolved graph instead of dismissing alerts without a defensible reason.
- Prefer dependency reductions and feature-pruning over advisory suppression.
- Preserve or improve the repository security posture:
  - secret scanning
  - push protection
  - dependency auditing
  - branch protection

## Workflow Hygiene

- Avoid duplicate GitHub Actions runs.
- Keep workflow names, run names, permissions, concurrency, and timeouts explicit.
- Prefer first-party or minimal custom workflow logic over aging third-party automation when the third-party dependency becomes operational debt.
- When replacing automation, keep behavior at least as strict as the previous setup.

## Documentation And Public Surface

- Keep the repo root curated.
- Public documentation belongs in tracked files and, where appropriate, under `docs/`.
- Internal scratch notes, local agent artifacts, and temporary working documents should not be committed.
- `REALITY.md` stays tracked because the repo should maintain truthful status reporting.
- Keep README, CLI docs, parity docs, and capability reporting aligned with the actual implementation.

## Vendoring And Licensing

- If vendoring third-party code, keep upstream license files intact.
- Add a short XENO-specific note explaining what was changed and why.
- Update `NOTICE` when vendored code or third-party licensing posture changes.
- Do not introduce dependencies or assets with unclear commercial-use posture.

## Media And Competitor Bar

- `xeno-lib` is being developed against an explicit "beat the incumbents" standard, especially versus FFmpeg-class workflows.
- Maintain the parity tracker and benchmark gates as first-class artifacts, not side notes.
- Capability claims must be backed by code paths, tests, and CI where feasible.

## Repository Maintenance Rules

- Do not revert unrelated user work.
- Do not delete historical or internal local files from disk unless explicitly asked.
- If a file should remain local-only, ignore it rather than deleting it.
- Keep the working tree clean after finishing.

## When In Doubt

- Choose the option that improves:
  - public reviewability
  - deterministic builds
  - honest documentation
  - branch and CI hygiene
  - long-term maintainability
