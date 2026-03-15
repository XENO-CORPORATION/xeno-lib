#!/usr/bin/env bash
# Build script for xeno-lib-napi
#
# Usage:
#   ./build.sh          # Release build for current platform
#   ./build.sh debug    # Debug build for current platform
#   ./build.sh test     # Run Rust tests
#   ./build.sh check    # Type-check without building

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "${1:-release}" in
  release)
    echo "==> Building xeno-lib-napi (release)..."
    cargo build --release
    echo "==> Build complete: target/release/"
    ;;
  debug)
    echo "==> Building xeno-lib-napi (debug)..."
    cargo build
    echo "==> Build complete: target/debug/"
    ;;
  test)
    echo "==> Running xeno-lib-napi tests..."
    cargo test
    ;;
  check)
    echo "==> Type-checking xeno-lib-napi..."
    cargo check
    ;;
  *)
    echo "Usage: $0 [release|debug|test|check]"
    exit 1
    ;;
esac
