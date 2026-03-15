#!/bin/bash
# Build xeno-lib-wasm for browser usage via wasm-pack.
#
# Prerequisites:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-pack
#
# Output:
#   xeno-lib-wasm/pkg/xeno_lib_wasm.js
#   xeno-lib-wasm/pkg/xeno_lib_wasm_bg.wasm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WASM_CRATE="$PROJECT_ROOT/xeno-lib-wasm"

echo "==> Building xeno-lib-wasm for browser target..."
cd "$WASM_CRATE"

# wasm-pack build for browser usage (ES modules)
wasm-pack build --target web --release

echo ""
echo "==> Build complete."
echo "    Output: $WASM_CRATE/pkg/"
ls -lh "$WASM_CRATE/pkg/"*.wasm "$WASM_CRATE/pkg/"*.js 2>/dev/null || true
