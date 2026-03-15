#!/bin/bash
# Run all Criterion benchmarks and save results.
#
# Usage:
#   ./scripts/bench.sh              # Run benchmarks, print to stdout
#   ./scripts/bench.sh --save       # Also save JSON baseline

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "==> Running xeno-lib benchmarks..."
echo ""

if [[ "${1:-}" == "--save" ]]; then
    cargo bench --bench transforms -- --save-baseline main
    echo ""
    echo "==> Baseline saved as 'main'. Compare future runs with:"
    echo "    cargo bench --bench transforms -- --baseline main"
else
    cargo bench --bench transforms
fi

echo ""
echo "==> Benchmarks complete."
echo "    HTML reports: target/criterion/report/index.html"
