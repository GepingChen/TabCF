#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
OUT_DIR="$ROOT_DIR/artifacts/paper_figures/reproduced"
mkdir -p "$OUT_DIR"

python "$ROOT_DIR/empirical/plot_results.py" \
  --ajr-dir "$ROOT_DIR/artifacts/aggregated_csv/empirical/ajr" \
  --fulton-dir "$ROOT_DIR/artifacts/aggregated_csv/empirical/fulton" \
  --card-dir "$ROOT_DIR/artifacts/aggregated_csv/empirical/card" \
  --cigarettes-dir "$ROOT_DIR/artifacts/aggregated_csv/empirical/cigarettes" \
  --output-stem "$OUT_DIR/empirical_comparison_grid"
