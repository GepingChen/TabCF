#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
OUT_DIR="$ROOT_DIR/artifacts/paper_figures/reproduced"
mkdir -p "$OUT_DIR"

python "$ROOT_DIR/interv_mean/pipeline/plot_results.py" \
  --csv "$ROOT_DIR/artifacts/aggregated_csv/interv_mean/mean_benchmark_results.csv" \
  --codes A3_B4 A3_B5 A3_B11 A9_B4 A9_B5 A9_B11 \
  --drop-methods tabpfn_cf \
  --output "$OUT_DIR/mean_benchmark_figure.png"
