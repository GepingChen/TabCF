#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
OUT_DIR="$ROOT_DIR/artifacts/paper_figures/reproduced"
mkdir -p "$OUT_DIR"

python "$ROOT_DIR/interv_qtl/viz/plot_results.py" \
  --input "$ROOT_DIR/artifacts/aggregated_csv/interv_qtl/quantile_median_curve_comparison.csv" \
  --output "$OUT_DIR/quantile_rmse_figure"
