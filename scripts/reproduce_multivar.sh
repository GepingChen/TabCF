#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
OUT_DIR="$ROOT_DIR/artifacts/paper_figures/reproduced"
mkdir -p "$OUT_DIR"

python "$ROOT_DIR/multivar/pipeline/plot_results.py" \
  --input "$ROOT_DIR/artifacts/aggregated_csv/multivar/multivar_wasserstein_curves.csv" \
  --output "$OUT_DIR/multivar_wasserstein_figure.png"
