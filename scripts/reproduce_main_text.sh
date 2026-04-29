#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

bash "$ROOT_DIR/scripts/reproduce_mean.sh"
bash "$ROOT_DIR/scripts/reproduce_quantile.sh"
bash "$ROOT_DIR/scripts/reproduce_multivar.sh"
bash "$ROOT_DIR/scripts/reproduce_empirical.sh"
