#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="${DATASET:-fulton}"
DATA_PATH="${DATA_PATH:-}"
OUT_DIR="${OUT_DIR:-}"
INSTRUMENT_SPEC="${INSTRUMENT_SPEC:-wave2}"
TAUS="${TAUS:-0.15,0.25,0.50,0.75,0.85}"
R_MODULE="${R_MODULE:-r/4.4.3-py311-xspgsan}"
CORE_BACKENDS="${CORE_BACKENDS:-tabpfn}"
GRID_POINTS="${GRID_POINTS:-100}"
N_V_POINTS="${N_V_POINTS:-101}"
Y_GRID_POINTS="${Y_GRID_POINTS:-1201}"
Y_GRID_PADDING="${Y_GRID_PADDING:-0.25}"
IVQR_GRID_MIN="${IVQR_GRID_MIN:-}"
IVQR_GRID_MAX="${IVQR_GRID_MAX:-}"
IVQR_GRID_POINTS="${IVQR_GRID_POINTS:-201}"
IVQR_SEED="${IVQR_SEED:-1}"
DIV_NUM_EPOCHS="${DIV_NUM_EPOCHS:-1000}"
DIV_NUM_LAYERS="${DIV_NUM_LAYERS:-4}"
DIV_LR="${DIV_LR:-1e-4}"
DIV_NSAMPLE="${DIV_NSAMPLE:-1000}"
DIV_SEED="${DIV_SEED:-1}"

cd "${REPO_DIR}"

if [ -f /etc/profile ]; then
  set +u
  # shellcheck disable=SC1091
  source /etc/profile >/dev/null 2>&1 || true
  set -u
fi

if command -v module >/dev/null 2>&1 && [ -n "${R_MODULE}" ]; then
  module load "${R_MODULE}" >/dev/null 2>&1 || true
fi

if [ -f "${REPO_DIR}/venv_latest_tabpfn/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/venv_latest_tabpfn/bin/activate"
else
  echo "Missing Python environment: ${REPO_DIR}/venv_latest_tabpfn" >&2
  exit 1
fi

export TABPFN_MODEL_VERSION="${TABPFN_MODEL_VERSION:-v2.5}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-${REPO_DIR}/tabpfn_home_config/models}"
export TABPFN_STATE_DIR="${TABPFN_STATE_DIR:-${REPO_DIR}/tabpfn_home_config}"
mkdir -p "${TABPFN_MODEL_CACHE_DIR}" "${TABPFN_STATE_DIR}"

if [ -z "${OUT_DIR}" ]; then
  case "${DATASET}" in
    fulton)
      OUT_DIR="${REPO_DIR}/empirical/outputs_fulton_fish_quantile"
      ;;
    *)
      OUT_DIR="${REPO_DIR}/empirical/outputs_${DATASET}_quantile"
      ;;
  esac
fi
mkdir -p "${OUT_DIR}"

echo "Running real-data quantile pipeline"
echo "DATASET=${DATASET}"
echo "INSTRUMENT_SPEC=${INSTRUMENT_SPEC}"
echo "OUT_DIR=${OUT_DIR}"
echo "CORE_BACKENDS=${CORE_BACKENDS}"
echo "TAUS=${TAUS}"

args=(
  --dataset "${DATASET}"
  --out-dir "${OUT_DIR}"
  --instrument-spec "${INSTRUMENT_SPEC}"
  --taus "${TAUS}"
  --core-backends ${CORE_BACKENDS}
  --grid-points "${GRID_POINTS}"
  --n-v-points "${N_V_POINTS}"
  --y-grid-points "${Y_GRID_POINTS}"
  --y-grid-padding "${Y_GRID_PADDING}"
  --ivqr-grid-points "${IVQR_GRID_POINTS}"
  --ivqr-seed "${IVQR_SEED}"
  --div-epochs "${DIV_NUM_EPOCHS}"
  --div-layers "${DIV_NUM_LAYERS}"
  --div-lr "${DIV_LR}"
  --div-nsample "${DIV_NSAMPLE}"
  --div-seed "${DIV_SEED}"
  --r-module "${R_MODULE}"
)

if [ -n "${IVQR_GRID_MIN}" ]; then
  args+=(--ivqr-grid-min "${IVQR_GRID_MIN}")
fi
if [ -n "${IVQR_GRID_MAX}" ]; then
  args+=(--ivqr-grid-max "${IVQR_GRID_MAX}")
fi
if [ -n "${DATA_PATH}" ]; then
  echo "DATA_PATH=${DATA_PATH}"
  args+=(--data-path "${DATA_PATH}")
fi

python empirical/run_empirical_quantile.py "${args[@]}" "$@"
