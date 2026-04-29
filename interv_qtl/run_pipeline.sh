#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_DIR="${REPO_DIR}/interv_qtl"
SIM_DATA_DIR="${SIM_DIR}/IV_datasets"
BATCH_DATA_DIR="${REPO_DIR}/interv_mean/IV_datasets"
TRAIN_DIR="${SIM_DATA_DIR}/train"
TEST_DIR="${SIM_DATA_DIR}/test"
STAGE1_DIR="${SIM_DATA_DIR}/stage1_output"
STAGE2_DIR="${SIM_DATA_DIR}/stage2_output"
TRUTH_DIR="${SIM_DATA_DIR}/ground_truth"
FIGURES_DIR="${SIM_DIR}/figures"

DEFAULT_TAUS="0.01,0.025,0.1,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.9,0.975,0.99"

DGP_CODES=(A3_B3)
TRAIN_SIZES=(1000 4000 10000)
SEEDS=($(seq 1 10))
METHODS=(tabpfn div ivqr)
TAUS="${DEFAULT_TAUS}"
OVERWRITE=0
SKIP_EXISTING=0
PLOT_RESULTS=1
CLEAN_STAGE1=0

split_override() {
  echo "${1:-}" | tr ',:' ' '
}

expand_numeric_list() {
  local input tokens=() tok start end
  input="$(split_override "${1:-}")"
  for tok in ${input}; do
    if [[ "${tok}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start=${BASH_REMATCH[1]}
      end=${BASH_REMATCH[2]}
      tokens+=($(seq "${start}" "${end}"))
    elif [[ -n "${tok}" ]]; then
      tokens+=("${tok}")
    fi
  done
  echo "${tokens[@]}"
}

parse_arg_list() {
  local values=()
  while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
    values+=("$1")
    shift
  done
  echo "${values[*]}"
}

normalize_methods() {
  local input=("$@")
  local out=()
  local method
  for method in "${input[@]}"; do
    case "${method}" in
      all)
        out=(tabpfn tabpfn_real tabicl div ivqr)
        ;;
      tabpfn|tabpfn_real|tabicl|div|ivqr)
        out+=("${method}")
        ;;
      "")
        ;;
      *)
        echo "Unknown method: ${method}" >&2
        exit 1
        ;;
    esac
  done
  printf '%s\n' "${out[@]}" | awk '!seen[$0]++'
}

usage() {
  cat <<'EOF'
Usage:
  bash interv_qtl/run_pipeline.sh [options]

Options:
  --codes CODE [CODE ...]          DGP codes to run
  --train-sizes N [N ...]          Train sizes
  --seeds S [S ...]                Seeds or ranges like 1-10
  --taus CSV                       Comma-separated tau list
  --methods M [M ...]              Any of: tabpfn tabpfn_real tabicl div ivqr all
  --overwrite                      Remove matching prior stage2 outputs, caches, and figures first
  --clean-stage1                   Also remove matching stage1 CSVs when used with --overwrite
  --skip-existing                  Pass through to method runners
  --no-plot                        Skip plotting after rerun
  --help                           Show this help

Environment overrides:
  DGP_CODES_OVERRIDE, TRAIN_SIZES_OVERRIDE, SEEDS_OVERRIDE, TAUS_OVERRIDE,
  METHODS_OVERRIDE, SOFTMAX_TEMPERATURE_OVERRIDE, OVERWRITE, SKIP_EXISTING, PLOT_RESULTS, CLEAN_STAGE1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --codes)
      shift
      read -r -a DGP_CODES <<< "$(parse_arg_list "$@")"
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do shift; done
      ;;
    --train-sizes)
      shift
      read -r -a TRAIN_SIZES <<< "$(parse_arg_list "$@")"
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do shift; done
      ;;
    --seeds)
      shift
      read -r -a SEEDS <<< "$(expand_numeric_list "$(parse_arg_list "$@")")"
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do shift; done
      ;;
    --taus)
      TAUS="$2"
      shift 2
      ;;
    --methods)
      shift
      readarray -t METHODS < <(normalize_methods $(parse_arg_list "$@"))
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do shift; done
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --clean-stage1)
      CLEAN_STAGE1=1
      shift
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    --no-plot)
      PLOT_RESULTS=0
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "${DGP_CODES_OVERRIDE:-}" ]]; then
  read -r -a DGP_CODES <<< "$(split_override "${DGP_CODES_OVERRIDE}")"
fi
if [[ -n "${TRAIN_SIZES_OVERRIDE:-}" ]]; then
  read -r -a TRAIN_SIZES <<< "$(split_override "${TRAIN_SIZES_OVERRIDE}")"
fi
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a SEEDS <<< "$(expand_numeric_list "${SEEDS_OVERRIDE}")"
fi
if [[ -n "${TAUS_OVERRIDE:-}" ]]; then
  TAUS="${TAUS_OVERRIDE}"
fi
if [[ -n "${METHODS_OVERRIDE:-}" ]]; then
  readarray -t METHODS < <(normalize_methods $(split_override "${METHODS_OVERRIDE}"))
fi
if [[ "${OVERWRITE:-0}" != "0" ]]; then
  OVERWRITE=1
fi
if [[ "${SKIP_EXISTING:-0}" != "0" ]]; then
  SKIP_EXISTING=1
fi
if [[ "${PLOT_RESULTS:-1}" == "0" ]]; then
  PLOT_RESULTS=0
fi
if [[ "${CLEAN_STAGE1:-0}" != "0" ]]; then
  CLEAN_STAGE1=1
fi

if [[ "${OVERWRITE}" -eq 1 && "${SKIP_EXISTING}" -eq 1 ]]; then
  echo "--overwrite and --skip-existing cannot be used together." >&2
  exit 1
fi

if [[ ${#DGP_CODES[@]} -eq 0 || ${#TRAIN_SIZES[@]} -eq 0 || ${#SEEDS[@]} -eq 0 ]]; then
  echo "codes/train-sizes/seeds must all be non-empty." >&2
  exit 1
fi

resolve_python_bin() {
  if [[ -x "${REPO_DIR}/venv_tabpfn_latest/bin/python" ]]; then
    echo "${REPO_DIR}/venv_tabpfn_latest/bin/python"
    return
  fi
  if [[ -x "${REPO_DIR}/venv_latest_tabpfn/bin/python" ]]; then
    echo "${REPO_DIR}/venv_latest_tabpfn/bin/python"
    return
  fi
  command -v python
}

PYTHON_BIN="$(resolve_python_bin)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "============================================"
echo "interv_qtl pipeline"
echo "============================================"
echo "Repo: ${REPO_DIR}"
echo "Codes: ${DGP_CODES[*]}"
echo "Train sizes: ${TRAIN_SIZES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Methods: ${METHODS[*]}"
echo "Taus: ${TAUS}"
echo "Overwrite: ${OVERWRITE}"
echo "Skip existing: ${SKIP_EXISTING}"
echo "Plot results: ${PLOT_RESULTS}"
echo "Clean stage1: ${CLEAN_STAGE1}"
echo "Python: ${PYTHON_BIN}"
echo "Start: $(date)"
echo "--------------------------------------------"

combined_plot_path() {
  local joined
  joined="$(IFS=_; echo "${DGP_CODES[*]}")"
  echo "${FIGURES_DIR}/rmse_boxplot_${joined}.png"
}

ensure_sim_inputs() {
  local code n seed src dst copied_count=0
  local -a missing_sources=()

  if [[ ! -d "${BATCH_DATA_DIR}/train" || ! -d "${BATCH_DATA_DIR}/test" ]]; then
    echo "interv_mean input data directory is missing: ${BATCH_DATA_DIR}" >&2
    exit 1
  fi

  mkdir -p "${TRAIN_DIR}" "${TEST_DIR}"

  for code in "${DGP_CODES[@]}"; do
    src="${BATCH_DATA_DIR}/test/test_data_${code}.csv"
    dst="${TEST_DIR}/test_data_${code}.csv"
    if [[ ! -f "${dst}" ]]; then
      if [[ ! -f "${src}" ]]; then
        missing_sources+=("${src}")
      else
        cp "${src}" "${dst}"
        copied_count=$((copied_count + 1))
      fi
    fi

    for n in "${TRAIN_SIZES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        src="${BATCH_DATA_DIR}/train/train_data_${code}_n${n}_seed${seed}.csv"
        dst="${TRAIN_DIR}/train_data_${code}_n${n}_seed${seed}.csv"
        if [[ ! -f "${dst}" ]]; then
          if [[ ! -f "${src}" ]]; then
            missing_sources+=("${src}")
          else
            cp "${src}" "${dst}"
            copied_count=$((copied_count + 1))
          fi
        fi
      done
    done
  done

  if [[ ${#missing_sources[@]} -gt 0 ]]; then
    printf 'Missing required input files:\n' >&2
    printf '  %s\n' "${missing_sources[@]}" >&2
    exit 1
  fi

  echo "Input sync complete. Copied ${copied_count} file(s) from interv_mean."
}

cleanup_outputs() {
  local code
  mkdir -p "${STAGE2_DIR}" "${TRUTH_DIR}" "${FIGURES_DIR}"
  for code in "${DGP_CODES[@]}"; do
    rm -f \
      "${STAGE2_DIR}/s2q_${code}_n"*"_seed"*"_predictions.csv" \
      "${STAGE2_DIR}/s2q_${code}_n"*"_seed"*"_summary.csv" \
      "${STAGE2_DIR}/s2q_tabpfn_real_${code}_n"*"_seed"*"_predictions.csv" \
      "${STAGE2_DIR}/s2q_tabpfn_real_${code}_n"*"_seed"*"_summary.csv" \
      "${STAGE2_DIR}/s2q_tabicl_${code}_n"*"_seed"*"_predictions.csv" \
      "${STAGE2_DIR}/s2q_tabicl_${code}_n"*"_seed"*"_summary.csv" \
      "${STAGE2_DIR}/s2q_div_${code}_n"*"_seed"*"_predictions.csv" \
      "${STAGE2_DIR}/s2q_div_${code}_n"*"_seed"*"_summary.csv" \
      "${STAGE2_DIR}/s2q_ivqr_${code}_n"*"_seed"*"_predictions.csv" \
      "${STAGE2_DIR}/s2q_ivqr_${code}_n"*"_seed"*"_summary.csv" \
      "${TRUTH_DIR}/true_quantiles_${code}_"* \
      "${FIGURES_DIR}/rmse_boxplot_${code}.png"
    if [[ "${CLEAN_STAGE1}" -eq 1 ]]; then
      rm -f "${STAGE1_DIR}/iv_stage1_train_${code}_n"*"_seed"*.csv
    fi
  done
  rm -f "$(combined_plot_path)"
}

run_method() {
  local method="$1"
  local script_path
  local -a env_args
  env_args=(
    "DGP_CODES_OVERRIDE=${DGP_CODES[*]}"
    "TRAIN_SIZES_OVERRIDE=${TRAIN_SIZES[*]}"
    "SEEDS_OVERRIDE=${SEEDS[*]}"
    "TAUS_OVERRIDE=${TAUS}"
    "SKIP_EXISTING=${SKIP_EXISTING}"
  )
  if [[ -n "${SOFTMAX_TEMPERATURE_OVERRIDE:-}" ]]; then
    env_args+=("SOFTMAX_TEMPERATURE_OVERRIDE=${SOFTMAX_TEMPERATURE_OVERRIDE}")
  fi
  case "${method}" in
    tabpfn)
      env_args+=("MODEL_BACKEND_OVERRIDE=tabpfn")
      script_path="${SIM_DIR}/slurm/run_interv_quantile.slurm"
      ;;
    tabpfn_real)
      env_args+=("MODEL_BACKEND_OVERRIDE=tabpfn_real")
      script_path="${SIM_DIR}/slurm/run_interv_quantile.slurm"
      ;;
    tabicl)
      env_args+=("MODEL_BACKEND_OVERRIDE=tabicl")
      script_path="${SIM_DIR}/slurm/run_interv_quantile.slurm"
      ;;
    div)
      script_path="${SIM_DIR}/slurm/run_div_baseline.slurm"
      ;;
    ivqr)
      script_path="${SIM_DIR}/slurm/run_ivqr_baseline.slurm"
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 1
      ;;
  esac
  echo ">>> Running method: ${method}"
  env "${env_args[@]}" bash "${script_path}"
}

plot_results() {
  local code
  for code in "${DGP_CODES[@]}"; do
    echo ">>> Plotting ${code}"
    "${PYTHON_BIN}" "${SIM_DIR}/viz/plot_rmse_boxplot.py" \
      --code "${code}" \
      --train-sizes "${TRAIN_SIZES[@]}" \
      --output "${FIGURES_DIR}/rmse_boxplot_${code}.png"
  done

  if [[ ${#DGP_CODES[@]} -gt 1 ]]; then
    echo ">>> Plotting combined figure"
    "${PYTHON_BIN}" "${SIM_DIR}/viz/plot_rmse_boxplot.py" \
      --codes "${DGP_CODES[@]}" \
      --train-sizes "${TRAIN_SIZES[@]}" \
      --output "$(combined_plot_path)"
  fi
}

if [[ "${OVERWRITE}" -eq 1 ]]; then
  echo ">>> Removing prior outputs for selected codes"
  cleanup_outputs
fi

echo ">>> Ensuring train/test inputs exist in interv_qtl"
ensure_sim_inputs

for method in "${METHODS[@]}"; do
  run_method "${method}"
done

if [[ "${PLOT_RESULTS}" -eq 1 ]]; then
  plot_results
fi

echo "--------------------------------------------"
echo "Pipeline completed: $(date)"
