#!/usr/bin/env python3
"""
Batch runner for backend-aware CDF-inversion quantile estimation.

Workflow for each (code, train_size, seed) combination:

1) Ensure Stage 1 outputs exist (control function V̂)
   Location: interv_qtl/IV_datasets/stage1_output/
   
2) Fit backend-specific conditional CDF F(y|X,V̂) on Stage 1 data
   
3) Integrate over V and invert CDF to obtain q_τ(x) on x_grid
   - x_grid is seed-invariant (default: test set quantiles)
   - Uses Gauss-Legendre (18) numerical integration by default
   
4) Monte Carlo ground truth on the same grid
   - Cached per (code, x_grid, taus) to avoid redundant computation
   
5) Save outputs with deterministic filenames
   - Legacy TabPFN:
       s2q_{code}_n{n}_seed{seed}_predictions.csv
       s2q_{code}_n{n}_seed{seed}_summary.csv
   - New backends:
       s2q_{backend}_{code}_n{n}_seed{seed}_predictions.csv
       s2q_{backend}_{code}_n{n}_seed{seed}_summary.csv

Key features:
- Supports multiple DGP codes, sample sizes, and seeds
- Environment variable overrides for flexible SLURM scheduling
- Ground truth caching to speed up repeated experiments
- Skip-existing mode for incremental runs
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
BATCH_DIR = REPO_ROOT / "interv_mean"
BATCH_CORE_DIR = REPO_ROOT / "tabcf_core"
for path in (BATCH_CORE_DIR, BATCH_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from stage1_control import Stage1Config, run_stage1_experiment
from stage2_outcome import ConditionalCDFEstimator, load_stage1_data
from dgp import DGPConfig, set_seed
from foundation_backends import (
    TABPFN_BACKEND,
    backend_metadata,
    normalize_backend_name,
    stage1_output_filename,
)
from mu_integrators import (
    DEFAULT_GAUSS_LEGENDRE_ORDER,
    DEFAULT_MU_INTEGRATOR,
    MuIntegratorConfig,
)
from cdf_to_quantiles import (
    build_y_grid,
    compute_quantiles_on_grid,
    compute_true_quantiles,
)

SIM_DATA_DIR = CURRENT_DIR / "IV_datasets"
BATCH_SIM_DATA_DIR = BATCH_DIR / "IV_datasets"
TRAIN_DIR = SIM_DATA_DIR / "train"
TEST_DIR = SIM_DATA_DIR / "test"
STAGE1_OUTPUT_DIR = SIM_DATA_DIR / "stage1_output"
STAGE2_OUTPUT_DIR = SIM_DATA_DIR / "stage2_output"
GROUND_TRUTH_DIR = SIM_DATA_DIR / "ground_truth"

DEFAULT_TAUS: Tuple[float, ...] = (
    0.01,
    0.025,
    0.1,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.9,
    0.975,
    0.99,
)
X_GRID_QUANTILE_MIN = 0.05
X_GRID_QUANTILE_MAX = 0.95


# ---------------------------------------------------------------------------
# Argument parsing and helpers
# ---------------------------------------------------------------------------
def parse_dgp_code(code: str) -> Tuple[str, str]:
    parts = code.split("_")
    if len(parts) != 2:
        raise ValueError(f"DGP code '{code}' must be formatted as 'A?_B?'.")
    return parts[0].upper(), parts[1].upper()


def parse_list_arg(values: Sequence[str] | None) -> List[str]:
    if not values:
        return []
    if isinstance(values, str):  # pragma: no cover - argparse passes list
        return values.replace(",", " ").split()
    parsed: List[str] = []
    for val in values:
        parsed.extend(val.replace(",", " ").split())
    # Expand numeric ranges like "1-5"
    expanded: List[str] = []
    for tok in parsed:
        if "-" in tok and tok.replace("-", "").isdigit():
            try:
                start_str, end_str = tok.split("-", 1)
                start_i, end_i = int(start_str), int(end_str)
                expanded.extend([str(i) for i in range(start_i, end_i + 1)])
                continue
            except Exception:
                pass  # fall back to original token
        expanded.append(tok)
    return expanded


def expected_stage1_csv(
    first_stage: str,
    second_stage: str,
    n: int,
    seed: int,
    *,
    backend_name: str = TABPFN_BACKEND,
    softmax_temperature: float | None = None,
) -> Path:
    codes = f"{first_stage}_{second_stage}"
    return STAGE1_OUTPUT_DIR / stage1_output_filename(
        "train",
        codes,
        train_sample_size=n,
        seed=seed,
        backend_name=backend_name,
        softmax_temperature=softmax_temperature,
        timestamp=None,
    )


def _stage2_quantile_stem(code: str, backend_name: str) -> str:
    backend = normalize_backend_name(backend_name)
    if backend == TABPFN_BACKEND:
        return f"s2q_{code}"
    return f"s2q_{backend}_{code}"


def expected_stage2_predictions(
    first_stage: str,
    second_stage: str,
    n: int,
    seed: int,
    *,
    output_dir: Path | None = None,
    backend_name: str = TABPFN_BACKEND,
) -> Path:
    codes = f"{first_stage}_{second_stage}"
    stem = _stage2_quantile_stem(codes, backend_name)
    base_dir = STAGE2_OUTPUT_DIR if output_dir is None else Path(output_dir)
    return base_dir / f"{stem}_n{n}_seed{seed}_predictions.csv"


def expected_stage2_summary(
    first_stage: str,
    second_stage: str,
    n: int,
    seed: int,
    *,
    output_dir: Path | None = None,
    backend_name: str = TABPFN_BACKEND,
) -> Path:
    codes = f"{first_stage}_{second_stage}"
    stem = _stage2_quantile_stem(codes, backend_name)
    base_dir = STAGE2_OUTPUT_DIR if output_dir is None else Path(output_dir)
    return base_dir / f"{stem}_n{n}_seed{seed}_summary.csv"


def summary_matches_requested_integrator(
    summary_path: Path,
    *,
    mu_integrator: str,
    gauss_legendre_order: int,
    softmax_temperature: float | None,
) -> bool:
    try:
        summary_df = pd.read_csv(summary_path, nrows=1)
    except Exception:
        return False

    if summary_df.empty:
        return False
    if "mu_integrator" not in summary_df.columns or "gauss_legendre_order" not in summary_df.columns:
        return False

    row = summary_df.iloc[0]
    method_raw = str(row["mu_integrator"]).strip().lower()
    if not method_raw or method_raw == "nan":
        return False
    try:
        order_val = int(float(row["gauss_legendre_order"]))
    except (TypeError, ValueError):
        return False

    if method_raw != str(mu_integrator).strip().lower() or order_val != int(gauss_legendre_order):
        return False

    if "softmax_temperature" not in summary_df.columns:
        return softmax_temperature is None

    temp_raw = row["softmax_temperature"]
    if pd.isna(temp_raw) or str(temp_raw).strip() == "":
        return softmax_temperature is None

    if softmax_temperature is None:
        return False

    try:
        return float(temp_raw) == float(softmax_temperature)
    except (TypeError, ValueError):
        return False


def load_test_data(first_stage: str, second_stage: str) -> pd.DataFrame:
    filename = f"test_data_{first_stage}_{second_stage}.csv"
    primary = TEST_DIR / filename
    fallback = BATCH_SIM_DATA_DIR / "test" / filename
    path = primary if primary.exists() else fallback
    if not path.exists():
        raise FileNotFoundError(
            f"Test data not found for {first_stage}_{second_stage}: {primary} or {fallback}"
        )
    return pd.read_csv(path)


def build_x_grid(
    train_data: dict,
    test_df: pd.DataFrame,
    *,
    mode: str,
    points: int,
    x_min_override: float | None = None,
    x_max_override: float | None = None,
) -> np.ndarray:
    """
    Construct evaluation grid for X (intervention points).
    
    Modes:
    - 'test_quantile': Use test-set X quantiles on [0.05, 0.95]
      (seed-invariant, recommended)
    - 'train_quantile': Use training-set X quantiles on [0.05, 0.95]
      (seed-dependent)
    - 'train_range': Use uniform grid over training set range
    
    Args:
        train_data: Training data dict with keys 'X', 'Y', 'V_hat'
        test_df: Test dataframe with column 'X'
        mode: Grid construction mode (see above)
        points: Number of grid points to generate
        x_min_override: Optional manual minimum (only for train_range)
        x_max_override: Optional manual maximum (only for train_range)
    
    Returns:
        np.ndarray: Unique x values, sorted ascending (length ≤ points)
    
    Note: 'test_quantile' mode is recommended for consistency across seeds,
          enabling fair comparison and easier aggregation.
    """
    mode = mode.lower()
    if points <= 1:
        raise ValueError("x_grid_points must be greater than 1.")

    if mode == "test_quantile":
        values = np.asarray(test_df["X"], dtype=float)
        quantiles = np.linspace(X_GRID_QUANTILE_MIN, X_GRID_QUANTILE_MAX, points)
        grid = np.quantile(values, quantiles)
    elif mode == "train_quantile":
        values = np.asarray(train_data["X"], dtype=float)
        quantiles = np.linspace(X_GRID_QUANTILE_MIN, X_GRID_QUANTILE_MAX, points)
        grid = np.quantile(values, quantiles)
    elif mode == "train_range":
        values = np.asarray(train_data["X"], dtype=float)
        x_min = x_min_override if x_min_override is not None else float(np.min(values))
        x_max = x_max_override if x_max_override is not None else float(np.max(values))
        if x_min >= x_max:
            raise ValueError(f"Invalid x-grid range: [{x_min}, {x_max}]")
        grid = np.linspace(x_min, x_max, points)
    else:
        raise ValueError(f"Unknown x_grid_mode '{mode}'. Choose from test_quantile, train_quantile, train_range.")

    grid = np.asarray(grid, dtype=float)
    unique_grid = np.unique(grid)
    if len(unique_grid) != len(grid):
        print(
            f"⚠️  x_grid contains duplicate values (mode={mode}); deduplicated from {len(grid)} to {len(unique_grid)}.",
            flush=True,
        )
        grid = unique_grid
    return grid


def _hash_array(arr: np.ndarray) -> str:
    arr64 = np.asarray(arr, dtype=np.float64)
    digest = hashlib.md5(arr64.tobytes()).hexdigest()
    return digest[:10]


def _hash_taus(taus: Sequence[float]) -> str:
    joined = ",".join(f"{t:.6f}" for t in taus)
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()
    return digest[:10]


def try_load_truth_cache(
    code: str,
    x_grid: np.ndarray,
    taus: Sequence[float],
    mc_samples: int,
    truth_seed: int,
) -> Tuple[np.ndarray | None, Path | None]:
    """
    Attempt to load cached ground-truth quantiles.
    
    Cache key includes:
    - code: DGP identifier (e.g., "A3_B3")
    - x_grid hash: MD5 hash of x values (ensures grid alignment)
    - tau hash: MD5 hash of quantile levels
    - mc_samples: Number of Monte Carlo samples used
    - truth_seed: Random seed for reproducibility
    
    Args:
        code: DGP code
        x_grid: X evaluation points
        taus: Quantile levels
        mc_samples: MC sample count
        truth_seed: RNG seed
    
    Returns:
        Tuple of (q_true array or None, cache file path or None)
    
    Benefits: Avoids recomputing expensive Monte Carlo truth for the same grid.
    """
    x_hash = _hash_array(x_grid)
    tau_hash = _hash_taus(taus)
    filename = f"true_quantiles_{code}_x{x_hash}_t{tau_hash}_mc{mc_samples}_rs{truth_seed}.csv"
    path = GROUND_TRUTH_DIR / filename
    if not path.exists():
        return None, None

    df = pd.read_csv(path)
    if not {"X", "tau", "q_true"}.issubset(df.columns):
        return None, None

    df_sorted = df.sort_values(["X", "tau"]).reset_index(drop=True)
    x_vals = df_sorted["X"].drop_duplicates().to_numpy()
    tau_vals = df_sorted["tau"].drop_duplicates().to_numpy()

    # Verify cache matches requested grid
    if len(x_vals) != len(x_grid) or len(tau_vals) != len(taus):
        return None, None
    if not np.allclose(x_vals, x_grid):
        return None, None
    if not np.allclose(tau_vals, np.asarray(taus, dtype=float)):
        return None, None

    q_true = df_sorted["q_true"].to_numpy().reshape(len(x_grid), len(taus))
    return q_true, path


def save_truth_cache(
    code: str,
    x_grid: np.ndarray,
    taus: Sequence[float],
    q_true: np.ndarray,
    mc_samples: int,
    truth_seed: int,
) -> Path:
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    x_hash = _hash_array(x_grid)
    tau_hash = _hash_taus(taus)
    filename = f"true_quantiles_{code}_x{x_hash}_t{tau_hash}_mc{mc_samples}_rs{truth_seed}.csv"
    path = GROUND_TRUTH_DIR / filename
    records = []
    for i, x_val in enumerate(x_grid):
        for j, tau in enumerate(taus):
            records.append({"X": float(x_val), "tau": float(tau), "q_true": float(q_true[i, j])})
    pd.DataFrame.from_records(records).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    dgp_codes: Sequence[str]
    train_sizes: Sequence[int]
    seeds: Sequence[int]
    backend_name: str
    stage2_output_dir: Path
    stage1_random_state: int
    stage2_random_state: int
    taus: Tuple[float, ...]
    n_v_grid: int
    mu_integrator: str
    gauss_legendre_order: int
    y_grid_points: int
    y_grid_padding: float
    x_grid_points: int
    x_grid_mode: str
    x_min_override: float | None
    x_max_override: float | None
    mc_samples: int
    max_points_per_batch: int | None
    softmax_temperature: float | None
    skip_existing: bool
    enable_truth_cache: bool

    def __post_init__(self) -> None:
        self.stage2_output_dir = Path(self.stage2_output_dir)
        integrator_cfg = MuIntegratorConfig(
            method=self.mu_integrator,
            n_v_points=self.n_v_grid,
            gauss_legendre_order=self.gauss_legendre_order,
        )
        self.mu_integrator = integrator_cfg.method
        self.n_v_grid = integrator_cfg.n_v_points
        self.gauss_legendre_order = integrator_cfg.gauss_legendre_order
        if self.softmax_temperature is not None:
            self.softmax_temperature = float(self.softmax_temperature)
        if self.backend_name == "tabicl" and self.softmax_temperature is not None:
            raise ValueError(
                "softmax_temperature is only supported for TabPFN backends ('tabpfn' and 'tabpfn_real')."
            )


def ensure_stage1_outputs(
    first_stage: str,
    second_stage: str,
    n: int,
    seed: int,
    random_state: int,
    *,
    backend_name: str,
    softmax_temperature: float | None,
) -> Path:
    STAGE1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    code = f"{first_stage}_{second_stage}"
    stage1_csv = expected_stage1_csv(
        first_stage,
        second_stage,
        n,
        seed,
        backend_name=backend_name,
        softmax_temperature=softmax_temperature,
    )
    if stage1_csv.exists():
        print(f"Stage1 output exists: {stage1_csv}", flush=True)
        return stage1_csv

    train_filename = f"train_data_{code}_n{n}_seed{seed}.csv"
    test_filename = f"test_data_{code}.csv"
    primary_train = TRAIN_DIR / train_filename
    primary_test = TEST_DIR / test_filename
    fallback_train = BATCH_SIM_DATA_DIR / "train" / train_filename
    fallback_test = BATCH_SIM_DATA_DIR / "test" / test_filename
    base_dir = SIM_DATA_DIR
    if not (primary_train.exists() and primary_test.exists()):
        if fallback_train.exists() and fallback_test.exists():
            base_dir = BATCH_SIM_DATA_DIR

    print(
        f"Stage1 output missing; running Stage 1 for {first_stage}_{second_stage}, "
        f"n={n}, seed={seed}, backend={backend_name}, base_dir={base_dir}",
        flush=True,
    )
    stage1_cfg = Stage1Config(
        random_state=random_state,
        backend_name=backend_name,
        softmax_temperature=softmax_temperature,
    )
    run_stage1_experiment(
        first_stage,
        second_stage,
        stage1_cfg,
        train_sample_size=n,
        seed=seed,
        output_dir=str(STAGE1_OUTPUT_DIR),
        base_dir=str(base_dir),
        save_outputs=True,
        use_timestamp=False,
    )
    if not stage1_csv.exists():
        raise FileNotFoundError(f"Stage1 output still missing after run: {stage1_csv}")
    return stage1_csv


def compute_truth_with_cache(
    dgp_cfg: DGPConfig,
    x_grid: np.ndarray,
    taus: Tuple[float, ...],
    mc_samples: int,
    truth_seed: int,
    use_cache: bool,
) -> Tuple[np.ndarray, Path | None, bool]:
    rng = np.random.default_rng(truth_seed)
    code = f"{dgp_cfg.first_stage}_{dgp_cfg.second_stage}"

    if use_cache:
        cached, cache_path = try_load_truth_cache(code, x_grid, taus, mc_samples, truth_seed)
        if cached is not None:
            print(f"Loaded ground truth from cache: {cache_path}", flush=True)
            return cached, cache_path, True

    q_true = compute_true_quantiles(dgp_cfg, x_grid, taus, mc_samples, rng)
    cache_path = None
    if use_cache:
        cache_path = save_truth_cache(code, x_grid, taus, q_true, mc_samples, truth_seed)
        print(f"Saved ground truth cache to {cache_path}", flush=True)
    return q_true, cache_path, False


def run_single_combo(
    first_stage: str,
    second_stage: str,
    n: int,
    seed_val: int,
    cfg: PipelineConfig,
) -> Tuple[Path, Path]:
    """
    Run quantile estimation for a single (code, n, seed) combination.
    
    Steps:
    1. Check if outputs already exist (optional skip)
    2. Load or generate Stage 1 outputs (control function V̂)
    3. Build x_grid and y_grid for evaluation
    4. Fit TabPFN CDF model and compute quantile estimates
    5. Load or compute ground truth (with caching)
    6. Calculate per-tau metrics (RMSE, MAE, MSE)
    7. Save long-format predictions and summary CSVs
    
    Args:
        first_stage: First-stage DGP code (e.g., "A3")
        second_stage: Second-stage DGP code (e.g., "B3")
        n: Training sample size
        seed_val: Random seed for data generation
        cfg: Pipeline configuration
    
    Returns:
        Tuple of (predictions_csv_path, summary_csv_path)
    """
    combo_start = time.time()
    code = f"{first_stage}_{second_stage}"
    backend_info = backend_metadata(cfg.backend_name, "auto")
    predictions_path = expected_stage2_predictions(
        first_stage,
        second_stage,
        n,
        seed_val,
        output_dir=cfg.stage2_output_dir,
        backend_name=cfg.backend_name,
    )
    summary_path = expected_stage2_summary(
        first_stage,
        second_stage,
        n,
        seed_val,
        output_dir=cfg.stage2_output_dir,
        backend_name=cfg.backend_name,
    )

    if cfg.skip_existing and predictions_path.exists() and summary_path.exists():
        if summary_matches_requested_integrator(
            summary_path,
            mu_integrator=cfg.mu_integrator,
            gauss_legendre_order=cfg.gauss_legendre_order,
            softmax_temperature=cfg.softmax_temperature,
        ):
            print(f"⏩ Skipping existing combo {code}, n={n}, seed={seed_val}, backend={cfg.backend_name}")
            return predictions_path, summary_path
        print(
            f"↻ Existing combo {code}, n={n}, seed={seed_val} uses missing or mismatched "
            "integrator metadata; recomputing.",
            flush=True,
        )

    stage1_csv = ensure_stage1_outputs(
        first_stage,
        second_stage,
        n,
        seed_val,
        cfg.stage1_random_state,
        backend_name=cfg.backend_name,
        softmax_temperature=cfg.softmax_temperature,
    )
    stage1_data = load_stage1_data(stage1_csv)
    test_df = load_test_data(first_stage, second_stage)

    x_grid = build_x_grid(
        stage1_data,
        test_df,
        mode=cfg.x_grid_mode,
        points=cfg.x_grid_points,
        x_min_override=cfg.x_min_override,
        x_max_override=cfg.x_max_override,
    )
    y_grid = build_y_grid(stage1_data["Y"], padding=cfg.y_grid_padding, n_points=cfg.y_grid_points)

    print(
        f"Running {cfg.backend_name} quantile estimation for {code}, n={n}, seed={seed_val} "
        f"(x_grid={len(x_grid)}, y_grid={len(y_grid)}, n_v={cfg.n_v_grid}, "
        f"softmax_temperature={cfg.softmax_temperature}, "
        f"integrator={cfg.mu_integrator})",
        flush=True,
    )

    set_seed(cfg.stage2_random_state)
    rng = np.random.default_rng(cfg.stage2_random_state)

    cdf_model = ConditionalCDFEstimator(
        use_tabpfn=cfg.backend_name != "tabicl",
        backend_name=cfg.backend_name,
        softmax_temperature=cfg.softmax_temperature,
    )
    _ = cdf_model.fit_full(stage1_data["X"], stage1_data["V_hat"], stage1_data["Y"])

    quantiles_est = compute_quantiles_on_grid(
        cdf_model,
        x_grid,
        y_grid,
        cfg.taus,
        cfg.n_v_grid,
        max_points_per_batch=cfg.max_points_per_batch,
        integrator_cfg=MuIntegratorConfig(
            method=cfg.mu_integrator,
            n_v_points=cfg.n_v_grid,
            gauss_legendre_order=cfg.gauss_legendre_order,
            max_points_per_batch=cfg.max_points_per_batch,
        ),
    )

    cache_allowed = cfg.enable_truth_cache and cfg.x_grid_mode == "test_quantile"
    truth_seed = cfg.stage2_random_state
    dgp_cfg = DGPConfig(
        n=n,
        seed=seed_val,
        first_stage=first_stage,
        second_stage=second_stage,
    )
    quantiles_true, cache_path, cache_hit = compute_truth_with_cache(
        dgp_cfg,
        x_grid,
        cfg.taus,
        cfg.mc_samples,
        truth_seed,
        cache_allowed,
    )

    sq_err = (quantiles_est - quantiles_true) ** 2
    mae = np.abs(quantiles_est - quantiles_true)
    rmse_per_tau = np.sqrt(np.mean(sq_err, axis=0))
    mae_per_tau = np.mean(mae, axis=0)
    mse_per_tau = np.mean(sq_err, axis=0)

    cfg.stage2_output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, x_val in enumerate(x_grid):
        for j, tau in enumerate(cfg.taus):
            records.append(
                {
                    "code": code,
                    "train_size": n,
                    "seed": seed_val,
                    "backend": backend_info["backend"],
                    "backend_package": backend_info["backend_package"],
                    "checkpoint_version": backend_info["checkpoint_version"],
                    "model_path": backend_info["model_path"],
                    "softmax_temperature": cfg.softmax_temperature if cfg.softmax_temperature is not None else "",
                    "X": float(x_val),
                    "tau": float(tau),
                    "q_pred": float(quantiles_est[i, j]),
                    "q_true": float(quantiles_true[i, j]),
                    "sq_err": float(sq_err[i, j]),
                }
            )
    pred_df = pd.DataFrame.from_records(records)
    pred_df = pred_df.sort_values(["tau", "X"]).reset_index(drop=True)
    pred_df.to_csv(predictions_path, index=False)
    print(f"✅ Saved predictions to {predictions_path}")

    summary_records = []
    for j, tau in enumerate(cfg.taus):
        summary_records.append(
            {
                "code": code,
                "train_size": n,
                "seed": seed_val,
                "backend": backend_info["backend"],
                "backend_package": backend_info["backend_package"],
                "checkpoint_version": backend_info["checkpoint_version"],
                "model_path": backend_info["model_path"],
                "softmax_temperature": cfg.softmax_temperature if cfg.softmax_temperature is not None else "",
                "tau": float(tau),
                "rmse": float(rmse_per_tau[j]),
                "mae": float(mae_per_tau[j]),
                "mse": float(mse_per_tau[j]),
                "n_x_grid": int(len(x_grid)),
                "n_y_grid": int(len(y_grid)),
                "n_v_grid": int(cfg.n_v_grid),
                "mu_integrator": cfg.mu_integrator,
                "gauss_legendre_order": int(cfg.gauss_legendre_order),
                "mc_samples": int(cfg.mc_samples),
                "x_grid_mode": cfg.x_grid_mode,
                "x_grid_points": int(cfg.x_grid_points),
                "y_grid_points": int(cfg.y_grid_points),
                "y_grid_padding": float(cfg.y_grid_padding),
                "max_points_per_batch": cfg.max_points_per_batch
                if cfg.max_points_per_batch is not None
                else "",
                "stage1_random_state": cfg.stage1_random_state,
                "stage2_random_state": cfg.stage2_random_state,
                "stage1_csv": str(stage1_csv),
                "truth_cache": str(cache_path) if cache_path else "",
                "truth_cache_hit": int(cache_hit),
                "elapsed_seconds": float(time.time() - combo_start),
            }
        )
    summary_df = pd.DataFrame.from_records(summary_records)
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved summary to {summary_path}")

    return predictions_path, summary_path


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Batch backend-aware quantile estimation (CDF inversion).")
    parser.add_argument("--dgp-codes", nargs="+", default=["A3_B3"], help="List of DGP codes, e.g., A3_B3.")
    parser.add_argument("--train-sizes", nargs="+", type=int, required=True, help="Training sample sizes (e.g., 1000 4000).")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Random seeds for training data.")
    parser.add_argument(
        "--backend",
        type=str,
        default=TABPFN_BACKEND,
        choices=["tabpfn", "tabpfn_real", "tabicl"],
        help="Regression backend used for Stage 1 and Stage 2.",
    )
    parser.add_argument(
        "--stage2-output-dir",
        type=Path,
        default=STAGE2_OUTPUT_DIR,
        help="Directory for quantile Stage 2 prediction/summary outputs.",
    )
    parser.add_argument("--stage1-random-state", type=int, default=1, help="Random state for Stage1 TabPFN.")
    parser.add_argument("--stage2-random-state", type=int, default=1, help="Random state for Stage2 components and truth MC.")
    parser.add_argument(
        "--taus",
        type=str,
        default=",".join(str(t) for t in DEFAULT_TAUS),
        help="Comma-separated quantile levels.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip combos with existing predictions + summary.")
    parser.add_argument(
        "--n-v-grid",
        type=int,
        default=101,
        help="Number of V integration points for the optional uniform-grid fallback integrator.",
    )
    parser.add_argument(
        "--stage2-mu-integrator",
        type=str,
        default=DEFAULT_MU_INTEGRATOR,
        choices=["simpson", "gauss_legendre"],
        help="Stage-2 integration rule. Default: gauss_legendre.",
    )
    parser.add_argument(
        "--stage2-gauss-legendre-order",
        type=int,
        default=DEFAULT_GAUSS_LEGENDRE_ORDER,
        help="Quadrature order when --stage2-mu-integrator=gauss_legendre.",
    )
    parser.add_argument("--y-grid-points", type=int, default=1201, help="Number of Y grid points.")
    parser.add_argument("--y-grid-padding", type=float, default=0.25, help="Padding fraction for Y grid construction.")
    parser.add_argument("--x-grid-points", type=int, default=200, help="Number of X grid points.")
    parser.add_argument(
        "--x-grid-mode",
        type=str,
        default="test_quantile",
        help="Grid construction: test_quantile (default, seed-invariant), train_quantile, or train_range.",
    )
    parser.add_argument("--x-grid-min", type=float, default=None, help="Optional X grid minimum (train_range mode).")
    parser.add_argument("--x-grid-max", type=float, default=None, help="Optional X grid maximum (train_range mode).")
    parser.add_argument("--mc-samples", type=int, default=2000, help="MC samples for ground-truth quantiles.")
    parser.add_argument(
        "--max-points-per-batch",
        type=int,
        default=None,
        help="Chunk size for TabPFN inference over (x,v); defaults to heuristic if unset.",
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=None,
        help="Optional TabPFN softmax_temperature passed to Stage 1 and Stage 2 when using TabPFN backends.",
    )
    parser.add_argument(
        "--disable-truth-cache",
        action="store_true",
        help="Disable reuse of cached ground-truth quantiles.",
    )
    args = parser.parse_args()
    env_codes_raw = getenv("DGP_CODES_OVERRIDE")
    env_train_raw = getenv("TRAIN_SIZES_OVERRIDE")
    env_seeds_raw = getenv("SEEDS_OVERRIDE")
    env_backend_raw = getenv("MODEL_BACKEND_OVERRIDE")
    env_stage2_output_dir_raw = getenv("STAGE2_OUTPUT_DIR_OVERRIDE")
    env_softmax_temperature_raw = getenv("SOFTMAX_TEMPERATURE_OVERRIDE")

    dgp_codes = parse_list_arg(args.dgp_codes)
    if env_codes_raw:
        override = parse_list_arg([env_codes_raw])
        if override:
            print(f"Override DGP codes from env: {override}")
            dgp_codes = override

    train_sizes = [int(v) for v in args.train_sizes]
    if env_train_raw:
        override = [int(v) for v in parse_list_arg([env_train_raw])]
        if override:
            print(f"Override train sizes from env: {override}")
            train_sizes = override

    seeds = [int(v) for v in args.seeds]
    if env_seeds_raw:
        override = [int(v) for v in parse_list_arg([env_seeds_raw])]
        if override:
            print(f"Override seeds from env: {override}")
            seeds = override

    backend_name = normalize_backend_name(args.backend)
    if env_backend_raw:
        backend_name = normalize_backend_name(env_backend_raw)
        print(f"Override backend from env: {backend_name}")

    stage2_output_dir = Path(args.stage2_output_dir)
    if env_stage2_output_dir_raw:
        stage2_output_dir = Path(env_stage2_output_dir_raw)
        print(f"Override stage2 output dir from env: {stage2_output_dir}")

    skip_existing = args.skip_existing
    env_skip = getenv("SKIP_EXISTING", "")
    if env_skip and env_skip.strip() not in ("0", "false", "False"):
        skip_existing = True
        print("SKIP_EXISTING detected; enabling skip-existing mode.")

    taus = tuple(float(t.strip()) for t in args.taus.split(",") if t.strip())
    if not taus:
        raise ValueError("At least one tau must be provided.")

    softmax_temperature = args.softmax_temperature
    if env_softmax_temperature_raw is not None and str(env_softmax_temperature_raw).strip() != "":
        softmax_temperature = float(env_softmax_temperature_raw)
        print(f"Override softmax_temperature from env: {softmax_temperature}")

    return PipelineConfig(
        dgp_codes=dgp_codes,
        train_sizes=train_sizes,
        seeds=seeds,
        backend_name=backend_name,
        stage2_output_dir=stage2_output_dir,
        stage1_random_state=args.stage1_random_state,
        stage2_random_state=args.stage2_random_state,
        taus=taus,
        n_v_grid=args.n_v_grid,
        mu_integrator=args.stage2_mu_integrator,
        gauss_legendre_order=args.stage2_gauss_legendre_order,
        y_grid_points=args.y_grid_points,
        y_grid_padding=args.y_grid_padding,
        x_grid_points=args.x_grid_points,
        x_grid_mode=args.x_grid_mode,
        x_min_override=args.x_grid_min,
        x_max_override=args.x_grid_max,
        mc_samples=args.mc_samples,
        max_points_per_batch=args.max_points_per_batch,
        softmax_temperature=softmax_temperature,
        skip_existing=skip_existing,
        enable_truth_cache=not args.disable_truth_cache,
    )


def getenv(key: str, default: str | None = None) -> str | None:
    try:
        return os.environ.get(key, default)
    except Exception:
        return default


def main() -> None:
    cfg = parse_args()
    print("============================================", flush=True)
    print("Backend-aware quantile batch runner (CDF inversion)", flush=True)
    print("============================================", flush=True)
    print(f"Backend: {cfg.backend_name}", flush=True)
    print(f"Stage2 output dir: {cfg.stage2_output_dir}", flush=True)
    print(f"DGP codes: {cfg.dgp_codes}", flush=True)
    print(f"Train sizes: {cfg.train_sizes}", flush=True)
    print(f"Seeds: {cfg.seeds}", flush=True)
    print(f"Taus: {cfg.taus}", flush=True)
    print(f"Softmax temperature: {cfg.softmax_temperature}", flush=True)
    print(
        f"x_grid_mode={cfg.x_grid_mode}, x_grid_points={cfg.x_grid_points}, "
        f"y_grid_points={cfg.y_grid_points}, n_v_grid={cfg.n_v_grid}, "
        f"integrator={cfg.mu_integrator}, gauss_legendre_order={cfg.gauss_legendre_order}",
        flush=True,
    )
    print("", flush=True)

    summary_overview = []
    for code in cfg.dgp_codes:
        first_stage, second_stage = parse_dgp_code(code)
        for n in cfg.train_sizes:
            for seed_val in cfg.seeds:
                try:
                    pred_path, summary_path = run_single_combo(
                        first_stage,
                        second_stage,
                        n,
                        seed_val,
                        cfg,
                    )
                    summary_overview.append(
                        {
                            "code": code,
                            "train_size": n,
                            "seed": seed_val,
                            "backend": cfg.backend_name,
                            "predictions": str(pred_path),
                            "summary": str(summary_path),
                            "status": "ok",
                        }
                    )
                except Exception as exc:
                    print(f"❌ Combo failed for {code}, n={n}, seed={seed_val}: {exc}", flush=True)
                    summary_overview.append(
                        {
                            "code": code,
                            "train_size": n,
                            "seed": seed_val,
                            "backend": cfg.backend_name,
                            "predictions": "",
                            "summary": "",
                            "status": f"error: {exc}",
                        }
                    )

    if summary_overview:
        overview_df = pd.DataFrame.from_records(summary_overview)
        print("\nCompleted combinations:")
        print(overview_df.to_string(index=False))


if __name__ == "__main__":
    main()
