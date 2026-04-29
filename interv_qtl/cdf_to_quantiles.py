"""
Quantile estimation pipeline for interventional distribution Y|do(X=x).

Core methodology:
1. Stage 1: Estimate control function V = F(X|Z) using TabPFN
   
2. Stage 2: Estimate conditional CDF F(y|X,V) using TabPFN full distribution output
   
3. Integrate over V to obtain interventional CDF:
   F(y|do(X=x)) = ∫[0,1] F(y|X=x,V=v) dv
   
4. Invert the CDF to extract quantile curves:
   q_τ(x) = inf{y: F(y|do(X=x)) ≥ τ}

This implementation uses:
- Numerical integration: Gauss-Legendre (18) for V integration by default
- CDF inversion: Binary search + linear interpolation on discrete y_grid
- Ground truth: Monte Carlo sampling from the DGP
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
BATCH_DIR = REPO_ROOT / "interv_mean"
BATCH_CORE_DIR = REPO_ROOT / "tabcf_core"
for path in (BATCH_CORE_DIR, BATCH_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

import numpy as np
import pandas as pd

# Import from Stage 2 batch variant
from stage2_outcome import (
    ConditionalCDFEstimator,
    _latest_matching_file,
    cdf_from_full_output,
    load_stage1_data,
    monte_carlo_y_given_x,
)
from mu_integrators import (
    DEFAULT_GAUSS_LEGENDRE_ORDER,
    DEFAULT_MU_INTEGRATOR,
    MuIntegratorConfig,
    integrate_values,
    resolve_max_points_per_batch,
    resolve_v_rule,
)

# Import from DGP
from dgp import DGPConfig, set_seed

# Import TabPFN lazily for Stage 3 smoothing only.
_TABPFN_IMPORT_ERROR: Optional[Exception] = None
try:
    from tabpfn.regressor import TabPFNRegressor

    _HAVE_TABPFN = True
    print("✅ TabPFNRegressor imported successfully")
except Exception as exc:  # pragma: no cover - surfaced during CLI run
    TabPFNRegressor = None  # type: ignore[assignment]
    _HAVE_TABPFN = False
    _TABPFN_IMPORT_ERROR = exc


def _require_tabpfn_regressor():
    if TabPFNRegressor is None:
        raise ImportError(f"TabPFNRegressor is required for smoothing but could not be imported: {_TABPFN_IMPORT_ERROR}")
    return TabPFNRegressor


@dataclass
class QuantileConfig:
    """Configuration for quantile estimation pipeline."""

    # Input/output
    input_dir: str = "IV_datasets/stage1_output"
    output_dir: str = "IV_datasets/quantile_output"
    dgp_base_dir: str = "IV_datasets"
    n_train_samples: Optional[int] = 2000

    random_state: int = 1

    # Quantile levels (default: 17 quantiles)
    quantile_levels: Tuple[float, ...] = (
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

    # V integration rule settings
    n_v_integration_points: int = 100
    mu_integrator: str = DEFAULT_MU_INTEGRATOR
    gauss_legendre_order: int = DEFAULT_GAUSS_LEGENDRE_ORDER

    # Model settings
    use_tabpfn: bool = True

    # DGP identifiers
    first_stage_code: str = "A3"
    second_stage_code: str = "B3"

    # Monte Carlo for true quantiles
    mc_samples: int = 2000

    # Stage 3 smoothing
    smooth_grid_points: int = 2000

    # X grid configuration (built from training data range unless overridden)
    x_grid_points: int = 200  # Number of points in x grid
    x_min_override: Optional[float] = None  # Optional manual lower bound
    x_max_override: Optional[float] = None  # Optional manual upper bound

    # Y grid used for CDF inversion
    y_grid_points: int = 1201
    y_grid_padding: float = 0.25

    def __post_init__(self) -> None:
        integrator_cfg = MuIntegratorConfig(
            method=self.mu_integrator,
            n_v_points=self.n_v_integration_points,
            gauss_legendre_order=self.gauss_legendre_order,
        )
        self.mu_integrator = integrator_cfg.method
        self.n_v_integration_points = integrator_cfg.n_v_points
        self.gauss_legendre_order = integrator_cfg.gauss_legendre_order


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_path(path_like: str | os.PathLike[str]) -> Path:
    """Resolve paths relative to repository root."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def set_project_root(path_like: str | os.PathLike[str]) -> None:
    """Add a project root to sys.path for script-style execution."""
    root = Path(path_like).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_y_grid(y_values: np.ndarray, *, padding: float, n_points: int) -> np.ndarray:
    """
    Construct a Y grid with symmetric padding around observed values.
    
    Args:
        y_values: Observed Y values from training data
        padding: Fraction of range to extend on each side (e.g., 0.25 = 25% extension)
        n_points: Number of grid points (minimum 10)
    
    Returns:
        np.ndarray: Uniformly spaced grid from (y_min - pad) to (y_max + pad)
    
    Note: Adequate padding is critical for extreme quantiles (e.g., τ=0.01, 0.99)
          to avoid boundary truncation during CDF inversion.
    """
    y_values = np.asarray(y_values, dtype=float)
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    span = y_max - y_min
    pad = max(padding * span, 1e-3) if span > 0 else 1.0
    grid = np.linspace(y_min - pad, y_max + pad, int(max(10, n_points)))
    return grid.astype(float)


def _invert_single_quantile(
    cdf_row: np.ndarray,
    y_grid: np.ndarray,
    tau: float,
) -> float:
    """Invert a single CDF row via linear interpolation."""

    if tau <= cdf_row[0]:
        return float(y_grid[0])
    if tau >= cdf_row[-1]:
        return float(y_grid[-1])

    idx = int(np.searchsorted(cdf_row, tau, side="left"))
    if idx == 0:
        return float(y_grid[0])
    if idx >= len(y_grid):
        return float(y_grid[-1])

    cdf_lo = float(cdf_row[idx - 1])
    cdf_hi = float(cdf_row[idx])
    y_lo = float(y_grid[idx - 1])
    y_hi = float(y_grid[idx])

    if cdf_hi <= cdf_lo + 1e-12:
        return y_hi

    weight = (tau - cdf_lo) / (cdf_hi - cdf_lo)
    return y_lo + weight * (y_hi - y_lo)


def invert_cdf_to_quantiles(
    cdf_matrix: np.ndarray,
    y_grid: np.ndarray,
    quantile_levels: Tuple[float, ...],
) -> np.ndarray:
    """
    Invert monotone CDF rows to obtain quantiles for requested levels.
    
    Algorithm:
    1. Enforce monotonicity: CDF[i] = max(CDF[0], ..., CDF[i])
    2. For each τ, find index where CDF crosses τ using binary search
    3. Linearly interpolate between adjacent y_grid points
    
    Args:
        cdf_matrix: Shape (n_x, n_y), CDF values at each (x, y) pair
        y_grid: Y values corresponding to CDF columns
        quantile_levels: Requested quantile levels τ ∈ (0,1)
    
    Returns:
        np.ndarray: Shape (n_x, n_quantiles), quantile values q_τ(x)
    """
    # Enforce monotonicity and clip to [0, 1]
    cdf_clipped = np.clip(np.maximum.accumulate(cdf_matrix, axis=1), 0.0, 1.0)
    n_x, _ = cdf_clipped.shape
    n_q = len(quantile_levels)
    quantiles = np.empty((n_x, n_q), dtype=float)

    tol = 1e-6
    warn_low = False
    warn_high = False

    for idx in range(n_x):
        row = cdf_clipped[idx]
        row_min = float(row[0])
        row_max = float(row[-1])
        for q_idx, tau in enumerate(quantile_levels):
            tau_f = float(tau)
            if tau_f <= row_min + tol:
                quantiles[idx, q_idx] = float(y_grid[0])
                if tau_f < row_min - tol:
                    warn_low = True
                continue
            if tau_f >= row_max - tol:
                quantiles[idx, q_idx] = float(y_grid[-1])
                if tau_f > row_max + tol:
                    warn_high = True
                continue
            quantiles[idx, q_idx] = _invert_single_quantile(row, y_grid, tau_f)

    if warn_low:
        print(
            "⚠️  Some requested quantiles fell below the evaluated y_grid support. "
            "Consider increasing y_grid_padding or y_grid_points.",
            flush=True,
        )
    if warn_high:
        print(
            "⚠️  Some requested quantiles exceeded the evaluated y_grid support. "
            "Consider increasing y_grid_padding or y_grid_points.",
            flush=True,
        )

    return quantiles


def quantile_integrator_config(
    cfg: QuantileConfig,
    *,
    max_points_per_batch: int | None = None,
) -> MuIntegratorConfig:
    return MuIntegratorConfig(
        method=cfg.mu_integrator,
        n_v_points=cfg.n_v_integration_points,
        gauss_legendre_order=cfg.gauss_legendre_order,
        max_points_per_batch=max_points_per_batch,
    )


def compute_quantiles_on_grid(
    cdf_model: ConditionalCDFEstimator,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    quantile_levels: Tuple[float, ...],
    n_v_points: int,
    max_points_per_batch: Optional[int] = None,
    *,
    integrator_cfg: MuIntegratorConfig | None = None,
) -> np.ndarray:
    """
    Compute Q_τ(x) by integrating CDFs over V and inverting them.
    
    Core computation:
    1. For each x in x_grid, compute F(y|X=x,V=v) for all v in v_grid and y in y_grid
    2. Integrate over V using the configured quadrature rule:
       F(y|do(X=x)) = ∫ F(y|X=x,V=v) dv
    3. Invert the resulting interventional CDF to get q_τ(x)
    
    Args:
        cdf_model: Fitted ConditionalCDFEstimator (TabPFN-based)
        x_grid: Intervention points to evaluate (shape: k)
        y_grid: Y values for CDF evaluation (shape: n_y)
        quantile_levels: Requested quantiles τ
        n_v_points: Number of V integration points for the uniform-grid fallback
        max_points_per_batch: Chunk size for batched prediction (memory control)
        integrator_cfg: Optional explicit integration configuration
    
    Returns:
        np.ndarray: Shape (k, n_quantiles), quantile estimates q_τ(x)
    """
    x_arr = np.asarray(x_grid, dtype=float).reshape(-1)
    k = len(x_arr)
    resolved_integrator = integrator_cfg or MuIntegratorConfig(
        n_v_points=int(n_v_points),
        max_points_per_batch=max_points_per_batch,
    )
    v_grid, v_weights = resolve_v_rule(resolved_integrator)
    n_v = len(v_grid)
    n_y = len(y_grid)

    print(
        "Computing CDF-integrated quantiles on grid: "
        f"{k} x values × {n_v} V points × {n_y} y points via "
        f"{resolved_integrator.method} integration",
        flush=True,
    )

    batch_size = resolve_max_points_per_batch(
        k,
        n_v,
        max_points_per_batch=resolved_integrator.max_points_per_batch,
    )

    cdf_matrix = np.empty((k, n_y), dtype=float)

    for start in range(0, k, batch_size):
        end = min(start + batch_size, k)
        x_chunk = np.asarray(x_arr[start:end], dtype=float)
        repeat_count = len(x_chunk)
        x_col = np.repeat(x_chunk, n_v).astype(float)
        v_col = np.tile(v_grid, repeat_count).astype(float)

        full_output = cdf_model.predict_full_distribution(x_col, v_col)
        cdf_vals = np.asarray(
            cdf_from_full_output(full_output, y_grid, squeeze_last=False),
            dtype=float,
        )

        if cdf_vals.ndim == 1:
            cdf_vals = cdf_vals.reshape(1, -1)

        expected = repeat_count * n_v
        if cdf_vals.shape[0] != expected:
            raise ValueError(
                f"Unexpected CDF predictions shape {cdf_vals.shape}; "
                f"expected ({expected}, {n_y})."
            )

        cdf_vals = cdf_vals.reshape(repeat_count, n_v, n_y)
        integrated = integrate_values(
            cdf_vals,
            v_grid,
            integrator_cfg=resolved_integrator,
            axis=1,
            v_weights=v_weights,
        )
        cdf_matrix[start:end, :] = integrated

    quantiles = invert_cdf_to_quantiles(cdf_matrix, y_grid, quantile_levels)
    print("✅ Quantiles computed from integrated CDFs", flush=True)
    return quantiles


def compute_true_quantiles(
    dgp_cfg: DGPConfig,
    x_grid: np.ndarray,
    quantile_levels: Tuple[float, ...],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute ground-truth quantiles of Y|do(X=x) via Monte Carlo.
    
    Procedure:
    1. For each x, sample (H, ε) from marginal distributions (independent of Z)
    2. Generate Y = g(x, H, ε) using the DGP second-stage equation
    3. Estimate q_τ(x) as the empirical τ-quantile of {Y_1, ..., Y_M}
    
    Args:
        dgp_cfg: DGP configuration specifying first/second stage equations
        x_grid: Intervention values (shape: k)
        quantile_levels: Requested quantiles τ
        n_samples: Number of Monte Carlo samples per x (larger = more accurate)
        rng: NumPy random generator for reproducibility
    
    Returns:
        np.ndarray: Shape (k, n_quantiles), true quantiles q_τ(x)
    
    Note: This ground truth represents the interventional distribution do(X=x),
          not the observational distribution Y|X=x.
    """
    k = len(x_grid)
    n_quantiles = len(quantile_levels)
    true_quantiles = np.empty((k, n_quantiles), dtype=float)

    print(
        f"Computing true quantiles via Monte Carlo: {k} x values × {n_samples} samples...",
        flush=True,
    )

    quantile_array = np.asarray(quantile_levels, dtype=float)

    for idx, x_val in enumerate(x_grid):
        y_samples, _ = monte_carlo_y_given_x(
            dgp_cfg,
            float(x_val),
            n_samples,
            rng,
        )
        true_quantiles[idx, :] = np.quantile(y_samples, quantile_array)

    print("✅ True quantiles computed", flush=True)
    return true_quantiles


def smooth_quantile_curves(
    x_grid: np.ndarray,
    quantile_estimates: np.ndarray,
    quantile_true: np.ndarray,
    quantile_levels: Tuple[float, ...],
    smoother_grid_points: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Stage 3 smoothing to quantile curves using TabPFNRegressor.
    
    Purpose: Smooth the raw quantile estimates for visualization (NOT for metrics).
    
    Method: For each τ, fit a separate TabPFN regressor on (x_grid, q_τ(x_grid))
            and predict on a denser smooth_x grid.
    
    Args:
        x_grid: Original evaluation points (shape: k)
        quantile_estimates: Estimated quantiles (shape: k × n_quantiles)
        quantile_true: True quantiles (shape: k × n_quantiles)
        quantile_levels: Quantile levels τ
        smoother_grid_points: Number of points in the smoothed curve
        seed: Random seed for TabPFN
    
    Returns:
        Tuple of (smooth_x, smooth_estimates, smooth_true)
    
    Warning: Only use smoothed curves for plotting. Metrics should use raw estimates.
    """
    n_quantiles = len(quantile_levels)

    if smoother_grid_points <= 1:
        raise ValueError("smoother_grid_points must be greater than 1.")

    x_min = float(np.min(x_grid))
    x_max = float(np.max(x_grid))
    smooth_x = np.linspace(x_min, x_max, int(smoother_grid_points))

    smooth_estimates = np.empty((smoother_grid_points, n_quantiles), dtype=float)
    smooth_true = np.empty((smoother_grid_points, n_quantiles), dtype=float)

    regressor_cls = _require_tabpfn_regressor()
    print(f"Smoothing {n_quantiles} quantile curves using TabPFNRegressor...", flush=True)

    features = np.asarray(x_grid, dtype=float).reshape(-1, 1)

    for q_idx in range(n_quantiles):
        reg_est = regressor_cls(
            random_state=seed if seed is not None else 1,
            ignore_pretraining_limits=True,
        )
        targets_est = np.asarray(quantile_estimates[:, q_idx], dtype=float)
        reg_est.fit(features, targets_est)
        smooth_estimates[:, q_idx] = reg_est.predict(smooth_x.reshape(-1, 1))

        reg_true = regressor_cls(
            random_state=(seed + 17) if seed is not None else 1,
            ignore_pretraining_limits=True,
        )
        targets_true = np.asarray(quantile_true[:, q_idx], dtype=float)
        reg_true.fit(features, targets_true)
        smooth_true[:, q_idx] = reg_true.predict(smooth_x.reshape(-1, 1))

    print("✅ Quantile curves smoothed", flush=True)
    return smooth_x, smooth_estimates, smooth_true


def create_quantile_plot(
    x_grid: np.ndarray,
    quantile_estimates: np.ndarray,
    quantile_true: np.ndarray,
    quantile_levels: Tuple[float, ...],
    smooth_x: Optional[np.ndarray],
    smooth_estimates: Optional[np.ndarray],
    smooth_true: Optional[np.ndarray],
    output_path: str,
) -> None:
    """Create a single large plot showing all quantile curves together."""

    import matplotlib.pyplot as plt

    n_quantiles = len(quantile_levels)
    cmap = plt.get_cmap("viridis")
    color_positions = np.linspace(0.1, 0.9, n_quantiles)
    colors = [cmap(pos) for pos in color_positions]

    fig, ax = plt.subplots(figsize=(14, 8))

    for q_idx, q_level in enumerate(quantile_levels):
        color = colors[q_idx]
        label_est = f"Est q={q_level:.2f}"
        label_true = f"True q={q_level:.2f}"

        if smooth_x is not None and smooth_estimates is not None and smooth_true is not None:
            ax.plot(
                smooth_x,
                smooth_estimates[:, q_idx],
                color=color,
                linewidth=2.2,
                label=label_est,
            )
            ax.plot(
                smooth_x,
                smooth_true[:, q_idx],
                color=color,
                linestyle="--",
                linewidth=1.8,
                label=label_true,
            )
        else:
            ax.plot(
                x_grid,
                quantile_estimates[:, q_idx],
                color=color,
                linewidth=2.2,
                label=label_est,
            )
            ax.plot(
                x_grid,
                quantile_true[:, q_idx],
                color=color,
                linestyle="--",
                linewidth=1.8,
                label=label_true,
            )

        ax.scatter(
            x_grid,
            quantile_estimates[:, q_idx],
            facecolors="none",
            edgecolors=color,
            linewidths=1.0,
            s=35,
            alpha=0.9,
            marker="o",
        )
        ax.scatter(
            x_grid,
            quantile_true[:, q_idx],
            color=color,
            linewidths=0.9,
            s=40,
            alpha=0.9,
            marker="x",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Quantile Value")
    ax.set_title("Estimated vs True Quantile Curves (smoothed lines, raw points)", fontsize=13)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(loc="best", fontsize=8.5, ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Quantile plot saved to: {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

def prepare_quantile_components(cfg: QuantileConfig) -> Dict[str, object]:
    """Load Stage 1 outputs and train the TabPFN CDF model."""

    input_dir = resolve_path(cfg.input_dir)
    codes = f"{cfg.first_stage_code}_{cfg.second_stage_code}"
    sample_tag = f"_{cfg.n_train_samples}" if cfg.n_train_samples is not None else ""
    train_prefix = f"iv_stage1_train_{codes}{sample_tag}_"
    train_csv = _latest_matching_file(input_dir, train_prefix)
    if train_csv is None:
        raise FileNotFoundError(
            f"No Stage-1 training CSV found in {input_dir} matching prefix {train_prefix}"
        )

    print(f"Resolved Stage-1 training CSV: {train_csv}", flush=True)

    train_data = load_stage1_data(train_csv)
    observed_train_samples = len(train_data["X"])
    if cfg.n_train_samples is not None and observed_train_samples != cfg.n_train_samples:
        raise ValueError(
            f"Resolved Stage-1 CSV ({train_csv}) has {observed_train_samples} rows; "
            f"expected {cfg.n_train_samples}."
        )

    dgp_cfg = DGPConfig(first_stage=cfg.first_stage_code, second_stage=cfg.second_stage_code)

    cdf_model = ConditionalCDFEstimator(use_tabpfn=cfg.use_tabpfn)
    _ = cdf_model.fit_full(train_data["X"], train_data["V_hat"], train_data["Y"])

    return {
        "codes": codes,
        "train_csv": train_csv,
        "train_data": train_data,
        "dgp_config": dgp_cfg,
        "cdf_model": cdf_model,
        "n_train_samples": observed_train_samples,
    }


def run_quantile_experiment(cfg: QuantileConfig) -> Dict[str, object]:
    """Execute the quantile estimation pipeline."""

    import sys

    print("Starting Quantile Estimation Experiment...", flush=True)
    print(f"Configuration: {asdict(cfg)}", flush=True)
    sys.stdout.flush()

    set_seed(cfg.random_state)
    rng = np.random.default_rng(cfg.random_state)

    components = prepare_quantile_components(cfg)
    train_data = components["train_data"]
    dgp_cfg = components["dgp_config"]
    cdf_model = components["cdf_model"]

    n_train = len(train_data["X"])
    X_train = train_data["X"]

    train_x_min = float(np.min(X_train))
    train_x_max = float(np.max(X_train))

    x_min = cfg.x_min_override if cfg.x_min_override is not None else train_x_min
    x_max = cfg.x_max_override if cfg.x_max_override is not None else train_x_max
    if x_min >= x_max:
        raise ValueError(
            f"x-range is invalid (x_min={x_min}, x_max={x_max}). "
            "Ensure overrides define a proper interval."
        )
    x_test_grid = np.linspace(x_min, x_max, cfg.x_grid_points)

    y_grid = build_y_grid(
        train_data["Y"],
        padding=cfg.y_grid_padding,
        n_points=cfg.y_grid_points,
    )

    print(f"\n[1/5] Components prepared: train={n_train} samples", flush=True)
    print(
        f"  X grid: {cfg.x_grid_points} points from {x_min:.3f} to {x_max:.3f} "
        f"({'overrides' if (cfg.x_min_override is not None or cfg.x_max_override is not None) else 'training range'})",
        flush=True,
    )
    print(
        f"  Y grid: {len(y_grid)} points from {y_grid.min():.3f} to {y_grid.max():.3f} (padding={cfg.y_grid_padding:.2f})",
        flush=True,
    )

    print("\n[2/5] Computing quantiles from integrated CDFs...", flush=True)
    quantiles_estimated = compute_quantiles_on_grid(
        cdf_model,
        x_test_grid,
        y_grid,
        cfg.quantile_levels,
        cfg.n_v_integration_points,
        integrator_cfg=quantile_integrator_config(cfg),
    )

    print("\n[3/5] Computing true quantiles via Monte Carlo...", flush=True)
    quantiles_true = compute_true_quantiles(
        dgp_cfg,
        x_test_grid,
        cfg.quantile_levels,
        cfg.mc_samples,
        rng,
    )

    print("\n[4/5] Applying Stage 3 smoothing...", flush=True)
    smooth_x, smooth_estimates, smooth_true = smooth_quantile_curves(
        x_test_grid,
        quantiles_estimated,
        quantiles_true,
        cfg.quantile_levels,
        cfg.smooth_grid_points,
        seed=cfg.random_state,
    )

    print("\n[5/5] Preparing outputs...", flush=True)

    predictions_dict = {"X": x_test_grid}
    for q_idx, q_level in enumerate(cfg.quantile_levels):
        col_name = f"quantile_{q_level:.2f}".replace(".", "_")
        predictions_dict[col_name] = quantiles_estimated[:, q_idx]

    predictions_df = pd.DataFrame(predictions_dict)

    smoothed_dict = {"x": smooth_x}
    for q_idx, q_level in enumerate(cfg.quantile_levels):
        col_name_est = f"quantile_{q_level:.2f}_estimated".replace(".", "_")
        col_name_true = f"quantile_{q_level:.2f}_true".replace(".", "_")
        smoothed_dict[col_name_est] = smooth_estimates[:, q_idx]
        smoothed_dict[col_name_true] = smooth_true[:, q_idx]

    smoothed_df = pd.DataFrame(smoothed_dict)

    return {
        "config": asdict(cfg),
        "data_stats": {
            "n_train_samples": n_train,
            "n_x_grid_points": cfg.x_grid_points,
            "x_grid_min": float(x_min),
            "x_grid_max": float(x_max),
            "n_y_grid_points": len(y_grid),
            "y_grid_min": float(y_grid.min()),
            "y_grid_max": float(y_grid.max()),
            "n_quantiles": len(cfg.quantile_levels),
            "quantile_levels": cfg.quantile_levels,
            "n_v_integration_points": cfg.n_v_integration_points,
            "mu_integrator": cfg.mu_integrator,
            "gauss_legendre_order": cfg.gauss_legendre_order,
            "mc_samples": cfg.mc_samples,
        },
        "predictions": predictions_df,
        "smoothed": smoothed_df,
        "x_grid": x_test_grid,
        "y_grid": y_grid,
        "quantiles_estimated": quantiles_estimated,
        "quantiles_true": quantiles_true,
        "smooth_x": smooth_x,
        "smooth_estimates": smooth_estimates,
        "smooth_true": smooth_true,
        "metadata": {
            "train_csv": components["train_csv"],
            "codes": components["codes"],
            "n_train_samples": components["n_train_samples"],
        },
    }


def save_quantile_results(results: Dict[str, object], output_dir: str | os.PathLike[str]) -> None:
    """Save quantile estimation outputs as CSV files and plots."""

    output_path = resolve_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    codes = results["metadata"]["codes"]
    n_train_samples_meta = results["metadata"].get("n_train_samples")
    sample_tag = f"_{int(n_train_samples_meta)}" if n_train_samples_meta is not None else ""
    base_prefix = f"quantile_{codes}{sample_tag}"
    artifact_names: List[str] = []

    predictions_df = results["predictions"].copy()
    predictions_df = predictions_df.sort_values("X").reset_index(drop=True)
    predictions_csv = output_path / f"{base_prefix}_predictions_{timestamp}.csv"
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"✅ Predictions saved to: {predictions_csv}")
    artifact_names.append(predictions_csv.name)

    smoothed_df = results["smoothed"].copy()
    smoothed_csv = output_path / f"{base_prefix}_smoothed_{timestamp}.csv"
    smoothed_df.to_csv(smoothed_csv, index=False)
    print(f"✅ Smoothed curves saved to: {smoothed_csv}")
    artifact_names.append(smoothed_csv.name)

    summary_rows = []
    for key, value in results["data_stats"].items():
        summary_rows.append({"key": key, "value": value})
    summary_rows.append({"key": "train_csv", "value": str(results["metadata"]["train_csv"])})
    summary_rows.append({"key": "timestamp", "value": timestamp})

    summary_csv = output_path / f"{base_prefix}_summary_{timestamp}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"✅ Summary saved to: {summary_csv}")
    artifact_names.append(summary_csv.name)

    plot_path = output_path / f"{base_prefix}_plot_{timestamp}.png"
    create_quantile_plot(
        x_grid=results["x_grid"],
        quantile_estimates=results["quantiles_estimated"],
        quantile_true=results["quantiles_true"],
        quantile_levels=tuple(results["data_stats"]["quantile_levels"]),
        smooth_x=results["smooth_x"],
        smooth_estimates=results["smooth_estimates"],
        smooth_true=results["smooth_true"],
        output_path=str(plot_path),
    )
    artifact_names.append(plot_path.name)

    print(f"\n📁 Outputs stored in: {output_path}/")
    print(f"🕒 Timestamp: {timestamp}")
    for name in artifact_names:
        print(f"   - {name}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantile estimation pipeline (CDF variant).")
    parser.add_argument("--first-stage", default="A3", help="First-stage code (e.g. A3).")
    parser.add_argument("--second-stage", default="B3", help="Second-stage code (e.g. B3).")
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=2000,
        help="Expected number of Stage 1 samples (used in file selection).",
    )
    parser.add_argument(
        "--stage1-dir",
        default="IV_datasets/stage1_output",
        help="Directory containing Stage 1 CSV outputs.",
    )
    parser.add_argument(
        "--dgp-dir",
        default="IV_datasets",
        help="Directory containing DGP train/test CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="IV_datasets/quantile_output",
        help="Directory to store quantile outputs.",
    )
    parser.add_argument(
        "--quantile-levels",
        type=str,
        default="0.01,0.025,0.1,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.9,0.975,0.99",
        help="Comma-separated quantile levels (e.g., '0.01,0.025,...,0.99').",
    )
    parser.add_argument("--random-state", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--n-v-grid",
        type=int,
        default=100,
        help="Number of V integration points for the optional uniform-grid fallback integrator.",
    )
    parser.add_argument(
        "--mu-integrator",
        type=str,
        default=DEFAULT_MU_INTEGRATOR,
        choices=["simpson", "gauss_legendre"],
        help="Stage-2 integration rule. Default: gauss_legendre.",
    )
    parser.add_argument(
        "--gauss-legendre-order",
        type=int,
        default=DEFAULT_GAUSS_LEGENDRE_ORDER,
        help="Quadrature order when --mu-integrator=gauss_legendre.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=2000,
        help="Monte Carlo samples for true quantile computation.",
    )
    parser.add_argument(
        "--smooth-grid-points",
        type=int,
        default=2000,
        help="Number of points for Stage 3 smoothing grid.",
    )
    parser.add_argument(
        "--x-grid-points",
        type=int,
        default=200,
        help="Number of X grid points built from training quantiles.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional lower bound for the X grid (defaults to min training X).",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional upper bound for the X grid (defaults to max training X).",
    )
    parser.add_argument(
        "--y-grid-points",
        type=int,
        default=1201,
        help="Number of Y grid points used when evaluating the CDF.",
    )
    parser.add_argument(
        "--y-grid-padding",
        type=float,
        default=0.25,
        help="Padding fraction added to Y min/max when building the evaluation grid.",
    )
    parser.add_argument(
        "--disable-tabpfn",
        action="store_true",
        help="Use RandomForest fallback instead of TabPFN (not recommended).",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Override project root for resolving relative paths.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()

    if args.project_root:
        set_project_root(args.project_root)

    print("=" * 60, flush=True)
    print("QUANTILE ESTIMATION PIPELINE (CDF variant): STARTING EXECUTION", flush=True)
    print("=" * 60, flush=True)
    print(f"TabPFN available: {_HAVE_TABPFN}", flush=True)

    quantile_levels = tuple(float(q.strip()) for q in args.quantile_levels.split(","))
    print(f"Quantile levels: {quantile_levels}", flush=True)

    cfg = QuantileConfig(
        input_dir=args.stage1_dir,
        output_dir=args.output_dir,
        dgp_base_dir=args.dgp_dir,
        n_train_samples=args.train_sample_size,
        random_state=args.random_state,
        quantile_levels=quantile_levels,
        n_v_integration_points=args.n_v_grid,
        mu_integrator=args.mu_integrator,
        gauss_legendre_order=args.gauss_legendre_order,
        use_tabpfn=not args.disable_tabpfn,
        first_stage_code=args.first_stage,
        second_stage_code=args.second_stage,
        mc_samples=args.mc_samples,
        smooth_grid_points=args.smooth_grid_points,
        x_grid_points=args.x_grid_points,
        x_min_override=args.x_min,
        x_max_override=args.x_max,
        y_grid_points=args.y_grid_points,
        y_grid_padding=args.y_grid_padding,
    )

    print(f"Configuration: {asdict(cfg)}", flush=True)
    print("", flush=True)

    try:
        results = run_quantile_experiment(cfg)
        save_quantile_results(results, cfg.output_dir)

        print("\n" + "=" * 60)
        print("QUANTILE ESTIMATION COMPLETED SUCCESSFULLY", flush=True)
        print("=" * 60)
    except Exception as exc:  # pragma: no cover - surfaced during CLI run
        print("\n❌ Quantile estimation failed with error:", exc, flush=True)
        import traceback

        traceback.print_exc()
