"""
IV Stage 2.9 Implementation: Density Diagnostics
================================================

Stage 2.9 extends Stage 2.8's full-distribution integration with diagnostics
targeted at the interventional density f(y | do(X = x)).

New in Stage 2.9:
- Align μ_c(x) evaluation points with the held-out test samples and export the
  paired `y_test` column for the μ_c curves CSV.
- Integrate criterion-based PDFs to recover f(y|do(X=x)) and compare with
  analytical ground-truth curves on the same grid.
- Produce a Figure-3-style kernel density visualization for representative test
  points, sourced from configurable X-quantiles.

Stage 2.8 foundations retained:
- μ_c(x) via Gauss-Legendre (18) integration over TabPFN structural predictions by default.
- F(y|do(X=x)) without quantile inversion, using criterion.cdf directly.
- Deterministic integration grids and reproducible output artefacts.
"""

from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("TABPFN_MODEL_VERSION", "v2.5")  # Default to latest TabPFN (requires HF token)
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(Path(__file__).resolve().parent.parent.parent / "tabpfn_home_config" / "models"))

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Tuple, Optional
from scipy.stats import norm, gaussian_kde
from datetime import datetime

# Import from Stage 1
from dgp import (
    DGPConfig,
    LATENT_H_WEIGHT,
    b10_response,
    set_seed,
    sigmaY_of_X,
)
from foundation_backends import (
    TABPFN_BACKEND,
    backend_metadata,
    cdf_from_distribution_output,
    make_regressor_backend,
    normalize_backend_name,
    pdf_from_distribution_output,
    predict_distribution,
    predict_mean as predict_backend_mean,
    stage1_output_filename,
    stage2_base_prefix,
)
from local_context_backends import (
    LocalContextConfig,
    ensure_local_context_config,
    local_context_metadata,
)
from dgp_test_utils import (
    DEFAULT_TEST_X_TRIM_QUANTILE_RANGE,
    normalize_test_x_trim_quantile_range,
    trim_test_array_dict_by_x,
)
from mu_integrators import (
    DEFAULT_GAUSS_LEGENDRE_ORDER,
    DEFAULT_MU_INTEGRATOR,
    MuIntegratorConfig,
    integrate_mean_function_over_v,
    integrate_values,
    resolve_v_rule,
)


BATCH_ROOT = Path(__file__).resolve().parents[1] / "interv_mean"
DEFAULT_DATA_DIR = BATCH_ROOT / "IV_datasets"
DEFAULT_STAGE1_OUTPUT_DIR = DEFAULT_DATA_DIR / "stage1_output"
DEFAULT_STAGE2_OUTPUT_DIR = DEFAULT_DATA_DIR / "stage2_output"

V_EPSILON = 1e-6  # Avoid norm.ppf endpoints
B4_SOFTPLUS_EPS = 1e-8  # Match numerical guard in DGP B4 branch

# =========================================================
# Stage 1 data loading helper
# =========================================================


# 1) Extended Configuration and I/O helpers
# =========================================================


def _latest_matching_file(directory: str | os.PathLike[str], prefix: str) -> Optional[Path]:
    """Return the most recently modified CSV in directory that starts with prefix."""
    directory_path = Path(directory)
    if not directory_path.exists():
        return None
    candidates = sorted(
        (p for p in directory_path.glob(f"{prefix}*.csv") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None

def load_stage1_data(csv_path: str | os.PathLike[str]) -> Dict[str, np.ndarray]:
    """Load Stage 1 outputs from CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Stage 1 CSV file not found: {csv_path}\n"
            "Please run IV_stage1_1.py first to generate Stage 1 outputs."
        )

    print(f"Loading Stage 1 data from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = ["X", "Y", "V_hat", "V_true"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in CSV: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    data = {
        "Z": df["Z"].to_numpy() if "Z" in df.columns else None,
        "X": df["X"].to_numpy(),
        "Y": df["Y"].to_numpy(),
        "V_hat": df["V_hat"].to_numpy(),
        "V_true": df["V_true"].to_numpy(),
        "eps": df["eps"].to_numpy() if "eps" in df.columns else None,
        "eta": df["eta"].to_numpy() if "eta" in df.columns else None,
    }

    V_hat = data["V_hat"]
    if np.any(V_hat < 0) or np.any(V_hat > 1):
        print("⚠️  Warning: V_hat has values outside [0,1]. Clipping to [0,1].")
        print(f"   Original range: [{V_hat.min():.4f}, {V_hat.max():.4f}]")
        data["V_hat"] = np.clip(V_hat, 0.0, 1.0)

    n = len(data["X"])
    print(f"✅ Loaded {n} samples from Stage 1")
    print(f"   X range: [{data['X'].min():.3f}, {data['X'].max():.3f}]")
    print(f"   Y range: [{data['Y'].min():.3f}, {data['Y'].max():.3f}]")
    print(f"   V_hat range: [{data['V_hat'].min():.3f}, {data['V_hat'].max():.3f}]")
    print(f"   V_true range: [{data['V_true'].min():.3f}, {data['V_true'].max():.3f}]")

    return data


def load_dgp_test_data(
    first_stage: str,
    second_stage: str,
    base_dir: str | os.PathLike[str] = DEFAULT_DATA_DIR,
    test_x_trim_quantile_range: tuple[float, float] | None = DEFAULT_TEST_X_TRIM_QUANTILE_RANGE,
) -> Tuple[Path, Dict[str, Optional[np.ndarray]]]:
    """
    Load immutable DGP test split from IV_datasets directory.

    Returns the resolved CSV path along with column arrays needed downstream.
    """
    base_path = Path(base_dir)
    test_file = base_path / "test" / f"test_data_{first_stage}_{second_stage}.csv"
    if not test_file.exists():
        raise FileNotFoundError(
            f"DGP test data not found: {test_file}\n"
            "Please ensure the pre-generated datasets are available under IV_datasets/test."
        )

    print(f"Loading DGP test data from: {test_file}")
    df = pd.read_csv(test_file)
    raw_test_rows = len(df)

    required_cols = ["Z", "X", "Y", "V_true"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"DGP test CSV is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    data: Dict[str, Optional[np.ndarray]] = {
        "Z": df["Z"].to_numpy(),
        "X": df["X"].to_numpy(),
        "Y": df["Y"].to_numpy(),
        "V_true": df["V_true"].to_numpy(),
        "V_hat": None,
        "eps": df["eps"].to_numpy() if "eps" in df.columns else None,
        "eta": df["eta"].to_numpy() if "eta" in df.columns else None,
    }

    normalized_trim_range = normalize_test_x_trim_quantile_range(test_x_trim_quantile_range)
    data = trim_test_array_dict_by_x(data, normalized_trim_range)
    selected_test_rows = len(data["X"]) if data["X"] is not None else 0
    if normalized_trim_range is None:
        print("Test X trim: disabled")
    else:
        print(
            "Test X trim range: "
            f"[{normalized_trim_range[0]:.2f}, {normalized_trim_range[1]:.2f}] "
            f"selected {selected_test_rows}/{raw_test_rows} rows"
        )

    return test_file, data

# =========================================================
# Structural function model from Stage 2.1
# =========================================================
class StructuralFunctionModel:
    """Estimate structural function m(x,v) = E[Y|X=x, V=v]."""

    def __init__(
        self,
        use_tabpfn: bool = True,
        *,
        backend_name: Optional[str] = None,
        model_path: str = "auto",
        random_state: int = 1,
        local_context: object | None = None,
        softmax_temperature: float | None = None,
    ):
        self.backend_name = normalize_backend_name(backend_name, use_tabpfn=use_tabpfn)
        self.use_tabpfn = self.backend_name in {TABPFN_BACKEND, "tabpfn_real"}
        raw_model_path = "" if model_path is None else str(model_path).strip()
        self.model_path = "auto" if not raw_model_path else raw_model_path
        self.random_state = int(random_state)
        self.local_context = ensure_local_context_config(local_context)
        self.softmax_temperature = None if softmax_temperature is None else float(softmax_temperature)
        self.model = None
        self._using_tabpfn = self.use_tabpfn

    def _new_regressor(self):
        """Create new regressor instance for the configured backend."""
        self._using_tabpfn = self.use_tabpfn
        return make_regressor_backend(
            self.backend_name,
            random_state=self.random_state,
            model_path=self.model_path,
            local_context=self.local_context,
            softmax_temperature=self.softmax_temperature,
        )

    def predict(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Predict m(x, v) for given arrays via explicit mean output."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit_full() first.")

        x = np.asarray(x).reshape(-1, 1)
        v = np.asarray(v).reshape(-1, 1)
        features = np.hstack([x, v])
        return predict_backend_mean(self.model, features, backend_name=self.backend_name)

    def __call__(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.predict(x, v)


class FullDataStructuralFunctionModel(StructuralFunctionModel):
    """Structural function model trained once on the full dataset.

    This augments the Stage 2.1 implementation by exposing a helper that
    fits the configured backend without cross validation.
    """

    def fit_full(self, X: np.ndarray, V: np.ndarray, Y: np.ndarray, *, verbose: bool = True) -> np.ndarray:
        X = np.asarray(X).reshape(-1, 1)
        V = np.asarray(V).reshape(-1, 1)
        Y = np.asarray(Y)
        features = np.hstack([X, V])

        if verbose:
            print("Training structural function on full dataset (no CV)...")
        reg = self._new_regressor()
        reg.fit(features, Y)
        self.model = reg

        preds = predict_backend_mean(self.model, features, backend_name=self.backend_name)
        method = self.backend_name
        if verbose:
            print(f"✅ {method} structural function fitted on {len(Y)} samples")
        return preds


@dataclass
class Stage2_9Config:
    """Configuration for Stage 2.9 diagnostics with numerical integration"""
    # Input/output
    input_dir: str = str(DEFAULT_STAGE1_OUTPUT_DIR)
    output_dir: str = str(DEFAULT_STAGE2_OUTPUT_DIR)
    dgp_base_dir: str = str(DEFAULT_DATA_DIR)
    n_train_samples: Optional[int] = None
    train_seed: Optional[int] = None
    
    random_state: int = 1
    
    # Test grids (uniform grids)
    n_y_grid: int = 100     # Number of Y points for CDF evaluation
    
    # V integration rule settings
    n_v_integration_points: int = 100  # Uniform-grid resolution used by Simpson fallback paths
    mu_integrator: str = DEFAULT_MU_INTEGRATOR
    gauss_legendre_order: int = DEFAULT_GAUSS_LEGENDRE_ORDER

    # Model settings
    use_tabpfn: bool = True
    backend_name: str = TABPFN_BACKEND
    model_path: str = "auto"
    local_context: LocalContextConfig = field(default_factory=LocalContextConfig)
    softmax_temperature: float | None = None

    # Density diagnostics
    kde_quantiles: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    kde_sample_size: int = 1000
    test_x_trim_quantile_range: tuple[float, float] | None = DEFAULT_TEST_X_TRIM_QUANTILE_RANGE

    # DGP identifiers (for file selection)
    first_stage_code: str = "A1"
    second_stage_code: str = "B1"

    # Monte Carlo estimation of y_clean
    y_clean_mc_samples: int = 5000

    def __post_init__(self) -> None:
        self.backend_name = normalize_backend_name(self.backend_name, use_tabpfn=self.use_tabpfn)
        raw_model_path = "" if self.model_path is None else str(self.model_path).strip()
        self.model_path = "auto" if not raw_model_path else raw_model_path
        self.local_context = ensure_local_context_config(self.local_context)
        if self.softmax_temperature is not None:
            self.softmax_temperature = float(self.softmax_temperature)
        self.test_x_trim_quantile_range = normalize_test_x_trim_quantile_range(
            self.test_x_trim_quantile_range
        )
        integrator_cfg = MuIntegratorConfig(
            method=self.mu_integrator,
            n_v_points=self.n_v_integration_points,
            gauss_legendre_order=self.gauss_legendre_order,
        )
        self.mu_integrator = integrator_cfg.method
        self.n_v_integration_points = integrator_cfg.n_v_points
        self.gauss_legendre_order = integrator_cfg.gauss_legendre_order


def prepare_stage2_components(cfg: Stage2_9Config):
    """
    Load Stage 1 outputs, train Stage 2 models, and return reusable components.
    """
    codes = f"{cfg.first_stage_code}_{cfg.second_stage_code}"
    sample_tag = f"_n{cfg.n_train_samples}" if cfg.n_train_samples is not None else ""
    seed_tag = f"_seed{cfg.train_seed}" if cfg.train_seed is not None else ""
    train_prefix = stage1_output_filename(
        "train",
        codes,
        train_sample_size=cfg.n_train_samples,
        seed=cfg.train_seed,
        backend_name=cfg.backend_name,
        softmax_temperature=cfg.softmax_temperature,
        timestamp=None,
    ).removesuffix(".csv")
    candidate = Path(cfg.input_dir) / f"{train_prefix}.csv"
    if candidate.exists():
        train_csv_path = candidate
    else:
        train_csv_path = _latest_matching_file(cfg.input_dir, train_prefix)

    if train_csv_path is None:
        raise FileNotFoundError(
            f"No Stage-1 training CSV found in {cfg.input_dir} matching prefix {train_prefix}"
        )

    print(f"Resolved Stage-1 training CSV: {train_csv_path}", flush=True)
    test_csv, test_data = load_dgp_test_data(
        cfg.first_stage_code,
        cfg.second_stage_code,
        base_dir=cfg.dgp_base_dir,
        test_x_trim_quantile_range=cfg.test_x_trim_quantile_range,
    )
    print(f"Resolved DGP test CSV: {test_csv}", flush=True)
    raw_test_samples = len(pd.read_csv(test_csv, usecols=["X"]))

    train_data = load_stage1_data(train_csv_path)
    observed_train_samples = len(train_data["X"])
    if cfg.n_train_samples is not None and observed_train_samples != cfg.n_train_samples:
        raise ValueError(
            f"Resolved Stage-1 CSV ({train_csv_path}) has {observed_train_samples} rows; "
            f"expected {cfg.n_train_samples}."
        )

    dgp_cfg = DGPConfig(first_stage=cfg.first_stage_code, second_stage=cfg.second_stage_code)

    m_model = FullDataStructuralFunctionModel(
        use_tabpfn=cfg.use_tabpfn,
        backend_name=cfg.backend_name,
        model_path=cfg.model_path,
        random_state=cfg.random_state,
        local_context=cfg.local_context,
        softmax_temperature=cfg.softmax_temperature,
    )
    _ = m_model.fit_full(train_data["X"], train_data["V_hat"], train_data["Y"])

    cdf_model = ConditionalCDFEstimator(
        use_tabpfn=cfg.use_tabpfn,
        backend_name=cfg.backend_name,
        model_path=cfg.model_path,
        random_state=cfg.random_state,
        local_context=cfg.local_context,
        softmax_temperature=cfg.softmax_temperature,
    )
    _ = cdf_model.fit_full(train_data["X"], train_data["V_hat"], train_data["Y"])

    return {
        "codes": codes,
        "train_csv": str(train_csv_path),
        "test_csv": str(test_csv),
        "train_data": train_data,
        "test_data": test_data,
        "n_test_samples_raw": raw_test_samples,
        "n_test_samples_selected": len(test_data["X"]) if test_data["X"] is not None else 0,
        "dgp_config": dgp_cfg,
        "m_model": m_model,
        "cdf_model": cdf_model,
        "n_train_samples": observed_train_samples,
    }

# =========================================================
# V Integration Grid Helper
# =========================================================
def create_v_integration_grid(n_points: int = 100) -> np.ndarray:
    """Create uniform grid over (0,1) for V integration.

    The grid is kept Simpson-compatible for the optional uniform-grid fallback.
    
    Args:
        n_points: Number of integration points (will be adjusted to odd if needed)
        
    Returns:
        v_grid: (n_points,) array of V values in (0,1)
    """
    if n_points % 2 == 0:
        n_points += 1  # Keep the optional Simpson fallback grid valid.
    return np.linspace(V_EPSILON, 1.0 - V_EPSILON, n_points)


def stage2_mu_integrator_config(
    cfg: Stage2_9Config,
    *,
    max_points_per_batch: int | None = None,
) -> MuIntegratorConfig:
    return MuIntegratorConfig(
        method=cfg.mu_integrator,
        n_v_points=cfg.n_v_integration_points,
        gauss_legendre_order=cfg.gauss_legendre_order,
        max_points_per_batch=max_points_per_batch,
    )


def select_kde_indices(n_points: int, quantiles: Tuple[float, ...]) -> np.ndarray:
    """
    Convert quantile requests into integer indices on the test grid.
    """
    if n_points <= 0:
        raise ValueError("Number of test points must be positive to select KDE indices.")
    q = np.asarray(quantiles, dtype=float)
    q = np.clip(q, 0.0, 1.0)
    idx = np.round(q * (n_points - 1)).astype(int)
    idx = np.clip(idx, 0, n_points - 1)
    ordered: list[int] = []
    seen = set()
    for val in idx:
        if int(val) not in seen:
            ordered.append(int(val))
            seen.add(int(val))
    if not ordered:
        ordered = [int((n_points - 1) // 2)]
    return np.asarray(ordered, dtype=int)

# =========================================================
# 2) Integrated Structural Function on Test Grid: μ_c(x)
# =========================================================
def compute_mu_c_on_grid(m_model: StructuralFunctionModel,
                         x_grid: np.ndarray,
                         n_v_points: int,
                         max_points_per_batch: int | None = None,
                         *,
                         integrator_cfg: MuIntegratorConfig | None = None,
                         verbose: bool = True) -> np.ndarray:
    """
    Compute μ_c(x) = ∫₀¹ m̂(x,v) dv using numerical integration.
    
    This integrates the structural function over the V distribution using
    the configured quadrature rule, giving E[Y|do(X=x)] at each test point x.
    
    Args:
        m_model: Fitted structural function model m̂(x,v)
        x_grid: (k,) array of NEW test x values
        n_v_points: Number of V integration points
        
    Returns:
        mu_c_grid: (k,) array of μ_c(x_j) for each test point
    """
    x_arr = np.asarray(x_grid, dtype=float).reshape(-1)
    k = len(x_arr)
    resolved_integrator = integrator_cfg or MuIntegratorConfig(
        n_v_points=int(n_v_points),
        max_points_per_batch=max_points_per_batch,
    )
    v_grid, _v_weights = resolve_v_rule(resolved_integrator)
    n_v = len(v_grid)

    if verbose:
        print(
            "Computing μ_c(x) on test grid: "
            f"{k} x values × {n_v} V points via {resolved_integrator.method} integration "
            "(vectorized backend inference)..."
        )

    mu_c_grid = integrate_mean_function_over_v(
        m_model.predict,
        x_arr,
        integrator_cfg=resolved_integrator,
    )

    if verbose:
        print(f"✅ μ_c computed on test grid: mean={np.mean(mu_c_grid):.4f}, std={np.std(mu_c_grid):.4f}")
    return mu_c_grid


def run_stage2_9_mean_only_experiment(
    cfg: Stage2_9Config,
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run the mean-only Stage 2 path without any diagnostics or file outputs."""
    required_train_cols = ["X", "Y", "V_hat"]
    missing_train = [col for col in required_train_cols if col not in train_df.columns]
    if missing_train:
        raise ValueError(
            f"Stage2 mean-only train frame missing columns {missing_train}; "
            f"available columns: {list(train_df.columns)}"
        )
    if "X" not in test_df.columns:
        raise ValueError(f"Stage2 mean-only test frame missing column 'X'; available columns: {list(test_df.columns)}")

    set_seed(cfg.random_state)

    x_test = np.asarray(test_df["X"], dtype=float)
    sort_idx = np.argsort(x_test)
    x_test_grid = x_test[sort_idx]

    if verbose:
        print("Starting IV Stage 2.9 mean-only experiment...", flush=True)
        print(f"Configuration: {asdict(cfg)}", flush=True)
        print(
            f"Mean-only evaluation on {len(train_df)} train samples and {len(x_test_grid)} test points",
            flush=True,
        )

    m_model = FullDataStructuralFunctionModel(
        use_tabpfn=cfg.use_tabpfn,
        backend_name=cfg.backend_name,
        model_path=cfg.model_path,
        random_state=cfg.random_state,
        local_context=cfg.local_context,
        softmax_temperature=cfg.softmax_temperature,
    )
    _ = m_model.fit_full(
        np.asarray(train_df["X"], dtype=float),
        np.asarray(train_df["V_hat"], dtype=float),
        np.asarray(train_df["Y"], dtype=float),
        verbose=verbose,
    )
    mu_c_sorted = compute_mu_c_on_grid(
        m_model,
        x_test_grid,
        cfg.n_v_integration_points,
        integrator_cfg=stage2_mu_integrator_config(cfg),
        verbose=verbose,
    )

    return {
        "config": asdict(cfg),
        "predictions": pd.DataFrame({"X": x_test_grid, "Y_do_pred": mu_c_sorted}),
        "metadata": {
            "backend": normalize_backend_name(cfg.backend_name, use_tabpfn=cfg.use_tabpfn),
            "n_train_samples": int(len(train_df)),
            "n_test_samples": int(len(test_df)),
            **local_context_metadata(
                cfg.local_context,
                n_train_samples=int(len(train_df)),
                resolved_k=getattr(m_model.model, "resolved_k_", None),
            ),
        },
    }

# =========================================================
# 3) Conditional CDF Estimation: F(y|X,V)
# =========================================================


def cdf_from_full_output(
    full_output: Dict[str, object],
    y_values: np.ndarray,
    *,
    squeeze_last: bool = True,
) -> np.ndarray:
    """Evaluate the conditional CDF using the backend-specific predictive distribution."""
    return cdf_from_distribution_output(full_output, y_values, squeeze_last=squeeze_last)


def pdf_from_full_output(
    full_output: Dict[str, object],
    y_values: np.ndarray,
    *,
    squeeze_last: bool = True,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Evaluate the conditional PDF using the backend-specific predictive distribution."""
    del epsilon
    return pdf_from_distribution_output(full_output, y_values, squeeze_last=squeeze_last)


class ConditionalCDFEstimator:
    """
    Estimate conditional CDF F_{Y|X,V}(y | x, v) using backend predictive distributions.
    """

    def __init__(
        self,
        use_tabpfn: bool = True,
        *,
        backend_name: Optional[str] = None,
        model_path: str = "auto",
        random_state: int = 1,
        local_context: object | None = None,
        softmax_temperature: float | None = None,
    ):
        self.backend_name = normalize_backend_name(backend_name, use_tabpfn=use_tabpfn)
        self.use_tabpfn = self.backend_name in {TABPFN_BACKEND, "tabpfn_real"}
        raw_model_path = "" if model_path is None else str(model_path).strip()
        self.model_path = "auto" if not raw_model_path else raw_model_path
        self.random_state = int(random_state)
        self.local_context = ensure_local_context_config(local_context)
        self.softmax_temperature = None if softmax_temperature is None else float(softmax_temperature)
        self.model = None
        self._using_tabpfn = self.use_tabpfn
        self.distribution_ = None

    def _new_regressor(self):
        """Create new regressor instance."""
        self._using_tabpfn = self.use_tabpfn
        return make_regressor_backend(
            self.backend_name,
            random_state=self.random_state,
            model_path=self.model_path,
            local_context=self.local_context,
            softmax_temperature=self.softmax_temperature,
        )

    def _predict_full_output(self, features: np.ndarray) -> Dict[str, object]:
        """Helper to grab the backend-specific predictive distribution output."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit_full() first.")

        full_output = predict_distribution(self.model, features, backend_name=self.backend_name)
        self.distribution_ = full_output.get("distribution", full_output)
        return full_output

    def fit_full(self, X: np.ndarray, V: np.ndarray, Y: np.ndarray, *, verbose: bool = True) -> None:
        """Fit the conditional CDF model on the full dataset (no cross-validation)."""
        X = np.asarray(X).reshape(-1, 1)
        V = np.asarray(V).reshape(-1, 1)
        Y = np.asarray(Y)
        features = np.hstack([X, V])

        model_type = self.backend_name
        if verbose:
            print(f"Training full-data CDF model with {model_type} (no CV)...", flush=True)

        reg = self._new_regressor()
        reg.fit(features, Y)
        self.model = reg

        if verbose:
            print(f"✅ Backend '{self.backend_name}' CDF model fitted on full dataset", flush=True)
        return None

    def predict_full_distribution(self, x: np.ndarray, v: np.ndarray) -> Dict[str, object]:
        """Return backend-specific predictive distribution output for provided (x, v) pairs."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit_full() first.")

        x = np.asarray(x, dtype=float).reshape(-1, 1)
        v = np.asarray(v, dtype=float).reshape(-1, 1)
        features = np.hstack([x, v])
        return self._predict_full_output(features)

    def predict_cdf(self, x: np.ndarray, v: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Predict F(y | x, v) for given (x, v, y) triples via predictive-distribution evaluation.

        Args:
            x: (n,) x values
            v: (n,) v values
            y: Array of values to evaluate the CDF at. Supports shapes:
               - (n,) matching each (x_i, v_i)
               - (m,) common evaluation grid shared across samples
               - (n, m) sample-specific evaluation grids

        Returns:
            Array of CDF evaluations aligned with the shape of `y`.
        """
        y = np.asarray(y)
        full_output = self.predict_full_distribution(x, v)
        return cdf_from_full_output(full_output, y, squeeze_last=True)




# =========================================================
# 4) Interventional CDF: F(y|do(X=x))
# =========================================================
def compute_interventional_cdf(cdf_model: ConditionalCDFEstimator,
                               x_grid: np.ndarray,
                               y_grid: np.ndarray,
                               n_v_points: int,
                               *,
                               integrator_cfg: MuIntegratorConfig | None = None) -> Dict[str, np.ndarray]:
    """
    Compute interventional CDF F(y|do(X=x)) by numerical integration over V.
    
    For each (x, y) pair on the grid:
    F(y|do(X=x)) = ∫₀¹ F(y|X=x,v) dv
    
    Args:
        cdf_model: Fitted conditional CDF model F(y|X,V)
        x_grid: (k_x,) array of x values to evaluate
        y_grid: (k_y,) array of y values to evaluate
        n_v_points: Number of V integration points
        
    Returns:
        Dictionary with:
            - x_grid: (k_x,) x values
            - y_grid: (k_y,) y values
            - F_interventional: (k_x, k_y) matrix of CDF values
    """
    k_x = len(x_grid)
    k_y = len(y_grid)
    resolved_integrator = integrator_cfg or MuIntegratorConfig(n_v_points=int(n_v_points))
    v_grid, v_weights = resolve_v_rule(resolved_integrator)
    
    F_interventional = np.zeros((k_x, k_y), dtype=float)
    
    print(f"\nComputing interventional CDF F(y|do(X=x)) on {k_x} × {k_y} grid...")
    print(f"  X grid: {k_x} points from {x_grid.min():.3f} to {x_grid.max():.3f}")
    print(f"  Y grid: {k_y} points from {y_grid.min():.3f} to {y_grid.max():.3f}")
    print(f"  Numerical integration over {len(v_grid)} V points via {resolved_integrator.method}...")
    
    # For each x on the grid reuse the predictive distribution across all y values
    for i, x0 in enumerate(x_grid):
        print(f"  Processing x = {x0:.3f} ({i+1}/{k_x})...")
        x_vec = np.full(len(v_grid), x0, dtype=float)
        full_output = cdf_model.predict_full_distribution(x_vec, v_grid)
        
        # Evaluate the full CDF grid from the backend predictive distribution
        cdf_vals = np.asarray(
            cdf_from_full_output(full_output, y_grid, squeeze_last=False),
            dtype=float,
        )
        if cdf_vals.ndim == 1:
            cdf_vals = cdf_vals.reshape(len(v_grid), 1)
        integrated = integrate_values(
            cdf_vals,
            v_grid,
            integrator_cfg=resolved_integrator,
            axis=0,
            v_weights=v_weights,
        )
        F_interventional[i, :] = integrated
    
    print(f"✅ Interventional CDF computed")
    print(f"   Range: [{F_interventional.min():.4f}, {F_interventional.max():.4f}]")
    
    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "F_interventional": F_interventional
    }


def compute_interventional_pdf(cdf_model: ConditionalCDFEstimator,
                               x_grid: np.ndarray,
                               y_grid: np.ndarray,
                               n_v_points: int,
                               *,
                               integrator_cfg: MuIntegratorConfig | None = None) -> Dict[str, np.ndarray]:
    """
    Compute interventional density f(y|do(X=x)) by integrating backend PDFs.
    """
    k_x = len(x_grid)
    k_y = len(y_grid)
    resolved_integrator = integrator_cfg or MuIntegratorConfig(n_v_points=int(n_v_points))
    v_grid, v_weights = resolve_v_rule(resolved_integrator)

    pdf_interventional = np.zeros((k_x, k_y), dtype=float)

    print(f"\nComputing interventional PDF f(y|do(X=x)) on {k_x} × {k_y} grid...")
    print(f"  Numerical integration over {len(v_grid)} V points via {resolved_integrator.method}...")

    for i, x0 in enumerate(x_grid):
        print(f"  Processing x = {x0:.3f} ({i+1}/{k_x}) for PDF...")
        x_vec = np.full(len(v_grid), x0, dtype=float)
        full_output = cdf_model.predict_full_distribution(x_vec, v_grid)
        pdf_vals = np.asarray(
            pdf_from_full_output(full_output, y_grid, squeeze_last=False),
            dtype=float,
        )

        if pdf_vals.ndim == 1:
            pdf_vals = pdf_vals.reshape(len(v_grid), 1)
        integrated = integrate_values(
            pdf_vals,
            v_grid,
            integrator_cfg=resolved_integrator,
            axis=0,
            v_weights=v_weights,
        )

        pdf_interventional[i, :] = np.maximum(integrated, 0.0)

    print("✅ Interventional PDF computed")
    rowsums = np.trapz(pdf_interventional, x=y_grid, axis=1)
    print(f"   Normalization check (∫ f(y) dy): min={rowsums.min():.4f}, max={rowsums.max():.4f}")

    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "pdf": pdf_interventional
    }

def sample_eps_marginal(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample ε from its marginal distribution.

    Under the joint normal construction used in the DGP, ε ~ N(0, 1).
    We keep this helper in case future DGP variants change the marginal.
    """
    return rng.standard_normal(size=n_samples)


def simulate_y_given_x_eps(cfg: DGPConfig,
                           x_value: float,
                           eps_draws: np.ndarray,
                           *,
                           h_draws: np.ndarray | None = None,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Evaluate Y given X=x and sampled ε draws.

    CRITICAL: eps_draws must come from marginal N(0,1), not ε|V.
    """
    eps_arr = np.asarray(eps_draws, dtype=float)
    x_arr = np.full_like(eps_arr, float(x_value), dtype=float)
    rng = np.random.default_rng() if rng is None else rng

    if cfg.second_stage == "B1":
        if h_draws is None:
            raise ValueError("B1 simulation now requires latent H draws.")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B1.")
        sigma_y = sigmaY_of_X(x_arr, cfg)
        m1 = cfg.beta1 * x_arr + cfg.beta2 * (x_arr ** 2)
        return m1 + sigma_y * (LATENT_H_WEIGHT * h_arr + eps_arr)
    elif cfg.second_stage == "B2":
        if h_draws is None:
            raise ValueError("B2 simulation now requires latent H draws.")
        # B2 uses bimodal mixture - generate Y using marginal ε and shared latent H
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B2.")
        n = len(eps_arr)
        mixture_indicators = rng.binomial(1, cfg.b2_mixture_weight, size=n)

        mu1 = np.sin(x_arr) + 0.3 * x_arr * h_arr
        mu2 = np.sin(x_arr + cfg.b2_beta_offset) + cfg.b2_peak_separation + 0.3 * x_arr * h_arr

        sigma1 = cfg.b2_sigma1 * (1.0 + 0.2 * np.abs(x_arr))
        sigma2 = cfg.b2_sigma2 * (1.0 + 0.3 * np.abs(x_arr))

        y1 = mu1 + sigma1 * rng.standard_normal(n)
        y2 = mu2 + sigma2 * rng.standard_normal(n)

        return mixture_indicators * y1 + (1 - mixture_indicators) * y2
    elif cfg.second_stage == "B3":
        if h_draws is None:
            raise ValueError("B3 simulation requires latent H draws.")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B3.")
        return x_arr - 3.0 * h_arr + eps_arr
    elif cfg.second_stage == "B4":
        if h_draws is None:
            raise ValueError("B4 simulation requires latent H draws.")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B4.")
        linear_branch = 0.2 * (5.5 + 2.0 * x_arr + 3.0 * h_arr + eps_arr)
        softplus_arg = (2.0 * x_arr + h_arr) ** 2 + eps_arr ** 2
        safe_arg = np.maximum(softplus_arg, B4_SOFTPLUS_EPS)
        softplus_branch = np.log(safe_arg)
        return np.where(x_arr <= 1.0, linear_branch, softplus_branch)
    elif cfg.second_stage == "B5":
        if h_draws is None:
            raise ValueError("B5 simulation requires latent H draws.")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B5.")
        return 3.0 * np.sin(2.0 * x_arr) + 2.0 * x_arr - 3.0 * h_arr + eps_arr
    elif cfg.second_stage == "B6":
        if h_draws is None:
            raise ValueError("B6 simulation requires latent H draws.")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B6.")
        return 1.0 + 2.0 * np.cos(2.0 * x_arr + h_arr) + x_arr * h_arr
    elif cfg.second_stage == "B7":
        if h_draws is None:
            raise ValueError("B7 simulation requires latent H draws.")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B7.")

        u = np.arctan(x_arr / float(cfg.b7_x_scale))
        h_t = np.tanh(h_arr)
        return (
            cfg.b7_beta1 * u
            + cfg.b7_beta2 * np.sin(3.0 * u)
            + cfg.b7_h_weight * h_t * np.cos(2.0 * u)
            + cfg.b7_eps_scale * eps_arr
        )
    elif cfg.second_stage == "B8":
        if h_draws is None:
            raise ValueError("B8 simulation requires latent H draws.")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B8.")
        linear_branch = 1.0 + x_arr + 2.0 * h_arr + eps_arr
        softplus_arg = 2.0 * (x_arr + h_arr) ** 2 + eps_arr ** 2
        safe_arg = np.maximum(softplus_arg, B4_SOFTPLUS_EPS)
        softplus_branch = np.log(safe_arg)
        return np.where(x_arr <= 1.0, linear_branch, softplus_branch)
    elif cfg.second_stage == "B9":
        if h_draws is None:
            raise ValueError("B9 simulation requires latent H draws.")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B9.")
        return (
            3.0 * np.sin(2.0 * x_arr)
            + np.cos(0.5 * x_arr)
            + 2.0 * x_arr
            - 3.0 * h_arr
            + eps_arr
        )
    elif cfg.second_stage == "B10":
        if h_draws is None:
            raise ValueError("B10 simulation requires latent H draws.")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B10.")
        return b10_response(cfg, x_arr, h_arr, eps_arr)
    elif cfg.second_stage == "B11":
        if h_draws is None:
            raise ValueError("B11 simulation requires latent H draws.")
        h_arr = np.asarray(h_draws, dtype=float)
        if h_arr.shape != eps_arr.shape:
            raise ValueError("Shape mismatch between eps_draws and h_draws for B11.")
        return 1.0 + 2.0 * x_arr + np.cos(2.0 * x_arr) + x_arr * h_arr - h_arr + eps_arr
    else:
        raise ValueError(f"Unknown second_stage: {cfg.second_stage}")


def monte_carlo_y_given_x(cfg: DGPConfig,
                          x_value: float,
                          n_samples: int,
                          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo sampler for Y|do(X=x):
      1. Draw ε from its marginal distribution (N(0,1) under current DGP)
      2. Evaluate Y with the DGP second-stage equation.

    Previous versions sampled ε | η; that path is retained in comments for reference.
    """
    needs_h = cfg.second_stage in {"B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"}
    h_samples = rng.standard_normal(size=n_samples) if needs_h else None
    eps_samples = sample_eps_marginal(n_samples, rng)
    y_samples = simulate_y_given_x_eps(cfg, x_value, eps_samples, h_draws=h_samples, rng=rng)
    return y_samples, eps_samples


def sample_backend_y_given_x(cdf_model: ConditionalCDFEstimator,
                             x_value: float,
                             n_samples: int,
                             y_support: np.ndarray,
                             rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw samples from the backend-estimated interventional distribution Y|do(X=x).

    We rely on inverse transform sampling using the predictive-distribution CDF evaluated
    on a shared y-support grid to stay aligned with the KDE diagnostics.
    """
    y_support = np.asarray(y_support, dtype=float)
    if y_support.ndim != 1:
        raise ValueError("y_support must be a 1-D array for inverse transform sampling.")

    v_samples = rng.uniform(low=V_EPSILON, high=1.0 - V_EPSILON, size=n_samples)
    u_samples = rng.uniform(low=0.0, high=1.0, size=n_samples)
    x_vec = np.full(n_samples, float(x_value), dtype=float)

    full_output = cdf_model.predict_full_distribution(x_vec, v_samples)
    cdf_vals = np.asarray(
        cdf_from_full_output(full_output, y_support, squeeze_last=False),
        dtype=float,
    )
    if cdf_vals.ndim == 1:
        cdf_vals = cdf_vals.reshape(1, -1)

    y_samples = np.empty(n_samples, dtype=float)
    for idx in range(n_samples):
        row = np.asarray(cdf_vals[idx], dtype=float)
        row = np.clip(row, 0.0, 1.0)
        row = np.maximum.accumulate(row)
        row[-1] = 1.0
        row[0] = max(row[0], 0.0)
        y_samples[idx] = np.interp(u_samples[idx], row, y_support)

    return y_samples, v_samples, u_samples


def sample_tabpfn_y_given_x(cdf_model: ConditionalCDFEstimator,
                            x_value: float,
                            n_samples: int,
                            y_support: np.ndarray,
                            rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible alias for legacy callers."""
    return sample_backend_y_given_x(cdf_model, x_value, n_samples, y_support, rng)


def compute_y_clean(cfg: DGPConfig,
                    x: np.ndarray,
                    n_samples: int = 100,
                    rng: np.random.Generator | None = None,
                    return_samples: bool = False):
    """
    Monte Carlo estimate of the structural expectation y_clean(x) = E_ε[m(x, ε)].

    For each x_j we draw ε_{j1}, ..., ε_{jM} ~ p(ε) (default M=100) and evaluate the
    structural model to obtain samples y_{jk} = m(x_j, ε_{jk}). We approximate
    E_ε[m(x_j, ε)] with the sample mean (1/M) sum_{k=1}^M y_{jk}. The same draws are
    later reused for diagnostics, so that per-point squared errors accumulate as
    sum_{k=1}^M (y_{jk} - y_pred_j)^2 and the global MSE diagnostic becomes
    sum_{j=1}^{n_test} sum_{k=1}^M (y_{jk} - y_pred_j)^2.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive when estimating y_clean.")

    rng = np.random.default_rng() if rng is None else rng
    x_raw = np.asarray(x, dtype=float)
    was_scalar = x_raw.ndim == 0
    x_arr = np.atleast_1d(x_raw).astype(float)

    n_x = x_arr.size
    samples = np.empty((n_x, n_samples), dtype=float)

    for idx, x_val in enumerate(x_arr):
        y_draws, _ = monte_carlo_y_given_x(cfg, float(x_val), n_samples, rng)
        samples[idx, :] = y_draws

    means = samples.mean(axis=1)

    if was_scalar:
        means_out = float(means[0])
        samples_out = samples[0]
    else:
        means_out = means
        samples_out = samples

    if return_samples:
        return means_out, samples_out
    return means_out


def create_kernel_density_plot(y_grid: np.ndarray,
                               pdf_estimated: np.ndarray,
                               pdf_true: Optional[np.ndarray],
                               x_values: np.ndarray,
                               y_observed: Optional[np.ndarray],
                               quantile_levels: Optional[np.ndarray],
                               output_path: str) -> None:
    """Render Figure-3-style KDE panels comparing estimated vs. true densities."""
    import matplotlib.pyplot as plt

    n_panels = len(x_values)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8.5, 8.5), sharex=True)
    if n_panels == 1:
        axes = [axes]

    fill_est_color = (231/255, 186/255, 82/255, 0.75)  # soft amber with alpha
    fill_true_color = (66/255, 133/255, 244/255, 0.35)  # muted blue with alpha
    line_est_color = (196/255, 139/255, 30/255)
    line_true_color = (38/255, 90/255, 136/255)

    for idx, ax in enumerate(axes):
        est_curve = pdf_estimated[idx]
        true_curve = pdf_true[idx] if pdf_true is not None else None

        ax.plot(y_grid, est_curve, color=line_est_color, linewidth=1.8)
        ax.fill_between(y_grid, 0.0, est_curve, color=fill_est_color, label="Estimated" if idx == 0 else None)

        if true_curve is not None:
            ax.plot(y_grid, true_curve, color=line_true_color, linewidth=1.5)
            ax.fill_between(y_grid, 0.0, true_curve, color=fill_true_color, label="True" if idx == 0 else None)

        if y_observed is not None:
            ax.axvline(y_observed[idx], color=line_true_color, linestyle=":", linewidth=1.0, alpha=0.7)

        quantile_text = ""
        if quantile_levels is not None:
            quantile_text = f" (q={quantile_levels[idx]:.2f})"

        ax.set_ylabel(r"$P_Y^{do(X=x)}$")
        ax.text(0.98, 0.85, f"x = {x_values[idx]:.2f}{quantile_text}", transform=ax.transAxes,
                ha="right", va="center", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    axes[-1].set_xlabel("Y")

    handles = []
    labels = []
    if pdf_estimated is not None:
        handles.append(plt.Line2D([0], [0], color=line_est_color, linewidth=1.8))
        labels.append("Estimated")
    if pdf_true is not None:
        handles.append(plt.Line2D([0], [0], color=line_true_color, linewidth=1.5))
        labels.append("True")
    if handles:
        axes[0].legend(handles, labels, loc="upper right", frameon=False)

    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.98))
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)



# =========================================================
# 6) Main execution pipeline
# =========================================================
def run_stage2_9_experiment(cfg: Stage2_9Config) -> Dict[str, object]:
    """Execute Stage 2.9 pipeline with numerical integration optimization.

    Steps:
      1. Load Stage 1 outputs
      2. Fit structural model m(x, v) on full data
      3. Integrate out V to obtain μ_c(x) on grids and at each test observation
      4. Fit conditional CDF models on full data
      5. Compute interventional CDF/PDF diagnostics (estimated vs. analytical ground truth)
    """
    import sys

    print("Starting IV Stage 2.9 Experiment (Numerical Integration Optimization)...", flush=True)
    print(f"Configuration: {asdict(cfg)}", flush=True)
    sys.stdout.flush()

    set_seed(cfg.random_state)
    rng = np.random.default_rng(cfg.random_state)

    components = prepare_stage2_components(cfg)
    train_data = components["train_data"]
    test_data = components["test_data"]
    dgp_cfg = components["dgp_config"]
    m_model = components["m_model"]
    cdf_model = components["cdf_model"]
    X_test = test_data["X"]
    Y_test = test_data["Y"]
    sort_idx = np.argsort(X_test)
    x_test_grid = X_test[sort_idx]
    y_test_grid = Y_test[sort_idx]
    n_train = len(train_data["X"])
    n_test_selected = len(X_test)
    n_test_raw = int(components["n_test_samples_raw"])

    print(
        "\n[1/5] Components prepared: "
        f"train={n_train} samples, test={n_test_selected}/{n_test_raw} selected/raw samples",
        flush=True,
    )

    print("\n[2/5] Aligning evaluation grid with held-out test samples...", flush=True)
    k_x = len(x_test_grid)
    x_min, x_max = float(np.min(x_test_grid)), float(np.max(x_test_grid))
    y_min, y_max = float(np.min(Y_test)), float(np.max(Y_test))
    y_grid = np.linspace(y_min, y_max, cfg.n_y_grid)
    print(f"  X grid: {k_x} test points (full test set) from {x_min:.3f} to {x_max:.3f}", flush=True)
    print(f"  Y grid: {cfg.n_y_grid} points from {y_min:.3f} to {y_max:.3f}", flush=True)

    mu_c_sorted = compute_mu_c_on_grid(
        m_model,
        x_test_grid,
        cfg.n_v_integration_points,
        integrator_cfg=stage2_mu_integrator_config(cfg),
    )
    mu_c_all_test = np.empty_like(X_test, dtype=float)
    mu_c_all_test[sort_idx] = mu_c_sorted

    print("\n[3/5] Computing ground-truth expectations for full test set...", flush=True)
    Y_clean_full = compute_y_clean(
        dgp_cfg,
        X_test,
        n_samples=cfg.y_clean_mc_samples,
        rng=rng,
        return_samples=False,
    )
    Y_clean_sorted = Y_clean_full[sort_idx]

    print("\n[4/5] Preparing outputs...", flush=True)

    kde_indices = select_kde_indices(len(x_test_grid), cfg.kde_quantiles)
    kde_x_values = x_test_grid[kde_indices]
    kde_y_observed = y_test_grid[kde_indices]

    kde_y_samples_true: list[np.ndarray] = []
    kde_eps_samples_true: list[np.ndarray] = []
    kde_y_samples_est: list[np.ndarray] = []
    kde_v_samples_est: list[np.ndarray] = []
    kde_u_samples_est: list[np.ndarray] = []
    kde_pdf_true_rows: list[np.ndarray] = []
    kde_pdf_est_rows: list[np.ndarray] = []

    for x0 in kde_x_values:
        y_draws_true, eps_draws_true = monte_carlo_y_given_x(
            dgp_cfg,
            float(x0),
            cfg.kde_sample_size,
            rng,
        )
        y_draws_est, v_draws_est, u_draws_est = sample_backend_y_given_x(
            cdf_model,
            float(x0),
            cfg.kde_sample_size,
            y_grid,
            rng,
        )

        kde_y_samples_true.append(y_draws_true)
        kde_eps_samples_true.append(eps_draws_true)
        kde_y_samples_est.append(y_draws_est)
        kde_v_samples_est.append(v_draws_est)
        kde_u_samples_est.append(u_draws_est)

        try:
            density_true = gaussian_kde(y_draws_true)
            pdf_true_row = density_true(y_grid)
        except np.linalg.LinAlgError:
            jitter = 1e-6 * rng.standard_normal(size=y_draws_true.shape)
            density_true = gaussian_kde(y_draws_true + jitter)
            pdf_true_row = density_true(y_grid)

        try:
            density_est = gaussian_kde(y_draws_est)
            pdf_est_row = density_est(y_grid)
        except np.linalg.LinAlgError:
            jitter_est = 1e-6 * rng.standard_normal(size=y_draws_est.shape)
            density_est = gaussian_kde(y_draws_est + jitter_est)
            pdf_est_row = density_est(y_grid)

        kde_pdf_true_rows.append(np.maximum(pdf_true_row, 0.0))
        kde_pdf_est_rows.append(np.maximum(pdf_est_row, 0.0))

    kde_pdf_true = np.vstack(kde_pdf_true_rows)
    kde_pdf_est = np.vstack(kde_pdf_est_rows)

    iae_per_x = np.trapz(np.abs(kde_pdf_est - kde_pdf_true), x=y_grid, axis=1)
    metrics = {
        "iae_mean": float(np.mean(iae_per_x)),
        "iae_max": float(np.max(iae_per_x)),
        "iae_per_x": iae_per_x,
        "y_clean_mc_samples": cfg.y_clean_mc_samples,
    }

    # Reuse the Monte Carlo draws: compare the sample mean Ê[m(x, ε)] with μ̂_c(x).
    # Only the averaged quantities enter the MSE, matching the structural comparison request.
    squared_error_mean_per_x = (Y_clean_full - mu_c_all_test) ** 2
    metrics["mse_do_pred_vs_clean"] = float(np.mean(squared_error_mean_per_x))
    metrics["mse_mean_per_x"] = squared_error_mean_per_x

    mse_sorted = (Y_clean_sorted - mu_c_sorted) ** 2
    # Map each sorted X to its empirical quantile strictly inside (0, 1)
    quantile_positions = (np.arange(k_x, dtype=float) + 1.0) / (k_x + 1.0)
    predictions_df = pd.DataFrame({
        "X": x_test_grid,
        "Y_clean_mean": Y_clean_sorted,
        "Y_do_pred": mu_c_sorted,
        "mse_per_x": mse_sorted,
        "X_quantile": quantile_positions,
    })

    return {
        "config": asdict(cfg),
        "data_stats": {
            "n_train_samples": n_train,
            "n_test_samples_raw": n_test_raw,
            "n_test_samples_selected": k_x,
            "n_y_grid": cfg.n_y_grid,
            "n_v_integration_points": cfg.n_v_integration_points,
            "mu_integrator": cfg.mu_integrator,
            "gauss_legendre_order": cfg.gauss_legendre_order,
            "test_x_trim_quantile_range": (
                "disabled"
                if cfg.test_x_trim_quantile_range is None
                else f"{cfg.test_x_trim_quantile_range[0]:.2f},{cfg.test_x_trim_quantile_range[1]:.2f}"
            ),
        },
        "mu_c": {
            "x_test_grid": x_test_grid,
            "y_test_grid": y_test_grid,
            "estimated": mu_c_sorted,
        },
        "predictions": predictions_df,
        "kde": {
            "indices": kde_indices,
            "x_values": kde_x_values,
            "y_observed": kde_y_observed,
            "pdf_estimated": kde_pdf_est,
            "pdf_true": kde_pdf_true,
            "y_grid": y_grid,
            "quantiles": np.asarray(cfg.kde_quantiles, dtype=float),
            "y_samples_true": np.asarray(kde_y_samples_true),
            "eps_samples_true": np.asarray(kde_eps_samples_true),
            "y_samples_est": np.asarray(kde_y_samples_est),
            "v_samples_est": np.asarray(kde_v_samples_est),
            "u_samples_est": np.asarray(kde_u_samples_est),
        },
        "metrics": metrics,
        "metadata": {
            "train_csv": components["train_csv"],
            "test_csv": components["test_csv"],
            "codes": components["codes"],
            "n_train_samples": components["n_train_samples"],
            **backend_metadata(cfg.backend_name, cfg.model_path),
            **local_context_metadata(
                cfg.local_context,
                n_train_samples=int(components["n_train_samples"]),
                resolved_k=getattr(m_model.model, "resolved_k_", None),
            ),
        }
    }




# =========================================================
# 7) Save results
# =========================================================
def save_stage2_9_results(
    results: Dict[str, object],
    output_dir: str | os.PathLike[str],
    *,
    use_timestamp: bool = False,
) -> Dict[str, Path]:
    """Persist Stage 2.9 outputs as CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_meta = results.get("config", {})
    codes = results["metadata"]["codes"]
    train_seed = config_meta.get("train_seed")
    backend_name = normalize_backend_name(config_meta.get("backend_name", TABPFN_BACKEND))
    local_context = ensure_local_context_config(config_meta.get("local_context"))
    n_train_samples_meta = results["metadata"].get("n_train_samples")
    if n_train_samples_meta is None:
        n_train_samples_meta = results["data_stats"].get("n_train_samples")
    resolved_local_k = results["metadata"].get("local_k_neighbors_resolved")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else None

    base_prefix = stage2_base_prefix(
        codes,
        train_sample_size=int(n_train_samples_meta) if n_train_samples_meta is not None else None,
        seed=int(train_seed) if train_seed is not None else None,
        backend_name=backend_name,
        local_context=local_context,
        resolved_local_k=int(resolved_local_k) if resolved_local_k is not None else None,
    )

    def _suffix(name: str) -> str:
        return f"{name}_{timestamp}" if timestamp else name

    artifact_paths: Dict[str, Path] = {}

    # Predictions CSV
    predictions_df = results["predictions"].copy()
    predictions_df = predictions_df.sort_values("X").reset_index(drop=True)
    predictions_csv = output_path / f"{_suffix(base_prefix + '_predictions')}.csv"
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"✅ Predictions saved to: {predictions_csv}")
    artifact_paths["predictions"] = predictions_csv

    # Summary CSV
    summary_rows = []
    for key, value in results["data_stats"].items():
        summary_rows.append({"key": key, "value": value})
    summary_rows.append({"key": "train_csv", "value": results["metadata"]["train_csv"]})
    summary_rows.append({"key": "test_csv", "value": results["metadata"]["test_csv"]})
    summary_rows.append({"key": "backend", "value": results["metadata"]["backend"]})
    summary_rows.append({"key": "backend_package", "value": results["metadata"]["backend_package"]})
    summary_rows.append({"key": "checkpoint_version", "value": results["metadata"]["checkpoint_version"]})
    summary_rows.append({"key": "model_path", "value": results["metadata"]["model_path"]})
    summary_rows.append({"key": "local_strategy", "value": results["metadata"].get("local_strategy", "global")})
    summary_rows.append(
        {
            "key": "local_k_neighbors_requested",
            "value": results["metadata"].get("local_k_neighbors_requested"),
        }
    )
    summary_rows.append(
        {
            "key": "local_k_neighbors_resolved",
            "value": results["metadata"].get("local_k_neighbors_resolved"),
        }
    )
    summary_rows.append(
        {
            "key": "local_retrieval_features",
            "value": results["metadata"].get("local_retrieval_features", "none"),
        }
    )
    summary_rows.append({"key": "local_metric", "value": results["metadata"].get("local_metric", "euclidean")})
    summary_rows.append(
        {
            "key": "local_scale_features",
            "value": results["metadata"].get("local_scale_features", True),
        }
    )
    summary_rows.append({"key": "timestamp", "value": timestamp})
    summary_rows.append({"key": "prediction_mode", "value": "mu_c_integrated"})
    summary_rows.append({"key": "mu_integrator", "value": config_meta.get("mu_integrator", DEFAULT_MU_INTEGRATOR)})
    summary_rows.append(
        {
            "key": "gauss_legendre_order",
            "value": config_meta.get("gauss_legendre_order", DEFAULT_GAUSS_LEGENDRE_ORDER),
        }
    )

    metrics = results.get("metrics")
    if metrics:
        summary_rows.append({"key": "metric_mse_do_pred_vs_clean", "value": f"{metrics['mse_do_pred_vs_clean']:.6f}"})
        summary_rows.append({"key": "metric_iae_mean", "value": f"{metrics['iae_mean']:.6f}"})
        summary_rows.append({"key": "metric_iae_max", "value": f"{metrics['iae_max']:.6f}"})
        iae_series = ";".join(f"{val:.6f}" for val in metrics['iae_per_x'])
        summary_rows.append({"key": "metric_iae_per_x", "value": iae_series})
        se_series = ";".join(f"{val:.6f}" for val in metrics["mse_mean_per_x"])
        summary_rows.append({"key": "metric_mse_mean_per_x", "value": se_series})
        summary_rows.append({"key": "metric_y_clean_mc_samples", "value": str(metrics["y_clean_mc_samples"])})

    kde = results.get("kde")
    if kde is not None:
        x_summary = ";".join(f"{val:.4f}" for val in kde["x_values"])
        summary_rows.append({"key": "kde_x_values", "value": x_summary})
    summary_csv = output_path / f"{_suffix(base_prefix + '_summary')}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"✅ Summary saved to: {summary_csv}")
    artifact_paths["summary"] = summary_csv

    # KDE Plot
    if kde is not None:
        kde_plot_path = output_path / f"{_suffix(base_prefix + '_kde')}.png"
        create_kernel_density_plot(
            y_grid=kde["y_grid"],
            pdf_estimated=kde["pdf_estimated"],
            pdf_true=kde["pdf_true"],
            x_values=kde["x_values"],
            y_observed=kde["y_observed"],
            quantile_levels=kde.get("quantiles"),
            output_path=str(kde_plot_path),
        )
        print(f"✅ Kernel density plot saved to: {kde_plot_path}")
        artifact_paths["kde_plot"] = kde_plot_path

    print(f"\n📁 Stage 2 outputs stored in: {output_path}/")
    if timestamp:
        print(f"🕒 Timestamp: {timestamp}")

    return artifact_paths




# =========================================================
# 8) Runner
# =========================================================
if __name__ == "__main__":
    """Command-line entry point for Stage 2.9."""

    import sys

    print("=" * 60, flush=True)
    print("IV STAGE 2.9: STARTING EXECUTION", flush=True)
    print("=" * 60, flush=True)

    cfg = Stage2_9Config(
        input_dir=str(DEFAULT_STAGE1_OUTPUT_DIR),
        output_dir=str(DEFAULT_STAGE2_OUTPUT_DIR),
        random_state=1,
        n_y_grid=100,
        n_v_integration_points=100,
        use_tabpfn=True,
        backend_name=TABPFN_BACKEND,
        first_stage_code="A3",
        second_stage_code="B4",
        kde_quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        kde_sample_size=1000,
        y_clean_mc_samples=5000,
        n_train_samples=500,
        train_seed=123,
    )

    print(f"Configuration: {cfg}", flush=True)
    print(f"Resolved backend: {cfg.backend_name}", flush=True)
    print("", flush=True)

    try:
        results = run_stage2_9_experiment(cfg)
        save_stage2_9_results(results, cfg.output_dir, use_timestamp=True)

        print("\n" + "=" * 60)
        print("STAGE 2.9 COMPLETED SUCCESSFULLY", flush=True)
        print("=" * 60)

        print("\nInterventional density diagnostics saved to output directory.")

    except Exception as exc:
        print("\n❌ Stage 2.9 failed with error:", exc, flush=True)
        raise

#2
