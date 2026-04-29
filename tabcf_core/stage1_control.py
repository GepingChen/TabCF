"""
IV Stage 1 Implementation: Pre-generated Data Loading and TabPFN Control Function Estimation (Version 4)
========================================================================================================

This file implements objectives 1 and 2 from the research plan:
1) Load pre-generated triangular DGP data with instrument Z, endogenous regressor X, and outcome Y
2) Estimate control function V = F_{X|Z}(X | Z) using TabPFNRegressor full-distribution output

Key Changes from IV_stage1_3.py:
- Load pre-generated data from IV_datasets/train and IV_datasets/test directories
- Remove dynamic data generation functions and DGPConfig dependency
- Accept first_stage and second_stage parameters to locate correct data files
- Export 7-column training set (Z, X, Y, V_true, eps, eta, V_hat) for Stage 2 consumption
- Save results to IV_datasets/stage1_output with DGP-aware naming

The control function is the conditional CDF of X given Z, estimated via:
- Use TabPFNRegressor to obtain the full conditional distribution of X given Z
- Evaluate criterion-based CDF values F̂(x|z) directly (no interpolation)
- Train on full training set; downstream stages source test data directly from IV_datasets/test
- Error if TabPFN full distribution API is unavailable

Reference: TabPFN_Demo_HPC.ipynb and TabPFN_Demo_Local.ipynb for proper TabPFN usage
"""

from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("TABPFN_MODEL_VERSION", "v2.5")  # Default to latest TabPFN (requires HF token)
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(Path(__file__).resolve().parent.parent.parent / "tabpfn_home_config" / "models"))

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple, Optional
from datetime import datetime

from foundation_backends import (
    TABPFN_BACKEND,
    cdf_from_distribution_output,
    make_regressor_backend,
    normalize_backend_name,
    predict_distribution,
    predict_quantiles as predict_backend_quantiles,
    stage1_output_filename,
)
from dgp_test_utils import (
    DEFAULT_TEST_X_TRIM_QUANTILE_RANGE,
    normalize_test_x_trim_quantile_range,
    trim_test_dataframe_by_x,
)


BATCH_ROOT = Path(__file__).resolve().parents[1] / "interv_mean"
DEFAULT_DATA_DIR = BATCH_ROOT / "IV_datasets"
DEFAULT_STAGE1_OUTPUT_DIR = DEFAULT_DATA_DIR / "stage1_output"


def cdf_from_full_output(
    full_output: Dict[str, object],
    x_values: np.ndarray,
    *,
    squeeze_last: bool = True,
) -> np.ndarray:
    """Evaluate F̂_{X|Z}(x | z) via the backend-specific predictive distribution."""
    return cdf_from_distribution_output(full_output, x_values, squeeze_last=squeeze_last)


# =========================================================
# 2.5) Output helpers
# =========================================================
def save_stage1_results(
    results: Dict[str, pd.DataFrame],
    first_stage: str,
    second_stage: str,
    *,
    train_sample_size: Optional[int] = None,
    seed: Optional[int] = None,
    backend_name: str = TABPFN_BACKEND,
    softmax_temperature: float | None = None,
    output_dir: str | os.PathLike[str] = DEFAULT_STAGE1_OUTPUT_DIR,
    use_timestamp: bool = False,
) -> Dict[str, Path]:
    """Persist Stage 1 outputs to disk with deterministic naming."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    codes = f"{first_stage}_{second_stage}"
    sample_tag = f"_n{train_sample_size}" if train_sample_size is not None else ""
    seed_tag = f"_seed{seed}" if seed is not None else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else None

    saved_paths: Dict[str, Path] = {}
    for subset, df in results.items():
        filename = stage1_output_filename(
            subset,
            codes,
            train_sample_size=train_sample_size,
            seed=seed,
            backend_name=backend_name,
            softmax_temperature=softmax_temperature,
            timestamp=timestamp,
        )
        csv_path = output_path / filename
        df.to_csv(csv_path, index=False)
        print(f"✅ Results saved to: {csv_path}")
        saved_paths[subset] = csv_path

    print(f"📁 Stage 1 outputs written to: {output_path}")
    return saved_paths


# =========================================================
# 0) Configuration Extension
# =========================================================
@dataclass
class Stage1Config:
    """Configuration for Stage 1 Control Function Estimation"""
    # Optional quantile levels to request alongside full TabPFN outputs
    quantiles: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
    random_state: int = 1
    backend_name: str = TABPFN_BACKEND
    model_path: str = "auto"
    softmax_temperature: float | None = None

    def __post_init__(self) -> None:
        self.backend_name = normalize_backend_name(self.backend_name)
        raw_model_path = "" if self.model_path is None else str(self.model_path).strip()
        self.model_path = "auto" if not raw_model_path else raw_model_path
        if self.softmax_temperature is not None:
            self.softmax_temperature = float(self.softmax_temperature)


# =========================================================
# 1) Data Loading Function
# =========================================================
def load_dgp_data(
    first_stage: str,
    second_stage: str,
    *,
    train_sample_size: int | None = None,
    seed: Optional[int] = None,
    base_dir: str | os.PathLike[str] = DEFAULT_DATA_DIR,
    test_x_trim_quantile_range: tuple[float, float] | None = DEFAULT_TEST_X_TRIM_QUANTILE_RANGE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load pre-generated training and test data from IV_datasets directory.
    
    Expected file structure:
        base_dir/train/train_data_{first_stage}_{second_stage}_n{train_sample_size}_seed{seed}.csv
        base_dir/test/test_data_{first_stage}_{second_stage}.csv
    
    Files must contain columns: Z, X, Y, V_true, eps, eta
    
    Args:
        first_stage: First stage DGP type (e.g., "A1", "A2")
        second_stage: Second stage DGP type (e.g., "B1", "B2")
        train_sample_size: If provided, append "_{train_sample_size}" when locating the training file
        seed: Random seed identifier appended to the training filename
        base_dir: Base directory containing train/ and test/ subdirectories
        test_x_trim_quantile_range: Optional [lower, upper] empirical X-rank range
            retained from the test split. None disables trimming.
    
    Returns:
        train_df, test_df: DataFrames with columns [Z, X, Y, V_true, eps, eta]
    
    Raises:
        FileNotFoundError: If matching files don't exist
        ValueError: If required columns are missing
    """
    base_path = Path(base_dir)
    train_dir = base_path / "train"
    test_dir = base_path / "test"
    codes = f"{first_stage}_{second_stage}"

    train_candidates: list[Path] = []
    if train_sample_size is not None and seed is not None:
        train_candidates.append(train_dir / f"train_data_{codes}_n{train_sample_size}_seed{seed}.csv")
    if train_sample_size is not None:
        train_candidates.append(train_dir / f"train_data_{codes}_{train_sample_size}.csv")
    train_candidates.append(train_dir / f"train_data_{codes}.csv")

    train_file: Optional[Path] = next((p for p in train_candidates if p.exists()), None)
    if train_file is None:
        expected = ", ".join(str(p) for p in train_candidates)
        raise FileNotFoundError(
            f"Training data not found for DGP {codes}. Looked for: {expected}"
        )

    test_file = test_dir / f"test_data_{codes}.csv"
    if not test_file.exists():
        raise FileNotFoundError(f"Test data not found: {test_file}")

    normalized_trim_range = normalize_test_x_trim_quantile_range(test_x_trim_quantile_range)

    print(f"Loading training data from: {train_file}")
    print(f"Loading test data from: {test_file}")
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    raw_test_rows = len(test_df)
    test_df = trim_test_dataframe_by_x(test_df, normalized_trim_range)
    
    # Validate columns
    required_cols = ["Z", "X", "Y", "V_true", "eps", "eta"]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Missing column '{col}' in {train_file}")
        if col not in test_df.columns:
            raise ValueError(f"Missing column '{col}' in {test_file}")
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape} (raw={raw_test_rows})")
    if normalized_trim_range is None:
        print("Test X trim: disabled")
    else:
        print(
            "Test X trim range: "
            f"[{normalized_trim_range[0]:.2f}, {normalized_trim_range[1]:.2f}] "
            f"selected {len(test_df)}/{raw_test_rows} rows"
        )
    
    if train_sample_size is not None and len(train_df) != train_sample_size:
        raise ValueError(
            f"Training data in {train_file} has {len(train_df)} rows, expected {train_sample_size}."
        )

    return train_df, test_df


class CondCDFModel:
    """
    Conditional CDF estimator F̂_{X|Z}(x | z) using TabPFNRegressor full distribution output.
    """
    
    def __init__(
        self,
        quantiles: Tuple[float, ...],
        *,
        backend_name: str = TABPFN_BACKEND,
        model_path: str = "auto",
        random_state: int = 1,
        softmax_temperature: float | None = None,
    ):
        self.quantiles = tuple(quantiles)
        self.backend_name = normalize_backend_name(backend_name)
        raw_model_path = "" if model_path is None else str(model_path).strip()
        self.model_path = "auto" if not raw_model_path else raw_model_path
        self.random_state = int(random_state)
        self.softmax_temperature = None if softmax_temperature is None else float(softmax_temperature)
        self.reg: Any | None = None

    def _predict_full_output(self, Z: np.ndarray) -> Dict[str, object]:
        """Return backend-specific predictive distribution output for provided Z values."""
        if self.reg is None:
            raise RuntimeError("Model must be fitted before prediction.")

        features = np.asarray(Z, dtype=float).reshape(-1, 1)
        return predict_distribution(
            self.reg,
            features,
            backend_name=self.backend_name,
            quantiles=self.quantiles,
        )

    def predict_quantiles(self, Z: np.ndarray) -> Dict[float, np.ndarray]:
        """Optional helper to extract requested quantiles alongside the full output."""
        if not self.quantiles:
            raise RuntimeError("Quantile levels were not configured for this estimator.")
        features = np.asarray(Z, dtype=float).reshape(-1, 1)
        quantile_matrix = predict_backend_quantiles(
            self.reg,
            features,
            backend_name=self.backend_name,
            quantiles=self.quantiles,
        )
        if quantile_matrix.shape != (len(features), len(self.quantiles)):
            raise RuntimeError(
                f"Quantile matrix shape mismatch: expected {(len(features), len(self.quantiles))}, "
                f"got {quantile_matrix.shape}."
            )
        return {tau: np.asarray(quantile_matrix[:, idx], dtype=float) for idx, tau in enumerate(self.quantiles)}

    def fit(self, Z_train: np.ndarray, X_train: np.ndarray, *, verbose: bool = True) -> None:
        """
        Train control function model on full training set.
        
        Args:
            Z_train: (n_train,) instrument values for training
            X_train: (n_train,) endogenous regressor values for training
        """
        Z_train = np.asarray(Z_train, dtype=float).reshape(-1, 1)

        if verbose:
            print(f"Training model with backend '{self.backend_name}' (distribution output)...")

        self.reg = make_regressor_backend(
            self.backend_name,
            random_state=self.random_state,
            model_path=self.model_path,
            softmax_temperature=self.softmax_temperature,
        )
        self.reg.fit(Z_train, X_train)

        # Validate that predictive distribution output is available
        sample_size = min(12, len(Z_train))
        _ = self._predict_full_output(Z_train[:sample_size])
        if verbose:
            print(f"  ✅ Backend '{self.backend_name}' predictive distribution verified")

    def predict(self, Z_test: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """
        Predict control function values for test data.
        
        Args:
            Z_test: (n_test,) instrument values
            X_test: (n_test,) endogenous regressor values
        
        Returns:
            V_hat: (n_test,) estimated control function values, uniformly distributed on [0,1]
        """
        if self.reg is None:
            raise RuntimeError("Model must be fitted before prediction.")

        full_output = self._predict_full_output(Z_test)
        V_hat = cdf_from_full_output(full_output, X_test, squeeze_last=True)
        V_hat = np.clip(np.asarray(V_hat, dtype=float), 0.0, 1.0)
        if V_hat.shape[0] != len(X_test):
            raise ValueError(
                f"Unexpected CDF prediction shape {V_hat.shape}; expected ({len(X_test)},)."
            )
        return V_hat


def build_stage1_training_frame(
    train_df: pd.DataFrame,
    stage1_cfg: Stage1Config,
    *,
    verbose: bool = True,
) -> Dict[str, object]:
    """Fit the Stage-1 control-function model in memory and return the augmented training frame."""
    required_cols = ["Z", "X", "Y", "V_true", "eps", "eta"]
    missing_cols = [col for col in required_cols if col not in train_df.columns]
    if missing_cols:
        raise ValueError(
            f"Training frame missing required columns {missing_cols}; "
            f"available columns: {list(train_df.columns)}"
        )

    Z_train = train_df["Z"].to_numpy()
    X_train = train_df["X"].to_numpy()
    Y_train = train_df["Y"].to_numpy()
    V_train_true = train_df["V_true"].to_numpy()

    if verbose:
        print(f"\n[2/3] Estimating control function V = F_{{X|Z}}(X | Z)...")
        print(f"Using backend '{stage1_cfg.backend_name}' predictive distribution for CDF estimation.")
        if stage1_cfg.quantiles:
            print(f"Requested supplemental quantiles ({len(stage1_cfg.quantiles)}): {stage1_cfg.quantiles}")
        print(f"Training X range: [{X_train.min():.3f}, {X_train.max():.3f}]")

    cdf_model = CondCDFModel(
        quantiles=stage1_cfg.quantiles,
        backend_name=stage1_cfg.backend_name,
        model_path=stage1_cfg.model_path,
        random_state=stage1_cfg.random_state,
        softmax_temperature=stage1_cfg.softmax_temperature,
    )
    cdf_model.fit(Z_train, X_train, verbose=verbose)

    if verbose:
        print(f"\n[3/3] Predicting control function on training data...")
    V_train_hat = cdf_model.predict(Z_train, X_train)
    if verbose:
        print(f"V̂_train statistics: mean={np.mean(V_train_hat):.3f}, std={np.std(V_train_hat):.3f}")

    train_result_df = pd.DataFrame(
        {
            "Z": Z_train,
            "X": X_train,
            "Y": Y_train,
            "V_true": V_train_true,
            "eps": train_df["eps"].to_numpy(),
            "eta": train_df["eta"].to_numpy(),
            "V_hat": V_train_hat,
        }
    )
    return {
        "train": train_result_df,
        "cdf_model": cdf_model,
    }


# =========================================================
# 3) Main execution pipeline
# =========================================================
def run_stage1_experiment(
    first_stage: str,
    second_stage: str,
    stage1_cfg: Stage1Config,
    *,
    train_sample_size: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: str | os.PathLike[str] = DEFAULT_STAGE1_OUTPUT_DIR,
    base_dir: str | os.PathLike[str] = DEFAULT_DATA_DIR,
    test_x_trim_quantile_range: tuple[float, float] | None = DEFAULT_TEST_X_TRIM_QUANTILE_RANGE,
    save_outputs: bool = True,
    use_timestamp: bool = False,
) -> Dict[str, object]:
    """
    Main pipeline implementing objectives 1 and 2:
    1) Load pre-generated triangular DGP data (training and test sets)
    2) Estimate control function using TabPFN

    Args:
        first_stage: First stage DGP type (e.g., "A1", "A2")
        second_stage: Second stage DGP type (e.g., "B1", "B2")
        stage1_cfg: Stage 1 estimation configuration
        train_sample_size: Optional training sample size identifier used in the CSV filename
        seed: Random seed identifier used to locate deterministic training data files
        output_dir: Directory for saving outputs
        base_dir: Base directory containing deterministic DGP datasets
        test_x_trim_quantile_range: Optional [lower, upper] empirical X-rank range
            retained from the test split. None disables trimming.
        save_outputs: Whether to persist Stage 1 results to disk
        use_timestamp: Append timestamps to filenames when saving (useful for debugging)

    Returns:
        Dictionary with:
            - "train": Stage 1 training DataFrame with V_hat
            - "output_paths": mapping of subset name to saved CSV paths
    """
    print("Starting IV Stage 1 Experiment (Version 4)")
    print(f"DGP Configuration: first_stage={first_stage}, second_stage={second_stage}")
    print(f"Stage 1 Configuration: {asdict(stage1_cfg)}")
    if train_sample_size is not None:
        print(f"Requested training sample size: {train_sample_size}")
    if seed is not None:
        print(f"Random seed tag: {seed}")

    print("\n[1/3] Loading pre-generated training data...")
    train_df, test_df = load_dgp_data(
        first_stage,
        second_stage,
        train_sample_size=train_sample_size,
        seed=seed,
        base_dir=base_dir,
        test_x_trim_quantile_range=test_x_trim_quantile_range,
    )

    Z_train = train_df["Z"].values
    X_train = train_df["X"].values
    Y_train = train_df["Y"].values
    Z_test = test_df["Z"].values
    X_test = test_df["X"].values
    Y_test = test_df["Y"].values
    V_test_true = test_df["V_true"].values

    print("Training data summary:")
    print(f"  Sample size: {len(Z_train)}")
    print(f"  X range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"  Y range: [{Y_train.min():.3f}, {Y_train.max():.3f}]")
    print(f"  Z range: [{Z_train.min():.3f}, {Z_train.max():.3f}]")

    print("Test data summary:")
    print(f"  Sample size: {len(Z_test)}")
    print(f"  X range: [{X_test.min():.3f}, {X_test.max():.3f}]")
    print(f"  Y range: [{Y_test.min():.3f}, {Y_test.max():.3f}]")
    print(f"  Z range: [{Z_test.min():.3f}, {Z_test.max():.3f}]")
    print(f"  V_true range: [{V_test_true.min():.3f}, {V_test_true.max():.3f}]")

    stage1_results = build_stage1_training_frame(train_df, stage1_cfg, verbose=True)
    train_result_df = stage1_results["train"]

    output_paths: Dict[str, Path] = {}
    if save_outputs:
        output_paths = save_stage1_results(
            {"train": train_result_df},
            first_stage,
            second_stage,
            train_sample_size=train_sample_size,
            seed=seed,
            backend_name=stage1_cfg.backend_name,
            softmax_temperature=stage1_cfg.softmax_temperature,
            output_dir=output_dir,
            use_timestamp=use_timestamp,
        )

    return {
        "train": train_result_df,
        "cdf_model": stage1_results["cdf_model"],
        "output_paths": output_paths,
    }


# =========================================================
# 4) Runner
# =========================================================
if __name__ == "__main__":
    """
    Main execution: demonstrate Stage 1 implementation with pre-generated data.
    Loads training and test data from IV_datasets directory based on DGP parameters.
    """
    
    # DGP parameters (default values)
    first_stage = "A3"
    second_stage = "B5"

    # Training data size identifier used in the CSV filename; set to None for legacy naming
    train_sample_size = 8000
    seed = 123
   
    # Stage 1 estimation configuration
    stage1_cfg = Stage1Config(
        random_state=1,
    )

    # Run experiment
    try:
        results = run_stage1_experiment(
            first_stage,
            second_stage,
            stage1_cfg,
            train_sample_size=train_sample_size,
            seed=seed,
            output_dir=DEFAULT_STAGE1_OUTPUT_DIR,
            base_dir=DEFAULT_DATA_DIR,
            save_outputs=True,
            use_timestamp=True,
        )
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Print basic statistics
        train_df = results["train"]
        print(f"\nTraining subset results:")
        print(f"  Sample size: {len(train_df)}")
        print(f"  X range: [{train_df['X'].min():.3f}, {train_df['X'].max():.3f}]")
        print(f"  Y range: [{train_df['Y'].min():.3f}, {train_df['Y'].max():.3f}]")
        print(f"  Z range: [{train_df['Z'].min():.3f}, {train_df['Z'].max():.3f}]")
        print(f"  V_true range: [{train_df['V_true'].min():.3f}, {train_df['V_true'].max():.3f}]")
        print(f"  V_hat range: [{train_df['V_hat'].min():.3f}, {train_df['V_hat'].max():.3f}]")
        
        output_paths = results.get("output_paths", {})
        for subset, path in output_paths.items():
            print(f"  Saved '{subset}' CSV to: {path}")
        
        print(f"\n📁 Output directory: {DEFAULT_STAGE1_OUTPUT_DIR}")
        print(f"🕒 Seed: {seed}")
        print(f"📊 DGP Configuration: {first_stage}_{second_stage}")
        
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()


#1
