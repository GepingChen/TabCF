#!/usr/bin/env python3
"""
Single-point A3_B3 PPD comparison at x=1.5 for global vs. local-kNN TabPFN.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

from dgp import DGPConfig, testdata_generation, training_data_generation
from stage1_control import (
    DEFAULT_DATA_DIR,
    DEFAULT_STAGE1_OUTPUT_DIR,
    Stage1Config,
    run_stage1_experiment,
)
from stage2_outcome import (
    ConditionalCDFEstimator,
    compute_interventional_pdf,
    load_dgp_test_data,
    load_stage1_data,
    monte_carlo_y_given_x,
)
from foundation_backends import TABPFN_BACKEND, stage1_output_filename
from local_context_backends import LocalContextConfig


DEFAULT_FIRST_STAGE = "A3"
DEFAULT_SECOND_STAGE = "B3"
DEFAULT_X_VALUE = 1.5
DEFAULT_TRAIN_SIZE = 1000
DEFAULT_SEEDS = tuple(range(1, 11))
DEFAULT_OUTPUT_DIR = Path(DEFAULT_DATA_DIR) / "ppd_x15_tabpfn_knn"
DEFAULT_TRUE_MC_SAMPLES = 20000
DEFAULT_TRUE_MC_SEED = 20260412


@dataclass(frozen=True)
class SinglePointPPDConfig:
    train_size: int = DEFAULT_TRAIN_SIZE
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    x_value: float = DEFAULT_X_VALUE
    local_k_neighbors: int | None = None
    n_y_grid: int = 200
    n_v_integration_points: int = 100
    stage1_random_state: int = 1
    stage2_random_state: int = 1
    test_seed: int = 999
    model_path: str = "auto"
    true_mc_samples: int = DEFAULT_TRUE_MC_SAMPLES
    true_mc_seed: int = DEFAULT_TRUE_MC_SEED
    data_dir: Path = Path(DEFAULT_DATA_DIR)
    stage1_output_dir: Path = Path(DEFAULT_STAGE1_OUTPUT_DIR)
    output_dir: Path = DEFAULT_OUTPUT_DIR
    first_stage_code: str = DEFAULT_FIRST_STAGE
    second_stage_code: str = DEFAULT_SECOND_STAGE

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_size", int(self.train_size))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "x_value", float(self.x_value))
        object.__setattr__(self, "n_y_grid", int(self.n_y_grid))
        object.__setattr__(self, "n_v_integration_points", int(self.n_v_integration_points))
        object.__setattr__(self, "stage1_random_state", int(self.stage1_random_state))
        object.__setattr__(self, "stage2_random_state", int(self.stage2_random_state))
        object.__setattr__(self, "test_seed", int(self.test_seed))
        object.__setattr__(self, "true_mc_samples", int(self.true_mc_samples))
        object.__setattr__(self, "true_mc_seed", int(self.true_mc_seed))
        if self.local_k_neighbors is not None:
            object.__setattr__(self, "local_k_neighbors", int(self.local_k_neighbors))
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "stage1_output_dir", Path(self.stage1_output_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


def analytic_a3_b3_mean(x_value: float) -> float:
    return float(x_value)


def analytic_a3_b3_variance() -> float:
    return 10.0


def analytic_a3_b3_pdf(y_values: np.ndarray, x_value: float) -> np.ndarray:
    y_arr = np.asarray(y_values, dtype=float)
    return norm.pdf(y_arr, loc=analytic_a3_b3_mean(x_value), scale=np.sqrt(analytic_a3_b3_variance()))


def analytic_b4_linear_mean(x_value: float) -> float:
    return float(0.2 * (5.5 + 2.0 * x_value))


def analytic_b4_linear_variance() -> float:
    return 0.4


def analytic_b4_linear_pdf(y_values: np.ndarray, x_value: float) -> np.ndarray:
    y_arr = np.asarray(y_values, dtype=float)
    return norm.pdf(y_arr, loc=analytic_b4_linear_mean(x_value), scale=np.sqrt(analytic_b4_linear_variance()))


def resolve_true_reference(config: SinglePointPPDConfig, y_grid: np.ndarray) -> dict[str, np.ndarray | float | str]:
    y_arr = np.asarray(y_grid, dtype=float)

    if config.first_stage_code == "A3" and config.second_stage_code == "B3":
        return {
            "pdf_true": analytic_a3_b3_pdf(y_arr, config.x_value),
            "mean_true": analytic_a3_b3_mean(config.x_value),
            "var_true": analytic_a3_b3_variance(),
            "true_method": "analytic_normal",
        }

    if config.first_stage_code == "A3" and config.second_stage_code == "B4" and float(config.x_value) <= 1.0:
        return {
            "pdf_true": analytic_b4_linear_pdf(y_arr, config.x_value),
            "mean_true": analytic_b4_linear_mean(config.x_value),
            "var_true": analytic_b4_linear_variance(),
            "true_method": "analytic_linear_branch",
        }

    dgp_cfg = DGPConfig(first_stage=config.first_stage_code, second_stage=config.second_stage_code)
    rng = np.random.default_rng(config.true_mc_seed)
    y_samples, _ = monte_carlo_y_given_x(
        dgp_cfg,
        float(config.x_value),
        int(config.true_mc_samples),
        rng,
    )
    y_samples = np.asarray(y_samples, dtype=float)
    try:
        density = gaussian_kde(y_samples)
        pdf_true = np.maximum(density(y_arr), 0.0)
    except np.linalg.LinAlgError:
        jitter = 1e-6 * rng.standard_normal(size=y_samples.shape)
        density = gaussian_kde(y_samples + jitter)
        pdf_true = np.maximum(density(y_arr), 0.0)
    return {
        "pdf_true": pdf_true,
        "mean_true": float(np.mean(y_samples)),
        "var_true": float(np.var(y_samples)),
        "true_method": "monte_carlo_kde",
    }


def _value_tag(value: float, *, prefix: str) -> str:
    numeric = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return f"{prefix}{numeric}"


def build_output_paths(config: SinglePointPPDConfig) -> dict[str, Path]:
    stem = f"{config.first_stage_code}_{config.second_stage_code}_n{config.train_size}_{_value_tag(config.x_value, prefix='x')}"
    return {
        "curves": config.output_dir / f"curves_{stem}.csv",
        "summary": config.output_dir / f"summary_{stem}.csv",
        "plot": config.output_dir / f"plot_{stem}.png",
    }


def _expected_train_csv(config: SinglePointPPDConfig, seed: int) -> Path:
    return config.data_dir / "train" / (
        f"train_data_{config.first_stage_code}_{config.second_stage_code}_n{config.train_size}_seed{int(seed)}.csv"
    )


def _expected_stage1_csv(config: SinglePointPPDConfig, seed: int) -> Path:
    return config.stage1_output_dir / stage1_output_filename(
        "train",
        f"{config.first_stage_code}_{config.second_stage_code}",
        train_sample_size=config.train_size,
        seed=int(seed),
        backend_name=TABPFN_BACKEND,
        timestamp=None,
    )


def ensure_test_dataset(config: SinglePointPPDConfig) -> Path:
    cfg = DGPConfig(
        n=10000,
        seed=config.test_seed,
        first_stage=config.first_stage_code,
        second_stage=config.second_stage_code,
    )
    test_csv = config.data_dir / "test" / f"test_data_{config.first_stage_code}_{config.second_stage_code}.csv"
    if not test_csv.exists():
        testdata_generation(
            cfg,
            test_dir=config.data_dir / "test",
            test_seed=config.test_seed,
            force_regenerate=False,
        )
    return test_csv


def ensure_stage1_csv(config: SinglePointPPDConfig, seed: int) -> Path:
    stage1_csv = _expected_stage1_csv(config, seed)
    if stage1_csv.exists():
        return stage1_csv

    train_csv = _expected_train_csv(config, seed)
    if not train_csv.exists():
        train_cfg = DGPConfig(
            n=config.train_size,
            seed=int(seed),
            first_stage=config.first_stage_code,
            second_stage=config.second_stage_code,
        )
        training_data_generation(
            train_cfg,
            save_csv=True,
            train_dir=config.data_dir / "train",
        )

    ensure_test_dataset(config)
    config.stage1_output_dir.mkdir(parents=True, exist_ok=True)
    run_stage1_experiment(
        config.first_stage_code,
        config.second_stage_code,
        Stage1Config(
            random_state=config.stage1_random_state,
            backend_name=TABPFN_BACKEND,
            model_path=config.model_path,
        ),
        train_sample_size=config.train_size,
        seed=int(seed),
        output_dir=config.stage1_output_dir,
        base_dir=config.data_dir,
        save_outputs=True,
        use_timestamp=False,
    )
    return stage1_csv


def build_y_grid(config: SinglePointPPDConfig) -> np.ndarray:
    _, test_data = load_dgp_test_data(
        config.first_stage_code,
        config.second_stage_code,
        base_dir=config.data_dir,
    )
    y_test = np.asarray(test_data["Y"], dtype=float)
    y_min = float(np.min(y_test))
    y_max = float(np.max(y_test))
    return np.linspace(y_min, y_max, int(config.n_y_grid))


def summarize_pdf_curve(y_grid: np.ndarray, pdf_values: np.ndarray) -> dict[str, float]:
    y_arr = np.asarray(y_grid, dtype=float)
    pdf_arr = np.maximum(np.asarray(pdf_values, dtype=float), 0.0)
    mass = float(np.trapz(pdf_arr, x=y_arr))
    if not np.isfinite(mass) or mass <= 0.0:
        return {"mass": float("nan"), "mean": float("nan"), "var": float("nan")}

    normalized_pdf = pdf_arr / mass
    mean_val = float(np.trapz(y_arr * normalized_pdf, x=y_arr))
    second_moment = float(np.trapz((y_arr ** 2) * normalized_pdf, x=y_arr))
    variance = max(second_moment - mean_val ** 2, 0.0)
    return {"mass": mass, "mean": mean_val, "var": float(variance)}


def compute_seed_curves(
    *,
    stage1_csv: Path,
    seed: int,
    config: SinglePointPPDConfig,
    y_grid: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    train_data = load_stage1_data(stage1_csv)
    x_vec = np.asarray([config.x_value], dtype=float)
    true_reference = resolve_true_reference(config, y_grid)

    global_model = ConditionalCDFEstimator(
        backend_name=TABPFN_BACKEND,
        model_path=config.model_path,
        random_state=config.stage2_random_state,
        local_context=LocalContextConfig(strategy="global"),
    )
    global_model.fit_full(train_data["X"], train_data["V_hat"], train_data["Y"], verbose=False)
    global_pdf = compute_interventional_pdf(
        global_model,
        x_vec,
        y_grid,
        config.n_v_integration_points,
    )["pdf"][0]

    local_model = ConditionalCDFEstimator(
        backend_name=TABPFN_BACKEND,
        model_path=config.model_path,
        random_state=config.stage2_random_state,
        local_context=LocalContextConfig(
            strategy="local_knn",
            k_neighbors=config.local_k_neighbors,
        ),
    )
    local_model.fit_full(train_data["X"], train_data["V_hat"], train_data["Y"], verbose=False)
    local_pdf = compute_interventional_pdf(
        local_model,
        x_vec,
        y_grid,
        config.n_v_integration_points,
    )["pdf"][0]

    true_pdf = np.asarray(true_reference["pdf_true"], dtype=float)
    global_stats = summarize_pdf_curve(y_grid, global_pdf)
    local_stats = summarize_pdf_curve(y_grid, local_pdf)

    curve_df = pd.DataFrame(
        {
            "seed": int(seed),
            "y": np.asarray(y_grid, dtype=float),
            "pdf_true": np.asarray(true_pdf, dtype=float),
            "pdf_global": np.asarray(global_pdf, dtype=float),
            "pdf_local": np.asarray(local_pdf, dtype=float),
        }
    )
    summary_row = {
        "seed": int(seed),
        "x_value": float(config.x_value),
        "mean_true": float(true_reference["mean_true"]),
        "var_true": float(true_reference["var_true"]),
        "mean_global": global_stats["mean"],
        "var_global": global_stats["var"],
        "mean_local": local_stats["mean"],
        "var_local": local_stats["var"],
        "mse_mean_global": float((global_stats["mean"] - float(true_reference["mean_true"])) ** 2),
        "mse_mean_local": float((local_stats["mean"] - float(true_reference["mean_true"])) ** 2),
        "iae_global": float(np.trapz(np.abs(global_pdf - true_pdf), x=y_grid)),
        "iae_local": float(np.trapz(np.abs(local_pdf - true_pdf), x=y_grid)),
        "var_gap_global": float(abs(global_stats["var"] - float(true_reference["var_true"]))),
        "var_gap_local": float(abs(local_stats["var"] - float(true_reference["var_true"]))),
        "mass_global": global_stats["mass"],
        "mass_local": local_stats["mass"],
        "local_k_neighbors_resolved": getattr(local_model.model, "resolved_k_", None),
        "true_method": str(true_reference["true_method"]),
    }
    return curve_df, summary_row


def aggregate_summary_rows(curves_df: pd.DataFrame, summary_df: pd.DataFrame, config: SinglePointPPDConfig) -> pd.DataFrame:
    y_grid = np.sort(curves_df["y"].unique())
    grouped = curves_df.groupby("y", sort=True, as_index=False).agg(
        pdf_true=("pdf_true", "mean"),
        pdf_global=("pdf_global", "mean"),
        pdf_local=("pdf_local", "mean"),
    )
    true_pdf = grouped["pdf_true"].to_numpy(dtype=float)
    global_pdf = grouped["pdf_global"].to_numpy(dtype=float)
    local_pdf = grouped["pdf_local"].to_numpy(dtype=float)
    global_stats = summarize_pdf_curve(y_grid, global_pdf)
    local_stats = summarize_pdf_curve(y_grid, local_pdf)
    mean_true = float(summary_df["mean_true"].iloc[0])
    var_true = float(summary_df["var_true"].iloc[0])
    true_method = str(summary_df["true_method"].iloc[0]) if "true_method" in summary_df.columns else ""
    mse_mean_global = float(summary_df["mse_mean_global"].mean()) if "mse_mean_global" in summary_df.columns else float("nan")
    mse_mean_local = float(summary_df["mse_mean_local"].mean()) if "mse_mean_local" in summary_df.columns else float("nan")

    resolved_values = summary_df["local_k_neighbors_resolved"].dropna().unique().tolist()
    resolved_local_k = resolved_values[0] if len(resolved_values) == 1 else ""

    aggregate_row = pd.DataFrame(
        [
            {
                "seed": "aggregate",
                "x_value": float(config.x_value),
                "mean_true": mean_true,
                "var_true": var_true,
                "mean_global": global_stats["mean"],
                "var_global": global_stats["var"],
                "mean_local": local_stats["mean"],
                "var_local": local_stats["var"],
                "mse_mean_global": mse_mean_global,
                "mse_mean_local": mse_mean_local,
                "iae_global": float(np.trapz(np.abs(global_pdf - true_pdf), x=y_grid)),
                "iae_local": float(np.trapz(np.abs(local_pdf - true_pdf), x=y_grid)),
                "var_gap_global": float(abs(global_stats["var"] - var_true)),
                "var_gap_local": float(abs(local_stats["var"] - var_true)),
                "mass_global": global_stats["mass"],
                "mass_local": local_stats["mass"],
                "local_k_neighbors_resolved": resolved_local_k,
                "true_method": true_method,
            }
        ]
    )
    return pd.concat([summary_df, aggregate_row], ignore_index=True)


def save_plot(curves_df: pd.DataFrame, output_path: Path, config: SinglePointPPDConfig) -> None:
    grouped = curves_df.groupby("y", sort=True)
    y_grid = np.asarray(sorted(curves_df["y"].unique()), dtype=float)
    true_pdf = grouped["pdf_true"].mean().to_numpy(dtype=float)
    global_mean = grouped["pdf_global"].mean().to_numpy(dtype=float)
    global_std = grouped["pdf_global"].std(ddof=0).fillna(0.0).to_numpy(dtype=float)
    local_mean = grouped["pdf_local"].mean().to_numpy(dtype=float)
    local_std = grouped["pdf_local"].std(ddof=0).fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(y_grid, true_pdf, color="#1f1f1f", linewidth=2.2, label="True")
    ax.plot(y_grid, global_mean, color="#1f77b4", linewidth=2.0, label="Global TabPFN")
    ax.fill_between(
        y_grid,
        np.maximum(global_mean - global_std, 0.0),
        global_mean + global_std,
        color="#1f77b4",
        alpha=0.2,
    )
    ax.plot(y_grid, local_mean, color="#d62728", linewidth=2.0, label="Local-kNN TabPFN")
    ax.fill_between(
        y_grid,
        np.maximum(local_mean - local_std, 0.0),
        local_mean + local_std,
        color="#d62728",
        alpha=0.2,
    )

    ax.set_title(
        f"{config.first_stage_code}_{config.second_stage_code} PPD at x={config.x_value:g} "
        f"(n={config.train_size}, seeds={len(config.seeds)})"
    )
    ax.set_xlabel("y")
    ax.set_ylabel("pdf")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def write_outputs(
    curves_df: pd.DataFrame,
    summary_seed_df: pd.DataFrame,
    config: SinglePointPPDConfig,
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    resolved_output_dir = Path(output_dir) if output_dir is not None else config.output_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    original_output_dir = config.output_dir
    if resolved_output_dir != original_output_dir:
        config = SinglePointPPDConfig(
            train_size=config.train_size,
            seeds=config.seeds,
            x_value=config.x_value,
            local_k_neighbors=config.local_k_neighbors,
            n_y_grid=config.n_y_grid,
            n_v_integration_points=config.n_v_integration_points,
            stage1_random_state=config.stage1_random_state,
            stage2_random_state=config.stage2_random_state,
            test_seed=config.test_seed,
            model_path=config.model_path,
            true_mc_samples=config.true_mc_samples,
            true_mc_seed=config.true_mc_seed,
            data_dir=config.data_dir,
            stage1_output_dir=config.stage1_output_dir,
            output_dir=resolved_output_dir,
            first_stage_code=config.first_stage_code,
            second_stage_code=config.second_stage_code,
        )

    curves_out = curves_df.sort_values(["seed", "y"]).reset_index(drop=True)
    summary_seed_out = summary_seed_df.sort_values("seed").reset_index(drop=True)
    summary_with_aggregate = aggregate_summary_rows(curves_out, summary_seed_out, config)
    output_paths = build_output_paths(config)
    curves_out.to_csv(output_paths["curves"], index=False)
    summary_with_aggregate.to_csv(output_paths["summary"], index=False)
    save_plot(curves_out, output_paths["plot"], config)
    return output_paths


def run_sideproject(config: SinglePointPPDConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.stage1_output_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / "train").mkdir(parents=True, exist_ok=True)
    (config.data_dir / "test").mkdir(parents=True, exist_ok=True)

    ensure_test_dataset(config)
    y_grid = build_y_grid(config)

    curve_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for seed in config.seeds:
        print(f"\n=== Seed {seed} ===")
        stage1_csv = ensure_stage1_csv(config, int(seed))
        curve_df, summary_row = compute_seed_curves(
            stage1_csv=stage1_csv,
            seed=int(seed),
            config=config,
            y_grid=y_grid,
        )
        curve_frames.append(curve_df)
        summary_rows.append(summary_row)

    curves_df = pd.concat(curve_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    return write_outputs(curves_df, summary_df, config)


def merge_worker_outputs(worker_output_dirs: Sequence[Path], config: SinglePointPPDConfig) -> dict[str, Path]:
    curve_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for worker_dir in worker_output_dirs:
        worker_paths = build_output_paths(
            SinglePointPPDConfig(
                train_size=config.train_size,
                seeds=config.seeds,
                x_value=config.x_value,
                local_k_neighbors=config.local_k_neighbors,
                n_y_grid=config.n_y_grid,
                n_v_integration_points=config.n_v_integration_points,
                stage1_random_state=config.stage1_random_state,
                stage2_random_state=config.stage2_random_state,
                test_seed=config.test_seed,
                model_path=config.model_path,
                true_mc_samples=config.true_mc_samples,
                true_mc_seed=config.true_mc_seed,
                data_dir=config.data_dir,
                stage1_output_dir=config.stage1_output_dir,
                output_dir=worker_dir,
                first_stage_code=config.first_stage_code,
                second_stage_code=config.second_stage_code,
            )
        )
        curves_path = worker_paths["curves"]
        summary_path = worker_paths["summary"]
        if not curves_path.exists() or not summary_path.exists():
            raise FileNotFoundError(
                f"Missing worker outputs in {worker_dir}: curves={curves_path.exists()} summary={summary_path.exists()}"
            )
        curve_frames.append(pd.read_csv(curves_path))
        summary_df = pd.read_csv(summary_path)
        summary_frames.append(summary_df.loc[summary_df["seed"].astype(str) != "aggregate"].copy())

    merged_curves = pd.concat(curve_frames, ignore_index=True)
    merged_curves = merged_curves.drop_duplicates(subset=["seed", "y"], keep="last")
    merged_summary = pd.concat(summary_frames, ignore_index=True)
    merged_summary = merged_summary.drop_duplicates(subset=["seed"], keep="last")
    return write_outputs(merged_curves, merged_summary, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare single-point PPDs using global vs local-kNN TabPFN.")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE, help="Training sample size.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), help="Training seeds to evaluate.")
    parser.add_argument("--x-value", type=float, default=DEFAULT_X_VALUE, help="Single intervention point x.")
    parser.add_argument("--first-stage-code", type=str, default=DEFAULT_FIRST_STAGE, help="First-stage DGP code.")
    parser.add_argument("--second-stage-code", type=str, default=DEFAULT_SECOND_STAGE, help="Second-stage DGP code.")
    parser.add_argument("--local-k-neighbors", type=int, default=None, help="Optional local-kNN neighborhood size.")
    parser.add_argument("--n-y-grid", type=int, default=200, help="Number of y-grid points.")
    parser.add_argument(
        "--n-v-integration-points",
        type=int,
        default=100,
        help="Number of V integration points used by the optional uniform-grid fallback integrator.",
    )
    parser.add_argument("--stage1-random-state", type=int, default=1, help="Random state for Stage 1 TabPFN.")
    parser.add_argument("--stage2-random-state", type=int, default=1, help="Random state for Stage 2 TabPFN.")
    parser.add_argument("--test-seed", type=int, default=999, help="Seed for the cached fixed test set.")
    parser.add_argument("--true-mc-samples", type=int, default=DEFAULT_TRUE_MC_SAMPLES, help="Monte Carlo draws for non-analytic true PPDs.")
    parser.add_argument("--true-mc-seed", type=int, default=DEFAULT_TRUE_MC_SEED, help="Random seed for non-analytic true PPD Monte Carlo.")
    parser.add_argument("--model-path", type=str, default="auto", help="Optional explicit TabPFN checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for sideproject outputs.")
    parser.add_argument(
        "--merge-worker-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Merge precomputed worker subdirectories into the canonical output files.",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = SinglePointPPDConfig(
        train_size=args.train_size,
        seeds=tuple(args.seeds),
        x_value=args.x_value,
        first_stage_code=str(args.first_stage_code).upper(),
        second_stage_code=str(args.second_stage_code).upper(),
        local_k_neighbors=args.local_k_neighbors,
        n_y_grid=args.n_y_grid,
        n_v_integration_points=args.n_v_integration_points,
        stage1_random_state=args.stage1_random_state,
        stage2_random_state=args.stage2_random_state,
        test_seed=args.test_seed,
        model_path=args.model_path,
        true_mc_samples=args.true_mc_samples,
        true_mc_seed=args.true_mc_seed,
        output_dir=args.output_dir,
    )
    if args.merge_worker_dirs:
        output_paths = merge_worker_outputs([Path(path) for path in args.merge_worker_dirs], config)
    else:
        output_paths = run_sideproject(config)
    print("\nSaved outputs:")
    for key, path in output_paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
