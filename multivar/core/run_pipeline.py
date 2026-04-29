#!/usr/bin/env python3
"""Batch runner for the multivariate benchmark."""

from __future__ import annotations

import argparse
import itertools
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multivar.core.dgp import parse_code_list
from multivar.core.experiment import ExperimentConfig
from multivar.core.experiment import run_single_experiment


COPULA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = COPULA_ROOT / "io" / "results"
DEFAULT_AGGREGATED_DIR = COPULA_ROOT / "io" / "aggregated"
DEFAULT_DIST_QUANTILE_LEVELS = [v / 100.0 for v in range(1, 100)]


def _rho_tag(value: float) -> str:
    sign = "m" if value < 0 else ""
    magnitude = re.sub(r"[^0-9a-zA-Z]+", "p", f"{abs(value):.3f}").strip("p")
    return f"{sign}{magnitude}"


def _run_tag(dgp_code: str, marginal_source: str, n_train: int, rho_eps: float, seed: int) -> str:
    return f"{dgp_code}_ms{marginal_source}_n{n_train}_rho{_rho_tag(rho_eps)}_seed{seed}"


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _summarise_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = (
        df.groupby(["dgp_code", "marginal_source", "n_train", "rho_eps"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            rho_abs_error_mean=("rho_abs_error", "mean"),
            rho_abs_error_std=("rho_abs_error", "std"),
            tau_abs_error_mean=("tau_abs_error", "mean"),
            tail_mae_estimated_mean=("estimated_tail_mae", "mean"),
            tail_mae_independence_mean=("independence_tail_mae", "mean"),
            joint_cdf_mae_estimated_mean=("estimated_joint_cdf_mae", "mean"),
            joint_cdf_mae_independence_mean=("independence_joint_cdf_mae", "mean"),
            pit_ks_u1_mean=("pit_ks_stat_u1", "mean"),
            pit_ks_u2_mean=("pit_ks_stat_u2", "mean"),
            xbin_corr_std_mean=("xbin_corr_std", "mean"),
        )
        .sort_values(["dgp_code", "marginal_source", "n_train", "rho_eps"])
        .reset_index(drop=True)
    )
    return grouped


def _coerce_int_list(values: Iterable[str]) -> list[int]:
    return [int(v) for v in values]


def _coerce_float_list(values: Iterable[str]) -> list[float]:
    return [float(v) for v in values]


def run_batch(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    dgp_codes = parse_code_list(args.dgp_codes)
    n_list = _coerce_int_list(args.train_sizes)
    seeds = _coerce_int_list(args.seeds)
    rho_list = _coerce_float_list(args.rho_eps_values)

    results_dir = args.results_dir.resolve()
    aggregated_dir = args.aggregated_dir.resolve()
    per_run_dir = results_dir / "per_run"
    _ensure_dirs(results_dir, aggregated_dir, per_run_dir)

    summary_rows = []
    combinations = list(itertools.product(dgp_codes, n_list, rho_list, seeds))
    total = len(combinations)
    if total == 0:
        raise SystemExit("No run combinations provided.")

    for idx, (dgp_code, n_train, rho_eps, seed) in enumerate(combinations, start=1):
        tag = _run_tag(dgp_code, args.marginal_source, n_train, rho_eps, seed)
        summary_path = per_run_dir / f"multivar_{tag}_summary.csv"
        xmetrics_path = per_run_dir / f"multivar_{tag}_xmetrics.csv"
        xbins_path = per_run_dir / f"multivar_{tag}_xbins.csv"
        dist_cdf_path = per_run_dir / f"multivar_{tag}_distribution_cdf.csv"
        dist_samples_path = per_run_dir / f"multivar_{tag}_distribution_samples.csv"

        required_paths = [summary_path, xmetrics_path, xbins_path]
        if args.save_distribution_output:
            required_paths.extend([dist_cdf_path, dist_samples_path])

        if args.skip_existing and all(path.exists() for path in required_paths):
            existing = pd.read_csv(summary_path)
            if len(existing) != 1:
                raise ValueError(f"Unexpected number of rows in {summary_path}: {len(existing)}")
            summary_rows.append(existing.iloc[0].to_dict())
            print(f"[{idx}/{total}] skip-existing: {tag}")
            continue

        start = time.time()
        exp_cfg = ExperimentConfig(
            dgp_code=dgp_code,
            n_train=n_train,
            seed=seed,
            rho_eps=rho_eps,
            x_grid_min=args.x_grid_min,
            x_grid_max=args.x_grid_max,
            x_grid_size=args.x_grid_size,
            x_bins=args.x_bins,
            clip_eps=args.clip_eps,
            tail_x_ref=args.tail_x_ref,
            y_quantile_levels=tuple(args.y_quantile_levels),
            marginal_source=args.marginal_source,
            deepcf_v_points=args.deepcf_v_points,
            deepcf_eval_batch_size=args.deepcf_eval_batch_size,
            save_distribution_output=args.save_distribution_output,
            dist_quantile_levels=tuple(args.dist_quantile_levels),
            dist_sample_size=args.dist_sample_size,
        )
        summary, x_metrics, x_bins, dist_cdf, dist_samples = run_single_experiment(exp_cfg)

        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        x_metrics.to_csv(xmetrics_path, index=False)
        x_bins.to_csv(xbins_path, index=False)
        if args.save_distribution_output:
            dist_cdf.to_csv(dist_cdf_path, index=False)
            dist_samples.to_csv(dist_samples_path, index=False)
        summary_rows.append(summary)
        elapsed = time.time() - start
        print(f"[{idx}/{total}] completed: {tag} ({elapsed:.2f}s)")

    results_df = pd.DataFrame(summary_rows)
    if results_df.empty:
        raise RuntimeError("No run summaries were produced.")
    if "marginal_source" not in results_df.columns:
        results_df["marginal_source"] = args.marginal_source
    results_df = results_df.sort_values(["dgp_code", "marginal_source", "n_train", "rho_eps", "seed"])
    grouped_df = _summarise_results(results_df)

    raw_output = args.results_csv or (results_dir / "multivar_batch_results.csv")
    grouped_output = args.grouped_csv or (aggregated_dir / "multivar_grouped_summary.csv")
    results_df.to_csv(raw_output, index=False)
    grouped_df.to_csv(grouped_output, index=False)
    print(f"Saved run-level results to {raw_output}")
    print(f"Saved grouped summary to {grouped_output}")
    return results_df, grouped_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multivariate batch experiments over DGP/size/rho/seed grids.")
    parser.add_argument(
        "--dgp-codes",
        nargs="+",
        default=[
            "DGP1_LINEAR",
            "DGP2_NONLINEAR",
            "DGP3_PRE_ADDITIVE",
            "DGP4_PIECEWISE",
            "DGP5_SOFTPLUS",
        ],
        help=(
            "DGP codes: DGP1_LINEAR, DGP2_NONLINEAR, DGP3_PRE_ADDITIVE, "
            "DGP4_PIECEWISE, DGP5_SOFTPLUS "
            "(aliases: D1, D2, D3, D4/piecewise, D5/softplus)."
        ),
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        default=["1000", "4000"],
        help="Training sample sizes to simulate.",
    )
    parser.add_argument(
        "--rho-eps-values",
        nargs="+",
        default=["0.0", "0.3", "0.6"],
        help="Outcome-noise correlations rho_eps.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[str(v) for v in range(1, 11)],
        help="Simulation seeds.",
    )
    parser.add_argument(
        "--marginal-source",
        choices=["deepcf", "tabpfn_real", "tabicl", "tabpfn-naive", "div", "oracle"],
        default="deepcf",
        help="Source of marginal CDFs for pseudo-uniform construction and joint plug-in evaluation.",
    )
    parser.add_argument(
        "--deepcf-v-points",
        type=int,
        default=101,
        help="Number of V integration points for TabCF marginal integration.",
    )
    parser.add_argument(
        "--deepcf-eval-batch-size",
        type=int,
        default=256,
        help="Batch size for TabCF marginal CDF pointwise evaluation.",
    )
    parser.add_argument("--x-grid-min", type=float, default=0.0)
    parser.add_argument("--x-grid-max", type=float, default=3.0)
    parser.add_argument("--x-grid-size", type=int, default=13)
    parser.add_argument("--x-bins", type=int, default=5, help="Number of quantile bins for X invariance diagnostics.")
    parser.add_argument("--tail-x-ref", type=float, default=None, help="Reference x used to define tail thresholds a,b.")
    parser.add_argument("--clip-eps", type=float, default=1e-6, help="Clipping epsilon for pseudo-uniform values.")
    parser.add_argument(
        "--y-quantile-levels",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="Quantile levels used for joint-CDF error grids.",
    )
    parser.add_argument(
        "--dist-quantile-levels",
        nargs="+",
        type=float,
        default=DEFAULT_DIST_QUANTILE_LEVELS,
        help="Quantile levels used for distributional-output CDF grids.",
    )
    parser.add_argument(
        "--dist-sample-size",
        type=int,
        default=10000,
        help="Number of joint samples per x and model for distributional-output sampling CSV.",
    )
    dist_output_group = parser.add_mutually_exclusive_group()
    dist_output_group.add_argument(
        "--save-distribution-output",
        dest="save_distribution_output",
        action="store_true",
        help="Write per-run distributional output CSV files (default: enabled).",
    )
    dist_output_group.add_argument(
        "--no-save-distribution-output",
        dest="save_distribution_output",
        action="store_false",
        help="Disable writing per-run distributional output CSV files.",
    )
    parser.set_defaults(save_distribution_output=True)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-run CSV files when available.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--aggregated-dir", type=Path, default=DEFAULT_AGGREGATED_DIR)
    parser.add_argument("--results-csv", type=Path, default=None, help="Optional run-level output CSV path.")
    parser.add_argument("--grouped-csv", type=Path, default=None, help="Optional grouped-summary output CSV path.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
