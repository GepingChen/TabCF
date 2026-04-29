#!/usr/bin/env python3
"""
Batch runner orchestrating DGP → Stage 1 → Stage 2 pipelines across multiple
random seeds, DGP specifications, and training sample sizes.

All inputs and outputs are scoped to the interv_mean sub-directory.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import pandas as pd

from dgp import DGPConfig, training_data_generation, testdata_generation
from stage1_control import (
    Stage1Config,
    run_stage1_experiment,
    DEFAULT_DATA_DIR as STAGE1_DATA_DIR,
    DEFAULT_STAGE1_OUTPUT_DIR,
)
from stage2_outcome import (
    Stage2_9Config,
    run_stage2_9_experiment,
    save_stage2_9_results,
    DEFAULT_STAGE2_OUTPUT_DIR,
)
from dgp_test_utils import (
    DEFAULT_TEST_X_TRIM_QUANTILE_RANGE,
    normalize_test_x_trim_quantile_range,
)
from foundation_backends import TABPFN_BACKEND, normalize_backend_name, stage1_output_filename, stage2_base_prefix
from local_context_backends import LocalContextConfig
from mu_integrators import DEFAULT_GAUSS_LEGENDRE_ORDER, DEFAULT_MU_INTEGRATOR


BATCH_ROOT = Path(__file__).resolve().parents[1] / "interv_mean"
DATA_DIR = Path(STAGE1_DATA_DIR)
STAGE1_OUTPUT = Path(DEFAULT_STAGE1_OUTPUT_DIR)
STAGE2_OUTPUT = Path(DEFAULT_STAGE2_OUTPUT_DIR)
# Default to A5 baseline plus new A6 variants; override via --dgp-codes or DGP_CODES_OVERRIDE in SLURM wrapper.
DEFAULT_DGP_CODES = ("A5_B3", "A6_B3", "A6_B4", "A6_B5", "A6_B6")


def parse_dgp_code(code: str) -> Tuple[str, str]:
    parts = code.split("_")
    if len(parts) != 2:
        raise ValueError(f"DGP code '{code}' must be formatted as 'A?_B?'.")
    return parts[0].upper(), parts[1].upper()


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
    return STAGE1_OUTPUT / stage1_output_filename(
        "train",
        codes,
        train_sample_size=n,
        seed=seed,
        backend_name=backend_name,
        softmax_temperature=softmax_temperature,
        timestamp=None,
    )


def expected_stage2_summary(
    first_stage: str,
    second_stage: str,
    n: int,
    seed: int,
    *,
    output_dir: str | Path = STAGE2_OUTPUT,
    backend_name: str = TABPFN_BACKEND,
    local_context: object | None = None,
) -> Path:
    codes = f"{first_stage}_{second_stage}"
    return Path(output_dir) / (
        f"{stage2_base_prefix(codes, train_sample_size=n, seed=seed, backend_name=backend_name, local_context=local_context)}_summary.csv"
    )


def ensure_test_dataset(first_stage: str, second_stage: str, test_seed: int, force: bool) -> None:
    cfg = DGPConfig(
        n=10000,
        seed=test_seed,
        first_stage=first_stage,
        second_stage=second_stage,
    )
    testdata_generation(
        cfg,
        test_dir=DATA_DIR / "test",
        test_seed=test_seed,
        force_regenerate=force,
    )


def extract_mse_from_summary(summary_path: Path) -> float:
    df = pd.read_csv(summary_path)
    metrics = {row["key"]: row["value"] for _, row in df.iterrows()}
    mse_str = metrics.get("metric_mse_do_pred_vs_clean")
    if mse_str is None:
        raise KeyError(f"metric_mse_do_pred_vs_clean not found in {summary_path}")
    return float(mse_str)


def run_pipeline(
    dgp_codes: Iterable[str],
    train_sizes: Iterable[int],
    seeds: Iterable[int],
    *,
    stage1_cfg: Stage1Config,
    stage2_local_context: LocalContextConfig,
    stage2_random_state: int,
    stage2_output_dir: Path,
    stage2_mu_integrator: str,
    stage2_gauss_legendre_order: int,
    test_seed: int,
    test_x_trim_quantile_range: tuple[float, float] | None,
    force_regenerate_test: bool,
    skip_existing: bool,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    ensured_tests: set[Tuple[str, str]] = set()
    backend_name = normalize_backend_name(stage1_cfg.backend_name)

    combinations = itertools.product(dgp_codes, train_sizes, seeds)
    for code, n_samples, seed in combinations:
        combo_start = time.time()
        first_stage, second_stage = parse_dgp_code(code)
        key = (first_stage, second_stage)
        print(f"\n=== Running combo: {code}, n={n_samples}, seed={seed} ===")

        if key not in ensured_tests:
            print("Ensuring test dataset is available...")
            ensure_test_dataset(first_stage, second_stage, test_seed, force_regenerate_test)
            ensured_tests.add(key)

        stage1_csv_path = expected_stage1_csv(
            first_stage,
            second_stage,
            n_samples,
            seed,
            backend_name=backend_name,
        )
        stage2_summary_path = expected_stage2_summary(
            first_stage,
            second_stage,
            n_samples,
            seed,
            output_dir=stage2_output_dir,
            backend_name=backend_name,
            local_context=stage2_local_context,
        )

        if skip_existing and stage1_csv_path.exists():
            print(f"Stage 1 output already exists at {stage1_csv_path}. Skipping regeneration.")
        else:
            print("Generating training dataset...")
            train_cfg = DGPConfig(
                n=n_samples,
                seed=seed,
                first_stage=first_stage,
                second_stage=second_stage,
            )
            training_data_generation(
                train_cfg,
                save_csv=True,
                train_dir=DATA_DIR / "train",
            )

            print("Running Stage 1...")
            run_stage1_experiment(
                first_stage,
                second_stage,
                stage1_cfg,
                train_sample_size=n_samples,
                seed=seed,
                output_dir=STAGE1_OUTPUT,
                base_dir=DATA_DIR,
                test_x_trim_quantile_range=test_x_trim_quantile_range,
                save_outputs=True,
                use_timestamp=False,
            )

        if skip_existing and stage2_summary_path.exists():
            print(f"Stage 2 summary already exists at {stage2_summary_path}. Skipping Stage 2 execution.")
        else:
            print("Running Stage 2...")
            stage2_cfg = Stage2_9Config(
                input_dir=str(STAGE1_OUTPUT),
                output_dir=str(stage2_output_dir),
                dgp_base_dir=str(DATA_DIR),
                n_train_samples=n_samples,
                train_seed=seed,
                random_state=stage2_random_state,
                backend_name=backend_name,
                model_path=stage1_cfg.model_path,
                local_context=stage2_local_context,
                mu_integrator=stage2_mu_integrator,
                gauss_legendre_order=stage2_gauss_legendre_order,
                first_stage_code=first_stage,
                second_stage_code=second_stage,
                test_x_trim_quantile_range=test_x_trim_quantile_range,
            )
            results = run_stage2_9_experiment(stage2_cfg)
            artifacts = save_stage2_9_results(results, stage2_output_dir, use_timestamp=False)
            stage2_summary_path = artifacts["summary"]

        mse_value = extract_mse_from_summary(stage2_summary_path)
        records.append(
            {
                "first_stage": first_stage,
                "second_stage": second_stage,
                "dgp_code": code,
                "backend": backend_name,
                "local_strategy": stage2_local_context.strategy,
                "mu_integrator": stage2_mu_integrator,
                "gauss_legendre_order": stage2_gauss_legendre_order,
                "test_x_trim_quantile_range": (
                    "disabled"
                    if test_x_trim_quantile_range is None
                    else f"{test_x_trim_quantile_range[0]:.2f},{test_x_trim_quantile_range[1]:.2f}"
                ),
                "train_sample_size": n_samples,
                "seed": seed,
                "stage1_csv": str(stage1_csv_path),
                "stage2_summary": str(stage2_summary_path),
                "mse_do_pred_vs_clean": mse_value,
            }
        )
        combo_elapsed = time.time() - combo_start
        print(
            f"Recorded MSE for combo {code}, n={n_samples}, seed={seed}: {mse_value:.6f} "
            f"(elapsed {combo_elapsed:.1f}s)"
        )

    return pd.DataFrame.from_records(records)


def summarise_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = (
        df.groupby(["backend", "first_stage", "second_stage", "train_sample_size"], as_index=False)
        .agg(
            mean_mse=("mse_do_pred_vs_clean", "mean"),
            std_mse=("mse_do_pred_vs_clean", "std"),
            min_mse=("mse_do_pred_vs_clean", "min"),
            max_mse=("mse_do_pred_vs_clean", "max"),
            n_runs=("mse_do_pred_vs_clean", "count"),
        )
        .sort_values(["backend", "first_stage", "second_stage", "train_sample_size"])
        .reset_index(drop=True)
    )
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Run batch IV simulations over multiple seeds.")
    parser.add_argument(
        "--dgp-codes",
        nargs="+",
        default=DEFAULT_DGP_CODES,
        help=(
            "List of DGP codes formatted as 'A?_B?' (e.g., A3_B2). "
            f"Default: {', '.join(DEFAULT_DGP_CODES)}"
        ),
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        required=True,
        help="Training sample sizes to simulate (e.g., 1000 2000 4000).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Random seeds for DGP → Stage1 → Stage2 pipeline.",
    )
    parser.add_argument(
        "--stage1-random-state",
        type=int,
        default=1,
        help="Random state for Stage 1 TabPFNRegressor.",
    )
    parser.add_argument(
        "--stage2-random-state",
        type=int,
        default=1,
        help="Random state for Stage 2 pipeline components.",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=999,
        help="Seed used when generating fixed test datasets.",
    )
    parser.add_argument(
        "--force-test-regenerate",
        action="store_true",
        help="Force regeneration of test datasets even if cached versions exist.",
    )
    parser.add_argument(
        "--test-x-trim-quantiles",
        nargs=2,
        type=float,
        default=list(DEFAULT_TEST_X_TRIM_QUANTILE_RANGE),
        metavar=("LOWER", "UPPER"),
        help="Empirical X-rank quantile range retained from the test split. Default: 0.05 0.95.",
    )
    parser.add_argument(
        "--disable-test-x-trim",
        action="store_true",
        help="Disable test-split X trimming and load the full cached test set.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip Stage 1/Stage 2 if deterministic outputs already exist.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Optional path to save aggregated MSE summary.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=TABPFN_BACKEND,
        choices=["tabpfn", "tabpfn_real", "tabicl"],
        help="Predictive backend for both Stage 1 and Stage 2.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="auto",
        help="Optional explicit checkpoint path. Defaults to backend-specific auto behavior.",
    )
    parser.add_argument(
        "--stage2-local-strategy",
        type=str,
        default="global",
        choices=["global", "local_knn"],
        help="Optional Stage 2 local-context strategy. Stage 1 remains global.",
    )
    parser.add_argument(
        "--stage2-local-k-neighbors",
        type=int,
        default=None,
        help="Optional k for Stage 2 local_knn neighborhoods. Defaults to an n_train-based heuristic.",
    )
    parser.add_argument(
        "--stage2-mu-integrator",
        type=str,
        default=DEFAULT_MU_INTEGRATOR,
        choices=["simpson", "gauss_legendre"],
        help="Integration rule for Stage 2 V integration. Default: gauss_legendre.",
    )
    parser.add_argument(
        "--stage2-gauss-legendre-order",
        type=int,
        default=DEFAULT_GAUSS_LEGENDRE_ORDER,
        help="Quadrature order when --stage2-mu-integrator=gauss_legendre.",
    )
    parser.add_argument(
        "--stage2-output-dir",
        type=Path,
        default=None,
        help="Optional explicit Stage 2 output directory. Required when overriding the default Gauss-Legendre (18) integrator.",
    )

    args = parser.parse_args()
    if args.stage2_gauss_legendre_order <= 0:
        raise SystemExit("--stage2-gauss-legendre-order must be positive.")
    test_x_trim_quantile_range = None
    if not args.disable_test_x_trim:
        test_x_trim_quantile_range = normalize_test_x_trim_quantile_range(args.test_x_trim_quantiles)

    if args.stage2_mu_integrator == DEFAULT_MU_INTEGRATOR:
        stage2_output_dir = STAGE2_OUTPUT if args.stage2_output_dir is None else Path(args.stage2_output_dir)
    else:
        if args.stage2_output_dir is None:
            raise SystemExit(
                "--stage2-output-dir is required when --stage2-mu-integrator is not the default gauss_legendre setting to avoid overwriting default results."
            )
        stage2_output_dir = Path(args.stage2_output_dir)

    STAGE1_OUTPUT.mkdir(parents=True, exist_ok=True)
    stage2_output_dir.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "train").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "test").mkdir(parents=True, exist_ok=True)

    stage1_cfg = Stage1Config(
        random_state=args.stage1_random_state,
        backend_name=args.backend,
        model_path=args.model_path,
    )
    stage2_local_context = LocalContextConfig(
        strategy=args.stage2_local_strategy,
        k_neighbors=args.stage2_local_k_neighbors,
    )

    results_df = run_pipeline(
        args.dgp_codes,
        args.train_sizes,
        args.seeds,
        stage1_cfg=stage1_cfg,
        stage2_local_context=stage2_local_context,
        stage2_random_state=args.stage2_random_state,
        stage2_output_dir=stage2_output_dir,
        stage2_mu_integrator=args.stage2_mu_integrator,
        stage2_gauss_legendre_order=args.stage2_gauss_legendre_order,
        test_seed=args.test_seed,
        test_x_trim_quantile_range=test_x_trim_quantile_range,
        force_regenerate_test=args.force_test_regenerate,
        skip_existing=args.skip_existing,
    )

    if results_df.empty:
        print("No runs executed. Verify configuration or disable --skip-existing.")
        return

    print("\n=== Individual run metrics ===")
    print(results_df.to_string(index=False))

    summary_df = summarise_results(results_df)
    if not summary_df.empty:
        print("\n=== Aggregated MSE summary ===")
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)))

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"\nAggregated summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
