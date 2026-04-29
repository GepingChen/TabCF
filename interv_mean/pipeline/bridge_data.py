#!/usr/bin/env python3
"""Bridge TabPFN datasets into DIV sec5.1 CSV format."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tabcf_core.dgp_test_utils import (
    DEFAULT_TEST_X_TRIM_QUANTILE_RANGE,
    normalize_test_x_trim_quantile_range,
    trim_test_dataframe_by_x,
)
from interv_mean.pipeline import utils


def _load_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    data = utils.load_manifest(path)
    runs = data.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"Malformed manifest at {path}: missing 'runs' list")
    return data


def _filter_runs(
    runs: Iterable[Dict[str, object]],
    codes: Optional[List[str]],
    sizes: Optional[List[int]],
    seeds: Optional[List[int]],
) -> List[Dict[str, object]]:
    filtered: List[Dict[str, object]] = []
    for run in runs:
        if codes and str(run["code"]) not in codes:
            continue
        if sizes and int(run["train_size"]) not in sizes:
            continue
        if seeds and int(run["seed"]) not in seeds:
            continue
        filtered.append(run)
    return filtered


def _ensure_required_columns(df: pd.DataFrame, required: List[str], path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")


def _compute_ground_truth_for_code(
    code: str,
    cfg_seed: int,
    test_df: pd.DataFrame,
    y_samples: int,
    rng_seed: int,
) -> Dict[str, np.ndarray]:
    cfg = utils.make_dgp_config(code, seed=cfg_seed)
    xint = test_df["X"].to_numpy()
    mean_int = utils.compute_y_clean_monte_carlo(cfg, xint, n_samples=y_samples, rng_seed=rng_seed)
    x_grid = np.sort(xint)
    mean_grid = utils.compute_y_clean_monte_carlo(cfg, x_grid, n_samples=y_samples, rng_seed=rng_seed)
    mean_grid_unconf = utils.compute_y_clean_monte_carlo(
        cfg, x_grid, n_samples=y_samples, rng_seed=rng_seed, force_h_zero=True
    )
    y_obs = test_df["Y"].to_numpy()
    return {
        "Xint": xint,
        "mean_int": mean_int,
        "Xtest_grid": x_grid,
        "mean_grid": mean_grid,
        "mean_grid_unconfounded": mean_grid_unconf,
        "Y": y_obs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge TabPFN outputs into DIV sec5.1 format.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=utils.DEFAULT_MANIFEST_PATH,
        help="Path to v2 manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=utils.DEFAULT_BRIDGE_DIR,
        help="Output directory for bridged CSVs.",
    )
    parser.add_argument(
        "--y-clean-samples",
        type=int,
        default=5000,
        help="Monte Carlo samples per X when computing mean_int.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=1,
        help="Seed for Monte Carlo draws.",
    )
    parser.add_argument(
        "--test-x-trim-quantiles",
        nargs=2,
        type=float,
        default=list(DEFAULT_TEST_X_TRIM_QUANTILE_RANGE),
        metavar=("LOWER_Q", "UPPER_Q"),
        help=(
            "Empirical X-rank range to retain from test_data_*.csv before bridging "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--disable-test-x-trim",
        action="store_true",
        help="Disable X-rank trimming and bridge the full test_data_*.csv rows.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing bridged CSVs.")
    parser.add_argument("--codes", nargs="+", default=None, help="Optional subset of DGP codes.")
    parser.add_argument("--sizes", nargs="+", type=int, default=None, help="Optional subset of train sizes.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional subset of seeds.")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    runs = _filter_runs(manifest["runs"], args.codes, args.sizes, args.seeds)
    if not runs:
        raise SystemExit("No runs matched the provided filters.")

    test_x_trim_quantile_range = None
    if not args.disable_test_x_trim:
        test_x_trim_quantile_range = normalize_test_x_trim_quantile_range(args.test_x_trim_quantiles)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_cache: Dict[str, Dict[str, np.ndarray]] = {}
    for idx, run in enumerate(runs, start=1):
        code = str(run["code"])
        size = int(run["train_size"])
        seed = int(run["seed"])

        resolved = utils.resolve_run_paths(run, manifest=manifest, bridge_dir=args.output_dir)
        train_path = resolved["train"]
        test_path = resolved["test"]
        bridge_train_path = resolved["bridge_train"]
        bridge_test_path = resolved["bridge_test"]
        assert train_path is not None and test_path is not None
        assert bridge_train_path is not None and bridge_test_path is not None

        if not args.overwrite and bridge_train_path.exists() and bridge_test_path.exists():
            print(f"[{idx}/{len(runs)}] Skipping {code}, n={size}, seed={seed} (already exists).")
            continue

        print(f"[{idx}/{len(runs)}] Bridging {code}, n={size}, seed={seed}")

        train_df = pd.read_csv(train_path)
        _ensure_required_columns(train_df, ["H", "Z", "X", "Y"], train_path)
        bridged_train = train_df[["H", "Z", "X", "Y"]].copy()
        bridged_train.to_csv(bridge_train_path, index=False)

        if code not in ground_truth_cache:
            test_df = pd.read_csv(test_path)
            _ensure_required_columns(test_df, ["X", "Y"], test_path)
            test_df = trim_test_dataframe_by_x(test_df, test_x_trim_quantile_range, x_col="X")
            ground_truth_cache[code] = _compute_ground_truth_for_code(
                code, seed, test_df, args.y_clean_samples, args.rng_seed
            )

        gt = ground_truth_cache[code]
        bridged_test = pd.DataFrame(
            {
                "Xint": gt["Xint"],
                "Yint": gt["Y"],
                "Yint2": gt["Y"],
                "mean_int": gt["mean_int"],
                "Xtest_grid": gt["Xtest_grid"],
                "mean_grid": gt["mean_grid"],
                "mean_grid_unconfounded": gt["mean_grid_unconfounded"],
            }
        )
        bridged_test.to_csv(bridge_test_path, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
