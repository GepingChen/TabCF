#!/usr/bin/env python3
"""
Trim extremes of X for prediction CSVs and recompute MSE.

For each method/code/train_size/seed combination, this script:
  1) Loads the corresponding *_pred.csv (same naming convention as check_results.py).
  2) Computes the untrimmed MSE vs mean_int (reference).
  3) Sorts by Xint and drops the first/last trim_pct% of rows (default 5%).
  4) Computes the trimmed MSE on the retained rows.
  5) Writes a summary CSV with both MSEs and sample counts.

Methods are aligned with check_results.py; missing files are skipped with a warning.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Repository-relative defaults.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "interv_mean" / "interv_mean" / "io" / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "check_results"
DEFAULT_CODES = ["A3_B3", "A3_B4", "A3_B5"]
DEFAULT_TRAIN_SIZES = [1000, 4000, 10000]

# Method metadata (consistent with check_results.py)
METHOD_CFG: Dict[str, Dict[str, str]] = {
    "div": {"pred_col": "pred"},
    "tabpfn_cf": {"pred_col": "pred"},
    "hsic": {"pred_col": "pred_mean"},
    "deepiv": {"pred_col": "pred"},
    "deepgmm": {"pred_col": "pred"},
    "linear_iv": {"pred_col": "pred"},
    "nonlinear_iv": {"pred_col": "pred"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim X extremes for prediction CSVs and recompute MSE."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_CFG.keys()),
        help=f"Methods to process (subset of {list(METHOD_CFG.keys())}).",
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        default=DEFAULT_CODES,
        help="DGP codes to include (default: %(default)s).",
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SIZES,
        help="Training sample sizes to include (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed to load predictions for (default: 1).",
    )
    parser.add_argument(
        "--trim-pct",
        type=float,
        default=5.0,
        help="Percentage to trim from each end of sorted X (default: 5%%).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing *_pred.csv files (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Default: results_dir/check_results/trim_report_seed{seed}_p{trim}.csv",
    )
    return parser.parse_args()


def _pred_path(results_dir: Path, method: str, code: str, size: int, seed: int) -> Path:
    return results_dir / f"{method}_{code}_n{size}_seed{seed}_pred.csv"


def _load_pred_df(path: Path, pred_col: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"Xint", "mean_int", pred_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(sorted(missing))}")
    df = df.loc[:, ["Xint", "mean_int", pred_col]].rename(columns={pred_col: "pred"})
    return df


def _trim_by_percentile(df: pd.DataFrame, trim_pct: float) -> pd.DataFrame:
    if trim_pct <= 0:
        return df
    if trim_pct >= 50:
        raise ValueError("trim_pct must be < 50 to retain data.")
    n = len(df)
    k = int(math.floor(n * trim_pct / 100.0))
    if k == 0:
        return df
    return df.iloc[k : n - k]


def _compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.shape != target.shape:
        raise ValueError(f"MSE shape mismatch: pred {pred.shape}, target {target.shape}")
    return float(np.mean((pred - target) ** 2))


def process(
    methods: Iterable[str],
    codes: Iterable[str],
    sizes: Iterable[int],
    seed: int,
    trim_pct: float,
    results_dir: Path,
) -> List[Dict[str, object]]:
    """
    Always computes the untrimmed MSE (trim=0) as reference, and the requested trim_pct.
    """
    rows: List[Dict[str, object]] = []
    for method in methods:
        if method not in METHOD_CFG:
            print(f"⚠️  Unknown method '{method}', skipping.")
            continue
        pred_col = METHOD_CFG[method]["pred_col"]
        for code in codes:
            for size in sizes:
                path = _pred_path(results_dir, method, code, size, seed)
                df = _load_pred_df(path, pred_col)
                if df is None:
                    print(f"⚠️  Missing pred file: {path}")
                    continue
                df_sorted = df.sort_values("Xint").reset_index(drop=True)
                # Reference: no trim
                mse_full = _compute_mse(df_sorted["pred"].to_numpy(), df_sorted["mean_int"].to_numpy())
                rows.append(
                    {
                        "method": method,
                        "code": code,
                        "train_size": size,
                        "seed": seed,
                        "trim_pct": 0.0,
                        "n_full": len(df_sorted),
                        "n_trim": len(df_sorted),
                        "mse_full": mse_full,
                        "mse_trim": mse_full,
                        "pred_path": str(path),
                    }
                )
                # Trimmed version
                if trim_pct > 0:
                    df_trim = _trim_by_percentile(df_sorted, trim_pct)
                    mse_trim = _compute_mse(df_trim["pred"].to_numpy(), df_trim["mean_int"].to_numpy())
                    rows.append(
                        {
                            "method": method,
                            "code": code,
                            "train_size": size,
                            "seed": seed,
                            "trim_pct": trim_pct,
                            "n_full": len(df_sorted),
                            "n_trim": len(df_trim),
                            "mse_full": mse_full,
                            "mse_trim": mse_trim,
                            "pred_path": str(path),
                        }
                    )
    return rows


def main() -> None:
    args = parse_args()
    out_path = (
        args.output
        if args.output is not None
        else DEFAULT_OUTPUT_DIR / f"trim_report_seed{args.seed}_p{args.trim_pct:.2f}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = process(
        methods=args.methods,
        codes=args.codes,
        sizes=args.train_sizes,
        seed=args.seed,
        trim_pct=args.trim_pct,
        results_dir=args.results_dir,
    )

    if not rows:
        raise SystemExit("No rows produced (missing files or empty selection).")

    df_out = pd.DataFrame(rows).sort_values(["method", "code", "train_size"])
    df_out.to_csv(out_path, index=False)

    print("\n=== Trim report ===")
    print(df_out.to_string(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
