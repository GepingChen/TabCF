#!/usr/bin/env python3
"""
Residual sanity-check plots for sec5.1 baselines.

This script gathers per-sample predictions (if available) for a single seed
across the three DGP codes (A3_B3, A3_B4, A3_B5) and training sizes
(1000, 4000, 10000). For each combination, it plots residuals
(prediction - mean_int) for the selected methods.

Requirements: pandas, matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# Repository-relative defaults.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "interv_mean" / "interv_mean" / "io" / "results"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "residual_seed1.png"
DEFAULT_CODES = ["A3_B3", "A3_B4", "A3_B5"]
DEFAULT_TRAIN_SIZES = [1000, 4000, 10000]

# Methods we know how to read. DeepIV/DeepGMM entries are included so the script
# can gracefully report missing predictions (they were not saved in past runs).
METHOD_CFG: Dict[str, Dict[str, object]] = {
    "div": {"label": "DIV", "pred_col": "pred", "color": "#8c564b"},
    "tabpfn_cf": {"label": "TabPFN-CF", "pred_col": "pred", "color": "#9b59b6"},
    "hsic": {"label": "HSIC-X", "pred_col": "pred_mean", "color": "#f28e2b"},
    "deepiv": {"label": "DeepIV", "pred_col": "pred", "color": "#f1c232"},
    "deepgmm": {"label": "DeepGMM", "pred_col": "pred", "color": "#6ab187"},
    "linear_iv": {"label": "Linear CF", "pred_col": "pred", "color": "#3b8adb"},
    "nonlinear_iv": {"label": "Nonlinear CF", "pred_col": "pred", "color": "#ff7f0e"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot residuals (pred - mean_int) for sec5.1 baselines."
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed to load predictions for.")
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
        "--methods",
        nargs="+",
        default=["div", "tabpfn_cf", "hsic", "deepiv", "deepgmm", "linear_iv", "nonlinear_iv"],
        help=f"Methods to plot (subset of {list(METHOD_CFG.keys())}).",
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
        default=DEFAULT_OUTPUT,
        help="Output image path (default: %(default)s).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Max points per method/combination to scatter (for readability).",
    )
    return parser.parse_args()


def _resolve_pred_path(results_dir: Path, model: str, code: str, size: int, seed: int) -> Path:
    return results_dir / f"{model}_{code}_n{size}_seed{seed}_pred.csv"


def _load_predictions(
    path: Path, pred_col: str
) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required_cols = {"Xint", "mean_int", pred_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(sorted(missing))}")
    df = df.loc[:, ["Xint", "mean_int", pred_col]].rename(columns={pred_col: "pred"})
    df["residual"] = df["pred"] - df["mean_int"]
    return df


def _make_axes(codes: List[str], train_sizes: List[int]) -> Tuple[plt.Figure, List[List[plt.Axes]]]:
    fig, axes = plt.subplots(
        nrows=len(codes),
        ncols=len(train_sizes),
        figsize=(4 * len(train_sizes), 3 * len(codes)),
        sharex=True,
        sharey=True,
    )
    # Normalize axes to 2D list for consistent indexing when len==1.
    if len(codes) == 1:
        axes = [axes]
    if len(train_sizes) == 1:
        axes = [[ax] for ax in axes]
    return fig, axes


def plot_residuals(
    seed: int,
    codes: Iterable[str],
    train_sizes: Iterable[int],
    methods: Iterable[str],
    results_dir: Path,
    output_path: Path,
    max_points: int,
) -> None:
    codes_list = list(codes)
    size_list = list(train_sizes)
    methods_list = [m for m in methods if m in METHOD_CFG]

    fig, axes = _make_axes(codes_list, size_list)
    legend_handles = {}
    missing_files: List[str] = []

    for r_idx, code in enumerate(codes_list):
        for c_idx, size in enumerate(size_list):
            ax = axes[r_idx][c_idx]
            for method in methods_list:
                cfg = METHOD_CFG[method]
                pred_col = cfg["pred_col"]  # type: ignore[index]
                label = cfg["label"]  # type: ignore[index]
                color = cfg["color"]  # type: ignore[index]

                pred_path = _resolve_pred_path(results_dir, method, code, size, seed)
                df = _load_predictions(pred_path, pred_col) if pred_path.exists() else None
                if df is None:
                    missing_files.append(str(pred_path))
                    continue

                if len(df) > max_points:
                    df = df.sample(n=max_points, random_state=seed)

                ax.scatter(df["Xint"], df["residual"], s=8, alpha=0.25, color=color, label=label)
                if label not in legend_handles:
                    legend_handles[label] = plt.Line2D(
                        [0], [0], marker="o", color="none", markerfacecolor=color, markersize=6, label=label
                    )

            ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
            ax.set_title(f"{code}, n={size}")
            if c_idx == 0:
                ax.set_ylabel("Residual (pred - mean_int)")
            if r_idx == len(codes_list) - 1:
                ax.set_xlabel("Xint")

    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="upper center",
            ncol=len(legend_handles),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    print(f"Saved residual grid to {output_path}")

    if missing_files:
        print("\n⚠️ Missing prediction files (skip):")
        for path in missing_files:
            print(f"  {path}")
        print(
            "DeepIV/DeepGMM predictions were not saved in the existing runs; "
            "rerun run_py_baselines.py with --save-predictions to include them."
        )


def main() -> None:
    args = parse_args()
    plot_residuals(
        seed=args.seed,
        codes=args.codes,
        train_sizes=args.train_sizes,
        methods=args.methods,
        results_dir=args.results_dir,
        output_path=args.output,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()
