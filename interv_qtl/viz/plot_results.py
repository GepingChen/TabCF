#!/usr/bin/env python3
"""Plot the paper-facing interv_qtl RMSE view from shipped report CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "aggregated_csv" / "interv_qtl" / "quantile_median_curve_comparison.csv"
DEFAULT_CODES = ("A3_B4", "A3_B5", "A3_B11", "A9_B4", "A9_B5", "A9_B11")


def default_input_path() -> Path:
    return DEFAULT_REPORT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input_path())
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "interv_qtl" / "figures" / "quantile_rmse_figure.png")
    parser.add_argument("--codes", nargs="+", default=list(DEFAULT_CODES))
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def load_report(path: Path, *, codes: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing report CSV: {path}")
    df = pd.read_csv(path)
    required = {"code", "tau", "rmse_new_median_seed"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    filtered = df[df["code"].isin(codes)].copy()
    if filtered.empty:
        raise ValueError(f"No rows matched requested codes {list(codes)} in {path}")
    return filtered.sort_values(["code", "tau"]).reset_index(drop=True)


def make_figure(df: pd.DataFrame, *, codes: tuple[str, ...]) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes_arr = plt.subplots(2, 3, figsize=(12.8, 7.4), facecolor="white")
    axes = list(axes_arr.reshape(-1))
    for ax, code in zip(axes, codes):
        panel = df[df["code"] == code].sort_values("tau")
        if panel.empty:
            ax.set_visible(False)
            continue
        ax.plot(
            panel["tau"],
            panel["rmse_new_median_seed"],
            color="#d1495b",
            linewidth=2.1,
            marker="o",
            markersize=4.6,
            markerfacecolor="white",
            markeredgewidth=1.1,
        )
        ax.set_title(code, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("tau")
        ax.set_ylabel("Median RMSE")
        ax.grid(axis="y", alpha=0.28, linestyle="--", linewidth=0.8)
        ax.grid(axis="x", alpha=0.16, linestyle=":", linewidth=0.7)
        ax.set_facecolor("#ffffff")
    fig.suptitle("Interventional-Quantile RMSE from Official Comparison Report", fontsize=13, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig, axes


def save_figure(fig: plt.Figure, output: Path, *, dpi: int) -> tuple[Path, Path]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    png_path = output.with_suffix(".png")
    pdf_path = output.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    codes = tuple(args.codes)
    df = load_report(args.input.resolve(), codes=codes)
    fig, _axes = make_figure(df, codes=codes)
    png_path, pdf_path = save_figure(fig, args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved PNG to: {png_path}")
    print(f"Saved PDF to: {pdf_path}")


if __name__ == "__main__":
    main()
