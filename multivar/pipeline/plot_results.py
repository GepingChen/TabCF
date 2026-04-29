#!/usr/bin/env python3
"""Create the official multi-DGP Wasserstein visualization for the paper."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multivar.core.dgp import normalize_dgp_code


DEFAULT_DGP_CODES = (
    "DGP1_LINEAR",
    "DGP3_PRE_ADDITIVE",
    "DGP4_PIECEWISE",
    "DGP5_SOFTPLUS",
)
PLOT_FONT_FAMILY = "Latin Modern Roman"
FONT_SCALE = 1.42
FIG_LEFT = 0.067
SUPYLABEL_X = 0.018


def _register_latin_modern_fonts() -> None:
    tectonic_cache = Path.home() / ".cache" / "Tectonic" / "bundles" / "data"
    if not tectonic_cache.exists():
        return
    for font_path in sorted(tectonic_cache.glob("*/lmroman*.otf")):
        font_manager.fontManager.addfont(str(font_path))


SERIES_SPECS = (
    ("core_vs_oracle_mean", "core_vs_oracle_std", "TabCF (TabPFNv2.5)", "#1f77b4", "-"),
    ("tabicl_vs_oracle_mean", "tabicl_vs_oracle_std", "TabCF (TabICLv2)", "#17becf", "-"),
    ("tabpfn_naive_vs_oracle_mean", "tabpfn_naive_vs_oracle_std", "TabPFN-naive", "#d55e00", "--"),
    ("independence_vs_oracle_mean", "independence_vs_oracle_std", "Independence", "#009e73", "-."),
    ("div_vs_oracle_mean", "div_vs_oracle_std", "DIV", "#cc79a7", ":"),
)

DGP_LABELS = {
    "DGP1_LINEAR": "Linear",
    "DGP3_PRE_ADDITIVE": "Nonlinear",
    "DGP4_PIECEWISE": "Piecewise",
    "DGP5_SOFTPLUS": "Softplus",
}


def _rho_tag(value: float) -> str:
    sign = "m" if value < 0 else ""
    text = f"{abs(value):.6f}".rstrip("0").rstrip(".")
    text = text.replace(".", "p")
    return f"{sign}{text or '0'}"


def _nice_upper(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(value))
    scaled = value / magnitude
    if scaled <= 1.0:
        nice = 1.0
    elif scaled <= 1.5:
        nice = 1.5
    elif scaled <= 2.0:
        nice = 2.0
    elif scaled <= 2.5:
        nice = 2.5
    elif scaled <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * magnitude


def _panel_upper(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return 1.0
    return value + max(0.01, 0.02 * value)


def _panel_lower(value: float, upper: float, *, force_zero: bool) -> float:
    if force_zero or not np.isfinite(value):
        return 0.0
    span = max(upper - value, 0.0)
    lower = value - max(0.01, 0.08 * max(span, upper))
    return max(0.0, lower)


def _load_and_filter(
    input_path: Path,
    *,
    dgp_codes: tuple[str, ...],
    n_train: int,
    rho_eps: float,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    required = {"dgp_code", "n_train", "rho_eps", "x"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {', '.join(missing)}")

    df["dgp_code"] = df["dgp_code"].map(normalize_dgp_code)
    df = df[df["dgp_code"].isin(dgp_codes)].copy()
    df = df[df["n_train"].astype(int) == int(n_train)].copy()
    df = df[np.isclose(df["rho_eps"].astype(float), float(rho_eps))].copy()
    if df.empty:
        raise ValueError(
            f"No rows found after filtering {input_path} for "
            f"dgp_codes={list(dgp_codes)}, n_train={n_train}, rho_eps={rho_eps}."
        )
    return df.sort_values(["dgp_code", "x"]).reset_index(drop=True)


def _series_for_frame(df: pd.DataFrame) -> list[tuple[str, str, str, str, str]]:
    series = []
    for mean_col, std_col, label, color, linestyle in SERIES_SPECS:
        if mean_col not in df.columns:
            continue
        if df[mean_col].isna().all():
            continue
        series.append((mean_col, std_col, label, color, linestyle))
    if not series:
        raise ValueError("No plottable metric columns were found in the filtered data.")
    return series


def _style_header_cell(ax: plt.Axes, *, facecolor: str = "#ffffff") -> None:
    ax.set_facecolor(facecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _x_tick_formatter(value: float, _: object) -> str:
    rounded = round(float(value))
    if np.isclose(value, rounded):
        return str(int(rounded))
    return f"{value:.1f}"


def make_figure(
    df: pd.DataFrame,
    *,
    dgp_codes: tuple[str, ...],
    n_train: int,
    rho_eps: float,
) -> tuple[plt.Figure, np.ndarray]:
    series = _series_for_frame(df)

    _register_latin_modern_fonts()
    font_path = font_manager.findfont(PLOT_FONT_FAMILY, fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.serif": [PLOT_FONT_FAMILY],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 11 * FONT_SCALE,
            "axes.labelsize": 13.5 * FONT_SCALE,
            "axes.titlesize": 13.0 * FONT_SCALE,
            "legend.fontsize": 10.5 * FONT_SCALE,
            "xtick.labelsize": 10.0 * FONT_SCALE,
            "ytick.labelsize": 10.0 * FONT_SCALE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    print(f"Using plot font: {PLOT_FONT_FAMILY} ({font_path})")

    fig = plt.figure(figsize=(15.2, 5.65), facecolor="white")
    fig.patch.set_edgecolor("none")
    fig.patch.set_linewidth(0.0)
    grid = fig.add_gridspec(
        2,
        len(dgp_codes),
        left=FIG_LEFT,
        right=0.995,
        bottom=0.19,
        top=0.775,
        hspace=0.10,
        wspace=0.14,
        height_ratios=[0.5, 4.8],
    )
    axes: list[plt.Axes] = []

    x_all = df["x"].to_numpy(dtype=float)
    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))
    xticks = np.arange(x_min, x_max + 1e-9, 0.5)

    legend_handles = []
    legend_labels = []

    for idx, dgp_code in enumerate(dgp_codes):
        header_ax = fig.add_subplot(grid[0, idx])
        _style_header_cell(header_ax)
        header_ax.text(
            0.5,
            0.5,
            DGP_LABELS.get(dgp_code, dgp_code),
            ha="center",
            va="center",
            fontsize=12.2 * FONT_SCALE,
            fontweight="bold",
            color="#2f2f2f",
        )

        ax = fig.add_subplot(grid[1, idx], sharex=axes[0] if axes else None)
        axes.append(ax)
        g = df[df["dgp_code"] == dgp_code].sort_values("x")
        if g.empty:
            raise ValueError(f"Filtered data is missing required DGP row: {dgp_code}")

        x = g["x"].to_numpy(dtype=float)
        upper_max = 0.0
        lower_min = math.inf
        for mean_col, std_col, label, color, linestyle in series:
            mean_vals = g[mean_col].to_numpy(dtype=float)
            std_vals = g[std_col].fillna(0.0).to_numpy(dtype=float) if std_col in g.columns else np.zeros_like(mean_vals)
            upper = mean_vals + std_vals
            lower = np.maximum(0.0, mean_vals - std_vals)
            upper_max = max(upper_max, float(np.max(upper)))
            lower_min = min(lower_min, float(np.min(lower)))
            (line,) = ax.plot(
                x,
                mean_vals,
                color=color,
                linewidth=2.2,
                linestyle=linestyle,
                label=label,
                solid_capstyle="round",
            )
            ax.fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=0.13,
                linewidth=0.0,
                zorder=1,
            )
            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(label)

        y_max = _panel_upper(upper_max)
        y_min = _panel_lower(lower_min, y_max, force_zero=(dgp_code != "DGP3_PRE_ADDITIVE"))
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(xticks)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.xaxis.set_major_formatter(FuncFormatter(_x_tick_formatter))
        ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.8)
        ax.grid(axis="x", alpha=0.10, linestyle=":", linewidth=0.7)
        ax.set_facecolor("#ffffff")
        ax.tick_params(axis="x", labelsize=9.8 * FONT_SCALE, length=0, colors="#4a4a46", pad=7)
        ax.tick_params(axis="y", labelsize=9.8 * FONT_SCALE, length=0, colors="#4a4a46", pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.supylabel("Wasserstein Distance", x=SUPYLABEL_X, fontsize=13.5 * FONT_SCALE)
    fig.supxlabel("Intervention value x", y=0.072, fontsize=13.5 * FONT_SCALE)
    if legend_handles:
        legend = fig.legend(
            legend_handles,
            legend_labels,
            ncol=min(len(legend_handles), 3),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            frameon=True,
            fancybox=False,
            edgecolor="#5d5d57",
            handlelength=2.8,
            columnspacing=1.35,
            borderpad=0.72,
            labelspacing=0.72,
            handletextpad=0.58,
            fontsize=10.2 * FONT_SCALE,
        )
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_linewidth(0.85)
        legend.get_frame().set_facecolor("#ffffff")

    return fig, np.asarray(axes, dtype=object)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the official 1x4 Wasserstein figure for the paper.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/aggregated_csv/multivar/multivar_wasserstein_curves.csv"),
        help="Input aggregated case-x CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/paper_figures/reproduced/multivar_wasserstein_figure.png"),
        help="Main output image path. A PDF with the same stem is also written.",
    )
    parser.add_argument(
        "--dgp-codes",
        nargs="+",
        default=list(DEFAULT_DGP_CODES),
        help="DGP codes to include, in row order.",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=2000,
        help="Training sample size to visualize.",
    )
    parser.add_argument(
        "--rho-eps",
        type=float,
        default=0.6,
        help="rho_eps value to visualize.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="PNG DPI.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dgp_codes = tuple(normalize_dgp_code(code) for code in args.dgp_codes)
    df = _load_and_filter(
        args.input,
        dgp_codes=dgp_codes,
        n_train=args.n_train,
        rho_eps=args.rho_eps,
    )

    fig, _ = make_figure(
        df,
        dgp_codes=dgp_codes,
        n_train=args.n_train,
        rho_eps=args.rho_eps,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")

    pdf_output = args.output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded {len(df)} filtered rows from {args.input}")
    print(f"Wrote PNG: {args.output}")
    print(f"Wrote PDF: {pdf_output}")
    print(
        "Figure spec: "
        f"dgp_codes={list(dgp_codes)}, n_train={args.n_train}, rho_eps={args.rho_eps}, "
        "layout=1x4_with_header_row, shared_x=True, shared_y=False"
    )


if __name__ == "__main__":
    main()
