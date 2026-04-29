#!/usr/bin/env python3
"""Create the paper-facing RMSE line figure for interv_qtl.

This official view keeps the quantile-RMSE line semantics from
`plot_rmse_boxplot.py`, but adopts the Section 5.1 DIV-style outer figure
language:

- only the interior quantiles with 0.1 <= tau <= 0.9 are shown;
- only the n=4000 regime is plotted;
- the deprecated `tabpfn_real` backend is excluded;
- each method is rendered as a line over tau using median seed-wise RMSE;
- the outer layout uses stage-aware row/column headers instead of per-panel titles.

Examples
--------
Default six-DGP official figure:
    python interv_qtl/viz/plot_rmse_boxplot_official.py

Explicit fixed code order:
    python interv_qtl/viz/plot_rmse_boxplot_official.py \
        --codes A3_B4 A3_B5 A3_B11 A9_B4 A9_B5 A9_B11
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from plot_rmse_boxplot import (  # noqa: E402
    _resolve_methods_and_colors,
    collect_records,
)


OFFICIAL_TRAIN_SIZE = 4000
DEFAULT_STAGE2_DIR = (
    REPO_ROOT
    / "interv_qtl"
    / "IV_datasets"
    / "stage2_output_official_tau9_xg0595_fullpipeline_20260415T001105"
)
LEGACY_STAGE2_DIR = REPO_ROOT / "interv_qtl" / "IV_datasets" / "stage2_output"
EXCLUDED_METHOD_KEYS = frozenset({"tabpfn_real"})
OFFICIAL_BOOTSTRAP_REPS = 2000
OFFICIAL_CI_LEVEL = 0.95
OFFICIAL_RNG_SEED = 20260413
OFFICIAL_TAUS: Tuple[float, ...] = tuple(round(tau, 1) for tau in np.arange(0.1, 1.0, 0.1))
OFFICIAL_METHOD_KEY_COLORS: Dict[str, str] = {
    "tabpfn": "#d62728",
    "tabicl": "#17becf",
    "div": "#4d8c57",
    "ivqr": "#c76aa9",
    "dopfn": "#d62728",
}
DEFAULT_OFFICIAL_CODES: Tuple[str, ...] = (
    "A3_B4",
    "A3_B5",
    "A3_B11",
    "A9_B4",
    "A9_B5",
    "A9_B11",
)
DEFAULT_FIRST_STAGE_ORDER: Tuple[str, ...] = ("A3", "A9")
DEFAULT_SECOND_STAGE_ORDER: Tuple[str, ...] = ("B4", "B5", "B11")
FIRST_STAGE_TABLE_LABELS: Dict[str, str] = {
    "A3": "Linear,\nAdditive",
    "A9": "Nonlinear,\nNonadditive",
}
SECOND_STAGE_TABLE_LABELS: Dict[str, str] = {
    "B4": "Piecewise",
    "B5": "Periodic",
    "B11": "Periodic,\nNonadditive",
}
PLOT_FONT_FAMILY = "Latin Modern Roman"
FONT_SCALE = 1.42


def _register_latin_modern_fonts() -> None:
    tectonic_cache = Path.home() / ".cache" / "Tectonic" / "bundles" / "data"
    if not tectonic_cache.exists():
        return
    for font_path in sorted(tectonic_cache.glob("*/lmroman*.otf")):
        font_manager.fontManager.addfont(str(font_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the official n=4000 RMSE line plot for the paper."
    )
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=DEFAULT_STAGE2_DIR if DEFAULT_STAGE2_DIR.exists() else LEGACY_STAGE2_DIR,
        help=(
            "Directory containing s2q*_summary.csv files. Defaults to the trimmed "
            "test-X official summaries under "
            "interv_qtl/IV_datasets/stage2_output_official_tau9_xg0595_fullpipeline_20260415T001105 "
            "when available."
        ),
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        default=None,
        help=(
            "Explicit DGP codes to plot. Defaults to the official six-DGP order "
            "A3_B4 A3_B5 A3_B11 A9_B4 A9_B5 A9_B11."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output figure path. Defaults to draft/figs_final/"
            "rmse_boxplot_A3A9_B11_4000_combined.pdf when available, "
            "otherwise interv_qtl/figures/."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="Raster DPI when saving PNG output.",
    )
    return parser.parse_args()


def _normalize_codes(raw_codes: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for raw_code in raw_codes:
        for part in str(raw_code).split(","):
            code = part.strip()
            if not code or code in seen:
                continue
            seen.add(code)
            normalized.append(code)
    return normalized


def _resolve_codes(args: argparse.Namespace) -> List[str]:
    if not args.codes:
        return list(DEFAULT_OFFICIAL_CODES)

    codes = _normalize_codes(args.codes)
    if not codes:
        raise SystemExit("No valid DGP codes were supplied via --codes.")
    return codes


def _default_output_dir() -> Path:
    preferred = REPO_ROOT / "draft" / "figs_final"
    if preferred.exists():
        return preferred
    return REPO_ROOT / "interv_qtl" / "figures"


def _default_output_path(codes: Sequence[str]) -> Path:
    output_dir = _default_output_dir()
    codes_tuple = tuple(codes)
    if codes_tuple == DEFAULT_OFFICIAL_CODES:
        stem = "rmse_boxplot_A3A9_B11_4000_combined"
    else:
        suffix = "_".join(codes)
        if len(suffix) > 80:
            suffix = f"selected_{len(codes)}dgps"
        stem = f"rmse_boxplot_official_{suffix}_n{OFFICIAL_TRAIN_SIZE}"
    return output_dir / f"{stem}.pdf"


def _filter_official_slice(df: pd.DataFrame) -> pd.DataFrame:
    tau_vals = df["tau"].astype(float)
    tau_mask = tau_vals.between(0.1 - 1e-12, 0.9 + 1e-12)
    train_mask = df["train_size"].astype(int) == OFFICIAL_TRAIN_SIZE
    return df.loc[tau_mask & train_mask].copy()


def _filter_official_methods(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "method_key" not in df.columns:
        return df.copy()
    method_keys = df["method_key"].astype(str).str.lower()
    return df.loc[~method_keys.isin(EXCLUDED_METHOD_KEYS)].copy()


def _warn_if_missing_official_taus(df: pd.DataFrame) -> None:
    available = np.asarray(
        sorted(float(tau) for tau in df["tau"].dropna().unique()),
        dtype=float,
    )
    missing = [
        tau
        for tau in OFFICIAL_TAUS
        if not np.any(np.isclose(available, tau, rtol=0.0, atol=1e-10))
    ]
    if missing:
        formatted = ", ".join(f"{tau:.3g}" for tau in missing)
        print(
            "Warning: missing official interior taus in the loaded summaries. "
            f"Empty x positions will be kept for: [{formatted}]."
        )


def _parse_code_stages(code: str) -> Tuple[str, str]:
    match = re.match(r"^(A\d+)_?(B\d+)$", code)
    if not match:
        raise SystemExit(
            f"Could not infer first/second-stage labels from code '{code}'. "
            "Expected codes like A3_B4."
        )
    return match.group(1), match.group(2)


def _natural_stage_key(token: str) -> Tuple[str, int, str]:
    match = re.match(r"([A-Za-z]+)(\d+)$", token)
    if not match:
        return (token, sys.maxsize, token)
    return (match.group(1), int(match.group(2)), token)


def _ordered_stage_values(
    values: Sequence[str],
    preferred_order: Sequence[str],
) -> List[str]:
    value_set = set(values)
    ordered = [value for value in preferred_order if value in value_set]
    ordered.extend(
        sorted((value for value in value_set if value not in set(ordered)), key=_natural_stage_key)
    )
    return ordered


def _resolve_grid_orders(codes: Sequence[str]) -> Tuple[List[str], List[str]]:
    first_stages = []
    second_stages = []
    for code in codes:
        first_stage, second_stage = _parse_code_stages(str(code))
        if first_stage not in first_stages:
            first_stages.append(first_stage)
        if second_stage not in second_stages:
            second_stages.append(second_stage)
    ordered_first = _ordered_stage_values(first_stages, DEFAULT_FIRST_STAGE_ORDER)
    ordered_second = _ordered_stage_values(second_stages, DEFAULT_SECOND_STAGE_ORDER)
    return ordered_first, ordered_second


def _code_lookup(codes: Sequence[str]) -> Dict[Tuple[str, str], str]:
    lookup: Dict[Tuple[str, str], str] = {}
    for code in codes:
        lookup[_parse_code_stages(str(code))] = str(code)
    return lookup


def _resolve_official_method_colors(
    df: pd.DataFrame,
    methods: Sequence[str],
    fallback_colors: Mapping[str, str],
) -> Dict[str, str]:
    method_colors: Dict[str, str] = {}
    for method in methods:
        method_key = str(df.loc[df["method"] == method, "method_key"].iloc[0]).lower()
        method_colors[method] = OFFICIAL_METHOD_KEY_COLORS.get(
            method_key,
            fallback_colors.get(method, "#999999"),
        )
    return method_colors


def _method_marker(method_index: int) -> str:
    markers = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
    return markers[method_index % len(markers)]


def _bootstrap_median_interval(
    vals: np.ndarray,
    *,
    rng: np.random.Generator,
    ci_level: float = OFFICIAL_CI_LEVEL,
    reps: int = OFFICIAL_BOOTSTRAP_REPS,
) -> Tuple[float, float, float]:
    if vals.size == 0:
        return (np.nan, np.nan, np.nan)

    median = float(np.median(vals))
    if vals.size == 1:
        return (median, median, median)

    sample_idx = rng.integers(0, vals.size, size=(reps, vals.size))
    boot_medians = np.median(vals[sample_idx], axis=1)
    alpha = 1.0 - ci_level
    lower = float(np.quantile(boot_medians, alpha / 2.0))
    upper = float(np.quantile(boot_medians, 1.0 - alpha / 2.0))
    return (median, lower, upper)


def _summarize_rmse_by_tau(
    df: pd.DataFrame,
    *,
    method: str,
    taus: Sequence[float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.full(len(taus), np.nan, dtype=float)
    lowers = np.full(len(taus), np.nan, dtype=float)
    uppers = np.full(len(taus), np.nan, dtype=float)
    tau_values = df["tau"].to_numpy(dtype=float)
    for idx, tau in enumerate(taus):
        tau_mask = np.isclose(tau_values, float(tau), rtol=0.0, atol=1e-10)
        vals = (
            df.loc[(df["method"] == method) & tau_mask, "rmse"]
            .dropna()
            .to_numpy(dtype=float)
        )
        median, lower, upper = _bootstrap_median_interval(vals, rng=rng)
        medians[idx] = median
        lowers[idx] = lower
        uppers[idx] = upper
    return medians, lowers, uppers


def _line_legend_handles(
    methods: Sequence[str],
    method_colors: Mapping[str, str],
) -> List[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=method_colors.get(method, "#999999"),
            linewidth=2.2,
            marker=_method_marker(idx),
            markersize=5.4,
            markerfacecolor="white",
            markeredgewidth=1.25,
            label=method,
        )
        for idx, method in enumerate(methods)
    ]


def _style_header_cell(ax, *, facecolor: str = "#ffffff") -> None:
    ax.set_facecolor(facecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _style_data_panel(ax, *, show_ylabel: bool, show_xlabel: bool) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, axis="y", color="#d9ddd6", linewidth=0.8, alpha=0.95)
    ax.grid(True, axis="x", color="#eceee9", linewidth=0.75, alpha=0.95)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9.8 * FONT_SCALE, length=0, pad=6, colors="#4a4a46")
    ax.tick_params(axis="x", labelsize=9.8 * FONT_SCALE, length=0, pad=6, colors="#4a4a46")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)


def _set_sparse_y_ticks(ax) -> None:
    y_min, y_max = ax.get_ylim()
    if y_max <= 1.05:
        step = 0.2
    elif y_max <= 2.05:
        step = 0.4
    else:
        step = 0.8
    ax.set_yticks(np.arange(0.0, y_max + 1e-12, step))
    ax.set_ylim(y_min, y_max)


def _draw_empty_panel(ax, taus: Sequence[float], *, show_ylabel: bool, show_xlabel: bool) -> None:
    positions = np.arange(len(taus), dtype=float)
    _style_data_panel(ax, show_ylabel=show_ylabel, show_xlabel=show_xlabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{tau:.3g}" for tau in taus],
        rotation=45,
        ha="right",
        fontsize=9.8 * FONT_SCALE,
    )
    ax.text(
        0.5,
        0.5,
        "No data",
        ha="center",
        va="center",
        fontsize=11.0 * FONT_SCALE,
        color="#8b8b86",
        transform=ax.transAxes,
    )
    _set_sparse_y_ticks(ax)


def _lineplot_grouped(
    ax,
    df: pd.DataFrame,
    *,
    methods: Sequence[str],
    taus: Sequence[float],
    method_colors: Mapping[str, str],
    rng: np.random.Generator,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    base_positions = np.arange(len(taus), dtype=float)

    for method_index, method in enumerate(methods):
        rmse_series, lower_band, upper_band = _summarize_rmse_by_tau(
            df,
            method=method,
            taus=taus,
            rng=rng,
        )
        valid_mask = np.isfinite(rmse_series)
        if not np.any(valid_mask):
            continue
        band_mask = np.isfinite(lower_band) & np.isfinite(upper_band)
        if np.any(band_mask):
            ax.fill_between(
                base_positions[band_mask],
                lower_band[band_mask],
                upper_band[band_mask],
                color=method_colors.get(method, "#999999"),
                alpha=0.14,
                linewidth=0.0,
                zorder=1,
            )
        ax.plot(
            base_positions[valid_mask],
            rmse_series[valid_mask],
            color=method_colors.get(method, "#999999"),
            linewidth=2.2,
            marker=_method_marker(method_index),
            markersize=5.4,
            markerfacecolor="white",
            markeredgewidth=1.25,
            label=method,
            zorder=2,
        )

    _style_data_panel(ax, show_ylabel=show_ylabel, show_xlabel=show_xlabel)
    ax.set_xticks(base_positions)
    ax.set_xticklabels(
        [f"{tau:.3g}" for tau in taus],
        rotation=45,
        ha="right",
        fontsize=9.8 * FONT_SCALE,
    )
    ax.margins(x=0.04, y=0.16)
    _set_sparse_y_ticks(ax)


def plot_official_rmse_lineplot(
    df: pd.DataFrame,
    *,
    codes: Sequence[str],
    output: Path,
    dpi: int = 320,
) -> None:
    methods, fallback_colors = _resolve_methods_and_colors(df)
    method_colors = _resolve_official_method_colors(df, methods, fallback_colors)
    if df.empty:
        raise SystemExit("No DGP records available for the requested official figure.")

    _warn_if_missing_official_taus(df)
    rng = np.random.default_rng(OFFICIAL_RNG_SEED)
    first_stage_order, second_stage_order = _resolve_grid_orders(codes)
    code_by_stage = _code_lookup(codes)

    if not first_stage_order or not second_stage_order:
        raise SystemExit("No plottable stage layout could be inferred from the requested codes.")

    _register_latin_modern_fonts()
    font_path = font_manager.findfont(PLOT_FONT_FAMILY, fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.serif": [PLOT_FONT_FAMILY],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "axes.labelsize": 13.5 * FONT_SCALE,
            "axes.titlesize": 13.0 * FONT_SCALE,
            "xtick.labelsize": 10.0 * FONT_SCALE,
            "ytick.labelsize": 10.0 * FONT_SCALE,
            "legend.fontsize": 10.5 * FONT_SCALE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    print(f"Using plot font: {PLOT_FONT_FAMILY} ({font_path})")

    n_treatments = len(first_stage_order)
    n_outcomes = len(second_stage_order)
    width = max(12.15, 1.55 + 3.55 * n_outcomes)
    height = max(8.45, 2.5 + 3.4 * n_treatments)

    fig = plt.figure(figsize=(width, height), facecolor="white")
    fig.patch.set_edgecolor("none")
    fig.patch.set_linewidth(0.0)
    grid = fig.add_gridspec(
        n_treatments + 1,
        n_outcomes + 1,
        left=0.04,
        right=0.994,
        bottom=0.14,
        top=0.845,
        hspace=0.055,
        wspace=0.105,
        width_ratios=[0.78] + [4.55] * n_outcomes,
        height_ratios=[0.32] + [4.16] * n_treatments,
    )

    corner_ax = fig.add_subplot(grid[0, 0])
    _style_header_cell(corner_ax, facecolor="#ffffff")

    for outcome_index, second_stage in enumerate(second_stage_order, start=1):
        header_ax = fig.add_subplot(grid[0, outcome_index])
        _style_header_cell(header_ax)
        header_ax.text(
            0.5,
            -0.5,
            SECOND_STAGE_TABLE_LABELS.get(second_stage, second_stage),
            ha="center",
            va="bottom",
            fontsize=12.2 * FONT_SCALE,
            fontweight="bold",
            color="#2f2f2f",
            wrap=True,
        )

    for treatment_index, first_stage in enumerate(first_stage_order, start=1):
        row_header_ax = fig.add_subplot(grid[treatment_index, 0])
        _style_header_cell(row_header_ax)
        row_header_ax.text(
            0.42,
            0.5,
            FIRST_STAGE_TABLE_LABELS.get(first_stage, first_stage).replace("\n", " "),
            ha="center",
            va="center",
            fontsize=11.8 * FONT_SCALE,
            fontweight="bold",
            color="#2f2f2f",
            rotation=90,
        )

        for outcome_index, second_stage in enumerate(second_stage_order, start=1):
            ax = fig.add_subplot(grid[treatment_index, outcome_index])
            code = code_by_stage.get((first_stage, second_stage))
            show_ylabel = outcome_index == 1
            show_xlabel = treatment_index == n_treatments

            if code is None:
                _draw_empty_panel(
                    ax,
                    OFFICIAL_TAUS,
                    show_ylabel=show_ylabel,
                    show_xlabel=show_xlabel,
                )
                continue

            sub = df[df["code"] == code]
            if sub.empty:
                _draw_empty_panel(
                    ax,
                    OFFICIAL_TAUS,
                    show_ylabel=show_ylabel,
                    show_xlabel=show_xlabel,
                )
                continue

            _lineplot_grouped(
                ax,
                sub,
                methods=methods,
                taus=OFFICIAL_TAUS,
                method_colors=method_colors,
                rng=rng,
                show_ylabel=show_ylabel,
                show_xlabel=show_xlabel,
            )

    legend_handles = _line_legend_handles(methods, method_colors)
    legend = fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(len(legend_handles), 5),
        frameon=True,
        fancybox=False,
        edgecolor="#5d5d57",
        fontsize=10.2 * FONT_SCALE,
        borderpad=0.72,
        labelspacing=0.72,
        columnspacing=1.0,
        handletextpad=0.7,
        handlelength=1.9,
    )
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_linewidth(0.85)
    legend.get_frame().set_facecolor("#ffffff")

    fig.supylabel("RMSE", fontsize=13.5 * FONT_SCALE, x=0.0295, color="#2f2f2f")
    fig.supxlabel(r"Quantile level $\tau$", fontsize=13.5 * FONT_SCALE, y=0.058)

    output.parent.mkdir(parents=True, exist_ok=True)
    savefig_kwargs: Dict[str, object] = {
        "bbox_inches": "tight",
        "pad_inches": 0.02,
        "facecolor": "white",
        "edgecolor": "none",
    }
    if output.suffix.lower() != ".pdf":
        savefig_kwargs["dpi"] = dpi
    fig.savefig(output, **savefig_kwargs)
    plt.close(fig)
    print(f"Saved official RMSE line plot to: {output}")


def main() -> None:
    args = parse_args()
    codes = _resolve_codes(args)

    df = collect_records(
        args.stage2_dir,
        codes=codes,
        train_sizes=[OFFICIAL_TRAIN_SIZE],
    )
    if df.empty:
        raise SystemExit(
            f"No summary files found for codes={codes} in {args.stage2_dir}. "
            "Ensure s2q*_summary.csv files exist for n=4000."
        )

    df = _filter_official_slice(df)
    df = _filter_official_methods(df)
    if df.empty:
        raise SystemExit(
            "No n="
            f"{OFFICIAL_TRAIN_SIZE} interior-tau RMSE records were found for "
            f"codes={codes} after filtering the official methods."
        )

    output = args.output if args.output is not None else _default_output_path(codes)
    plot_official_rmse_lineplot(
        df,
        codes=codes,
        output=output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
