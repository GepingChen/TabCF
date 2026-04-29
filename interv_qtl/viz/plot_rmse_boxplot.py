#!/usr/bin/env python3
"""
RMSE boxplot for interventional quantile estimators.

Purpose: Visualize method comparison across different quantile levels (tau).

Input data:
- Reads per-tau RMSE from Stage 2 summary CSVs in interv_qtl/IV_datasets/stage2_output
- Expected filename patterns:
    * s2q_{code}_n{n}_seed{seed}_summary.csv       -> TabCF (TabPFN)
    * s2q_tabpfn_real_{code}_n{n}_seed{seed}_summary.csv -> TabCF (Real-TabPFN)
    * s2q_tabicl_{code}_n{n}_seed{seed}_summary.csv -> TabCF (TabICL)
    * s2q_div_{code}_n{n}_seed{seed}_summary.csv   -> DIV
    * s2q_ivqr_{code}_n{n}_seed{seed}_summary.csv  -> IVQR

Output format:
- Single-code mode:
    * one row of subplots, with one subplot per train_size
- Multi-code mode (`--codes`):
    * one outer column of DGP panels
    * one row per DGP
    * within each DGP row, subplots are aligned by train_size

Usage:
  python plot_rmse_boxplot.py --code A3_B3 --train-sizes 1000 4000 10000
  python plot_rmse_boxplot.py --code A3_B3 A3_B4 A3_B5 A3_B9 --train-sizes 4000 --output interv_qtl/figures/rmse_boxplot_A3.pdf
  python plot_rmse_boxplot.py --codes A3_B3 A3_B4 A6_B3
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SUMMARY_FILENAME_RE = re.compile(
    r"^s2q_(?P<body>.+)_n(?P<n>\d+)_seed(?P<seed>\d+)_summary\.csv$"
)
SUMMARY_BODY_RE = re.compile(r"^(?:(?P<method>.+)_)?(?P<code>A\d+_B\d+)$")


def parse_filename(path: Path) -> Optional[Dict]:
    """
    Parse Stage 2 summary filenames to extract metadata.
    
    Recognized patterns:
      - s2q_{code}_n{n}_seed{seed}_summary.csv           -> TabPFN (default)
      - s2q_tabpfn_real_{code}_n{n}_seed{seed}_summary.csv -> Real-TabPFN
      - s2q_tabicl_{code}_n{n}_seed{seed}_summary.csv    -> TabICL
      - s2q_div_{code}_n{n}_seed{seed}_summary.csv       -> DIV
      - s2q_ivqr_{code}_n{n}_seed{seed}_summary.csv      -> IVQR
      - s2q_{method}_{code}_n{n}_seed{seed}_summary.csv  -> generic
    
    Args:
        path: Path to summary CSV file
    
    Returns:
        Dict with keys {method, code, n, seed}, or None if parse fails
    
    Example:
        parse_filename("s2q_div_A3_B3_n1000_seed1_summary.csv")
        -> {"method": "div", "code": "A3_B3", "n": 1000, "seed": 1}
    """
    name = path.name
    match = SUMMARY_FILENAME_RE.match(name)
    if not match:
        return None

    body_match = SUMMARY_BODY_RE.match(match.group("body"))
    if not body_match:
        return None

    try:
        n_val = int(match.group("n"))
        seed_val = int(match.group("seed"))
    except ValueError:
        return None

    method = body_match.group("method") or "tabpfn"
    code = body_match.group("code")
    return {"method": method, "code": code, "n": n_val, "seed": seed_val}

# User-friendly labels for known methods
METHOD_LABELS: Dict[str, str] = {
    "tabpfn": "TabCF (TabPFNv2.5)",
    "tabpfn_real": "TabCF (Real-TabPFNv2.5)",
    "tabicl": "TabCF (TabICLv2)",
    "dopfn": "TabCF",
    "div": "DIV",
    "ivqr": "IVQR",
}
METHOD_ORDER: tuple[str, ...] = ("tabpfn", "tabpfn_real", "tabicl", "dopfn", "div", "ivqr")

# Color palette (fallback colors are generated from a colormap)
METHOD_COLORS: Dict[str, str] = {
    "tabpfn": "#1f77b4",
    "tabpfn_real": "#8c564b",
    "tabicl": "#17becf",
    "dopfn": "#1f77b4",
    "div": "#d62728",
    "ivqr": "#2ca02c",
}

DEFAULT_CODE = "A3_B3"
DEFAULT_TRAIN_SIZES: tuple[int, ...] = (1000, 4000, 10000)
PLOT_TAUS: tuple[float, ...] = (
    0.01,
    0.025,
    0.1,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.9,
    0.975,
    0.99,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RMSE boxplots by tau and method.")
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "IV_datasets" / "stage2_output",
        help="Directory containing s2q*_summary.csv files.",
    )
    parser.add_argument(
        "--code",
        nargs="+",
        default=[DEFAULT_CODE],
        help=(
            "One or more DGP codes to plot. "
            "When multiple codes are provided, the script automatically uses multi-code mode. "
            "Ignored when --codes is provided."
        ),
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        default=None,
        help=(
            "Optional list of DGP codes to combine into one figure. "
            "Accepts space-separated or comma-separated values. "
            "When provided, the output uses one outer column with one row per DGP."
        ),
    )
    parser.add_argument(
        "--train-sizes",
        nargs="*",
        type=int,
        default=list(DEFAULT_TRAIN_SIZES),
        help=(
            "Optional list of train sizes to include. "
            "Defaults to 1000 4000 10000."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path. Defaults to interv_qtl/figures/rmse_boxplot_{code}.png "
            "in single-code mode, or a combined rmse_boxplot_{code1}_{code2}_... .png "
            "file in multi-code mode."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for the saved figure.",
    )
    return parser.parse_args()


def friendly_method(method_key: str) -> str:
    key = method_key.lower()
    return METHOD_LABELS.get(key, method_key)


def color_for_method(method_key: str, colormap="tab10", idx: int = 0) -> str:
    key = method_key.lower()
    if key in METHOD_COLORS:
        return METHOD_COLORS[key]
    cmap = plt.get_cmap(colormap)
    return cmap(idx % cmap.N)


def collect_records(
    stage2_dir: Path, codes: Sequence[str], train_sizes: Optional[Sequence[int]] = None
) -> pd.DataFrame:
    code_set = set(codes)
    records: List[Dict] = []
    for path in stage2_dir.glob("s2q*_*summary.csv"):
        info = parse_filename(path)
        if not info:
            continue
        file_code = info["code"]
        if file_code not in code_set:
            continue
        train_size = int(info["n"])
        if train_sizes and train_size not in train_sizes:
            continue
        method = str(info["method"]).lower()
        seed = int(info["seed"])

        df = pd.read_csv(path)
        if "tau" not in df.columns or "rmse" not in df.columns:
            continue
        for _, row in df.iterrows():
            try:
                tau_val = float(row["tau"])
                rmse_val = float(row["rmse"])
            except Exception:
                continue
            records.append(
                {
                    "code": file_code,
                    "train_size": train_size,
                    "seed": seed,
                    "tau": tau_val,
                    "rmse": rmse_val,
                    "method_key": method,
                    "method": friendly_method(method),
                }
            )
    return pd.DataFrame(records)


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
    if args.codes:
        codes = _normalize_codes(args.codes)
        if not codes:
            raise SystemExit("No valid DGP codes were supplied via --codes.")
        return codes
    codes = _normalize_codes(args.code)
    if not codes:
        raise SystemExit("No valid DGP codes were supplied via --code.")
    return codes


def _resolve_train_sizes(df: pd.DataFrame, requested: Optional[Sequence[int]]) -> List[int]:
    if requested:
        return [int(n) for n in requested]
    return list(DEFAULT_TRAIN_SIZES)


def _resolve_methods_and_colors(df: pd.DataFrame) -> tuple[List[str], Dict[str, str]]:
    method_info = (
        df[["method", "method_key"]]
        .drop_duplicates()
        .assign(method=lambda frame: frame["method"].astype(str), method_key=lambda frame: frame["method_key"].astype(str).str.lower())
    )
    order_rank = {key: idx for idx, key in enumerate(METHOD_ORDER)}
    method_info["order_rank"] = method_info["method_key"].map(lambda key: order_rank.get(key, len(order_rank)))
    method_info = method_info.sort_values(["order_rank", "method"]).reset_index(drop=True)
    methods = method_info["method"].tolist()
    method_colors: Dict[str, str] = {}
    for idx, method in enumerate(methods):
        method_key = str(df[df["method"] == method]["method_key"].iloc[0])
        method_colors[method] = color_for_method(method_key, idx=idx)
    return methods, method_colors


def _legend_patches(methods: Sequence[str], method_colors: Dict[str, str]) -> List[Patch]:
    return [
        Patch(
            facecolor=method_colors.get(method, "#999999"),
            edgecolor="black",
            linewidth=1.0,
            alpha=0.7,
            label=method,
        )
        for method in methods
    ]


def _tau_mask(series: pd.Series, tau: float) -> np.ndarray:
    return np.isclose(series.to_numpy(dtype=float), tau, rtol=0.0, atol=1e-10)


def _missing_plot_taus(df: pd.DataFrame) -> List[float]:
    available = np.asarray(sorted(float(tau) for tau in df["tau"].dropna().unique()), dtype=float)
    return [tau for tau in PLOT_TAUS if not np.any(np.isclose(available, tau, rtol=0.0, atol=1e-10))]


def _warn_if_missing_plot_taus(df: pd.DataFrame) -> None:
    missing = _missing_plot_taus(df)
    if missing:
        formatted = ", ".join(f"{tau:.3g}" for tau in missing)
        print(
            "Warning: the loaded stage2 summaries do not contain all requested plot taus. "
            f"Missing taus: [{formatted}]. Empty x positions will be shown for them. "
            "To populate those positions with boxplots, regenerate the summaries using the updated tau defaults."
        )


def _boxplot_grouped(
    ax,
    df: pd.DataFrame,
    methods: List[str],
    taus: List[float],
    method_colors: Dict[str, str],
    title: str,
    *,
    show_legend: bool = True,
):
    base_positions = np.arange(len(taus), dtype=float)
    n_methods = len(methods)
    offset = 0.2
    width = 0.16
    legend_handles = []

    for j, method in enumerate(methods):
        pos_offset = (j - (n_methods - 1) / 2.0) * offset
        data_list: List[List[float]] = []
        pos_list: List[float] = []
        for i, tau in enumerate(taus):
            tau_mask = _tau_mask(df["tau"], tau)
            vals = df[(df["method"] == method) & tau_mask]["rmse"].dropna().tolist()
            if len(vals) == 0:
                continue
            data_list.append(vals)
            pos_list.append(base_positions[i] + pos_offset)
        if not data_list:
            continue
        bp = ax.boxplot(
            data_list,
            positions=pos_list,
            widths=width,
            patch_artist=True,
            showfliers=False,
            boxprops={
                "facecolor": method_colors.get(method, "#999999"),
                "edgecolor": "black",
                "linewidth": 1.0,
                "alpha": 0.7,
            },
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
        )
        legend_handles.append((bp["boxes"][0], method))

    ax.set_xticks(base_positions)
    ax.set_xticklabels([f"{tau:.3g}" for tau in taus], rotation=45, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Build legend from handles (avoid duplicates)
    if show_legend and legend_handles:
        seen = set()
        handles = []
        labels = []
        for handle, label in legend_handles:
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
        ax.legend(handles, labels, title="Method", loc="upper right")


def _select_plot_taus(taus: Sequence[float]) -> List[float]:
    del taus
    return list(PLOT_TAUS)


def _draw_empty_panel(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_ylabel("RMSE")
    ax.set_xticks([])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10)


def _add_bottom_label_and_legend(
    fig: plt.Figure,
    methods: Sequence[str],
    method_colors: Dict[str, str],
) -> None:
    fig.text(0.5, 0.085, "tau", ha="center", va="center")
    legend_handles = _legend_patches(methods, method_colors)
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            labels=[handle.get_label() for handle in legend_handles],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=max(1, len(legend_handles)),
            frameon=False,
        )


def plot_rmse_boxplot_single(
    df: pd.DataFrame,
    code: str,
    output: Path,
    dpi: int = 200,
    train_sizes: Optional[Sequence[int]] = None,
) -> None:
    train_sizes_resolved = _resolve_train_sizes(df, train_sizes)
    methods, method_colors = _resolve_methods_and_colors(df)
    _warn_if_missing_plot_taus(df)

    n_cols = len(train_sizes_resolved)
    figure_width = 1.2 * (1.2 * max(4 * n_cols, 5))
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(figure_width, 4.5),
        sharey=True if n_cols > 1 else False,
    )
    if n_cols == 1:
        axes = [axes]  # type: ignore

    for ax, n in zip(axes, train_sizes_resolved):
        sub = df[df["train_size"] == n]
        if sub.empty:
            _draw_empty_panel(ax, title=f"{code} (n={n})")
            continue
        taus = _select_plot_taus(sub["tau"].unique())
        _boxplot_grouped(
            ax,
            sub,
            methods=methods,
            taus=taus,
            method_colors=method_colors,
            title=f"{code} (n={n})",
            show_legend=False,
        )

    _add_bottom_label_and_legend(fig, methods, method_colors)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"Saved RMSE boxplot to: {output}")


def plot_rmse_boxplot_multi(
    df: pd.DataFrame,
    codes: Sequence[str],
    output: Path,
    dpi: int = 200,
    train_sizes: Optional[Sequence[int]] = None,
) -> None:
    train_sizes_resolved = _resolve_train_sizes(df, train_sizes)
    methods, method_colors = _resolve_methods_and_colors(df)
    codes_resolved = [code for code in codes if code in set(df["code"].unique())]
    if not codes_resolved:
        raise SystemExit("No DGP records available for the requested codes.")
    _warn_if_missing_plot_taus(df)

    n_rows = len(codes_resolved)
    n_cols = len(train_sizes_resolved)
    figure_width = 1.2 * (1.2 * max(4 * n_cols, 5))
    fig = plt.figure(figsize=(figure_width, max(3.6 * n_rows, 4.5)))
    outer = fig.add_gridspec(n_rows, 1, hspace=0.35)

    for row_idx, code in enumerate(codes_resolved):
        code_df = df[df["code"] == code]
        inner = outer[row_idx].subgridspec(1, n_cols, wspace=0.18)
        row_shared_y_ax = None

        for col_idx, n in enumerate(train_sizes_resolved):
            share_target = row_shared_y_ax if row_shared_y_ax is not None else None
            ax = fig.add_subplot(inner[0, col_idx], sharey=share_target)
            if row_shared_y_ax is None:
                row_shared_y_ax = ax

            sub = code_df[code_df["train_size"] == n]
            title = f"{code} (n={n})"
            if sub.empty:
                _draw_empty_panel(ax, title=title)
            else:
                taus = _select_plot_taus(sub["tau"].unique())
                _boxplot_grouped(
                    ax,
                    sub,
                    methods=methods,
                    taus=taus,
                    method_colors=method_colors,
                    title=title,
                    show_legend=False,
                )

            if col_idx > 0:
                ax.set_ylabel("")

    _add_bottom_label_and_legend(fig, methods, method_colors)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"Saved RMSE boxplot to: {output}")


def _default_output_path(codes: Sequence[str]) -> Path:
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    if len(codes) == 1:
        filename = f"rmse_boxplot_{codes[0]}.png"
    else:
        suffix = "_".join(codes)
        if len(suffix) > 80:
            suffix = f"selected_{len(codes)}dgps"
        filename = f"rmse_boxplot_{suffix}.png"
    return figures_dir / filename


def main() -> None:
    args = parse_args()
    stage2_dir: Path = args.stage2_dir
    codes = _resolve_codes(args)
    df = collect_records(stage2_dir, codes=codes, train_sizes=args.train_sizes)
    if df.empty:
        raise SystemExit(
            f"No summary files found for codes={codes} in {stage2_dir}. "
            "Ensure s2q*_summary.csv files exist."
        )
    found_codes = set(str(code) for code in df["code"].unique())
    missing_codes = [code for code in codes if code not in found_codes]
    if missing_codes:
        raise SystemExit(
            f"No summary files found for requested code(s): {missing_codes} in {stage2_dir}."
        )

    output = args.output if args.output is not None else _default_output_path(codes)
    if len(codes) == 1:
        plot_rmse_boxplot_single(
            df,
            code=codes[0],
            output=output,
            dpi=args.dpi,
            train_sizes=args.train_sizes,
        )
    else:
        plot_rmse_boxplot_multi(
            df,
            codes=codes,
            output=output,
            dpi=args.dpi,
            train_sizes=args.train_sizes,
        )


if __name__ == "__main__":
    main()
