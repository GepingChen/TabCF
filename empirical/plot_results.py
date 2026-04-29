#!/usr/bin/env python3
"""Create the official 2x2 real-data visualization for the paper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from empirical import run_empirical_mean as pipeline


DEFAULT_OUTPUT_STEM = pipeline.REPO_ROOT / "empirical" / "official_plots" / "empirical_comparison_grid"
PLOT_FONT_FAMILY = "Latin Modern Roman"
FONT_SCALE = 1.42


def _register_latin_modern_fonts() -> None:
    tectonic_cache = Path.home() / ".cache" / "Tectonic" / "bundles" / "data"
    if not tectonic_cache.exists():
        return
    for font_path in sorted(tectonic_cache.glob("*/lmroman*.otf")):
        font_manager.fontManager.addfont(str(font_path))


def _apply_official_plot_style() -> None:
    _register_latin_modern_fonts()
    font_path = font_manager.findfont(PLOT_FONT_FAMILY, fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.serif": [PLOT_FONT_FAMILY],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 11.0 * FONT_SCALE,
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


def _style_data_panel(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, axis="y", color="#d9ddd6", linewidth=0.8, alpha=0.95)
    ax.grid(True, axis="x", color="#eceee9", linewidth=0.75, alpha=0.95)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9.8 * FONT_SCALE, length=0, pad=6, colors="#4a4a46")
    ax.tick_params(axis="x", labelsize=9.8 * FONT_SCALE, length=0, pad=6, colors="#4a4a46")
    for spine in ax.spines.values():
        spine.set_visible(False)


@dataclass(frozen=True)
class OfficialTaskSpec:
    dataset_key: str
    title: str
    out_dir: Path

    @property
    def dataset_spec(self) -> pipeline.DatasetSpec:
        return pipeline.DATASET_SPECS[self.dataset_key]


@dataclass(frozen=True)
class LoadedPanel:
    task: OfficialTaskSpec
    clean_df: pd.DataFrame
    preds_df: pd.DataFrame


OFFICIAL_TASKS = (
    OfficialTaskSpec("ajr", "AJR colonial origins", pipeline.DEFAULT_AJR_OUTPUT_DIR),
    OfficialTaskSpec("fulton", "Fulton Fish Market", pipeline.DEFAULT_FULTON_OUTPUT_DIR),
    OfficialTaskSpec("card", "Card college proximity", pipeline.DEFAULT_CARD_OUTPUT_DIR),
    OfficialTaskSpec("cigarettes", "CigarettesSW", pipeline.DEFAULT_CIGARETTES_OUTPUT_DIR),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ajr-dir", type=Path, default=pipeline.DEFAULT_AJR_OUTPUT_DIR)
    parser.add_argument("--fulton-dir", type=Path, default=pipeline.DEFAULT_FULTON_OUTPUT_DIR)
    parser.add_argument("--card-dir", type=Path, default=pipeline.DEFAULT_CARD_OUTPUT_DIR)
    parser.add_argument("--cigarettes-dir", type=Path, default=pipeline.DEFAULT_CIGARETTES_OUTPUT_DIR)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def required_prediction_columns() -> tuple[str, ...]:
    return (
        "X",
        "ols_pred",
        "tsls_pred",
        "div_pred",
    )


def _resolve_tabcf_column(preds_df: pd.DataFrame) -> pd.DataFrame:
    preferred = pipeline.prediction_column_for_backend(pipeline.TABPFN_BACKEND)
    if preferred in preds_df.columns:
        return preds_df
    legacy = "tabcf_pred"
    if legacy in preds_df.columns:
        resolved = preds_df.copy()
        resolved[preferred] = resolved[legacy]
        return resolved
    raise ValueError(
        "Predictions CSV is missing the TabCF curve column. Expected either "
        f"'{preferred}' or legacy '{legacy}'."
    )


def resolve_tasks(args: argparse.Namespace) -> tuple[OfficialTaskSpec, ...]:
    return (
        OfficialTaskSpec("ajr", "AJR colonial origins", args.ajr_dir.resolve()),
        OfficialTaskSpec("fulton", "Fulton Fish Market", args.fulton_dir.resolve()),
        OfficialTaskSpec("card", "Card college proximity", args.card_dir.resolve()),
        OfficialTaskSpec("cigarettes", "CigarettesSW", args.cigarettes_dir.resolve()),
    )


def load_panel(task: OfficialTaskSpec) -> LoadedPanel:
    output_paths = pipeline.build_output_paths(task.dataset_spec, task.out_dir)
    clean_csv = output_paths["clean_csv"]
    merged_csv = output_paths["merged_pred_csv"]

    if not clean_csv.exists():
        raise FileNotFoundError(
            f"Missing clean CSV for {task.title}: {clean_csv}. "
            "Run empirical/run_empirical_mean.py for this dataset first."
        )
    if not merged_csv.exists():
        raise FileNotFoundError(
            f"Missing predictions CSV for {task.title}: {merged_csv}. "
            "Run empirical/run_empirical_mean.py for this dataset first."
        )

    clean_df = pd.read_csv(clean_csv)
    preds_df = pd.read_csv(merged_csv).sort_values("X", kind="mergesort").reset_index(drop=True)

    preds_df = _resolve_tabcf_column(preds_df)
    required = required_prediction_columns() + (
        pipeline.prediction_column_for_backend(pipeline.TABPFN_BACKEND),
    )
    missing = [column for column in required if column not in preds_df.columns]
    if missing:
        raise ValueError(
            f"Predictions CSV for {task.title} is missing required columns {missing}: {merged_csv}. "
            "Regenerate outputs with the real-data pipeline."
        )

    return LoadedPanel(task=task, clean_df=clean_df, preds_df=preds_df)


def load_panels(tasks: tuple[OfficialTaskSpec, ...]) -> tuple[LoadedPanel, ...]:
    return tuple(load_panel(task) for task in tasks)


def make_figure(panels: tuple[LoadedPanel, ...]) -> tuple[plt.Figure, list[plt.Axes], plt.Legend]:
    core_method_specs = pipeline.resolve_core_method_specs(
        [pipeline.TABPFN_BACKEND],
        legacy_core_label="TabCF (TabPFNv2.5)",
    )

    _apply_official_plot_style()

    fig, axes_arr = plt.subplots(2, 2, figsize=(15.0, 11.0), facecolor="white")
    axes = list(axes_arr.reshape(-1))
    legend_handles = []
    legend_labels: list[str] = []

    for index, (ax, panel) in enumerate(zip(axes, panels)):
        spec = panel.task.dataset_spec
        preds = panel.preds_df
        clean = panel.clean_df
        line_kwargs = {"marker": "o", "markersize": 4.0} if spec.discrete_x else {}

        ax.scatter(clean["X"], clean["Y"], color="grey", alpha=0.5, s=28, linewidths=0, label="_scatter")
        panel_handles = []
        panel_labels = []

        for method_spec in core_method_specs:
            (line,) = ax.plot(
                preds["X"],
                preds[method_spec.pred_column],
                color=method_spec.color,
                linewidth=2.2,
                label=method_spec.label,
                **line_kwargs,
            )
            panel_handles.append(line)
            panel_labels.append(method_spec.label)

        for label, column, color, width in (
            ("DIV", "div_pred", pipeline.BASELINE_COLORS["DIV"], 2.2),
            ("2SLS", "tsls_pred", pipeline.BASELINE_COLORS["2SLS"], 2.0),
            ("OLS", "ols_pred", pipeline.BASELINE_COLORS["OLS"], 2.0),
        ):
            (line,) = ax.plot(preds["X"], preds[column], color=color, linewidth=width, label=label, **line_kwargs)
            panel_handles.append(line)
            panel_labels.append(label)

        if index == 0:
            legend_handles = panel_handles
            legend_labels = panel_labels

        ax.set_title(panel.task.title, fontsize=13.0 * FONT_SCALE, fontweight="bold")
        ax.set_xlabel(spec.x_label)
        ax.set_ylabel(spec.y_label)
        ax.set_xlim(float(preds["X"].min()), float(preds["X"].max()))
        if spec.discrete_x:
            ax.set_xticks(preds["X"].to_numpy(dtype=float))
        _style_data_panel(ax)

    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.978),
        frameon=True,
        fancybox=False,
        edgecolor="#5d5d57",
        columnspacing=1.1,
        handlelength=2.4,
        borderpad=0.72,
    )
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_linewidth(0.85)
    legend.get_frame().set_facecolor("#ffffff")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return fig, axes, legend


def save_figure(fig: plt.Figure, output_stem: Path) -> tuple[Path, Path]:
    output_stem = output_stem.resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    tasks = resolve_tasks(args)
    panels = load_panels(tasks)
    fig, _axes, _legend = make_figure(panels)
    png_path, pdf_path = save_figure(fig, args.output_stem)
    plt.close(fig)
    print(f"Saved official PNG to: {png_path}")
    print(f"Saved official PDF to: {pdf_path}")


if __name__ == "__main__":
    main()
